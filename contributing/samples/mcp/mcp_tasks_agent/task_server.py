# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An MCP server that answers a slow tool call with a task.

Servers implementing the Tasks extension are still scarce, so this one exists
to have something to point `enable_tasks=True` at. It keeps a slow tool that
takes longer than any sensible HTTP write timeout, hands back a task handle
instead of blocking, and serves `tasks/get` and `tasks/cancel` against an
in-memory store.

Run it with:

  python contributing/samples/mcp/mcp_tasks_agent/task_server.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import datetime
from typing import Any
from typing import Literal
import uuid

from mcp.server.extension import Extension
from mcp.server.extension import MethodBinding
from mcp.server.mcpserver import MCPServer
import mcp_types

TASKS_EXTENSION_ID = "io.modelcontextprotocol/tasks"

# Long enough that holding the request open would be the wrong answer, short
# enough to watch.
_WORK_DURATION_SECONDS = 20.0
_POLL_INTERVAL_MS = 2000
_TTL_MS = 300000


def _now() -> str:
  return (
      datetime.datetime.now(datetime.timezone.utc)
      .isoformat(timespec="seconds")
      .replace("+00:00", "Z")
  )


@dataclass
class _Task:
  """One tracked operation, and whatever it has produced so far."""

  task_id: str
  status: str = "working"
  status_message: str | None = None
  created_at: str = field(default_factory=_now)
  last_updated_at: str = field(default_factory=_now)
  result: dict[str, Any] | None = None
  error: dict[str, Any] | None = None
  worker: asyncio.Task[None] | None = None

  def to_wire(self) -> dict[str, Any]:
    wire: dict[str, Any] = {
        "taskId": self.task_id,
        "status": self.status,
        "createdAt": self.created_at,
        "lastUpdatedAt": self.last_updated_at,
        "ttlMs": _TTL_MS,
        "pollIntervalMs": _POLL_INTERVAL_MS,
    }
    if self.status_message is not None:
      wire["statusMessage"] = self.status_message
    if self.result is not None:
      wire["result"] = self.result
    if self.error is not None:
      wire["error"] = self.error
    return wire


class _TaskParams(mcp_types.RequestParams):
  task_id: str


class TasksExtension(Extension):
  """Serves the Tasks extension over an in-memory task store."""

  identifier = TASKS_EXTENSION_ID

  def __init__(self) -> None:
    self._tasks: dict[str, _Task] = {}

  def methods(self):
    return (
        MethodBinding(
            method="tasks/get",
            params_type=_TaskParams,
            handler=self._handle_get,
        ),
        MethodBinding(
            method="tasks/cancel",
            params_type=_TaskParams,
            handler=self._handle_cancel,
        ),
    )

  async def _handle_get(self, ctx: Any, params: _TaskParams) -> dict[str, Any]:
    del ctx
    task = self._tasks.get(params.task_id)
    if task is None:
      raise mcp_types.MCPError(
          code=mcp_types.jsonrpc.INVALID_PARAMS,
          message=f"unknown taskId {params.task_id!r}",
      )
    return task.to_wire()

  async def _handle_cancel(
      self, ctx: Any, params: _TaskParams
  ) -> dict[str, Any]:
    del ctx
    task = self._tasks.get(params.task_id)
    if task is not None and task.worker is not None:
      task.worker.cancel()
    return {}

  async def intercept_tool_call(self, params, ctx, call_next):
    """Turns a call to the slow tool into a task, when the client can take one.

    A server must not hand a task to a client that did not advertise the
    extension on this very request, so anything else falls through to the
    ordinary handler.
    """
    if params.name != "slow_operation" or not _client_supports_tasks(ctx):
      return await call_next(ctx)

    task = _Task(task_id=str(uuid.uuid4()))
    task.status_message = "The operation is now in progress."
    self._tasks[task.task_id] = task
    task.worker = asyncio.create_task(self._run(task, params))

    return {"resultType": "task", **task.to_wire()}

  async def _run(self, task: _Task, params: Any) -> None:
    """Does the slow work, then leaves the outcome on the task."""
    label = (params.arguments or {}).get("label", "the job")
    try:
      await asyncio.sleep(_WORK_DURATION_SECONDS)
    except asyncio.CancelledError:
      task.status = "cancelled"
      task.last_updated_at = _now()
      raise
    except Exception as e:  # pylint: disable=broad-except
      task.status = "failed"
      task.error = {"code": -32000, "message": str(e)}
      task.last_updated_at = _now()
      return
    task.status = "completed"
    task.status_message = "Done."
    message = f"Finished {label} after {int(_WORK_DURATION_SECONDS)} seconds."
    # A task's result is an ordinary tool result and is held to the tool's
    # output schema, so it needs the structured half too -- the client
    # validates it exactly as it would an inline answer.
    task.result = {
        "content": [{"type": "text", "text": message}],
        "structuredContent": {"result": message},
    }
    task.last_updated_at = _now()


def _client_supports_tasks(ctx: Any) -> bool:
  """Whether this request advertised the tasks extension."""
  meta = getattr(ctx, "meta", None) or {}
  capabilities = meta.get("io.modelcontextprotocol/clientCapabilities") or {}
  extensions = capabilities.get("extensions") or {}
  return TASKS_EXTENSION_ID in extensions


server = MCPServer(
    name="Task Server",
    instructions="Runs an operation that takes longer than one request.",
    extensions=[TasksExtension()],
)


@server.tool()
async def slow_operation(label: str = "the job") -> str:
  """Runs a long operation. Served as a task when the client supports it.

  Args:
    label: What to call the job in the final message.
  """
  # Reached only by a client without the tasks extension, which has no choice
  # but to hold the request open.
  await asyncio.sleep(_WORK_DURATION_SECONDS)
  return f"Finished {label} after {int(_WORK_DURATION_SECONDS)} seconds."


if __name__ == "__main__":
  print("Task server listening on http://localhost:3000/mcp")
  server.run(transport="streamable-http", host="localhost", port=3000)
