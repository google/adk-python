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

"""Client support for the MCP Tasks extension.

A server that expects an operation to be slow may answer `tools/call` with a
task handle instead of a result. The work then outlives the request: the
client polls `tasks/get` until the task reaches a terminal state and reads
the result from there. That is what makes an operation survive a dropped
connection, and what removes the need to split one tool into a start/status/
result triplet driven by the model.

The wire models live here because no Python implementation of the extension
exists yet -- neither the SDK nor a published package -- and the task types
the SDK does carry are the older, wire-incompatible design that predates it.
Everything in this module is private so it can be replaced by an upstream
implementation without changing ADK's own surface.

MCP SDK 2.x only, and unimportable on 1.x: `ClaimContext` and `ResultClaim`
have no 1.x counterpart. `McpToolset` refuses `enable_tasks=True` there, and
imports this module only after that check.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import Literal

from ...dependencies._mcp import CallToolResult
from ...dependencies._mcp import ClaimContext
from ...dependencies._mcp import McpError
from ...dependencies._mcp import ResultClaim
from ...dependencies._mcp import types as mcp_types

logger = logging.getLogger('google_adk.' + __name__)

TASKS_EXTENSION_ID = 'io.modelcontextprotocol/tasks'

# A task in one of these states will not change again, so polling stops.
_TERMINAL_STATUSES = frozenset({'completed', 'failed', 'cancelled'})

# What to wait between polls when the server states no preference. Tasks exist
# for minute-scale work, so a second costs nothing and keeps idle load low.
_DEFAULT_POLL_INTERVAL_SECONDS = 1.0

# The floor stops a server that reports 0 from turning the loop into a spin.
# The ceiling bounds how long a finished task can go unnoticed; it overrides
# the server's preference, but only in the direction of noticing sooner.
_MIN_POLL_INTERVAL_SECONDS = 0.1
_MAX_POLL_INTERVAL_SECONDS = 30.0

# Cancellation is best-effort by specification, and it runs while the caller
# is already being torn down, so it gets one short round trip and no more.
_CANCEL_TIMEOUT_SECONDS = 2.0

# How often a still-running task says so, counted in polls. Without it a task
# that never finishes is indistinguishable from a client that stopped asking.
_PROGRESS_LOG_EVERY_N_POLLS = 30


class _TaskResult(mcp_types.Result):
  """The `resultType: "task"` answer to `tools/call`.

  Flat by specification: the task's fields sit alongside `resultType` rather
  than under a nested object. Field names are snake_case with the wire
  spelling supplied by the base model's alias generator.
  """

  result_type: Literal['task'] = 'task'
  task_id: str
  status: mcp_types.TaskStatus
  status_message: str | None = None
  created_at: str
  last_updated_at: str
  ttl_ms: int | None
  poll_interval_ms: int | None = None


class _TaskParams(mcp_types.RequestParams):
  """Parameters shared by every `tasks/*` request."""

  task_id: str


class _GetTaskRequest(mcp_types.Request[_TaskParams, Literal['tasks/get']]):
  """Reads the current state of a task.

  `name_param` is what puts the task id in the `Mcp-Name` header, which the
  specification requires on `tasks/*` over streamable HTTP so an intermediary
  can route the request to whoever holds the task.
  """

  method: Literal['tasks/get'] = 'tasks/get'
  params: _TaskParams
  name_param = 'taskId'


class _CancelTaskRequest(
    mcp_types.Request[_TaskParams, Literal['tasks/cancel']]
):
  """Asks the server to stop working on a task."""

  method: Literal['tasks/cancel'] = 'tasks/cancel'
  params: _TaskParams
  name_param = 'taskId'


class _GetTaskResult(mcp_types.Result):
  """The state of a task, plus whatever its current state carries.

  Only one of `result`, `error` and `input_requests` is ever populated, and
  which one follows from `status`.
  """

  task_id: str
  status: mcp_types.TaskStatus
  status_message: str | None = None
  created_at: str
  last_updated_at: str
  ttl_ms: int | None = None
  poll_interval_ms: int | None = None
  result: dict[str, Any] | None = None
  error: dict[str, Any] | None = None
  input_requests: dict[str, Any] | None = None


class _EmptyResult(mcp_types.Result):
  """An acknowledgement with no payload, as `tasks/cancel` returns."""


def _error_result(message: str) -> CallToolResult:
  """Builds the failed tool result the agent will see."""
  return CallToolResult(
      content=[mcp_types.TextContent(type='text', text=message)], is_error=True
  )


def _poll_interval_seconds(poll_interval_ms: int | None) -> float:
  """Turns the server's stated interval into one worth sleeping for."""
  if poll_interval_ms is None:
    return _DEFAULT_POLL_INTERVAL_SECONDS
  return min(
      max(poll_interval_ms / 1000.0, _MIN_POLL_INTERVAL_SECONDS),
      _MAX_POLL_INTERVAL_SECONDS,
  )


def _describe_error(error: dict[str, Any] | None) -> str:
  """Renders a JSON-RPC error object for a human reading a tool result."""
  if not error:
    return 'no error detail was provided'
  code, message = error.get('code'), error.get('message')
  if code is None and message is None:
    return str(error)
  return f'{message or "unknown error"} (code {code})'


async def _cancel_task(ctx: ClaimContext, task_id: str) -> None:
  """Tells the server to drop a task, without ever raising.

  Called from a cancellation path, so it must neither fail nor hang: whatever
  goes wrong here, the caller's own cancellation is what matters.
  """
  try:
    await asyncio.wait_for(
        asyncio.shield(
            ctx.session.send_request(
                _CancelTaskRequest(params=_TaskParams(task_id=task_id)),
                _EmptyResult,
            )
        ),
        timeout=_CANCEL_TIMEOUT_SECONDS,
    )
  except (Exception, asyncio.CancelledError) as e:  # pylint: disable=broad-except
    logger.debug('Best-effort tasks/cancel for %s did not land: %s', task_id, e)


def _completed_to_result(task: _GetTaskResult) -> CallToolResult:
  """Reads the tool's result out of a completed task."""
  if task.result is None:
    return _error_result(f'MCP task {task.task_id} completed without a result.')
  try:
    return CallToolResult.model_validate(task.result)
  except Exception as e:  # pylint: disable=broad-except
    # The task did finish; it is the payload that is unusable. Report that as
    # a failed tool call rather than letting a validation error escape a
    # resolver the caller has no way to guard.
    return _error_result(
        f'MCP task {task.task_id} returned a result that is not a valid'
        f' tool result: {e}'
    )


async def _resolve_task(
    claimed: _TaskResult, ctx: ClaimContext
) -> CallToolResult:
  """Polls a task to completion and returns what the tool call produced.

  Blocks until the task reaches a terminal state, so the agent sees the same
  result it would have seen from a tool that answered inline.

  A task that fails, is cancelled by the server, or asks for input it cannot
  be given comes back as a failed tool result rather than an exception: a
  tool that fails reports `isError`, and going through a task should not
  change that.
  """
  task_id = claimed.task_id
  status: str = claimed.status
  interval = _poll_interval_seconds(claimed.poll_interval_ms)
  polls = 0

  try:
    while True:
      # A task that is already terminal still has to be read once: the result
      # only ever travels on tasks/get.
      if status not in _TERMINAL_STATUSES:
        await asyncio.sleep(interval)

      polls += 1
      task = await ctx.session.send_request(
          _GetTaskRequest(params=_TaskParams(task_id=task_id)),
          _GetTaskResult,
          request_read_timeout_seconds=ctx.read_timeout_seconds,
      )
      status = task.status
      interval = _poll_interval_seconds(task.poll_interval_ms)

      if status == 'completed':
        return _completed_to_result(task)
      if status == 'failed':
        return _error_result(
            f'MCP task {task_id} failed: {_describe_error(task.error)}'
        )
      if status == 'cancelled':
        return _error_result(f'MCP task {task_id} was cancelled by the server.')
      if status == 'input_required':
        # Answering would mean bridging tasks/update to an interaction
        # channel, which this does not do yet. Release the task rather than
        # leaving the server holding it until its ttl runs out.
        await _cancel_task(ctx, task_id)
        return _error_result(
            f'MCP task {task_id} requires additional input, which is not'
            ' supported yet.'
        )

      if polls % _PROGRESS_LOG_EVERY_N_POLLS == 0:
        logger.debug(
            'MCP task %s still %s after %d polls', task_id, status, polls
        )
  except asyncio.CancelledError:
    await _cancel_task(ctx, task_id)
    raise
  except McpError as e:
    # An expired or forgotten task answers with an error. Retrying would not
    # help: the handle is what is gone.
    return _error_result(
        f'MCP task {task_id} could not be read (last known status'
        f' {status!r}): {e}'
    )


def make_tasks_claim() -> ResultClaim[_TaskResult]:
  """Builds the claim that turns a task handle into a tool result."""
  return ResultClaim(
      result_type='task', model=_TaskResult, resolve=_resolve_task
  )
