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

"""Agent-callable tools for reading/writing the scratchpad."""

from __future__ import annotations

from typing import Any
from typing import Optional

from google.genai import types
from typing_extensions import override

from .base_tool import BaseTool
from .tool_context import ToolContext


def _get_db_memory_service(tool_context: ToolContext):
  """Return the DatabaseMemoryService from the invocation context, or raise."""
  # Import here to avoid circular imports at module load time.
  # pylint: disable=g-import-not-at-top
  from ..memory.database_memory_service import DatabaseMemoryService

  svc = tool_context._invocation_context.memory_service
  if not isinstance(svc, DatabaseMemoryService):
    raise ValueError(
        "Scratchpad tools require the agent's memory_service to be a "
        f"DatabaseMemoryService, got: {type(svc).__name__}"
    )
  return svc


def _session_scope(tool_context: ToolContext) -> tuple[str, str, str]:
  """Return (app_name, user_id, session_id) from the invocation context."""
  ic = tool_context._invocation_context
  return ic.app_name, ic.session.user_id, ic.session.id


class ScratchpadGetTool(BaseTool):
  """Read a value from the agent scratchpad KV store."""

  def __init__(self):
    super().__init__(
        name="scratchpad_get",
        description=(
            "Read a value stored in the scratchpad KV store by key."
            " Returns null if the key does not exist."
        ),
    )

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "key": types.Schema(
                    type=types.Type.STRING,
                    description="The key to read.",
                ),
            },
            required=["key"],
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    svc = _get_db_memory_service(tool_context)
    app_name, user_id, session_id = _session_scope(tool_context)
    return await svc.get_scratchpad(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        key=args["key"],
    )


class ScratchpadSetTool(BaseTool):
  """Write a value to the agent scratchpad KV store."""

  def __init__(self):
    super().__init__(
        name="scratchpad_set",
        description=(
            "Write a value to the scratchpad KV store. "
            "Overwrites any existing value for the same key."
        ),
    )

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "key": types.Schema(
                    type=types.Type.STRING,
                    description="The key to write.",
                ),
                "value": types.Schema(
                    description=(
                        "The value to store (any JSON-serialisable type)."
                    ),
                ),
            },
            required=["key", "value"],
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> str:
    svc = _get_db_memory_service(tool_context)
    app_name, user_id, session_id = _session_scope(tool_context)
    await svc.set_scratchpad(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        key=args["key"],
        value=args["value"],
    )
    return "ok"


class ScratchpadAppendLogTool(BaseTool):
  """Append an observation or note to the agent scratchpad log."""

  def __init__(self):
    super().__init__(
        name="scratchpad_append_log",
        description=(
            "Append a text observation or note to the scratchpad log. "
            "Entries are stored in insertion order and can be filtered by tag."
        ),
    )

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "content": types.Schema(
                    type=types.Type.STRING,
                    description="The text content to log.",
                ),
                "tag": types.Schema(
                    type=types.Type.STRING,
                    description="Optional category label for filtering.",
                ),
            },
            required=["content"],
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> str:
    svc = _get_db_memory_service(tool_context)
    app_name, user_id, session_id = _session_scope(tool_context)
    await svc.append_log(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        content=args["content"],
        tag=args.get("tag"),
        agent_name=tool_context.agent_name,
    )
    return "ok"


class ScratchpadGetLogTool(BaseTool):
  """Read recent entries from the agent scratchpad log."""

  def __init__(self):
    super().__init__(
        name="scratchpad_get_log",
        description=(
            "Read recent entries from the scratchpad log, "
            "optionally filtered by tag."
        ),
    )

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tag": types.Schema(
                    type=types.Type.STRING,
                    description="Optional category label to filter by.",
                ),
                "limit": types.Schema(
                    type=types.Type.INTEGER,
                    description=(
                        "Maximum number of entries to return (default 50)."
                    ),
                ),
            },
        ),
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> list[dict]:
    svc = _get_db_memory_service(tool_context)
    app_name, user_id, session_id = _session_scope(tool_context)
    return await svc.get_log(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        tag=args.get("tag"),
        limit=int(args.get("limit", 50)),
    )


# Ready-to-use singleton instances
scratchpad_get_tool = ScratchpadGetTool()
scratchpad_set_tool = ScratchpadSetTool()
scratchpad_append_log_tool = ScratchpadAppendLogTool()
scratchpad_get_log_tool = ScratchpadGetLogTool()
