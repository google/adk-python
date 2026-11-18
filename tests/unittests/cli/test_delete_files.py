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

"""Tests that delete_files is gated by framework human confirmation."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.cli.built_in_agents.adk_agent_builder_assistant import AgentBuilderAssistant
from google.adk.sessions.session import Session
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext


def _tool_context(
    root: Path,
    tool_confirmation: ToolConfirmation | None = None,
) -> ToolContext:
  invocation_context = mock.MagicMock(spec=InvocationContext)
  invocation_context._state_schema = None
  invocation_context.session = mock.MagicMock(spec=Session)
  invocation_context.session.state = {"root_directory": str(root)}
  return ToolContext(
      invocation_context=invocation_context,
      function_call_id="call-1",
      tool_confirmation=tool_confirmation,
  )


def _delete_files_tool():
  agent = AgentBuilderAssistant.create_agent(model="gemini-2.0-flash")
  return next(tool for tool in agent.tools if tool.name == "delete_files")


async def test_delete_files_is_registered_with_require_confirmation(tmp_path):
  tool = _delete_files_tool()

  assert await tool.check_require_confirmation({}, _tool_context(tmp_path))


async def test_model_supplied_argument_cannot_bypass_confirmation(tmp_path):
  victim = tmp_path / "orphan.py"
  victim.write_text("x")
  tool = _delete_files_tool()

  result = await tool.run_async(
      args={
          "file_paths": [str(victim)],
          # A stale argument from the old signature must not act as approval.
          "confirm_deletion": True,
      },
      tool_context=_tool_context(tmp_path),
  )

  assert result == {
      "error": "This tool call requires confirmation, please approve or reject."
  }
  assert victim.exists()


async def test_approved_call_deletes_the_file(tmp_path):
  victim = tmp_path / "orphan.py"
  victim.write_text("x")
  tool = _delete_files_tool()

  result = await tool.run_async(
      args={"file_paths": [str(victim)]},
      tool_context=_tool_context(tmp_path, ToolConfirmation(confirmed=True)),
  )

  assert result["success"]
  assert not victim.exists()
