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

from __future__ import annotations

from unittest import mock

from google.adk.tools import bash_tool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
import pytest


@pytest.mark.asyncio
async def test_execute_bash_tool_reports_unsupported_platform(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
  """A confirmed Bash command returns a clear error on non-POSIX hosts."""
  tool = bash_tool.ExecuteBashTool(workspace=tmp_path)
  tool_context = mock.create_autospec(ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = True
  tool_context.tool_confirmation = confirmation
  monkeypatch.setattr(bash_tool.os, "name", "nt")

  result = await tool.run_async(
      args={"command": "echo hello"}, tool_context=tool_context
  )

  assert result == {
      "error": "ExecuteBashTool is only supported on POSIX systems."
  }
