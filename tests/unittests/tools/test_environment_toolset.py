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

"""Tests for EnvironmentToolset and configurable output limits."""

from pathlib import Path
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.tools.environment._environment_toolset import EnvironmentToolset
from google.adk.tools.tool_context import ToolContext
import pytest
import pytest_asyncio


class _FakeEnvironment(BaseEnvironment):
  """Fake environment to return customized execution and read results."""

  def __init__(self, *, stdout: str, file_content: bytes):
    self._stdout = stdout
    self._file_content = file_content

  @property
  def working_dir(self) -> Path:
    return Path("/workspace")

  async def initialize(self) -> None:
    pass

  async def close(self) -> None:
    pass

  async def execute(
      self, command: str, *, timeout: Optional[float] = None
  ) -> ExecutionResult:
    return ExecutionResult(
        exit_code=0,
        stdout=self._stdout,
        stderr="",
        timed_out=False,
    )

  async def read_file(self, path: Path) -> bytes:
    return self._file_content

  async def write_file(self, path: Path, content: str | bytes) -> None:
    pass


@pytest.mark.asyncio
async def test_default_truncation_limit():
  """Verify tools default to the standard 30k limit."""
  long_text = "a" * 40_000
  env = _FakeEnvironment(
      stdout=long_text, file_content=long_text.encode("utf-8")
  )
  toolset = EnvironmentToolset(environment=env)
  tools = await toolset.get_tools()

  # 1. Check ExecuteTool
  execute_tool = next(t for t in tools if t.name == "Execute")
  res = await execute_tool.run_async(
      args={"command": "dummy"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"
  assert len(res["stdout"]) == 30_000 + len(
      "\n... (truncated, 40000 total chars)"
  )
  assert res["stdout"].endswith("\n... (truncated, 40000 total chars)")

  # 2. Check ReadFileTool
  read_file_tool = next(t for t in tools if t.name == "ReadFile")
  res = await read_file_tool.run_async(
      args={"path": "dummy.txt"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"
  assert len(res["content"]) == 30_000 + len(
      "\n... (truncated, 40000 total chars)"
  )


@pytest.mark.asyncio
async def test_custom_truncation_limit():
  """Verify tools honor custom max_output_chars limits."""
  long_text = "a" * 40_000
  env = _FakeEnvironment(
      stdout=long_text, file_content=long_text.encode("utf-8")
  )
  toolset = EnvironmentToolset(environment=env, max_output_chars=10_000)
  tools = await toolset.get_tools()

  # 1. Check ExecuteTool
  execute_tool = next(t for t in tools if t.name == "Execute")
  res = await execute_tool.run_async(
      args={"command": "dummy"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"
  assert len(res["stdout"]) == 10_000 + len(
      "\n... (truncated, 40000 total chars)"
  )

  # 2. Check ReadFileTool
  read_file_tool = next(t for t in tools if t.name == "ReadFile")
  res = await read_file_tool.run_async(
      args={"path": "dummy.txt"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"
  assert len(res["content"]) == 10_000 + len(
      "\n... (truncated, 40000 total chars)"
  )


@pytest.mark.asyncio
async def test_no_truncation_under_limit():
  """Verify short outputs are not truncated."""
  short_text = "a" * 100
  env = _FakeEnvironment(
      stdout=short_text, file_content=short_text.encode("utf-8")
  )
  toolset = EnvironmentToolset(environment=env, max_output_chars=10_000)
  tools = await toolset.get_tools()

  execute_tool = next(t for t in tools if t.name == "Execute")
  res = await execute_tool.run_async(
      args={"command": "dummy"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"
  assert res["stdout"] == short_text
