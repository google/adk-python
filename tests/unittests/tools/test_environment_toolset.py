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

import asyncio
from pathlib import Path
import pickle
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.environment._edit_file_tool import EditFileTool
from google.adk.tools.environment._environment_toolset import EnvironmentToolset
from google.adk.tools.environment._execute_tool import ExecuteTool
from google.adk.tools.environment._read_file_tool import ReadFileTool
from google.adk.tools.environment._write_file_tool import WriteFileTool
from google.adk.tools.tool_context import ToolContext
import pytest
import pytest_asyncio


class _FakeEnvironment(BaseEnvironment):
  """Fake environment to return customized execution and read results."""

  def __init__(self, *, stdout: str, file_content: bytes):
    self._stdout = stdout
    self._file_content = file_content
    self.contexts: list[tuple[str, Any]] = []

  @property
  def working_dir(self) -> Path:
    return Path("/workspace")

  async def initialize(self, *, ctx: Any = None) -> None:
    self.contexts.append(("initialize", ctx))

  async def close(self) -> None:
    pass

  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
      ctx: Any = None,
  ) -> ExecutionResult:
    self.contexts.append(("execute", ctx))
    return ExecutionResult(
        exit_code=0,
        stdout=self._stdout,
        stderr="",
        timed_out=False,
    )

  async def read_file(self, path: Path, *, ctx: Any = None) -> bytes:
    self.contexts.append(("read_file", ctx))
    return self._file_content

  async def write_file(
      self, path: Path, content: str | bytes, *, ctx: Any = None
  ) -> None:
    self.contexts.append(("write_file", ctx))


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


@pytest.mark.asyncio
async def test_get_tools_does_not_initialize_environment():
  """get_tools leaves initialization to the first tool call."""
  env = _FakeEnvironment(stdout="", file_content=b"")
  toolset = EnvironmentToolset(environment=env)

  await toolset.get_tools(mock.MagicMock(spec=ReadonlyContext))

  assert not env.contexts


async def _instruction(toolset: EnvironmentToolset) -> str:
  llm_request = LlmRequest()
  await toolset.process_llm_request(
      tool_context=mock.MagicMock(spec=ToolContext), llm_request=llm_request
  )
  return llm_request.config.system_instruction


@pytest.mark.asyncio
async def test_process_llm_request_does_not_initialize_environment():
  """Before initialization the instruction has the rules, not the directory."""
  env = _FakeEnvironment(stdout="", file_content=b"")
  toolset = EnvironmentToolset(environment=env)

  instruction = await _instruction(toolset)

  assert not env.contexts
  assert "# Environment Rules" in instruction
  assert "Your environment is at" not in instruction


@pytest.mark.asyncio
async def test_instruction_includes_working_dir_after_tool_call():
  """Once a tool call initialized the environment, the directory is shown."""
  env = _FakeEnvironment(stdout="", file_content=b"")
  toolset = EnvironmentToolset(environment=env)
  await _run_execute(toolset)

  instruction = await _instruction(toolset)

  assert "Your environment is at /workspace/" in instruction
  assert "# Environment Rules" in instruction


@pytest.mark.asyncio
async def test_instruction_includes_working_dir_of_shared_environment():
  """An environment another toolset initialized shows its directory too."""
  env = _FakeEnvironment(stdout="", file_content=b"")
  env.is_initialized = True
  toolset = EnvironmentToolset(environment=env)

  instruction = await _instruction(toolset)

  assert "Your environment is at /workspace/" in instruction
  assert "# Environment Rules" in instruction
  assert not env.contexts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "args", "expected_calls"),
    [
        ("Execute", {"command": "ls"}, ["execute"]),
        ("ReadFile", {"path": "a.txt"}, ["read_file"]),
        ("WriteFile", {"path": "a.txt", "content": "x"}, ["write_file"]),
        (
            "EditFile",
            {"path": "a.txt", "old_string": "old", "new_string": "new"},
            ["read_file", "write_file"],
        ),
    ],
)
async def test_tools_pass_tool_context_to_environment(
    tool_name: str, args: dict[str, Any], expected_calls: list[str]
):
  """Every call initializes first; all calls pass the call's tool context."""
  env = _FakeEnvironment(stdout="", file_content=b"old")
  toolset = EnvironmentToolset(environment=env)
  tools = await toolset.get_tools()
  tool = next(t for t in tools if t.name == tool_name)
  tool_context = mock.MagicMock(spec=ToolContext)
  second_tool_context = mock.MagicMock(spec=ToolContext)

  res = await tool.run_async(args=args, tool_context=tool_context)
  second_res = await tool.run_async(args=args, tool_context=second_tool_context)

  assert res["status"] == "ok"
  assert second_res["status"] == "ok"
  assert env.contexts == (
      [("initialize", tool_context)]
      + [(call, tool_context) for call in expected_calls]
      + [("initialize", second_tool_context)]
      + [(call, second_tool_context) for call in expected_calls]
  )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_class", "args"),
    [
        (ExecuteTool, {"command": "ls"}),
        (ReadFileTool, {"path": "a.txt"}),
        (WriteFileTool, {"path": "a.txt", "content": "x"}),
        (
            EditFileTool,
            {"path": "a.txt", "old_string": "old", "new_string": "new"},
        ),
    ],
)
async def test_tool_without_toolset_initializes_environment(
    tool_class: type[BaseTool], args: dict[str, Any]
):
  """A tool constructed directly initializes the environment on first use."""
  env = _FakeEnvironment(stdout="", file_content=b"old")
  tool = tool_class(env)
  tool_context = mock.MagicMock(spec=ToolContext)

  res = await tool.run_async(args=args, tool_context=tool_context)

  assert res["status"] == "ok"
  assert env.contexts[0] == ("initialize", tool_context)
  assert env.is_initialized


async def _run_execute(toolset: EnvironmentToolset) -> None:
  tools = await toolset.get_tools()
  execute_tool = next(t for t in tools if t.name == "Execute")
  res = await execute_tool.run_async(
      args={"command": "ls"}, tool_context=mock.MagicMock(spec=ToolContext)
  )
  assert res["status"] == "ok"


class _AsyncCountingEnvironment(BaseEnvironment):
  """Environment that yields to the event loop during initialize/close."""

  def __init__(self) -> None:
    self.init_count = 0
    self.close_count = 0

  @property
  def working_dir(self) -> Path:
    return Path("/workspace")

  async def initialize(self, *, ctx: Any = None) -> None:
    del ctx
    await asyncio.sleep(0)
    self.init_count += 1

  async def close(self) -> None:
    await asyncio.sleep(0)
    self.close_count += 1

  async def execute(
      self,
      command: str,
      *,
      timeout: float | None = None,
      ctx: Any = None,
  ) -> ExecutionResult:
    del timeout, ctx
    await asyncio.sleep(0)
    return ExecutionResult(
        exit_code=0,
        stdout=f"exec:{command}",
        stderr="",
        timed_out=False,
    )

  async def read_file(self, path: Path, *, ctx: Any = None) -> bytes:
    del ctx
    return f"content:{path}".encode("utf-8")

  async def write_file(
      self, path: Path, content: str | bytes, *, ctx: Any = None
  ) -> None:
    del path, content, ctx


@pytest.mark.asyncio
async def test_environment_toolset_concurrent_initialization():
  """Concurrent tool calls each initialize the environment first."""
  env = _AsyncCountingEnvironment()
  toolset = EnvironmentToolset(environment=env, tool_name_prefix="env")
  tools = await toolset.get_tools()
  calls = {
      "Execute": {"command": "ls"},
      "ReadFile": {"path": "a.txt"},
      "WriteFile": {"path": "a.txt", "content": "x"},
      "EditFile": {"path": "a.txt", "old_string": "a", "new_string": "b"},
  }

  results = await asyncio.gather(*(
      tool.run_async(
          args=calls[tool.name], tool_context=mock.MagicMock(spec=ToolContext)
      )
      for tool in tools * 2
  ))

  assert [r["status"] for r in results] == ["ok"] * 8
  assert env.init_count == 8


@pytest.mark.asyncio
async def test_environment_toolset_pickle_roundtrip():
  """An unpickled toolset initializes its environment again on first use."""
  env = _AsyncCountingEnvironment()
  toolset = EnvironmentToolset(environment=env, tool_name_prefix="env")
  tools = await toolset.get_tools_with_prefix()
  await _run_execute(toolset)
  assert env.init_count == 1

  restored = pickle.loads(pickle.dumps(toolset))
  assert not restored._environment.is_initialized
  restored_tools = await restored.get_tools_with_prefix()
  await _run_execute(restored)
  assert len(restored_tools) == len(tools)
  assert restored._environment.is_initialized
  assert restored._environment.init_count == 2

  await restored.close()
  await _run_execute(restored)
  assert restored._environment.init_count == 3
  # The original toolset's environment is untouched by pickling.
  assert env.is_initialized


class _ReadOnlyFlagEnvironment(_AsyncCountingEnvironment):
  """Manages its own flag behind a read-only `is_initialized` property."""

  def __init__(self) -> None:
    super().__init__()
    self._initialized = False

  @property
  def is_initialized(self) -> bool:
    return self._initialized

  async def initialize(self, *, ctx: Any = None) -> None:
    await super().initialize(ctx=ctx)
    self._initialized = True


@pytest.mark.asyncio
async def test_unpickling_tolerates_a_read_only_is_initialized():
  """A read-only `is_initialized` is left for the environment to manage."""
  toolset = EnvironmentToolset(environment=_ReadOnlyFlagEnvironment())
  await _run_execute(toolset)

  restored = pickle.loads(pickle.dumps(toolset))

  assert restored._environment.is_initialized


@pytest.mark.asyncio
async def test_environment_toolset_close_resets_and_allows_reinitialization():
  """close() skips uninitialized env, closes one a tool call initialized."""
  env = _AsyncCountingEnvironment()
  toolset = EnvironmentToolset(environment=env, tool_name_prefix="env")
  await toolset.close()
  assert env.close_count == 0

  tools = await toolset.get_tools_with_prefix()
  await _instruction(toolset)
  await toolset.close()
  assert env.close_count == 0

  await _run_execute(toolset)
  assert env.init_count == 1

  await asyncio.gather(*(toolset.close() for _ in range(8)))
  assert env.close_count == 1

  new_tools = await toolset.get_tools_with_prefix()
  await _run_execute(toolset)
  assert len(new_tools) == len(tools)
  assert env.init_count == 2
