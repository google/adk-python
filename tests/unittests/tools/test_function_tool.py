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

import inspect
from typing import Any
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import _build_declaration_cached
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
import pytest


@pytest.fixture
def mock_tool_context() -> ToolContext:
  """Fixture that provides a mock ToolContext for testing."""
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  mock_invocation_context.session.state = MagicMock()
  return ToolContext(invocation_context=mock_invocation_context)


def function_for_testing_with_no_args():
  """Function for testing with no args."""
  pass


async def async_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
):
  """Async function for testing with 1 arg and tool context."""
  assert arg1
  assert tool_context
  return arg1


async def async_function_for_testing_with_2_arg_and_no_tool_context(arg1, arg2):
  """Async function for testing with 2 args and no tool context."""
  assert arg1
  assert arg2
  return arg1


class AsyncCallableWith2ArgsAndNoToolContext:

  def __init__(self):
    self.__name__ = "Async callable name"
    self.__doc__ = "Async callable doc"

  async def __call__(self, arg1, arg2):
    assert arg1
    assert arg2
    return arg1


def function_for_testing_with_1_arg_and_tool_context(arg1, tool_context):
  """Function for testing with 1 arg and tool context."""
  assert arg1
  assert tool_context
  return arg1


class AsyncCallableWith1ArgAndToolContext:

  async def __call__(self, arg1, tool_context):
    """Async call doc"""
    assert arg1
    assert tool_context
    return arg1


def function_for_testing_with_2_arg_and_no_tool_context(arg1, arg2):
  """Function for testing with 2 args and no tool context."""
  assert arg1
  assert arg2
  return arg1


async def async_function_for_testing_with_4_arg_and_no_tool_context(
    arg1, arg2, arg3, arg4
):
  """Async function for testing with 4 args."""
  pass


def function_for_testing_with_4_arg_and_no_tool_context(arg1, arg2, arg3, arg4):
  """Function for testing with 4 args."""
  pass


def function_returning_none() -> None:
  """Function for testing with no return value."""
  return None


def function_returning_empty_dict() -> dict[str, str]:
  """Function for testing with empty dict return value."""
  return {}


def test_init():
  """Test that the FunctionTool is initialized correctly."""
  tool = FunctionTool(function_for_testing_with_no_args)
  assert tool.name == "function_for_testing_with_no_args"
  assert tool.description == "Function for testing with no args."
  assert tool.func == function_for_testing_with_no_args


@pytest.mark.asyncio
async def test_function_returning_none():
  """Test that the function returns with None actually returning None."""
  tool = FunctionTool(function_returning_none)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert result is None


@pytest.mark.asyncio
async def test_function_returning_empty_dict():
  """Test that the function returns with empty dict actually returning empty dict."""
  tool = FunctionTool(function_returning_empty_dict)
  result = await tool.run_async(args={}, tool_context=MagicMock())
  assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_run_async_with_tool_context_async_func():
  """Test that run_async calls the function with tool_context when tool_context is in signature (async function)."""

  tool = FunctionTool(async_function_for_testing_with_1_arg_and_tool_context)
  args = {"arg1": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"


@pytest.mark.asyncio
async def test_run_async_with_tool_context_async_callable():
  """Test that run_async calls the callable with tool_context when tool_context is in signature (async callable)."""

  tool = FunctionTool(AsyncCallableWith1ArgAndToolContext())
  args = {"arg1": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"
  assert tool.name == "AsyncCallableWith1ArgAndToolContext"
  assert tool.description == "Async call doc"


@pytest.mark.asyncio
async def test_run_async_without_tool_context_async_func():
  """Test that run_async calls the function without tool_context when tool_context is not in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_2_arg_and_no_tool_context)
  args = {"arg1": "test_value_1", "arg2": "test_value_2"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"


@pytest.mark.asyncio
async def test_run_async_without_tool_context_async_callable():
  """Test that run_async calls the callable without tool_context when tool_context is not in signature (async callable)."""
  tool = FunctionTool(AsyncCallableWith2ArgsAndNoToolContext())
  args = {"arg1": "test_value_1", "arg2": "test_value_2"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"
  assert tool.name == "Async callable name"
  assert tool.description == "Async callable doc"


@pytest.mark.asyncio
async def test_run_async_with_tool_context_sync_func():
  """Test that run_async calls the function with tool_context when tool_context is in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_1_arg_and_tool_context)
  args = {"arg1": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"


@pytest.mark.asyncio
async def test_run_async_without_tool_context_sync_func():
  """Test that run_async calls the function without tool_context when tool_context is not in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_2_arg_and_no_tool_context)
  args = {"arg1": "test_value_1", "arg2": "test_value_2"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1"


@pytest.mark.asyncio
async def test_run_async_1_missing_arg_sync_func():
  """Test that run_async calls the function with 1 missing arg in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_2_arg_and_no_tool_context)
  args = {"arg1": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `function_for_testing_with_2_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg2
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_1_missing_arg_async_func():
  """Test that run_async calls the function with 1 missing arg in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_2_arg_and_no_tool_context)
  args = {"arg2": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `async_function_for_testing_with_2_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_3_missing_arg_sync_func():
  """Test that run_async calls the function with 3 missing args in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_4_arg_and_no_tool_context)
  args = {"arg2": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_3_missing_arg_async_func():
  """Test that run_async calls the function with 3 missing args in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_4_arg_and_no_tool_context)
  args = {"arg3": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `async_function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_missing_all_arg_sync_func():
  """Test that run_async calls the function with all missing args in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_4_arg_and_no_tool_context)
  args = {}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_missing_all_arg_async_func():
  """Test that run_async calls the function with all missing args in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_4_arg_and_no_tool_context)
  args = {}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": (
          """Invoking `async_function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      )
  }


@pytest.mark.asyncio
async def test_run_async_with_optional_args_not_set_sync_func():
  """Test that run_async calls the function for sync function with optional args not set."""

  def func_with_optional_args(arg1, arg2=None, *, arg3, arg4=None, **kwargs):
    return f"{arg1},{arg3}"

  tool = FunctionTool(func_with_optional_args)
  args = {"arg1": "test_value_1", "arg3": "test_value_3"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1,test_value_3"


@pytest.mark.asyncio
async def test_run_async_with_optional_args_not_set_async_func():
  """Test that run_async calls the function for async function with optional args not set."""

  async def async_func_with_optional_args(
      arg1, arg2=None, *, arg3, arg4=None, **kwargs
  ):
    return f"{arg1},{arg3}"

  tool = FunctionTool(async_func_with_optional_args)
  args = {"arg1": "test_value_1", "arg3": "test_value_3"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1,test_value_3"


@pytest.mark.asyncio
async def test_run_async_with_unexpected_argument():
  """Test that run_async filters out unexpected arguments."""

  def sample_func(expected_arg: str):
    return {"received_arg": expected_arg}

  tool = FunctionTool(sample_func)
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  # Add the missing state attribute to the session mock
  mock_invocation_context.session.state = MagicMock()
  tool_context_mock = ToolContext(invocation_context=mock_invocation_context)

  result = await tool.run_async(
      args={"expected_arg": "hello", "parameters": "should_be_filtered"},
      tool_context=tool_context_mock,
  )
  assert result == {"received_arg": "hello"}


@pytest.mark.asyncio
async def test_run_async_with_tool_context_and_unexpected_argument():
  """Test that run_async handles tool_context and filters out unexpected arguments."""

  def sample_func_with_context(expected_arg: str, tool_context: ToolContext):
    return {"received_arg": expected_arg, "context_present": bool(tool_context)}

  tool = FunctionTool(sample_func_with_context)
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  # Add the missing state attribute to the session mock
  mock_invocation_context.session.state = MagicMock()
  mock_tool_context = ToolContext(invocation_context=mock_invocation_context)

  result = await tool.run_async(
      args={
          "expected_arg": "world",
          "parameters": "should_also_be_filtered",
      },
      tool_context=mock_tool_context,
  )
  assert result == {
      "received_arg": "world",
      "context_present": True,
  }


@pytest.mark.asyncio
async def test_run_async_with_require_confirmation():
  """Test that run_async handles require_confirmation flag."""

  def sample_func(arg1: str):
    return {"received_arg": arg1}

  tool = FunctionTool(sample_func, require_confirmation=True)
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
  mock_invocation_context.session = MagicMock(spec=Session)
  mock_invocation_context.session.state = MagicMock()
  mock_invocation_context.agent = MagicMock()
  mock_invocation_context.agent.name = "test_agent"
  tool_context_mock = ToolContext(invocation_context=mock_invocation_context)
  tool_context_mock.function_call_id = "test_function_call_id"

  # First call, should request confirmation
  result = await tool.run_async(
      args={"arg1": "hello"},
      tool_context=tool_context_mock,
  )
  assert result == {
      "error": "This tool call requires confirmation, please approve or reject."
  }
  assert tool_context_mock._event_actions.requested_tool_confirmations[
      "test_function_call_id"
  ].hint == (
      "Please approve or reject the tool call sample_func() by responding with"
      " a FunctionResponse with an expected ToolConfirmation payload."
  )

  # Second call, user rejects
  tool_context_mock.tool_confirmation = ToolConfirmation(confirmed=False)
  result = await tool.run_async(
      args={"arg1": "hello"},
      tool_context=tool_context_mock,
  )
  assert result == {"error": "This tool call is rejected."}

  # Third call, user approves
  tool_context_mock.tool_confirmation = ToolConfirmation(confirmed=True)
  result = await tool.run_async(
      args={"arg1": "hello"},
      tool_context=tool_context_mock,
  )
  assert result == {"received_arg": "hello"}


@pytest.mark.asyncio
async def test_run_async_parameter_filtering(mock_tool_context):
  """Test that parameter filtering works correctly for functions with explicit parameters."""

  def explicit_params_func(arg1: str, arg2: int):
    """Function with explicit parameters (no **kwargs)."""
    return {"arg1": arg1, "arg2": arg2}

  tool = FunctionTool(explicit_params_func)

  # Test that unexpected parameters are still filtered out for non-kwargs functions
  result = await tool.run_async(
      args={
          "arg1": "test",
          "arg2": 42,
          "unexpected_param": "should_be_filtered",
      },
      tool_context=mock_tool_context,
  )

  assert result == {"arg1": "test", "arg2": 42}
  # Explicitly verify that unexpected_param was filtered out and not passed to the function
  assert "unexpected_param" not in result


def test_context_param_detection_with_context_type():
  """Test that FunctionTool detects context parameter by Context type annotation."""

  def my_tool(query: str, ctx: Context) -> str:
    return query

  tool = FunctionTool(my_tool)
  assert tool._context_param_name == "ctx"
  assert tool._ignore_params == ["ctx", "input_stream"]


def test_context_param_detection_with_tool_context_type():
  """Test that FunctionTool detects context parameter by ToolContext type annotation."""

  def my_tool(query: str, tool_context: ToolContext) -> str:
    return query

  tool = FunctionTool(my_tool)
  assert tool._context_param_name == "tool_context"
  assert tool._ignore_params == ["tool_context", "input_stream"]


def test_context_param_detection_with_custom_name():
  """Test that FunctionTool detects context parameter with any name if type is Context."""

  def my_tool(query: str, my_custom_context: Context) -> str:
    return query

  tool = FunctionTool(my_tool)
  assert tool._context_param_name == "my_custom_context"
  assert tool._ignore_params == ["my_custom_context", "input_stream"]


def test_context_param_detection_fallback_to_name():
  """Test that FunctionTool falls back to 'tool_context' name when no type annotation."""

  def my_tool(query: str, tool_context) -> str:
    return query

  tool = FunctionTool(my_tool)
  assert tool._context_param_name == "tool_context"
  assert tool._ignore_params == ["tool_context", "input_stream"]


def test_context_param_detection_no_context():
  """Test that FunctionTool defaults to 'tool_context' when no context param exists."""

  def my_tool(query: str, count: int) -> str:
    return query

  tool = FunctionTool(my_tool)
  assert tool._context_param_name == "tool_context"
  assert tool._ignore_params == ["tool_context", "input_stream"]


@pytest.mark.asyncio
async def test_run_async_with_custom_context_param_name(mock_tool_context):
  """Test that run_async correctly injects context with custom parameter name."""

  def my_tool(query: str, ctx: Context) -> dict:
    return {"query": query, "has_context": ctx is not None}

  tool = FunctionTool(my_tool)
  result = await tool.run_async(
      args={"query": "test"},
      tool_context=mock_tool_context,
  )

  assert result == {"query": "test", "has_context": True}


@pytest.mark.asyncio
async def test_run_async_with_context_type_annotation(mock_tool_context):
  """Test that run_async works with Context type annotation."""

  async def async_tool(query: str, context: Context) -> dict:
    return {"query": query, "context_type": type(context).__name__}

  tool = FunctionTool(async_tool)
  result = await tool.run_async(
      args={"query": "hello"},
      tool_context=mock_tool_context,
  )

  assert result["query"] == "hello"
  assert result["context_type"] == "Context"


def test_get_declaration_is_cached_and_returns_independent_copies():
  """_get_declaration caches the build and hands out independent copies."""

  def sample_tool(a: int, b: str) -> str:
    """A sample tool."""
    return b * a

  _build_declaration_cached.cache_clear()
  tool = FunctionTool(func=sample_tool)

  d1 = tool._get_declaration()  # pylint: disable=protected-access
  d2 = tool._get_declaration()  # pylint: disable=protected-access

  # The expensive build runs once; the second call is served from cache.
  info = _build_declaration_cached.cache_info()
  assert info.misses == 1
  assert info.hits >= 1

  assert d1.name == d2.name == "sample_tool"

  # Callers (e.g. toolset prefixing) mutate the returned declaration, so each
  # call must return an independent copy rather than the shared cached object.
  d1.name = "prefixed_sample_tool"
  d3 = tool._get_declaration()  # pylint: disable=protected-access
  assert d3.name == "sample_tool"


@pytest.mark.asyncio
async def test_run_async_with_async_generator_streaming_tool(mock_tool_context):
  """Test that run_async returns an AsyncGenerator when wrapped function is an async generator."""

  async def streaming_tool(val: int, tool_context: Context):
    yield f"item_{val}"
    yield f"item_{val + 1}"

  tool = FunctionTool(streaming_tool)
  result = await tool.run_async(
      args={"val": 10},
      tool_context=mock_tool_context,
  )

  items = []
  async for item in result:
    items.append(item)

  assert items == ["item_10", "item_11"]


@pytest.mark.asyncio
async def test_run_async_with_streaming_tool_and_input_stream(
    mock_tool_context,
):
  """Test that run_async injects input_stream into args_to_call for a streaming tool."""
  mock_stream = mock.MagicMock()
  mock_stream.read.return_value = "stream_data"

  mock_tool_context._invocation_context = mock.MagicMock()
  mock_tool_context._invocation_context.active_streaming_tools = {
      "streaming_tool_input": mock.MagicMock(stream=mock_stream)
  }

  async def streaming_tool_input(val: int, input_stream: Any):
    data = input_stream.read()
    yield f"{data}_{val}"

  tool = FunctionTool(streaming_tool_input)
  result = await tool.run_async(
      args={"val": 42},
      tool_context=mock_tool_context,
  )

  items = [item async for item in result]
  assert items == ["stream_data_42"]


@pytest.mark.asyncio
async def test_run_async_with_streaming_tool_require_confirmation(
    mock_tool_context,
):
  """Test e2e confirmation lifecycle for a streaming tool in run_async."""

  async def streaming_tool_conf(val: int):
    yield f"confirmed_{val}"

  tool = FunctionTool(streaming_tool_conf, require_confirmation=True)
  mock_tool_context.function_call_id = "test_function_call_id"

  # Stage 1: Call without confirmation should request confirmation and return error dict
  mock_tool_context.tool_confirmation = None
  res_unconfirmed = await tool.run_async(
      args={"val": 1},
      tool_context=mock_tool_context,
  )
  assert isinstance(res_unconfirmed, dict)
  assert "error" in res_unconfirmed
  assert "requires confirmation" in res_unconfirmed["error"]
  assert (
      "test_function_call_id"
      in mock_tool_context.actions.requested_tool_confirmations
  )

  # Stage 2: Call with rejected confirmation
  mock_tool_context.tool_confirmation = ToolConfirmation(confirmed=False)
  res_rejected = await tool.run_async(
      args={"val": 1},
      tool_context=mock_tool_context,
  )
  assert res_rejected == {"error": "This tool call is rejected."}

  # Stage 3: Call with approved confirmation should return the AsyncGenerator
  mock_tool_context.tool_confirmation = ToolConfirmation(confirmed=True)
  res_confirmed = await tool.run_async(
      args={"val": 1},
      tool_context=mock_tool_context,
  )
  assert inspect.isasyncgen(res_confirmed)
  items = [item async for item in res_confirmed]
  assert items == ["confirmed_1"]


@pytest.mark.asyncio
async def test_run_async_with_streaming_tool_missing_mandatory_arg(
    mock_tool_context,
):
  """Test that missing mandatory parameters in a streaming tool return an error dict."""

  async def streaming_tool_req(req_param: str):
    yield req_param

  tool = FunctionTool(streaming_tool_req)
  result = await tool.run_async(
      args={},
      tool_context=mock_tool_context,
  )
  assert isinstance(result, dict)
  assert "error" in result
  assert "mandatory input parameters are not present" in result["error"]


@pytest.mark.asyncio
async def test_run_async_coerces_integral_float_to_int_param(mock_tool_context):
  """A proto Struct round-trip turns an int arg into a float; it is coerced back."""

  async def tool_with_int(component_id: int):
    return {"got": component_id, "type": type(component_id).__name__}

  tool = FunctionTool(tool_with_int)
  result = await tool.run_async(
      args={"component_id": 1396683.0},
      tool_context=mock_tool_context,
  )
  assert result == {"got": 1396683, "type": "int"}


@pytest.mark.asyncio
async def test_run_async_coerces_integral_float_to_optional_int_param(
    mock_tool_context,
):
  """Optional[int] is unwrapped before the check, so it is coerced too."""

  async def tool_with_optional_int(component_id: Optional[int] = None):
    return {"type": type(component_id).__name__}

  tool = FunctionTool(tool_with_optional_int)
  result = await tool.run_async(
      args={"component_id": 7.0},
      tool_context=mock_tool_context,
  )
  assert result == {"type": "int"}


@pytest.mark.asyncio
async def test_run_async_passes_through_non_integral_float_for_int_param(
    mock_tool_context,
):
  """A float that is not a whole number is not silently truncated."""

  async def tool_with_int(component_id: int):
    return {"got": component_id}

  tool = FunctionTool(tool_with_int)
  result = await tool.run_async(
      args={"component_id": 1.5},
      tool_context=mock_tool_context,
  )
  assert result == {"got": 1.5}


@pytest.mark.asyncio
async def test_run_async_leaves_float_param_alone(mock_tool_context):
  """A float-typed parameter keeps its float, so the coercion is int-only."""

  async def tool_with_float(ratio: float):
    return {"type": type(ratio).__name__}

  tool = FunctionTool(tool_with_float)
  result = await tool.run_async(
      args={"ratio": 2.0},
      tool_context=mock_tool_context,
  )
  assert result == {"type": "float"}


@pytest.mark.asyncio
async def test_run_async_leaves_bool_arg_for_int_param_alone(mock_tool_context):
  """bool is an int subclass but not a float, so it is untouched."""

  async def tool_with_int(flag: int):
    return {"type": type(flag).__name__}

  tool = FunctionTool(tool_with_int)
  result = await tool.run_async(
      args={"flag": True},
      tool_context=mock_tool_context,
  )
  assert result == {"type": "bool"}
