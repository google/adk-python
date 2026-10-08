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

from enum import Enum
import inspect
from typing import Any
from typing import Optional
from typing import Union
from unittest import mock
from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import _build_declaration_cached
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
import pydantic
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
  mock_tool_context._invocation_context.live_request_queue = mock.MagicMock()

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
  mock_tool_context._invocation_context.live_request_queue = mock.MagicMock()

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


def test_function_tool_init_type_hints():
  """Test that get_type_hints on FunctionTool.__init__ resolves without NameError."""
  from typing import get_type_hints

  hints = get_type_hints(FunctionTool.__init__)
  assert "require_confirmation" in hints


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_coercion(mock_tool_context):
  """Test that argument type coercion and enum conversion work when validation is enabled."""

  class Color(Enum):
    RED = "red"
    BLUE = "blue"

  def sample_func(num: int, color: Color, flag: bool) -> dict:
    return {"num": num, "color": color.value, "flag": flag}

  tool = FunctionTool(sample_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result = await tool.run_async(
        args={"num": "42", "color": "red", "flag": 1},
        tool_context=mock_tool_context,
    )
    assert result == {"num": 42, "color": "red", "flag": True}


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_error(mock_tool_context):
  """Test that invalid argument types return a validation error dict to the LLM."""

  def sample_func(num: int) -> int:
    return num

  tool = FunctionTool(sample_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result = await tool.run_async(
        args={"num": "not_an_int"},
        tool_context=mock_tool_context,
    )
    assert isinstance(result, dict)
    assert "error" in result
    assert "validation error" in result["error"].lower()
    assert "num" in result["error"]


@pytest.mark.asyncio
async def test_run_async_arg_validation_disabled_by_default(mock_tool_context):
  """Test that argument validation is disabled by default and allows lax arguments."""

  def sample_func(zip_code: str) -> str:
    return zip_code

  tool = FunctionTool(sample_func)
  # Flag is disabled by default; int passed for str is not rejected
  result = await tool.run_async(
      args={"zip_code": 123},
      tool_context=mock_tool_context,
  )
  assert result == 123


def test_preprocess_args_with_unhandled_annotation_skipped():
  """Test that unhandled/invalid type annotations gracefully skip validation."""

  def invalid_type_func(x: 123) -> int:
    return x

  tool = FunctionTool(invalid_type_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    args, errors = tool._preprocess_args_with_validation({"x": "some_value"})
    assert errors == []
    assert args["x"] == "some_value"


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_pydantic_model(mock_tool_context):
  """Test that BaseModel arguments are validated and converted when flag is enabled."""

  class UserModel(pydantic.BaseModel):
    name: str
    age: int

  def sample_func(user: UserModel) -> dict:
    return {"name": user.name, "age": user.age}

  tool = FunctionTool(sample_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # Valid dict converted to BaseModel
    result = await tool.run_async(
        args={"user": {"name": "Alice", "age": 30}},
        tool_context=mock_tool_context,
    )
    assert result == {"name": "Alice", "age": 30}

    # Invalid dict returns validation error
    result_err = await tool.run_async(
        args={"user": {"name": "Alice", "age": "not_an_int"}},
        tool_context=mock_tool_context,
    )
    assert isinstance(result_err, dict)
    assert "error" in result_err
    assert "validation error" in result_err["error"].lower()
    assert "user" in result_err["error"]


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_list_of_pydantic_models(
    mock_tool_context,
):
  """Test that list[BaseModel] arguments are validated and converted when flag is enabled."""

  class ItemModel(pydantic.BaseModel):
    item_id: str
    price: float

  def sample_func(items: list[ItemModel]) -> float:
    return sum(item.price for item in items)

  tool = FunctionTool(sample_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # Valid list of dicts converted
    result = await tool.run_async(
        args={
            "items": [
                {"item_id": "a", "price": 10.5},
                {"item_id": "b", "price": "20.0"},
            ]
        },
        tool_context=mock_tool_context,
    )
    assert result == 30.5

    # Invalid list item returns validation error
    result_err = await tool.run_async(
        args={"items": [{"item_id": "a", "price": "invalid_price"}]},
        tool_context=mock_tool_context,
    )
    assert isinstance(result_err, dict)
    assert "error" in result_err
    assert "validation error" in result_err["error"].lower()
    assert "items" in result_err["error"]


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_union_of_pydantic_models(
    mock_tool_context,
):
  """Test that Union[BaseModel, ...] arguments are validated when flag is enabled."""

  class UserProfile(pydantic.BaseModel):
    username: str

  class OrgProfile(pydantic.BaseModel):
    org_name: str

  def sample_func(entity: Union[UserProfile, OrgProfile]) -> str:
    return type(entity).__name__

  tool = FunctionTool(sample_func)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # Valid dict matching UserProfile
    result_user = await tool.run_async(
        args={"entity": {"username": "alice"}},
        tool_context=mock_tool_context,
    )
    assert result_user == "UserProfile"

    # Valid dict matching OrgProfile
    result_org = await tool.run_async(
        args={"entity": {"org_name": "Google"}},
        tool_context=mock_tool_context,
    )
    assert result_org == "OrgProfile"

    # Invalid dict matching neither returns validation error
    result_err = await tool.run_async(
        args={"entity": {"unrelated": "data"}},
        tool_context=mock_tool_context,
    )
    assert isinstance(result_err, dict)
    assert "error" in result_err
    assert "validation error" in result_err["error"].lower()
    assert "entity" in result_err["error"]


@pytest.mark.asyncio
async def test_run_async_with_arg_validation_confirmation_predicate(
    mock_tool_context,
):
  """Test that check_require_confirmation receives preprocessed args."""
  received_args = []

  def confirm_predicate(amount: float) -> bool:
    received_args.append(amount)
    return amount > 100.0

  def transfer(amount: float) -> float:
    return amount

  tool = FunctionTool(transfer, require_confirmation=confirm_predicate)
  mock_tool_context.function_call_id = "test_call_id"
  mock_tool_context._invocation_context.agent = MagicMock()
  mock_tool_context._invocation_context.agent.name = "test_agent"
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # String "150.0" is coerced to float 150.0 so predicate receives 150.0
    result = await tool.run_async(
        args={"amount": "150.0"}, tool_context=mock_tool_context
    )
    assert received_args == [150.0]
    assert isinstance(result, dict)
    assert "requires confirmation" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_check_require_confirmation_with_coerced_args(
    mock_tool_context,
):
  """Test that direct check_require_confirmation calls preprocess and coerce args."""
  received_args = []

  def confirm_predicate(amount: float) -> bool:
    received_args.append(amount)
    return amount > 100.0

  def transfer(amount: float) -> float:
    return amount

  tool = FunctionTool(transfer, require_confirmation=confirm_predicate)
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # Calling check_require_confirmation directly with "150.0" coerces to float 150.0
    requires = await tool.check_require_confirmation(
        args={"amount": "150.0"}, tool_context=mock_tool_context
    )
    assert requires is True
    assert received_args == [150.0]


@pytest.mark.asyncio
async def test_subclass_check_require_confirmation_override(mock_tool_context):
  """Test that subclass overriding check_require_confirmation works in run_async."""

  class CustomFunctionTool(FunctionTool):

    async def check_require_confirmation(
        self, args: dict[str, Any], tool_context: ToolContext
    ) -> bool:
      return args.get("amount", 0) > 50

  def transfer(amount: int) -> int:
    return amount

  tool = CustomFunctionTool(transfer)
  mock_tool_context.function_call_id = "test_call_id"
  mock_tool_context._invocation_context.agent = MagicMock()
  mock_tool_context._invocation_context.agent.name = "test_agent"

  # Test with flag on
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result = await tool.run_async(
        args={"amount": 100}, tool_context=mock_tool_context
    )
    assert isinstance(result, dict)
    assert "requires confirmation" in result.get("error", "").lower()

  # Test with flag off (default)
  result_off = await tool.run_async(
      args={"amount": 100}, tool_context=mock_tool_context
  )
  assert isinstance(result_off, dict)
  assert "requires confirmation" in result_off.get("error", "").lower()


@pytest.mark.asyncio
async def test_subclass_preprocess_args_override_with_confirmation_predicate(
    mock_tool_context,
):
  """Test that subclass _preprocess_args override is applied for confirmation predicate and func."""
  predicate_args = []
  func_args = []

  class ScalingTool(FunctionTool):

    def _preprocess_args(self, args: dict[str, Any]) -> dict[str, Any]:
      args = super()._preprocess_args(args)
      args = args.copy()
      if "amount" in args:
        args["amount"] = args["amount"] * 10
      return args

  def confirm_predicate(amount: int) -> bool:
    predicate_args.append(amount)
    return False

  def transfer(amount: int) -> int:
    func_args.append(amount)
    return amount

  tool = ScalingTool(transfer, require_confirmation=confirm_predicate)

  # With feature flag enabled:
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    await tool.run_async(args={"amount": 100}, tool_context=mock_tool_context)
    assert predicate_args == [1000]
    assert func_args == [1000]

  # With feature flag disabled:
  predicate_args.clear()
  func_args.clear()
  await tool.run_async(args={"amount": 100}, tool_context=mock_tool_context)
  assert predicate_args == [1000]
  assert func_args == [1000]


@pytest.mark.asyncio
async def test_subclass_check_require_confirmation_receives_raw_args(
    mock_tool_context,
):
  """Test that subclass check_require_confirmation receives raw dict args."""
  received_args = []

  class UserModel(pydantic.BaseModel):
    name: str

  class CustomFunctionTool(FunctionTool):

    async def check_require_confirmation(
        self, args: dict[str, Any], tool_context: ToolContext
    ) -> bool:
      received_args.append(args.get("user"))
      return args["user"]["name"] == "admin"

  def update_user(user: UserModel) -> str:
    return user.name

  tool = CustomFunctionTool(update_user)
  mock_tool_context.function_call_id = "test_call_id"
  mock_tool_context._invocation_context.agent = MagicMock()
  mock_tool_context._invocation_context.agent.name = "test_agent"

  # Flag disabled (default)
  result = await tool.run_async(
      args={"user": {"name": "admin"}}, tool_context=mock_tool_context
  )
  assert isinstance(result, dict)
  assert "requires confirmation" in result.get("error", "").lower()
  assert isinstance(received_args[-1], dict)

  # Flag enabled
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result_enabled = await tool.run_async(
        args={"user": {"name": "admin"}}, tool_context=mock_tool_context
    )
    assert isinstance(result_enabled, dict)
    assert "requires confirmation" in result_enabled.get("error", "").lower()
    assert isinstance(received_args[-1], dict)


@pytest.mark.asyncio
async def test_subclass_preprocess_args_override_called_in_run_async(
    mock_tool_context,
):
  """Test that subclass overriding _preprocess_args is called during run_async."""

  class CustomPreprocessTool(FunctionTool):

    def _preprocess_args(self, args: dict[str, Any]) -> dict[str, Any]:
      args = super()._preprocess_args(args)
      args["custom"] = "preprocessed"
      return args

  def sample_func(custom: str = "default") -> str:
    return custom

  tool = CustomPreprocessTool(sample_func)

  # Flag disabled (default)
  result = await tool.run_async(args={}, tool_context=mock_tool_context)
  assert result == "preprocessed"

  # Flag enabled
  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result_enabled = await tool.run_async(
        args={}, tool_context=mock_tool_context
    )
    assert result_enabled == "preprocessed"


@pytest.mark.asyncio
async def test_nested_check_require_confirmation_preprocesses_args(
    mock_tool_context,
):
  """Test that nested check_require_confirmation on another tool preprocesses its args."""
  tool2_received_args = []

  def tool2_confirm(limit: int) -> bool:
    tool2_received_args.append(limit)
    return limit > 50

  def func2(limit: int) -> int:
    return limit

  tool2 = FunctionTool(func2, require_confirmation=tool2_confirm)

  async def tool1_confirm(amount: float) -> bool:
    # Nested check_require_confirmation on tool2 with raw string argument "100"
    tool2_requires = await tool2.check_require_confirmation(
        args={"limit": "100"}, tool_context=mock_tool_context
    )
    return tool2_requires and amount > 50.0

  def func1(amount: float) -> float:
    return amount

  tool1 = FunctionTool(func1, require_confirmation=tool1_confirm)
  mock_tool_context.function_call_id = "test_call_id"
  mock_tool_context._invocation_context.agent = MagicMock()
  mock_tool_context._invocation_context.agent.name = "test_agent"

  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    result = await tool1.run_async(
        args={"amount": "150.0"}, tool_context=mock_tool_context
    )
    # tool2's predicate must have received coerced int 100, not raw str "100"
    assert tool2_received_args == [100]
    assert isinstance(result, dict)
    assert "requires confirmation" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_monkeypatched_check_require_confirmation(mock_tool_context):
  """Test that monkeypatching FunctionTool.check_require_confirmation is invoked during run_async."""
  patch_called = []

  original_method = FunctionTool.check_require_confirmation

  async def patched_check(self, args, tool_context):
    patch_called.append(True)
    return True

  def sample_func(x: int) -> int:
    return x

  tool = FunctionTool(sample_func)
  mock_tool_context.function_call_id = "test_call_id"
  mock_tool_context._invocation_context.agent = MagicMock()
  mock_tool_context._invocation_context.agent.name = "test_agent"

  try:
    FunctionTool.check_require_confirmation = patched_check
    result = await tool.run_async(args={"x": 1}, tool_context=mock_tool_context)
    assert patch_called == [True]
    assert isinstance(result, dict)
    assert "requires confirmation" in result.get("error", "").lower()
  finally:
    FunctionTool.check_require_confirmation = original_method


@pytest.mark.asyncio
async def test_run_async_coerces_integral_floats_in_list_int_param(
    mock_tool_context,
):
  """Integral floats inside a list[int] argument are coerced back to int."""

  async def tool_with_list_int(component_ids: list[int]):
    return {"types": [type(i).__name__ for i in component_ids]}

  tool = FunctionTool(tool_with_list_int)
  result = await tool.run_async(
      args={"component_ids": [1396683.0, 7.0]},
      tool_context=mock_tool_context,
  )
  assert result == {"types": ["int", "int"]}


@pytest.mark.asyncio
async def test_run_async_coerces_integral_floats_in_optional_list_int_param(
    mock_tool_context,
):
  """Optional[list[int]] is unwrapped before the check, so it is coerced too."""

  async def tool_with_optional_list_int(
      component_ids: Optional[list[int]] = None,
  ):
    return {"types": [type(i).__name__ for i in component_ids]}

  tool = FunctionTool(tool_with_optional_list_int)
  result = await tool.run_async(
      args={"component_ids": [7.0]},
      tool_context=mock_tool_context,
  )
  assert result == {"types": ["int"]}


@pytest.mark.asyncio
async def test_run_async_passes_through_non_integral_float_in_list_int_param(
    mock_tool_context,
    caplog,
):
  """A list element that is not a whole number is not silently truncated."""

  async def tool_with_list_int(component_ids: list[int]):
    return {"got": component_ids}

  tool = FunctionTool(tool_with_list_int)
  with caplog.at_level("WARNING"):
    result = await tool.run_async(
        args={"component_ids": [1.5, 2.0]},
        tool_context=mock_tool_context,
    )
  assert result == {"got": [1.5, 2]}
  assert "Argument 'component_ids' is typed list[int]" in caplog.text
  assert "1.5" in caplog.text


@pytest.mark.asyncio
async def test_run_async_leaves_list_float_param_alone(mock_tool_context):
  """A list[float] parameter keeps its floats, so the coercion is int-only."""

  async def tool_with_list_float(ratios: list[float]):
    return {"types": [type(r).__name__ for r in ratios]}

  tool = FunctionTool(tool_with_list_float)
  result = await tool.run_async(
      args={"ratios": [2.0]},
      tool_context=mock_tool_context,
  )
  assert result == {"types": ["float"]}


@pytest.mark.asyncio
async def test_run_async_leaves_untyped_list_param_alone(mock_tool_context):
  """A bare list annotation carries no element type, so nothing is coerced."""

  async def tool_with_bare_list(values: list):
    return {"types": [type(v).__name__ for v in values]}

  tool = FunctionTool(tool_with_bare_list)
  result = await tool.run_async(
      args={"values": [2.0]},
      tool_context=mock_tool_context,
  )
  assert result == {"types": ["float"]}


@pytest.mark.asyncio
async def test_async_generator_tool_emits_intermediate_events_and_returns_final_output():
  """An async generator tool emits intermediate events on a sub-branch and returns its final output to the model."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.events.event import Event
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  async def analyze(query: str):
    yield Event(message=f"Starting {query}...")
    yield Event(message=f"Finished {query}.")
    yield {"rows": 42}

  fc = types.Part.from_function_call(name="analyze", args={"query": "sales"})
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="All done.")]
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[analyze])
  runner = testing_utils.InMemoryRunner(agent)

  events = await runner.run_async("run sales")

  intermediate = [
      e
      for e in events
      if e.author == "analyze"
      and e.content
      and e.content.parts
      and e.content.parts[0].text in ("Starting sales...", "Finished sales.")
  ]
  assert len(intermediate) == 2
  assert all(
      e.branch is not None and e.branch.startswith("analyze@")
      for e in intermediate
  )
  assert len(mock_model.requests) == 2
  second_req_responses = [
      p.function_response.response
      for c in mock_model.requests[1].contents
      for p in c.parts or []
      if p.function_response
  ]
  assert {"rows": 42} in second_req_responses
  assert not [
      c
      for c in mock_model.requests[1].contents
      if "Starting sales..." in str(c)
  ]


@pytest.mark.asyncio
async def test_generator_tool_intermediate_events_stay_out_of_next_turn_context():
  """A root agent does not see a generator tool's intermediate events in a later turn."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.events.event import Event
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  async def analyze(query: str):
    yield Event(message=f"Starting {query}...")
    yield Event(message=f"Finished {query}.")
    yield {"rows": 42}

  fc = types.Part.from_function_call(name="analyze", args={"query": "sales"})
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="All done.")]
          )
      ),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="42 rows.")]
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[analyze])
  runner = testing_utils.InMemoryRunner(agent)

  await runner.run_async("run sales")
  await runner.run_async("what happened?")

  assert len(mock_model.requests) == 3
  third_req = str(mock_model.requests[2].contents)
  assert "Starting sales" not in third_req
  assert "Finished sales" not in third_req
  assert "] said:" not in third_req
  third_req_responses = [
      p.function_response.response
      for c in mock_model.requests[2].contents
      for p in c.parts or []
      if p.function_response
  ]
  assert {"rows": 42} in third_req_responses


@pytest.mark.asyncio
async def test_sync_generator_tool_emits_events_and_persists_state():
  """A sync generator tool emits intermediate events, persists state deltas, and returns its final output."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.events.event import Event
  from google.adk.events.event_actions import EventActions
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  def sync_analyze(query: str, tool_context: ToolContext):
    tool_context.state["last_query"] = query
    yield Event(
        message=f"Sync step for {query}",
        actions=EventActions(state_delta={"step_count": 1}),
    )
    yield {"status": "ok", "query": query}

  fc = types.Part.from_function_call(
      name="sync_analyze", args={"query": "revenue"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="Completed.")]
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[sync_analyze])
  runner = testing_utils.InMemoryRunner(agent)

  events = await runner.run_async("check revenue")

  texts = [
      p.text
      for e in events
      for p in (e.content.parts if e.content and e.content.parts else [])
      if p.text
  ]
  assert "Sync step for revenue" in texts
  assert runner.session.state.get("last_query") == "revenue"
  assert runner.session.state.get("step_count") == 1


@pytest.mark.asyncio
async def test_generator_tool_stamps_tool_author_and_propagates_actions():
  """A generator tool stamps the tool name as event author and propagates EventActions to the tool response."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.events.event import Event
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  async def summarize_and_skip(query: str, tool_context: ToolContext):
    tool_context.actions.skip_summarization = True
    yield Event(author="custom_reporter", message=f"Progress on {query}")
    yield {"done": query}

  fc = types.Part.from_function_call(
      name="summarize_and_skip", args={"query": "report"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[summarize_and_skip])
  runner = testing_utils.InMemoryRunner(agent)

  events = await runner.run_async("generate report")

  tool_events = [e for e in events if e.author == "summarize_and_skip"]
  assert len(tool_events) == 2
  assert not any(e.author == "custom_reporter" for e in events)
  fn_response_events = [e for e in events if e.get_function_responses()]
  assert len(fn_response_events) == 1
  assert fn_response_events[0].actions.skip_summarization is True
  assert len(mock_model.requests) == 1


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_generator_tool_rejects_multiple_outputs(is_async: bool):
  """A generator tool that yields more than one non-Event output raises ValueError and deterministically closes the generator."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  closed = False

  async def async_analyze(query: str):
    nonlocal closed
    del query
    try:
      yield {"rows": 1}
      yield {"rows": 2}
    finally:
      closed = True

  def sync_analyze(query: str):
    nonlocal closed
    del query
    try:
      yield {"rows": 1}
      yield {"rows": 2}
    finally:
      closed = True

  tool_fn = async_analyze if is_async else sync_analyze
  fc = types.Part.from_function_call(
      name=tool_fn.__name__, args={"query": "sales"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[tool_fn])
  runner = testing_utils.InMemoryRunner(agent)

  with pytest.raises(
      ValueError, match="Output already set. A node can produce at most one"
  ):
    await runner.run_async("run sales")

  assert closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable", [False, True])
async def test_generator_tool_pauses_on_request_input_and_resumes(
    resumable: bool,
):
  """A generator tool yielding RequestInput pauses the invocation and resumes with user input.

  Setup: App (resumable or non-resumable) with an Agent whose async generator
  tool yields RequestInput on first run.
  Act: Run the invocation until paused, then resume with
  create_request_input_response.
  Assert: The tool receives the resume input via tool_context.resume_inputs and
  completes the turn.
  """
  from google.adk.agents.llm_agent import Agent
  from google.adk.apps.app import App
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.events.event import Event
  from google.adk.events.request_input import RequestInput
  from google.adk.models.llm_response import LlmResponse
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
  from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
  from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.genai import types

  from .. import testing_utils

  async def ask_and_analyze(query: str, tool_context: ToolContext):
    clarification = tool_context.resume_inputs.get("clarify_region")
    if clarification is None:
      yield Event(message=f"Preparing {query}, need region...")
      yield RequestInput(
          interrupt_id="clarify_region",
          message="Which region should I analyze?",
      )
      return
    region = (
        clarification.get("text")
        if isinstance(clarification, dict)
        else clarification
    )
    yield Event(message=f"Analyzing {query} for {region}...")
    yield {"query": query, "region": region}

  fc = types.Part.from_function_call(
      name="ask_and_analyze", args={"query": "sales"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="Region analyzed.")],
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[ask_and_analyze])
  app = App(
      name="test_app",
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  events1 = await runner.run_async("start")
  req_events = [
      e
      for e in events1
      if any(
          p.function_call
          and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
          for p in (e.content.parts if e.content and e.content.parts else [])
      )
  ]
  interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
  invocation_id = req_events[0].invocation_id
  resume_part = create_request_input_response(interrupt_id, {"text": "APAC"})
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(resume_part),
      invocation_id=invocation_id,
  )

  texts2 = [
      p.text
      for e in events2
      for p in (e.content.parts if e.content and e.content.parts else [])
      if p.text
  ]
  assert "Analyzing sales for APAC..." in texts2
  assert "Region analyzed." in texts2


@pytest.mark.asyncio
async def test_generator_tool_preserves_non_context_ctx_parameter():
  """A generator tool with a non-Context parameter named 'ctx' receives the LLM argument rather than a Context object."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.events.event import Event
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  async def search_in_scope(ctx: str, query: str):
    yield Event(message=f"Searching {query} in {ctx}")
    yield {"ctx": ctx, "query": query, "ctx_type": type(ctx).__name__}

  fc = types.Part.from_function_call(
      name="search_in_scope", args={"ctx": "billing_domain", "query": "invoice"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="Done.")]
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[search_in_scope])
  runner = testing_utils.InMemoryRunner(agent)

  await runner.run_async("search invoice")

  second_req_responses = [
      p.function_response.response
      for c in mock_model.requests[1].contents
      for p in c.parts or []
      if p.function_response
  ]
  assert {
      "ctx": "billing_domain",
      "query": "invoice",
      "ctx_type": "str",
  } in second_req_responses


def test_generator_tool_caches_function_node_across_invocations():
  """_get_generator_node reuses the cached FunctionNode unless tool metadata changes."""

  async def gen_tool(query: str):
    yield query

  tool = FunctionTool(gen_tool)

  node1 = tool._get_generator_node()
  node2 = tool._get_generator_node()
  tool.name = "renamed_gen_tool"
  node3 = tool._get_generator_node()

  assert node1 is node2
  assert node3 is not node1
  assert node3.name == "renamed_gen_tool"


@pytest.mark.asyncio
async def test_generator_tool_repeated_calls_isolate_node_path_and_hitl_state():
  """Repeated calls to the same generator tool in one invocation get distinct node_info.path values and do not leak resume_inputs."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.apps.app import App
  from google.adk.apps.app import ResumabilityConfig
  from google.adk.events.event import Event
  from google.adk.events.request_input import RequestInput
  from google.adk.models.llm_response import LlmResponse
  from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
  from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
  from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
  from google.genai import types

  from .. import testing_utils

  async def ask_and_analyze(query: str, tool_context: ToolContext):
    clarification = tool_context.resume_inputs.get("clarify_region")
    if clarification is None:
      yield Event(message=f"Need region for {query}")
      yield RequestInput(
          interrupt_id="clarify_region",
          message=f"Which region for {query}?",
      )
      return
    region = (
        clarification.get("text")
        if isinstance(clarification, dict)
        else clarification
    )
    yield Event(message=f"Analyzed {query} in {region}")
    yield {"query": query, "region": region}

  fc1 = types.Part.from_function_call(
      name="ask_and_analyze", args={"query": "sales"}
  )
  fc2 = types.Part.from_function_call(
      name="ask_and_analyze", args={"query": "marketing"}
  )
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc1])),
      LlmResponse(content=types.Content(role="model", parts=[fc2])),
      LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text="Both analyzed.")],
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[ask_and_analyze])
  app = App(
      name="test_app",
      root_agent=agent,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Step 1: First call ("sales") pauses for input.
  events1 = await runner.run_async("start")
  req_event1 = next(
      e
      for e in events1
      if any(
          p.function_call
          and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
          for p in (e.content.parts if e.content and e.content.parts else [])
      )
  )
  interrupt_id1 = get_request_input_interrupt_ids(req_event1)[0]
  invocation_id = req_event1.invocation_id

  # Step 2: Resume first call ("APAC"); model then calls ask_and_analyze("marketing"),
  # which must pause for its own input rather than reusing "APAC".
  events2 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id1, {"text": "APAC"})
      ),
      invocation_id=invocation_id,
  )
  req_event2 = next(
      e
      for e in events2
      if any(
          p.function_call
          and p.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME
          for p in (e.content.parts if e.content and e.content.parts else [])
      )
  )
  interrupt_id2 = get_request_input_interrupt_ids(req_event2)[0]

  # Step 3: Resume second call ("EMEA") and complete.
  events3 = await runner.run_async(
      new_message=testing_utils.UserContent(
          create_request_input_response(interrupt_id2, {"text": "EMEA"})
      ),
      invocation_id=invocation_id,
  )

  texts3 = [
      p.text
      for e in events3
      for p in (e.content.parts if e.content and e.content.parts else [])
      if p.text
  ]
  assert req_event1.branch != req_event2.branch
  assert req_event1.node_info is not None and req_event2.node_info is not None
  assert req_event1.node_info.path != req_event2.node_info.path
  assert "Analyzed marketing in EMEA" in texts3
  assert "Both analyzed." in texts3


@pytest.mark.parametrize(
    "llm_args",
    [{"query": "sales"}, {"query": "sales", "tag": None}],
    ids=["omitted", "explicit_none"],
)
@pytest.mark.asyncio
async def test_generator_tool_binds_none_default_without_revalidation(
    llm_args: dict[str, Any],
):
  """A generator tool passes a `str = None` parameter through as-is instead of re-validating it against `str`."""
  from google.adk.agents.llm_agent import Agent
  from google.adk.models.llm_response import LlmResponse
  from google.genai import types

  from .. import testing_utils

  async def tagged_search(query: str, tag: str = None):
    yield {"query": query, "tag": tag}

  fc = types.Part.from_function_call(name="tagged_search", args=llm_args)
  mock_model = testing_utils.MockModel.create([
      LlmResponse(content=types.Content(role="model", parts=[fc])),
      LlmResponse(
          content=types.Content(
              role="model", parts=[types.Part.from_text(text="Done.")]
          )
      ),
  ])
  agent = Agent(name="root_agent", model=mock_model, tools=[tagged_search])
  runner = testing_utils.InMemoryRunner(agent)

  await runner.run_async("search sales")

  second_req_responses = [
      p.function_response.response
      for c in mock_model.requests[1].contents
      for p in c.parts or []
      if p.function_response
  ]
  assert second_req_responses == [{"query": "sales", "tag": None}]
