# Copyright 2025 Google LLC
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

from unittest.mock import MagicMock

from google.adk.tools.function_tool import FunctionTool
import pytest


def function_for_testing_with_no_args():
  """Function for testing with no args."""
  pass


async def async_function_for_testing_with_1_arg_and_tool_context(
    arg1, tool_context
):
  """Async function for testing with 1 arge and tool context."""
  assert arg1
  assert tool_context
  return arg1


async def async_function_for_testing_with_2_arg_and_no_tool_context(arg1, arg2):
  """Async function for testing with 2 arge and no tool context."""
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
  """Function for testing with 1 arge and tool context."""
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
  """Function for testing with 2 arge and no tool context."""
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
      "error": """Invoking `function_for_testing_with_2_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg2
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_1_missing_arg_async_func():
  """Test that run_async calls the function with 1 missing arg in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_2_arg_and_no_tool_context)
  args = {"arg2": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": """Invoking `async_function_for_testing_with_2_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_3_missing_arg_sync_func():
  """Test that run_async calls the function with 3 missing args in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_4_arg_and_no_tool_context)
  args = {"arg2": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": """Invoking `function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_3_missing_arg_async_func():
  """Test that run_async calls the function with 3 missing args in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_4_arg_and_no_tool_context)
  args = {"arg3": "test_value_1"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": """Invoking `async_function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_missing_all_arg_sync_func():
  """Test that run_async calls the function with all missing args in signature (synchronous function)."""
  tool = FunctionTool(function_for_testing_with_4_arg_and_no_tool_context)
  args = {}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": """Invoking `function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_missing_all_arg_async_func():
  """Test that run_async calls the function with all missing args in signature (async function)."""
  tool = FunctionTool(async_function_for_testing_with_4_arg_and_no_tool_context)
  args = {}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == {
      "error": """Invoking `async_function_for_testing_with_4_arg_and_no_tool_context()` failed as the following mandatory input parameters are not present:
arg1
arg2
arg3
arg4
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
  }


@pytest.mark.asyncio
async def test_run_async_with_optional_args_not_set_sync_func():
  """Test that run_async calls the function for sync funciton with optional args not set."""

  def func_with_optional_args(arg1, arg2=None, *, arg3, arg4=None, **kwargs):
    return f"{arg1},{arg3}"

  tool = FunctionTool(func_with_optional_args)
  args = {"arg1": "test_value_1", "arg3": "test_value_3"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1,test_value_3"


@pytest.mark.asyncio
async def test_run_async_with_optional_args_not_set_async_func():
  """Test that run_async calls the function for async funciton with optional args not set."""

  async def async_func_with_optional_args(
      arg1, arg2=None, *, arg3, arg4=None, **kwargs
  ):
    return f"{arg1},{arg3}"

  tool = FunctionTool(async_func_with_optional_args)
  args = {"arg1": "test_value_1", "arg3": "test_value_3"}
  result = await tool.run_async(args=args, tool_context=MagicMock())
  assert result == "test_value_1,test_value_3"
