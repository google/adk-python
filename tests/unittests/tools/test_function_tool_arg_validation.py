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

"""Tests for FunctionTool argument type validation and coercion."""

from enum import Enum
from typing import Optional
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
import pytest


class Color(Enum):
  RED = "red"
  GREEN = "green"
  BLUE = "blue"


def int_func(num: int) -> int:
  return num


def float_func(val: float) -> float:
  return val


def bool_func(flag: bool) -> bool:
  return flag


def enum_func(color: Color) -> str:
  return color.value


def list_int_func(nums: list[int]) -> list[int]:
  return nums


def optional_int_func(num: Optional[int] = None) -> Optional[int]:
  return num


def multi_param_func(name: str, count: int, flag: bool) -> dict:
  return {"name": name, "count": count, "flag": flag}


# --- _preprocess_args coercion tests ---


class TestArgCoercion:

  def test_string_to_int(self):
    tool = FunctionTool(int_func)
    args, errors = tool._preprocess_args_with_validation({"num": "42"})
    assert errors == []
    assert args["num"] == 42
    assert isinstance(args["num"], int)

  def test_float_to_int(self):
    """Pydantic lax mode truncates float to int."""
    tool = FunctionTool(int_func)
    args, errors = tool._preprocess_args_with_validation({"num": 3.0})
    assert errors == []
    assert args["num"] == 3
    assert isinstance(args["num"], int)

  def test_string_to_float(self):
    tool = FunctionTool(float_func)
    args, errors = tool._preprocess_args_with_validation({"val": "3.14"})
    assert errors == []
    assert abs(args["val"] - 3.14) < 1e-9

  def test_int_to_float(self):
    tool = FunctionTool(float_func)
    args, errors = tool._preprocess_args_with_validation({"val": 5})
    assert errors == []
    assert args["val"] == 5.0
    assert isinstance(args["val"], float)

  def test_enum_valid_value(self):
    tool = FunctionTool(enum_func)
    args, errors = tool._preprocess_args_with_validation({"color": "red"})
    assert errors == []
    assert args["color"] == Color.RED

  def test_enum_invalid_value(self):
    tool = FunctionTool(enum_func)
    args, errors = tool._preprocess_args_with_validation({"color": "purple"})
    assert len(errors) == 1
    assert "color" in errors[0]

  def test_list_int_coercion(self):
    tool = FunctionTool(list_int_func)
    args, errors = tool._preprocess_args_with_validation({"nums": ["1", "2", "3"]})
    assert errors == []
    assert args["nums"] == [1, 2, 3]

  def test_optional_none_skipped(self):
    tool = FunctionTool(optional_int_func)
    args, errors = tool._preprocess_args_with_validation({"num": None})
    assert errors == []
    assert args["num"] is None

  def test_optional_value_coerced(self):
    tool = FunctionTool(optional_int_func)
    args, errors = tool._preprocess_args_with_validation({"num": "7"})
    assert errors == []
    assert args["num"] == 7

  def test_bool_from_int(self):
    tool = FunctionTool(bool_func)
    args, errors = tool._preprocess_args_with_validation({"flag": 1})
    assert errors == []
    assert args["flag"] is True


# --- _preprocess_args validation error tests ---


class TestArgValidationErrors:

  def test_string_for_int_returns_error(self):
    tool = FunctionTool(int_func)
    args, errors = tool._preprocess_args_with_validation({"num": "foobar"})
    assert len(errors) == 1
    assert "num" in errors[0]

  def test_none_for_required_int_returns_error(self):
    """None for a non-Optional int should be flagged."""
    tool = FunctionTool(int_func)
    # None passed for a required int param. The Optional unwrap won't
    # trigger because the annotation is plain `int`, not Optional[int].
    # TypeAdapter(int).validate_python(None) raises ValidationError.
    args, errors = tool._preprocess_args_with_validation({"num": None})
    assert len(errors) == 1
    assert "num" in errors[0]

  def test_multiple_param_errors(self):
    tool = FunctionTool(multi_param_func)
    args, errors = tool._preprocess_args_with_validation(
        {"name": 123, "count": "not_a_number", "flag": "not_a_bool"}
    )
    # All three fail: pydantic rejects int->str, "not_a_number"->int,
    # and "not_a_bool"->bool.
    assert len(errors) == 3
    assert any("name" in e for e in errors)
    assert any("count" in e for e in errors)
    assert any("flag" in e for e in errors)


# --- run_async integration tests ---


def _make_tool_context():
  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock
  return tool_context_mock


class TestRunAsyncValidation:

  @pytest.mark.asyncio
  async def test_invalid_arg_returns_error_to_llm(self):
    tool = FunctionTool(int_func)
    result = await tool.run_async(
        args={"num": "foobar"}, tool_context=_make_tool_context()
    )
    assert isinstance(result, dict)
    assert "error" in result
    assert "validation error" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_valid_coercion_invokes_function(self):
    tool = FunctionTool(int_func)
    result = await tool.run_async(
        args={"num": "42"}, tool_context=_make_tool_context()
    )
    assert result == 42

  @pytest.mark.asyncio
  async def test_enum_invalid_returns_error(self):
    tool = FunctionTool(enum_func)
    result = await tool.run_async(
        args={"color": "purple"}, tool_context=_make_tool_context()
    )
    assert isinstance(result, dict)
    assert "error" in result

  @pytest.mark.asyncio
  async def test_enum_valid_invokes_function(self):
    tool = FunctionTool(enum_func)
    result = await tool.run_async(
        args={"color": "green"}, tool_context=_make_tool_context()
    )
    assert result == "green"
