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

"""Tests for context_utils module."""

from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.tools.tool_context import ToolContext
from google.adk.utils.context_utils import find_context_parameter


class TestFindContextParameter:
  """Tests for find_context_parameter function."""

  def test_find_context_parameter_with_context_type(self):
    """Test detection of Context type annotation."""

    def my_tool(query: str, ctx: Context) -> str:
      return query

    assert find_context_parameter(my_tool) == 'ctx'

  def test_find_context_parameter_with_string_annotation(self):
    """Test detection of string annotation 'Context'."""

    def my_tool(query: str, ctx: 'Context') -> str:
      return query

    assert find_context_parameter(my_tool) == 'ctx'

  def test_find_context_parameter_with_string_tool_context(self):
    """Test detection of string annotation 'ToolContext'."""

    def my_tool(query: str, ctx: 'ToolContext') -> str:
      return query

    assert find_context_parameter(my_tool) == 'ctx'

  def test_find_context_parameter_with_string_optional_context(self):
    """Test detection of string annotation 'Optional[Context]'."""

    def my_tool(query: str, ctx: 'Optional[Context]' = None) -> str:
      return query

    assert find_context_parameter(my_tool) == 'ctx'

  def test_find_context_parameter_with_tool_context_type(self):
    """Test detection of ToolContext type annotation."""

    def my_tool(query: str, tool_context: ToolContext) -> str:
      return query

    assert find_context_parameter(my_tool) == 'tool_context'

  def test_find_context_parameter_with_callback_context_type(self):
    """Test detection of CallbackContext type annotation."""

    def my_callback(ctx: CallbackContext) -> None:
      pass

    assert find_context_parameter(my_callback) == 'ctx'

  def test_find_context_parameter_with_optional_context(self):
    """Test detection of Optional[Context] type annotation."""

    def my_tool(query: str, context: Optional[Context] = None) -> str:
      return query

    assert find_context_parameter(my_tool) == 'context'

  def test_find_context_parameter_with_custom_name(self):
    """Test that any parameter name works with Context type."""

    def my_tool(query: str, my_custom_ctx: Context) -> str:
      return query

    assert find_context_parameter(my_tool) == 'my_custom_ctx'

  def test_find_context_parameter_no_context(self):
    """Test function without context parameter returns None."""

    def my_tool(query: str, count: int) -> str:
      return query

    assert find_context_parameter(my_tool) is None

  def test_find_context_parameter_no_annotations(self):
    """Test function without type annotations returns None."""

    def my_tool(query, ctx):
      return query

    assert find_context_parameter(my_tool) is None

  def test_find_context_parameter_with_none_func(self):
    """Test that None function returns None."""
    assert find_context_parameter(None) is None

  def test_find_context_parameter_returns_first_match(self):
    """Test that first context parameter is returned if multiple exist."""

    def my_tool(first_ctx: Context, second_ctx: Context) -> str:
      return 'test'

    assert find_context_parameter(my_tool) == 'first_ctx'

  def test_find_context_parameter_with_mixed_params(self):
    """Test context parameter detection with various other parameters."""

    def my_tool(
        query: str,
        count: int,
        ctx: Context,
        optional_param: Optional[str] = None,
    ) -> str:
      return query

    assert find_context_parameter(my_tool) == 'ctx'
