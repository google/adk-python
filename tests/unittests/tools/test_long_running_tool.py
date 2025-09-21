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

from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
import pytest


def sample_long_running_function(arg1: str, tool_context: ToolContext) -> str:
  """Sample function for testing long running operations.

  Args:
    arg1: First argument
    tool_context: Tool context for the operation

  Returns:
    A string result
  """
  return f"Processing {arg1}"


def sample_function_without_tool_context(arg1: str) -> str:
  """Sample function without tool context.

  Args:
    arg1: First argument

  Returns:
    A string result
  """
  return f"Result: {arg1}"


class TestLongRunningFunctionTool:
  """Test cases for LongRunningFunctionTool class."""

  def test_init(self):
    """Test that the LongRunningFunctionTool is initialized correctly."""
    tool = LongRunningFunctionTool(sample_long_running_function)
    assert tool.name == "sample_long_running_function"
    # The description includes the full docstring
    assert (
        "Sample function for testing long running operations."
        in tool.description
    )
    assert tool.func == sample_long_running_function
    assert tool.is_long_running is True

  def test_is_long_running_property(self):
    """Test that is_long_running property is set to True."""
    tool = LongRunningFunctionTool(sample_long_running_function)
    assert tool.is_long_running is True

  def test_get_declaration_with_description(self):
    """Test that _get_declaration adds warning message to existing description."""
    tool = LongRunningFunctionTool(sample_long_running_function)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == "sample_long_running_function"

    # Check that the original description is preserved
    assert (
        "Sample function for testing long running operations."
        in declaration.description
    )

    # Check that the warning message is added
    expected_warning = (
        "\n\nNOTE: This is a long-running operation. Do not call this tool "
        "again if it has already returned some intermediate or pending status."
    )
    assert expected_warning in declaration.description

  def test_get_declaration_without_description(self):
    """Test that _get_declaration handles functions without descriptions."""

    def no_doc_function():
      pass

    tool = LongRunningFunctionTool(no_doc_function)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert declaration.name == "no_doc_function"

    # Check that the warning message is added as the description
    expected_warning = (
        "NOTE: This is a long-running operation. Do not call this tool "
        "again if it has already returned some intermediate or pending status."
    )
    assert declaration.description == expected_warning

  def test_get_declaration_returns_none_when_parent_returns_none(self):
    """Test that _get_declaration returns None when parent method returns None."""
    tool = LongRunningFunctionTool(sample_long_running_function)

    # Mock the parent method to return None
    with pytest.MonkeyPatch.context() as m:
      m.setattr(
          tool.__class__.__bases__[0], "_get_declaration", lambda self: None
      )
      declaration = tool._get_declaration()
      assert declaration is None

  @pytest.mark.asyncio
  async def test_run_async_functionality(self):
    """Test that run_async works correctly with long running function."""
    tool = LongRunningFunctionTool(sample_long_running_function)
    args = {"arg1": "test_value"}
    result = await tool.run_async(args=args, tool_context=MagicMock())
    assert result == "Processing test_value"

  @pytest.mark.asyncio
  async def test_run_async_without_tool_context(self):
    """Test that run_async works with functions that don't require tool_context."""
    tool = LongRunningFunctionTool(sample_function_without_tool_context)
    args = {"arg1": "test_value"}
    result = await tool.run_async(args=args, tool_context=MagicMock())
    assert result == "Result: test_value"

  def test_inheritance_from_function_tool(self):
    """Test that LongRunningFunctionTool properly inherits from FunctionTool."""
    from google.adk.tools.function_tool import FunctionTool

    tool = LongRunningFunctionTool(sample_long_running_function)
    assert isinstance(tool, FunctionTool)

  def test_description_modification_preserves_original(self):
    """Test that the original description is preserved when adding warning."""
    original_description = (
        "This is a test function for long running operations."
    )

    def test_function():
      pass

    test_function.__doc__ = original_description

    tool = LongRunningFunctionTool(test_function)
    declaration = tool._get_declaration()

    assert declaration is not None
    assert original_description in declaration.description
    assert "NOTE: This is a long-running operation" in declaration.description

  def test_warning_message_format(self):
    """Test that the warning message has the correct format and content."""
    tool = LongRunningFunctionTool(sample_long_running_function)
    declaration = tool._get_declaration()

    assert declaration is not None

    expected_warning = (
        "\n\nNOTE: This is a long-running operation. Do not call this tool "
        "again if it has already returned some intermediate or pending status."
    )

    # Check that the warning appears at the end of the description
    assert declaration.description.endswith(expected_warning)

    # Check for key phrases in the warning
    assert "long-running operation" in declaration.description
    assert "Do not call this tool again" in declaration.description
    assert "intermediate or pending status" in declaration.description
