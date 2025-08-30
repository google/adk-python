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

from __future__ import annotations

from unittest.mock import MagicMock

from google.adk.tools.griptape_tool import GriptapeTool
from griptape.artifacts import BaseArtifact
from griptape.artifacts import TextArtifact
from griptape.tools import BaseTool
from griptape.utils.decorators import activity
import pytest
from schema import Literal
from schema import Schema


class AdditionTool(BaseTool):
  """A simple Griptape tool that adds two numbers."""

  @activity(
      config={
          "description": "Can be used to add two numbers together",
          "schema": Schema({
              Literal("x", description="First number to add"): int,
              Literal("y", description="Second number to add"): int,
          }),
      }
  )
  def add_numbers(self, params: dict) -> BaseArtifact:
    """Add two numbers and return the result."""
    x = params["x"]
    y = params["y"]
    result = x + y
    return TextArtifact(str(result))


class GreetingTool(BaseTool):
  """A Griptape tool that creates greetings with different parameters."""

  @activity(
      config={
          "description": "Generate a personalized greeting message",
          "schema": Schema({
              Literal("name", description="Name of the person to greet"): str,
              Literal(
                  "city", description="City where the person is located"
              ): str,
              Literal(
                  "formal", description="Whether to use formal greeting"
              ): bool,
          }),
      }
  )
  def create_greeting(self, params: dict) -> BaseArtifact:
    """Create a personalized greeting."""
    name = params["name"]
    city = params["city"]
    formal = params["formal"]

    if formal:
      greeting = (
          f"Good day, {name}. I hope you are enjoying your time in {city}."
      )
    else:
      greeting = f"Hey {name}! How's life in {city}?"

    return TextArtifact(greeting)

  @activity(
      config={
          "description": "Generate a farewell message",
          "schema": Schema({
              Literal(
                  "name", description="Name of the person to bid farewell"
              ): str,
          }),
      }
  )
  def create_farewell(self, params: dict) -> BaseArtifact:
    """Create a farewell message."""
    name = params["name"]
    return TextArtifact(f"Goodbye, {name}! See you later!")


class TestGriptapeTool:
  """Test cases for GriptapeTool wrapper."""

  def test_griptape_tool_initialization(self):
    """Test that GriptapeTool can be initialized with a real Griptape tool."""
    addition_tool = AdditionTool()
    wrapped_tool = GriptapeTool(addition_tool)

    assert wrapped_tool.name == "add_numbers"  # Should use activity method name
    assert wrapped_tool._griptape_tool == addition_tool

  def test_griptape_tool_with_custom_name_and_description(self):
    """Test GriptapeTool with overridden name and description."""
    addition_tool = AdditionTool()
    wrapped_tool = GriptapeTool(
        addition_tool, name="CustomAdd", description="Custom addition tool"
    )

    assert wrapped_tool.name == "CustomAdd"
    assert wrapped_tool.description == "Custom addition tool"

  def test_griptape_tool_invalid_tool(self):
    """Test that GriptapeTool raises error for invalid tools."""
    invalid_tool = object()  # No run method

    with pytest.raises(ValueError, match="Tool must be a Griptape BaseTool"):
      GriptapeTool(invalid_tool)

  def test_griptape_tool_function_declaration(self):
    """Test that function declaration can be generated."""
    addition_tool = AdditionTool()
    wrapped_tool = GriptapeTool(addition_tool)

    declaration = wrapped_tool._get_declaration()

    assert declaration.name == "add_numbers"  # Should use activity method name
    assert hasattr(declaration, "description")

  @pytest.mark.asyncio
  async def test_griptape_tool_execution(self):
    """Test that the wrapped tool can be executed."""
    addition_tool = AdditionTool()
    wrapped_tool = GriptapeTool(addition_tool)

    result = await wrapped_tool.run_async(
        args={"x": 5, "y": 3}, tool_context=MagicMock()
    )

    # The result should be the sum as a string (since Griptape returns TextArtifact)
    assert result == "8"

  def test_griptape_tool_different_schema(self):
    """Test GriptapeTool with a different schema (string, string, bool parameters)."""
    greeting_tool = GreetingTool()
    wrapped_tool = GriptapeTool(
        greeting_tool, activity_method="create_greeting"
    )

    # Check that it detects the correct method name
    assert wrapped_tool.name == "create_greeting"

    # Check function declaration works
    declaration = wrapped_tool._get_declaration()
    assert declaration.name == "create_greeting"

  @pytest.mark.asyncio
  async def test_griptape_tool_different_schema_execution(self):
    """Test execution of GriptapeTool with different parameter types."""
    greeting_tool = GreetingTool()
    wrapped_tool = GriptapeTool(
        greeting_tool, activity_method="create_greeting"
    )

    result = await wrapped_tool.run_async(
        args={"name": "Alice", "city": "Paris", "formal": True},
        tool_context=MagicMock(),
    )

    assert (
        result == "Good day, Alice. I hope you are enjoying your time in Paris."
    )

    # Test with informal greeting
    result = await wrapped_tool.run_async(
        args={"name": "Bob", "city": "Tokyo", "formal": False},
        tool_context=MagicMock(),
    )

    assert result == "Hey Bob! How's life in Tokyo?"

  def test_griptape_tool_specific_activity_method(self):
    """Test GriptapeTool with specific activity method selection."""
    greeting_tool = GreetingTool()

    # Wrap with specific activity method
    wrapped_tool = GriptapeTool(
        greeting_tool, activity_method="create_farewell"
    )

    # Check that it uses the specified method
    assert wrapped_tool.name == "create_farewell"

    # Check function declaration works
    declaration = wrapped_tool._get_declaration()
    assert declaration.name == "create_farewell"

  @pytest.mark.asyncio
  async def test_griptape_tool_specific_activity_method_execution(self):
    """Test execution of GriptapeTool with specific activity method."""
    greeting_tool = GreetingTool()
    wrapped_tool = GriptapeTool(
        greeting_tool, activity_method="create_farewell"
    )

    result = await wrapped_tool.run_async(
        args={"name": "Charlie"}, tool_context=MagicMock()
    )

    assert result == "Goodbye, Charlie! See you later!"

  def test_griptape_tool_invalid_activity_method(self):
    """Test that GriptapeTool raises AttributeError for invalid activity method."""
    greeting_tool = GreetingTool()

    with pytest.raises(
        AttributeError, match="Tool does not have method 'nonexistent_method'"
    ):
      GriptapeTool(greeting_tool, activity_method="nonexistent_method")

  def test_griptape_tool_no_activity_methods(self):
    """Test that GriptapeTool raises AttributeError when no activity methods found."""

    # Create a simple class without activity methods that inherits from BaseTool
    class EmptyTool(BaseTool):
      pass

    empty_tool = EmptyTool()

    with pytest.raises(AttributeError, match="No activity method found"):
      GriptapeTool(empty_tool)
