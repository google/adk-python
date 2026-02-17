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

from unittest.mock import MagicMock

import pytest

# Skip entire module if Python < 3.10 (must be before crewai_tool import)
pytest.importorskip(
    "google.adk.tools.crewai_tool", reason="Requires Python 3.10+"
)

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.crewai_tool import CrewaiTool
from google.adk.tools.tool_context import ToolContext


@pytest.fixture
def mock_tool_context() -> ToolContext:
  """Fixture that provides a mock ToolContext for testing."""
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context.session = MagicMock(spec=Session)
  mock_invocation_context.session.state = MagicMock()
  return ToolContext(invocation_context=mock_invocation_context)


def _simple_crewai_tool(*args, **kwargs):
  """Simple CrewAI-style tool that accepts any keyword arguments."""
  return {
      "search_query": kwargs.get("search_query"),
      "other_param": kwargs.get("other_param"),
  }


def _crewai_tool_with_context(tool_context: ToolContext, *args, **kwargs):
  """CrewAI tool with explicit tool_context parameter."""
  return {
      "search_query": kwargs.get("search_query"),
      "tool_context_present": bool(tool_context),
  }


class MockCrewaiBaseTool:
  """Mock CrewAI BaseTool for testing."""

  def __init__(self, run_func, name="mock_tool", description="Mock tool"):
    self.run = run_func
    self.name = name
    self.description = description
    self.args_schema = MagicMock()
    self.args_schema.model_json_schema.return_value = {
        "type": "object",
        "properties": {
            "search_query": {"type": "string", "description": "Search query"}
        },
    }


def test_crewai_tool_initialization():
  """Test CrewaiTool initialization with various parameters."""
  mock_crewai_tool = MockCrewaiBaseTool(_simple_crewai_tool)

  # Test with custom name and description
  tool = CrewaiTool(
      mock_crewai_tool,
      name="custom_search_tool",
      description="Custom search tool description",
  )

  assert tool.name == "custom_search_tool"
  assert tool.description == "Custom search tool description"
  assert tool.tool == mock_crewai_tool


def test_crewai_tool_initialization_with_tool_defaults():
  """Test CrewaiTool initialization using tool's default name and description."""
  mock_crewai_tool = MockCrewaiBaseTool(
      _simple_crewai_tool,
      name="Serper Dev Tool",
      description="Search the internet with Serper",
  )

  # Test with empty name and description (should use tool defaults)
  tool = CrewaiTool(mock_crewai_tool, name="", description="")

  assert (
      tool.name == "serper_dev_tool"
  )  # Spaces replaced with underscores, lowercased
  assert tool.description == "Search the internet with Serper"


@pytest.mark.asyncio
async def test_crewai_tool_basic_functionality(mock_tool_context):
  """Test basic CrewaiTool functionality with **kwargs parameter passing."""
  mock_crewai_tool = MockCrewaiBaseTool(_simple_crewai_tool)
  tool = CrewaiTool(mock_crewai_tool, name="test_tool", description="Test tool")

  # Test that **kwargs parameters are passed through correctly
  result = await tool.run_async(
      args={"search_query": "test query", "other_param": "test value"},
      tool_context=mock_tool_context,
  )

  assert result["search_query"] == "test query"
  assert result["other_param"] == "test value"


@pytest.mark.asyncio
async def test_crewai_tool_with_tool_context(mock_tool_context):
  """Test CrewaiTool with a tool that has explicit tool_context parameter."""
  mock_crewai_tool = MockCrewaiBaseTool(_crewai_tool_with_context)
  tool = CrewaiTool(
      mock_crewai_tool, name="context_tool", description="Context tool"
  )

  # Test that tool_context is properly injected
  result = await tool.run_async(
      args={"search_query": "test query"},
      tool_context=mock_tool_context,
  )

  assert result["search_query"] == "test query"
  assert result["tool_context_present"] is True


@pytest.mark.asyncio
async def test_crewai_tool_parameter_filtering(mock_tool_context):
  """Test that CrewaiTool filters parameters for non-**kwargs functions."""

  def explicit_params_func(arg1: str, arg2: int):
    """Function with explicit parameters (no **kwargs)."""
    return {"arg1": arg1, "arg2": arg2}

  mock_crewai_tool = MockCrewaiBaseTool(explicit_params_func)
  tool = CrewaiTool(
      mock_crewai_tool, name="explicit_tool", description="Explicit tool"
  )

  # Test that unexpected parameters are filtered out
  result = await tool.run_async(
      args={
          "arg1": "test",
          "arg2": 42,
          "unexpected_param": "should_be_filtered",
      },
      tool_context=mock_tool_context,
  )

  assert result == {"arg1": "test", "arg2": 42}
  # Verify unexpected parameter was filtered out
  assert "unexpected_param" not in result


@pytest.mark.asyncio
async def test_crewai_tool_get_declaration():
  """Test that CrewaiTool properly builds function declarations."""
  mock_crewai_tool = MockCrewaiBaseTool(_simple_crewai_tool)
  tool = CrewaiTool(mock_crewai_tool, name="test_tool", description="Test tool")

  # Test function declaration generation
  declaration = tool._get_declaration()

  # Verify the declaration object structure and content
  assert declaration is not None
  assert declaration.name == "test_tool"
  assert declaration.description == "Test tool"
  assert declaration.parameters is not None

  # Verify that the args_schema was used to build the declaration
  mock_crewai_tool.args_schema.model_json_schema.assert_called_once()
