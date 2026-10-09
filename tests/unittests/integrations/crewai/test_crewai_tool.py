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

# Skip the module when the optional crewai dependency is not installed. Guard on
# the third-party dep itself rather than the adk wrapper, so a real import bug in
# crewai_tool surfaces as a failure instead of being silently skipped.
pytest.importorskip(
    "crewai.tools", reason="Requires crewai (google-adk[extensions])"
)

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.features import FeatureName
from google.adk.features._feature_registry import temporary_feature_override
from google.adk.integrations.crewai import CrewaiTool
from google.adk.sessions.session import Session
from google.adk.tools.tool_context import ToolContext
import pydantic


@pytest.fixture
def mock_tool_context() -> ToolContext:
  """Fixture that provides a mock ToolContext for testing."""
  mock_invocation_context = MagicMock(spec=InvocationContext)
  mock_invocation_context._state_schema = None
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


def _crewai_tool_with_context_type(ctx: Context, *args, **kwargs):
  """CrewAI tool with Context type annotation."""
  return {
      "search_query": kwargs.get("search_query"),
      "context_present": bool(ctx),
  }


class MockCrewaiBaseTool:
  """Mock CrewAI BaseTool for testing."""

  def __init__(
      self,
      run_func,
      name="mock_tool",
      description="Mock tool",
      args_schema=None,
  ):
    self.run = run_func
    self.name = name
    self.description = description
    if args_schema is not None:
      self.args_schema = args_schema
    else:
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


@pytest.mark.asyncio
async def test_crewai_tool_with_context_type_annotation(mock_tool_context):
  """Test CrewaiTool with Context type annotation and custom parameter name."""
  mock_crewai_tool = MockCrewaiBaseTool(_crewai_tool_with_context_type)
  tool = CrewaiTool(
      mock_crewai_tool,
      name="context_type_tool",
      description="Context type tool",
  )

  # Verify the context parameter is detected by type
  assert tool._context_param_name == "ctx"

  # Test that context is properly injected
  result = await tool.run_async(
      args={"search_query": "test query"},
      tool_context=mock_tool_context,
  )

  assert result["search_query"] == "test query"
  assert result["context_present"]


@pytest.mark.asyncio
async def test_crewai_tool_invalid_argument_validation(mock_tool_context):
  """Test that CrewaiTool returns validation errors when argument types are invalid."""

  class TypedArgsSchema(pydantic.BaseModel):
    arg1: str
    arg2: int

  def mock_crewai_run(*args, **kwargs):
    """CrewAI tool run signature is (*args, **kwargs)."""
    return {"arg1": kwargs.get("arg1"), "arg2": kwargs.get("arg2")}

  mock_crewai_tool = MockCrewaiBaseTool(
      mock_crewai_run,
      name="typed_tool",
      description="Typed tool",
      args_schema=TypedArgsSchema,
  )
  tool = CrewaiTool(
      mock_crewai_tool, name="typed_tool", description="Typed tool"
  )

  with temporary_feature_override(
      FeatureName.FUNCTION_TOOL_ARG_VALIDATION, True
  ):
    # Pass an invalid type (non-numeric string for int parameter)
    result = await tool.run_async(
        args={"arg1": "test", "arg2": "invalid_int"},
        tool_context=mock_tool_context,
    )

    assert isinstance(result, dict)
    assert "error" in result
    assert "validation error" in result["error"].lower()
    assert "arg2" in result["error"]

    # Verify valid coercion succeeds
    result_valid = await tool.run_async(
        args={"arg1": "test", "arg2": "42"},
        tool_context=mock_tool_context,
    )
    assert result_valid == {"arg1": "test", "arg2": 42}


@pytest.mark.asyncio
async def test_crewai_tool_validation_disabled_by_default(mock_tool_context):
  """Test that CrewaiTool allows lax argument types when flag is disabled."""

  class TypedArgsSchema(pydantic.BaseModel):
    arg1: str
    arg2: int

  def mock_crewai_run(*args, **kwargs):
    return {"arg1": kwargs.get("arg1"), "arg2": kwargs.get("arg2")}

  mock_crewai_tool = MockCrewaiBaseTool(
      mock_crewai_run,
      name="typed_tool",
      description="Typed tool",
      args_schema=TypedArgsSchema,
  )
  tool = CrewaiTool(
      mock_crewai_tool, name="typed_tool", description="Typed tool"
  )

  # Flag is disabled by default; unvalidated arguments pass through to the tool
  result = await tool.run_async(
      args={"arg1": "test", "arg2": "invalid_int"},
      tool_context=mock_tool_context,
  )
  assert result == {"arg1": "test", "arg2": "invalid_int"}
