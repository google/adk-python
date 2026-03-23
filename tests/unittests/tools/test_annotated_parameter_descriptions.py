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

"""Tests for per-parameter descriptions via Annotated[T, Field(description=...)]."""

from typing import Annotated
from typing import Optional

from google.adk.tools._automatic_function_calling_util import _extract_base_type_from_annotated
from google.adk.tools._automatic_function_calling_util import _extract_field_info_from_annotated
from google.adk.tools._automatic_function_calling_util import _get_fields_dict
from google.adk.tools._automatic_function_calling_util import build_function_declaration
from google.adk.utils.variant_utils import GoogleLLMVariant
from pydantic import Field
import pytest


class TestExtractFieldInfoFromAnnotated:
  """Tests for _extract_field_info_from_annotated helper function."""

  def test_extract_field_info_with_description(self):
    """Test extracting FieldInfo with description from Annotated type."""
    annotation = Annotated[str, Field(description="A test description")]
    field_info = _extract_field_info_from_annotated(annotation)

    assert field_info is not None
    assert field_info.description == "A test description"

  def test_extract_field_info_without_description(self):
    """Test extracting FieldInfo without description from Annotated type."""
    annotation = Annotated[str, Field()]
    field_info = _extract_field_info_from_annotated(annotation)

    assert field_info is not None
    assert field_info.description is None

  def test_extract_field_info_not_annotated(self):
    """Test that non-Annotated types return None."""
    field_info = _extract_field_info_from_annotated(str)
    assert field_info is None

  def test_extract_field_info_annotated_without_field(self):
    """Test that Annotated without Field returns None."""
    annotation = Annotated[str, "some_metadata"]
    field_info = _extract_field_info_from_annotated(annotation)
    assert field_info is None

  def test_extract_field_info_with_multiple_metadata(self):
    """Test that Field is found even with multiple metadata items."""
    annotation = Annotated[
        str,
        "some_string_metadata",
        Field(description="Found it!"),
        42,
    ]
    field_info = _extract_field_info_from_annotated(annotation)

    assert field_info is not None
    assert field_info.description == "Found it!"


class TestExtractBaseTypeFromAnnotated:
  """Tests for _extract_base_type_from_annotated helper function."""

  def test_extract_base_type_from_annotated(self):
    """Test extracting base type from Annotated."""
    annotation = Annotated[str, Field(description="test")]
    base_type = _extract_base_type_from_annotated(annotation)
    assert base_type is str

  def test_extract_base_type_from_non_annotated(self):
    """Test that non-Annotated types are returned as-is."""
    base_type = _extract_base_type_from_annotated(int)
    assert base_type is int

  def test_extract_base_type_complex_annotated(self):
    """Test extracting complex base types from Annotated."""
    from typing import List

    annotation = Annotated[List[str], Field(description="A list of strings")]
    base_type = _extract_base_type_from_annotated(annotation)
    assert base_type == List[str]


class TestGetFieldsDict:
  """Tests for _get_fields_dict with Annotated parameter descriptions."""

  def test_get_fields_dict_with_annotated_description(self):
    """Test that _get_fields_dict extracts descriptions from Annotated."""

    def sample_func(
        repo: Annotated[
            str,
            Field(description="Repository URL from get_repository_info"),
        ],
        branch: Annotated[
            str,
            Field(description="Base branch for development"),
        ],
    ) -> dict:
      return {}

    fields = _get_fields_dict(sample_func)

    assert "repo" in fields
    assert "branch" in fields

    # Check that descriptions are extracted
    repo_type, repo_field = fields["repo"]
    branch_type, branch_field = fields["branch"]

    assert repo_type is str
    assert repo_field.description == "Repository URL from get_repository_info"
    assert branch_type is str
    assert branch_field.description == "Base branch for development"

  def test_get_fields_dict_mixed_annotations(self):
    """Test _get_fields_dict with mix of Annotated and regular params."""

    def sample_func(
        annotated_param: Annotated[
            str,
            Field(description="This has a description"),
        ],
        regular_param: str,
    ) -> None:
      pass

    fields = _get_fields_dict(sample_func)

    annotated_type, annotated_field = fields["annotated_param"]
    regular_type, regular_field = fields["regular_param"]

    assert annotated_type is str
    assert annotated_field.description == "This has a description"
    assert regular_type is str
    assert regular_field.description is None

  def test_get_fields_dict_with_default_values(self):
    """Test that default values are preserved with Annotated types."""

    def sample_func(
        required_param: Annotated[
            str,
            Field(description="Required parameter"),
        ],
        optional_param: Annotated[
            str,
            Field(description="Optional parameter"),
        ] = "default_value",
    ) -> None:
      pass

    fields = _get_fields_dict(sample_func)

    _, required_field = fields["required_param"]
    _, optional_field = fields["optional_param"]

    assert required_field.description == "Required parameter"
    assert optional_field.description == "Optional parameter"
    assert optional_field.default == "default_value"

  def test_get_fields_dict_with_optional_annotated(self):
    """Test Annotated with Optional type."""

    def sample_func(
        optional_param: Annotated[
            Optional[str],
            Field(description="Optional string parameter"),
        ] = None,
    ) -> None:
      pass

    fields = _get_fields_dict(sample_func)

    param_type, param_field = fields["optional_param"]
    assert param_field.description == "Optional string parameter"
    assert param_field.default is None


class TestBuildFunctionDeclaration:
  """Tests for build_function_declaration with Annotated descriptions."""

  def test_build_declaration_with_annotated_params(self):
    """Test that build_function_declaration includes parameter descriptions."""

    def create_task(
        repository: Annotated[
            str,
            Field(
                description=(
                    "Full GitLab repository URL. "
                    "MUST be obtained from get_repository_info."
                )
            ),
        ],
        base_branch: Annotated[
            str,
            Field(
                description=(
                    "Base branch for development (e.g. 'main', 'develop'). "
                    "MUST be obtained from get_repository_info."
                )
            ),
        ],
    ) -> dict:
      """Create a new task in the repository."""
      return {}

    declaration = build_function_declaration(
        create_task,
        variant=GoogleLLMVariant.GEMINI_API,
    )

    assert declaration.name == "create_task"
    assert declaration.description == "Create a new task in the repository."
    assert declaration.parameters is not None
    assert declaration.parameters.properties is not None

    # Check that descriptions are in the schema
    repo_schema = declaration.parameters.properties.get("repository")
    branch_schema = declaration.parameters.properties.get("base_branch")

    assert repo_schema is not None
    assert branch_schema is not None

    # The descriptions should be present in the schema
    assert repo_schema.description is not None
    assert "GitLab repository URL" in repo_schema.description
    assert branch_schema.description is not None
    assert "Base branch for development" in branch_schema.description

  def test_build_declaration_without_annotated_params(self):
    """Test build_function_declaration without Annotated still works."""

    def simple_func(name: str, count: int) -> str:
      """A simple function."""
      return name * count

    declaration = build_function_declaration(
        simple_func,
        variant=GoogleLLMVariant.GEMINI_API,
    )

    assert declaration.name == "simple_func"
    assert declaration.parameters is not None
    assert declaration.parameters.properties is not None
    assert "name" in declaration.parameters.properties
    assert "count" in declaration.parameters.properties


class TestIntegrationWithFunctionTool:
  """Integration tests for FunctionTool with Annotated descriptions."""

  def test_function_tool_with_annotated_params(self):
    """Test that FunctionTool works with Annotated parameter descriptions."""
    from google.adk.tools.function_tool import FunctionTool

    def search_repos(
        query: Annotated[
            str,
            Field(description="Search query for repositories"),
        ],
        limit: Annotated[
            int,
            Field(description="Maximum number of results to return"),
        ] = 10,
    ) -> list:
      """Search for repositories matching the query."""
      return []

    tool = FunctionTool(search_repos)

    assert tool.name == "search_repos"
    assert tool.description == "Search for repositories matching the query."

    # Get the function declaration (internal method)
    declaration = tool._get_declaration()
    assert declaration is not None
    assert declaration.parameters is not None

  @pytest.mark.asyncio
  async def test_function_tool_execution_with_annotated_params(self):
    """Test that FunctionTool executes correctly with Annotated params."""
    from unittest.mock import MagicMock

    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.sessions.session import Session
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.tools.tool_context import ToolContext

    def greet(
        name: Annotated[
            str,
            Field(description="Name of the person to greet"),
        ],
        greeting: Annotated[
            str,
            Field(description="Greeting to use"),
        ] = "Hello",
    ) -> str:
      """Greet a person."""
      return f"{greeting}, {name}!"

    tool = FunctionTool(greet)

    mock_invocation_context = MagicMock(spec=InvocationContext)
    mock_invocation_context.session = MagicMock(spec=Session)
    mock_invocation_context.session.state = MagicMock()
    tool_context = ToolContext(invocation_context=mock_invocation_context)

    result = await tool.run_async(
        args={"name": "World"},
        tool_context=tool_context,
    )

    assert result == "Hello, World!"

    result_custom = await tool.run_async(
        args={"name": "Alice", "greeting": "Hi"},
        tool_context=tool_context,
    )

    assert result_custom == "Hi, Alice!"
