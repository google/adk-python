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


class TestNestedPydanticModels:
  """Tests for nested Pydantic BaseModel support with Field descriptions."""

  def test_single_level_nested_model(self):
    """Test that nested Pydantic model Field descriptions are inlined."""
    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class Address(BaseModel):
      """User's address information."""

      street: str = Field(description="Street name and number")
      city: str = Field(description="City name")
      zipcode: str = Field(description="Postal code (5 digits)")

    def create_user(
        address: Annotated[
            Address, Field(description="User's residential address")
        ],
    ) -> dict:
      """Create a new user with address."""
      return {}

    schema = _get_pydantic_schema(create_user)

    # Check that address parameter exists
    assert "properties" in schema
    assert "address" in schema["properties"]

    address_schema = schema["properties"]["address"]

    # Check that the parameter-level description is preserved
    assert address_schema.get("description") == "User's residential address"

    # Check that nested properties are inlined (not using $ref)
    assert (
        "properties" in address_schema
    ), "Nested properties should be inlined, not using $ref"
    assert "$ref" not in address_schema, "Should not have $ref after resolution"
    assert (
        "allOf" not in address_schema
    ), "Should not have allOf after resolution"

    # Check that nested Field descriptions are present
    nested_props = address_schema["properties"]
    assert "street" in nested_props
    assert "city" in nested_props
    assert "zipcode" in nested_props

    assert nested_props["street"].get("description") == "Street name and number"
    assert nested_props["city"].get("description") == "City name"
    assert (
        nested_props["zipcode"].get("description") == "Postal code (5 digits)"
    )

    # Verify $defs is removed after inlining
    assert "$defs" not in schema, "$defs should be removed after inlining"

  def test_multi_level_nested_model(self):
    """Test that doubly-nested Pydantic models preserve all descriptions."""
    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class ContactInfo(BaseModel):
      """Contact information."""

      email: str = Field(description="Email address in format user@domain.com")
      phone: str = Field(description="Phone number with country code")

    class Person(BaseModel):
      """Person information."""

      name: str = Field(description="Person's full name")
      age: int = Field(description="Person's age in years")
      contact: ContactInfo = Field(description="Contact information")

    def create_user(
        person: Annotated[
            Person, Field(description="User personal information")
        ],
    ) -> dict:
      """Create a new user."""
      return {}

    schema = _get_pydantic_schema(create_user)

    # Check first level (person parameter)
    person_schema = schema["properties"]["person"]
    assert person_schema.get("description") == "User personal information"
    assert "properties" in person_schema

    # Check second level (name, age, contact)
    person_props = person_schema["properties"]
    assert person_props["name"].get("description") == "Person's full name"
    assert person_props["age"].get("description") == "Person's age in years"
    assert person_props["contact"].get("description") == "Contact information"

    # Check third level (email, phone within contact)
    assert "properties" in person_props["contact"]
    contact_props = person_props["contact"]["properties"]
    assert (
        contact_props["email"].get("description")
        == "Email address in format user@domain.com"
    )
    assert (
        contact_props["phone"].get("description")
        == "Phone number with country code"
    )

  def test_nested_model_with_list(self):
    """Test that List of nested Pydantic models works correctly."""
    from typing import List

    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class Tag(BaseModel):
      """A tag."""

      name: str = Field(description="Tag name")
      color: str = Field(description="Tag color in hex format")

    def create_item(
        tags: Annotated[
            List[Tag], Field(description="List of tags for the item")
        ],
    ) -> dict:
      """Create an item with tags."""
      return {}

    schema = _get_pydantic_schema(create_item)

    tags_schema = schema["properties"]["tags"]
    assert tags_schema.get("description") == "List of tags for the item"
    assert tags_schema.get("type") == "array"
    assert "items" in tags_schema

    # Check that items schema has inlined properties
    items_schema = tags_schema["items"]
    assert "properties" in items_schema
    assert items_schema["properties"]["name"].get("description") == "Tag name"
    assert (
        items_schema["properties"]["color"].get("description")
        == "Tag color in hex format"
    )

  def test_nested_model_with_optional(self):
    """Test that Optional nested Pydantic models preserve descriptions."""
    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class Metadata(BaseModel):
      """Metadata information."""

      key: str = Field(description="Metadata key")
      value: str = Field(description="Metadata value")

    def create_item(
        metadata: Annotated[
            Optional[Metadata], Field(description="Optional metadata")
        ] = None,
    ) -> dict:
      """Create an item with optional metadata."""
      return {}

    schema = _get_pydantic_schema(create_item)

    metadata_schema = schema["properties"]["metadata"]

    # Optional handling might use anyOf, but descriptions should still be there
    # Check if properties are accessible (could be in anyOf structure)
    if "properties" in metadata_schema:
      # Direct properties
      assert (
          metadata_schema["properties"]["key"].get("description")
          == "Metadata key"
      )
    elif "anyOf" in metadata_schema:
      # Look for the object definition in anyOf
      for variant in metadata_schema["anyOf"]:
        if variant.get("type") == "object" and "properties" in variant:
          assert (
              variant["properties"]["key"].get("description") == "Metadata key"
          )
          break

  def test_mixed_nested_and_simple_params(self):
    """Test function with both nested models and simple parameters."""
    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class Config(BaseModel):
      """Configuration."""

      timeout: int = Field(description="Timeout in seconds")
      retries: int = Field(description="Number of retries")

    def execute_task(
        task_name: Annotated[str, Field(description="Name of the task")],
        config: Annotated[Config, Field(description="Task configuration")],
        dry_run: Annotated[
            bool, Field(description="Run in dry-run mode")
        ] = False,
    ) -> dict:
      """Execute a task with configuration."""
      return {}

    schema = _get_pydantic_schema(execute_task)

    # Check simple parameters
    assert (
        schema["properties"]["task_name"].get("description")
        == "Name of the task"
    )
    assert (
        schema["properties"]["dry_run"].get("description")
        == "Run in dry-run mode"
    )

    # Check nested model
    config_schema = schema["properties"]["config"]
    assert config_schema.get("description") == "Task configuration"
    assert "properties" in config_schema
    assert (
        config_schema["properties"]["timeout"].get("description")
        == "Timeout in seconds"
    )
    assert (
        config_schema["properties"]["retries"].get("description")
        == "Number of retries"
    )

  def test_nested_model_circular_reference_handling(self):
    """Test that circular references in nested models don't cause infinite loops."""
    from typing import List

    from google.adk.tools._automatic_function_calling_util import _get_pydantic_schema
    from pydantic import BaseModel

    class TreeNode(BaseModel):
      """A tree node."""

      value: str = Field(description="Node value")
      children: List["TreeNode"] = Field(
          default_factory=list, description="Child nodes"
      )

    def create_tree(
        root: Annotated[TreeNode, Field(description="Root node of the tree")],
    ) -> dict:
      """Create a tree structure."""
      return {}

    # This should not raise an error or hang
    schema = _get_pydantic_schema(create_tree)

    # Verify schema was generated
    assert "properties" in schema
    assert "root" in schema["properties"]

    # The function should handle the circular reference gracefully
    # (implementation may vary: could inline first level, use ref, or break cycle)
    root_schema = schema["properties"]["root"]
    assert root_schema.get("description") == "Root node of the tree"

  def test_function_declaration_with_nested_models(self):
    """Test that build_function_declaration works with nested Pydantic models."""
    from pydantic import BaseModel

    class Credentials(BaseModel):
      """API credentials."""

      api_key: str = Field(description="API key for authentication")
      secret: str = Field(description="API secret")

    def authenticate(
        creds: Annotated[
            Credentials, Field(description="Authentication credentials")
        ],
    ) -> dict:
      """Authenticate with API credentials."""
      return {}

    declaration = build_function_declaration(
        authenticate,
        variant=GoogleLLMVariant.GEMINI_API,
    )

    assert declaration.name == "authenticate"
    assert declaration.parameters is not None
    assert declaration.parameters.properties is not None

    creds_schema = declaration.parameters.properties.get("creds")
    assert creds_schema is not None
    assert creds_schema.description == "Authentication credentials"

    # Check that nested properties are accessible
    assert creds_schema.properties is not None
    assert "api_key" in creds_schema.properties
    assert "secret" in creds_schema.properties

    # Verify nested descriptions are present
    assert (
        creds_schema.properties["api_key"].description
        == "API key for authentication"
    )
    assert creds_schema.properties["secret"].description == "API secret"
