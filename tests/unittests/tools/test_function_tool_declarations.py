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

"""Tests for the Pydantic-based function declaration builder.

These tests verify that the simplified Pydantic approach generates correct
JSON schemas for various function signatures, including edge cases.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any
from typing import AsyncGenerator
from typing import Generator
from typing import Literal
from typing import Optional

from absl.testing import parameterized
from google.adk.tools._function_tool_declarations import build_function_declaration_with_json_schema
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel
from pydantic import Field
from pydantic.dataclasses import dataclass as pyd_dataclass


class Color(Enum):
  """A simple enum for testing."""

  RED = "red"
  GREEN = "green"
  BLUE = "blue"


class Priority(Enum):
  """An integer enum for testing."""

  LOW = 1
  MEDIUM = 2
  HIGH = 3


class Address(BaseModel):
  """A Pydantic model for nested object testing."""

  street: str = Field(..., description="Street address")
  city: str = Field(..., description="City name")
  zip_code: str = Field(..., pattern=r"^\d{5}$", description="US ZIP code")


class Person(BaseModel):
  """A Pydantic model with nested model."""

  name: str
  age: int
  address: Optional[Address] = None


@pyd_dataclass
class Window:
  """A Pydantic dataclass for testing."""

  width: int
  height: int


class TestBasicTypes(parameterized.TestCase):
  """Tests for basic Python types."""

  @parameterized.named_parameters(
      (
          "string",
          lambda name: f"Hello, {name}!",
          {"name": {"title": "Name", "type": "string"}},
          {"type": "string"},
      ),
      (
          "integer",
          lambda n: n * 2,
          {"n": {"title": "N", "type": "integer"}},
          {"type": "integer"},
      ),
      (
          "float",
          lambda x: x * x,
          {"x": {"title": "X", "type": "number"}},
          {"type": "number"},
      ),
      (
          "boolean",
          lambda enabled: not enabled,
          {"enabled": {"title": "Enabled", "type": "boolean"}},
          {"type": "boolean"},
      ),
  )
  def test_basic_parameter_types(
      self, func, expected_param_props, expected_response_schema
  ):
    """Test functions with single basic type parameters."""
    # We need to define the functions within the test or use types from typing
    # to properly capture annotations. For simplicity, we'll define them here.
    if func.__code__.co_varnames[0] == "name":

      def test_func(name: str) -> str:
        return func(name)

    elif func.__code__.co_varnames[0] == "n":

      def test_func(n: int) -> int:
        return func(n)

    elif func.__code__.co_varnames[0] == "x":

      def test_func(x: float) -> float:
        return func(x)

    elif func.__code__.co_varnames[0] == "enabled":

      def test_func(enabled: bool) -> bool:
        return func(enabled)

    else:
      raise ValueError("Unexpected function signature")

    decl = build_function_declaration_with_json_schema(test_func)

    self.assertIsNotNone(decl.parameters_json_schema)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"], expected_param_props)
    self.assertEqual(decl.response_json_schema, expected_response_schema)
    self.assertEqual(set(schema["required"]), set(expected_param_props.keys()))

  def test_string_parameter_details(self):
    """Test function with string parameter details."""

    def greet(name: str) -> str:
      """Greet someone by name."""
      return f"Hello, {name}!"

    decl = build_function_declaration_with_json_schema(greet)

    self.assertEqual(decl.name, "greet")
    self.assertEqual(decl.description, "Greet someone by name.")
    self.assertEqual(
        decl.parameters_json_schema,
        {
            "type": "object",
            "properties": {
                "name": {
                    "title": "Name",
                    "type": "string",
                }
            },
            "required": ["name"],
            "title": "greetParams",
        },
    )

    self.assertEqual(decl.response_json_schema, {"type": "string"})

  def test_multiple_parameters(self):
    """Test function with multiple parameters of different types."""

    def create_user(name: str, age: int, active: bool) -> str:
      """Create a new user."""
      return f"Created {name}"

    decl = build_function_declaration_with_json_schema(create_user)
    schema = decl.parameters_json_schema

    self.assertLen(schema["properties"], 3)
    self.assertEqual(schema["properties"]["name"]["type"], "string")
    self.assertEqual(schema["properties"]["age"]["type"], "integer")
    self.assertEqual(schema["properties"]["active"]["type"], "boolean")
    self.assertEqual(set(schema["required"]), {"name", "age", "active"})
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )


class TestDefaultValues(parameterized.TestCase):
  """Tests for parameters with default values."""

  def test_string_with_default(self):
    """Test string parameter with default value."""

    def greet(name: str = "World") -> str:
      """Greet someone."""
      return f"Hello, {name}!"

    decl = build_function_declaration_with_json_schema(greet)
    schema = decl.parameters_json_schema

    assert schema["properties"]["name"]["default"] == "World"
    self.assertNotIn("name", schema.get("required", []))
    assert decl.response_json_schema == {
        "type": "string",
    }

  def test_int_with_default(self):
    """Test integer parameter with default value."""

    def repeat(text: str, times: int = 3) -> str:
      """Repeat text."""
      return text * times

    decl = build_function_declaration_with_json_schema(repeat)
    schema = decl.parameters_json_schema

    # times should have default, text should be required
    assert "text" in schema["required"]
    assert schema["properties"]["times"]["default"] == 3
    self.assertNotIn("times", schema.get("required", []))
    assert decl.response_json_schema == {
        "type": "string",
    }

  def test_none_default(self):
    """Test parameter with None as default."""

    def search(query: str, limit: Optional[int] = None) -> str:
      """Search for something."""
      return query

    decl = build_function_declaration_with_json_schema(search)
    schema = decl.parameters_json_schema

    assert "query" in schema["required"]
    # limit should not be required since it has default None
    self.assertNotIn("limit", schema.get("required", []))
    assert schema["properties"]["limit"]["default"] is None
    assert decl.response_json_schema == {
        "type": "string",
    }


class TestCollectionTypes(parameterized.TestCase):
  """Tests for list, dict, and other collection types."""

  @parameterized.named_parameters(
      (
          "strings",
          ", ".join,
          "items",
          str,
          "string",
          "string",
      ),
      (
          "integers",
          sum,
          "numbers",
          int,
          "integer",
          "integer",
      ),
  )
  def test_list_parameters(
      self,
      func_impl,
      param_name,
      item_type,
      expected_item_schema_type,
      expected_response_schema_type,
  ):
    """Test list parameters with different item types."""

    if item_type == str:

      def test_func(items: list[str]) -> str:
        return func_impl(items)

      test_func.__name__ = "join_strings"
    elif item_type == int:

      def test_func(numbers: list[int]) -> int:
        return func_impl(numbers)

      test_func.__name__ = "sum_numbers"
    else:
      raise ValueError("Unsupported item type")

    decl = build_function_declaration_with_json_schema(test_func)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"][param_name]["type"], "array")
    self.assertEqual(
        schema["properties"][param_name]["items"]["type"],
        expected_item_schema_type,
    )
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": expected_response_schema_type,
        },
    )

  def test_dict_parameter(self):
    """Test dict[str, Any] parameter."""

    def process_data(data: dict[str, Any]) -> str:
      """Process a dictionary."""
      return str(data)

    decl = build_function_declaration_with_json_schema(process_data)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"]["data"]["type"], "object")
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_dict_with_typed_values(self):
    """Test dict[str, int] parameter."""

    def sum_scores(scores: dict[str, int]) -> int:
      """Sum all scores."""
      return sum(scores.values())

    decl = build_function_declaration_with_json_schema(sum_scores)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"]["scores"]["type"], "object")
    # additionalProperties should specify int type
    self.assertEqual(
        schema["properties"]["scores"]
        .get("additionalProperties", {})
        .get("type"),
        "integer",
    )
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )

  def test_sequence_type(self):
    """Test Sequence[str] parameter (from collections.abc)."""

    def process_items(items: Sequence[str]) -> int:
      """Process items and return count."""
      return len(list(items))

    decl = build_function_declaration_with_json_schema(process_items)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"]["items"]["type"], "array")
    self.assertEqual(schema["properties"]["items"]["items"]["type"], "string")
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )

  def test_tuple_fixed_length(self):
    """Test tuple[int, int] parameter (fixed length)."""

    def add_point(coords: tuple[int, int]) -> int:
      """Add coordinates."""
      x, y = coords
      return x + y

    decl = build_function_declaration_with_json_schema(add_point)
    schema = decl.parameters_json_schema

    # Fixed-length tuples use prefixItems
    coords_schema = schema["properties"]["coords"]
    self.assertEqual(coords_schema["type"], "array")
    self.assertIn("prefixItems", coords_schema)
    self.assertLen(coords_schema["prefixItems"], 2)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )


class TestEnumAndLiteral(parameterized.TestCase):
  """Tests for Enum and Literal types."""

  def test_string_enum(self):
    """Test Enum parameter with string values."""

    def set_color(color: Color) -> str:
      """Set the color."""
      return color.value

    decl = build_function_declaration_with_json_schema(set_color)
    schema = decl.parameters_json_schema

    self.assertIn("$defs", schema)
    self.assertIn("color", schema["properties"])
    color_schema = schema["properties"]["color"]
    self.assertIn("$ref", color_schema)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_literal_type(self):
    """Test Literal type parameter."""

    def set_mode(mode: Literal["fast", "slow", "auto"]) -> str:
      """Set the mode."""
      return mode

    decl = build_function_declaration_with_json_schema(set_mode)
    schema = decl.parameters_json_schema

    mode_schema = schema["properties"]["mode"]
    self.assertEqual(mode_schema.get("enum"), ["fast", "slow", "auto"])
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_literal_with_default(self):
    """Test Literal type with default value."""

    def configure(mode: Literal["on", "off"] = "on") -> str:
      """Configure something."""
      return mode

    decl = build_function_declaration_with_json_schema(configure)
    schema = decl.parameters_json_schema

    self.assertEqual(schema["properties"]["mode"]["default"], "on")
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )


class TestOptionalAndUnion(parameterized.TestCase):
  """Tests for Optional and Union types."""

  def test_optional_string(self):
    """Test Optional[str] parameter."""

    def greet(name: Optional[str] = None) -> str:
      """Greet someone."""
      return f"Hello, {name or 'World'}!"

    decl = build_function_declaration_with_json_schema(greet)
    schema = decl.parameters_json_schema

    # Optional should be represented with anyOf including null
    name_schema = schema["properties"]["name"]
    self.assertIn("anyOf", name_schema)
    self.assertLen(name_schema["anyOf"], 2)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_union_of_primitives(self):
    """Test Union[int, str] parameter."""

    def process(value: int | str) -> str:
      """Process a value."""
      return str(value)

    decl = build_function_declaration_with_json_schema(process)
    schema = decl.parameters_json_schema

    value_schema = schema["properties"]["value"]
    self.assertIn("anyOf", value_schema)
    self.assertLen(value_schema["anyOf"], 2)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_complex_union(self):
    """Test Union[int, str, dict[str, float]] parameter."""

    def flexible_input(
        payload: int | str | dict[str, float] = 0,
    ) -> str:
      """Accept flexible input."""
      return str(payload)

    decl = build_function_declaration_with_json_schema(flexible_input)
    schema = decl.parameters_json_schema

    payload_schema = schema["properties"]["payload"]
    self.assertIn("anyOf", payload_schema)
    self.assertLen(payload_schema["anyOf"], 3)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )


class TestNestedObjects(parameterized.TestCase):
  """Tests for nested Pydantic models and dataclasses."""

  def test_pydantic_model_parameter(self):
    """Test parameter that is a Pydantic model."""

    def save_address(address: Address) -> str:
      """Save an address."""
      return f"Saved address in {address.city}"

    decl = build_function_declaration_with_json_schema(save_address)
    schema = decl.parameters_json_schema

    # Should have $defs for the nested model
    self.assertIn("address", schema["properties"])
    self.assertIn("$ref", schema["properties"]["address"])

    address_def = schema["$defs"]["Address"]
    self.assertEqual(address_def["type"], "object")
    self.assertIn("street", address_def["properties"])
    self.assertEqual(
        address_def["properties"]["zip_code"]["pattern"], r"^\d{5}$"
    )
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_nested_pydantic_model(self):
    """Test Pydantic model with nested model."""

    def save_person(person: Person) -> str:
      """Save a person."""
      return f"Saved {person.name}"

    decl = build_function_declaration_with_json_schema(save_person)
    schema = decl.parameters_json_schema

    # Should handle nested Address model
    self.assertIn("$defs", schema)
    person_defs = schema["$defs"]["Person"]
    self.assertEqual(person_defs["type"], "object")
    self.assertIn("address", person_defs["properties"])
    self.assertIn("person", schema["properties"])
    self.assertIn("$ref", schema["properties"]["person"])
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_pydantic_dataclass_parameter(self):
    """Test parameter that is a Pydantic dataclass."""

    def resize_window(window: Window) -> str:
      """Resize a window."""
      return f"Resized to {window.width}x{window.height}"

    decl = build_function_declaration_with_json_schema(resize_window)
    schema = decl.parameters_json_schema

    self.assertIn("window", schema["properties"])
    self.assertIn("$ref", schema["properties"]["window"])
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_list_of_pydantic_models(self):
    """Test list of Pydantic models."""

    def save_addresses(addresses: list[Address]) -> int:
      """Save multiple addresses."""
      return len(addresses)

    decl = build_function_declaration_with_json_schema(save_addresses)
    schema = decl.parameters_json_schema

    addr_schema = schema["properties"]["addresses"]
    self.assertEqual(addr_schema["type"], "array")
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )


class TestSpecialCases(parameterized.TestCase):
  """Tests for special cases and edge cases."""

  def test_no_parameters(self):
    """Test function with no parameters."""

    def get_time() -> str:
      """Get current time."""
      return "12:00"

    decl = build_function_declaration_with_json_schema(get_time)

    self.assertEqual(decl.name, "get_time")
    self.assertIsNone(decl.parameters_json_schema)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_no_type_annotations(self):
    """Test function with no type annotations."""

    def legacy_function(x, y):
      """A legacy function without types."""
      return x + y

    decl = build_function_declaration_with_json_schema(legacy_function)
    schema = decl.parameters_json_schema

    # Should still generate schema, with Any type
    self.assertIn("x", schema["properties"])
    self.assertIsNone(schema["properties"]["x"].get("type"))
    self.assertIn("y", schema["properties"])
    self.assertIsNone(schema["properties"]["y"].get("type"))
    # No return type annotation, so response schema should be None
    self.assertIsNone(decl.response_json_schema)

  def test_any_type_parameter(self):
    """Test parameter with Any type."""

    def process_any(data: Any) -> str:
      """Process any data."""
      return str(data)

    decl = build_function_declaration_with_json_schema(process_any)
    schema = decl.parameters_json_schema

    # Any type should be represented somehow
    self.assertIn("data", schema["properties"])
    self.assertIsNone(schema["properties"]["data"].get("type"))
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_tool_context_ignored_via_ignore_params(self):
    """Test that tool_context parameter is ignored when passed in ignore_params."""

    def my_tool(query: str, tool_context: ToolContext) -> str:
      """A tool that uses context."""
      return query

    decl = build_function_declaration_with_json_schema(
        my_tool, ignore_params=["tool_context"]
    )
    schema = decl.parameters_json_schema

    self.assertIn("query", schema["properties"])
    self.assertNotIn("tool_context", schema["properties"])
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_ignore_params(self):
    """Test ignoring specific parameters."""

    def complex_func(a: str, b: int, c: float, internal: str) -> str:
      """A function with internal parameter."""
      return a

    decl = build_function_declaration_with_json_schema(
        complex_func, ignore_params=["internal"]
    )
    schema = decl.parameters_json_schema

    self.assertIn("a", schema["properties"])
    self.assertIn("b", schema["properties"])
    self.assertIn("c", schema["properties"])
    self.assertNotIn("internal", schema["properties"])
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "string",
        },
    )

  def test_docstring_preserved(self):
    """Test that docstring is preserved as description."""

    def well_documented(x: int) -> int:
      """This is a well-documented function.

      It does something useful.

      Args:
        x: The number to square.

      Returns:
        The squared number.
      """
      return x

    decl = build_function_declaration_with_json_schema(well_documented)

    self.assertIn("well-documented function", decl.description)
    self.assertIn("something useful", decl.description)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )

  def test_no_docstring(self):
    """Test function without docstring."""

    def undocumented(x: int) -> int:
      return x

    decl = build_function_declaration_with_json_schema(undocumented)

    self.assertIsNone(decl.description)
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "integer",
        },
    )


class TestComplexFunction(parameterized.TestCase):
  """Test the complex function from the user's prototype."""

  def test_complex_function_schema(self):
    """Test the complex function with many type variations."""

    def complex_fn(
        color: Color,
        tags: list[str],
        mode: Literal["fast", "slow"] = "fast",
        count: Optional[int] = None,
        address: Optional[Address] = None,
        window: Optional[Window] = None,
        payload: int | str | dict[str, float] = 0,
        colors: Optional[list[Color]] = None,
    ) -> None:
      """A complex function with many parameter types."""
      del color, tags, mode, count, address, window, payload, colors

    decl = build_function_declaration_with_json_schema(complex_fn)

    self.assertEqual(decl.name, "complex_fn")
    self.assertIsNotNone(decl.parameters_json_schema)

    schema = decl.parameters_json_schema
    props = schema["properties"]

    # Verify all parameters are present
    self.assertIn("color", props)
    self.assertIn("tags", props)
    self.assertIn("mode", props)
    self.assertIn("count", props)
    self.assertIn("address", props)
    self.assertIn("window", props)
    self.assertIn("payload", props)
    self.assertIn("colors", props)

    # tags should be array of strings
    self.assertEqual(props["tags"]["type"], "array")

    # mode should have enum
    self.assertEqual(props["mode"].get("enum"), ["fast", "slow"])
    # Return type is None, which maps to JSON schema null type
    self.assertEqual(
        decl.response_json_schema,
        {
            "type": "null",
        },
    )


class TestPydanticModelAsFunction(parameterized.TestCase):
  """Tests for using Pydantic BaseModel directly."""

  def test_base_model_class(self):
    """Test passing a Pydantic BaseModel class directly."""

    class CreateUserRequest(BaseModel):
      """Request to create a user."""

      name: str
      email: str
      age: Optional[int] = None

    decl = build_function_declaration_with_json_schema(CreateUserRequest)

    self.assertEqual(decl.name, "CreateUserRequest")
    self.assertIsNotNone(decl.parameters_json_schema)

    schema = decl.parameters_json_schema
    self.assertIn("name", schema["properties"])
    self.assertIn("email", schema["properties"])
    self.assertIn("age", schema["properties"])
    # When passing a BaseModel, there is no function return, so response schema
    # is None
    self.assertIsNone(decl.response_json_schema)


class TestStreamingReturnTypes(parameterized.TestCase):
  """Tests for AsyncGenerator and Generator return types (streaming tools)."""

  def test_async_generator_string_yield(self):
    """Test AsyncGenerator[str, None] return type extracts str as response."""

    async def streaming_tool(param: str) -> AsyncGenerator[str, None]:
      """A streaming tool that yields strings."""
      yield param

    decl = build_function_declaration_with_json_schema(streaming_tool)

    self.assertEqual(decl.name, "streaming_tool")
    self.assertIsNotNone(decl.parameters_json_schema)
    self.assertEqual(
        decl.parameters_json_schema["properties"]["param"]["type"], "string"
    )
    # Should extract str from AsyncGenerator[str, None]
    self.assertEqual(decl.response_json_schema, {"type": "string"})

  def test_async_generator_int_yield(self):
    """Test AsyncGenerator[int, None] return type extracts int as response."""

    async def counter(start: int) -> AsyncGenerator[int, None]:
      """A streaming counter."""
      yield start

    decl = build_function_declaration_with_json_schema(counter)

    self.assertEqual(decl.name, "counter")
    # Should extract int from AsyncGenerator[int, None]
    self.assertEqual(decl.response_json_schema, {"type": "integer"})

  def test_async_generator_dict_yield(self):
    """Test AsyncGenerator[dict[str, str], None] return type."""

    async def streaming_dict(
        param: str,
    ) -> AsyncGenerator[dict[str, str], None]:
      """A streaming tool that yields dicts."""
      yield {"result": param}

    decl = build_function_declaration_with_json_schema(streaming_dict)

    self.assertEqual(decl.name, "streaming_dict")
    # Should extract dict[str, str] from AsyncGenerator
    self.assertEqual(
        decl.response_json_schema,
        {"additionalProperties": {"type": "string"}, "type": "object"},
    )

  def test_generator_string_yield(self):
    """Test Generator[str, None, None] return type extracts str as response."""

    def sync_streaming_tool(param: str) -> Generator[str, None, None]:
      """A sync streaming tool that yields strings."""
      yield param

    decl = build_function_declaration_with_json_schema(sync_streaming_tool)

    self.assertEqual(decl.name, "sync_streaming_tool")
    # Should extract str from Generator[str, None, None]
    self.assertEqual(decl.response_json_schema, {"type": "string"})

  def test_generator_int_yield(self):
    """Test Generator[int, None, None] return type extracts int as response."""

    def sync_counter(start: int) -> Generator[int, None, None]:
      """A sync streaming counter."""
      yield start

    decl = build_function_declaration_with_json_schema(sync_counter)

    self.assertEqual(decl.name, "sync_counter")
    # Should extract int from Generator[int, None, None]
    self.assertEqual(decl.response_json_schema, {"type": "integer"})
