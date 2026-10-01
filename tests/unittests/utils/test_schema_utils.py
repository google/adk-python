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

"""Tests for _schema_utils module."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
import functools
import inspect
import json
import signal
import time
from typing import Annotated
from typing import Optional

from google.adk.utils._callable_utils import get_type_hints_cached
from google.adk.utils._schema_utils import _strip_json_code_fence
from google.adk.utils._schema_utils import get_list_inner_type
from google.adk.utils._schema_utils import is_basemodel_schema
from google.adk.utils._schema_utils import is_list_of_basemodel
from google.adk.utils._schema_utils import lowercase_schema_types
from google.adk.utils._schema_utils import preprocess_args
from google.adk.utils._schema_utils import schema_to_json_schema
from google.adk.utils._schema_utils import validate_node_data
from google.adk.utils._schema_utils import validate_schema
from google.adk.workflow._base_node import BaseNode
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from pydantic import PlainSerializer
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
import pytest


class SampleModel(BaseModel):
  """Sample model for testing."""

  name: str
  value: int


class OtherModel(BaseModel):
  """Another model for testing."""

  tag: str


class TestIsBasemodelSchema:
  """Tests for is_basemodel_schema function."""

  def test_basemodel_class_returns_true(self):
    """Test that a BaseModel class returns True."""
    assert is_basemodel_schema(SampleModel)

  def test_list_of_basemodel_returns_false(self):
    """Test that list[BaseModel] returns False."""
    assert not is_basemodel_schema(list[SampleModel])

  def test_list_of_str_returns_false(self):
    """Test that list[str] returns False."""
    assert not is_basemodel_schema(list[str])

  def test_dict_returns_false(self):
    """Test that dict types return False."""
    assert not is_basemodel_schema(dict[str, int])

  def test_plain_str_returns_false(self):
    """Test that plain str returns False."""
    assert not is_basemodel_schema(str)

  def test_plain_int_returns_false(self):
    """Test that plain int returns False."""
    assert not is_basemodel_schema(int)

  def test_annotated_basemodel_returns_true(self):
    """Test that Annotated[BaseModel, ...] returns True."""
    assert is_basemodel_schema(
        Annotated[SampleModel, Field(description="A sample")]
    )


class TestIsListOfBasemodel:
  """Tests for is_list_of_basemodel function."""

  def test_list_of_basemodel_returns_true(self):
    """Test that list[BaseModel] returns True."""
    assert is_list_of_basemodel(list[SampleModel])

  def test_basemodel_class_returns_false(self):
    """Test that a plain BaseModel class returns False."""
    assert not is_list_of_basemodel(SampleModel)

  def test_list_of_str_returns_false(self):
    """Test that list[str] returns False."""
    assert not is_list_of_basemodel(list[str])

  def test_list_of_int_returns_false(self):
    """Test that list[int] returns False."""
    assert not is_list_of_basemodel(list[int])

  def test_dict_returns_false(self):
    """Test that dict types return False."""
    assert not is_list_of_basemodel(dict[str, int])

  def test_plain_list_returns_false(self):
    """Test that plain list (no type arg) returns False."""
    assert not is_list_of_basemodel(list)

  def test_is_list_of_basemodel_with_annotated(self):
    """Test is_list_of_basemodel unwraps Annotated inside list."""
    assert is_list_of_basemodel(
        list[Annotated[SampleModel, Field(description="A sample")]]
    )

  def test_is_list_of_basemodel_with_annotated_list(self):
    """Test is_list_of_basemodel unwraps outer Annotated on list[Model]."""
    schema = Annotated[list[SampleModel], Field(description="A list of models")]
    assert is_list_of_basemodel(schema)


class TestGetListInnerType:
  """Tests for get_list_inner_type function."""

  def test_list_of_basemodel_returns_inner_type(self):
    """Test that list[BaseModel] returns the inner type."""
    assert get_list_inner_type(list[SampleModel]) is SampleModel

  def test_get_list_inner_type_with_annotated(self):
    """Test get_list_inner_type unwraps Annotated inside list."""
    assert (
        get_list_inner_type(
            list[Annotated[SampleModel, Field(description="A sample")]]
        )
        is SampleModel
    )

  def test_get_list_inner_type_with_annotated_list(self):
    """Test get_list_inner_type unwraps outer Annotated on list[Model]."""
    schema = Annotated[list[SampleModel], Field(description="A list of models")]
    assert get_list_inner_type(schema) is SampleModel

  def test_basemodel_class_returns_none(self):
    """Test that a plain BaseModel class returns None."""
    assert get_list_inner_type(SampleModel) is None

  def test_list_of_str_returns_none(self):
    """Test that list[str] returns None."""
    assert get_list_inner_type(list[str]) is None

  def test_dict_returns_none(self):
    """Test that dict types return None."""
    assert get_list_inner_type(dict[str, int]) is None


class TestValidateSchema:
  """Tests for validate_schema function."""

  def test_basemodel_schema(self):
    """Test validation with a BaseModel schema."""
    json_text = '{"name": "test", "value": 42}'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_basemodel_schema_excludes_none(self):
    """Test that None values are excluded from the result."""

    class ModelWithOptional(BaseModel):
      name: str
      optional_field: str | None = None

    json_text = '{"name": "test", "optional_field": null}'
    result = validate_schema(ModelWithOptional, json_text)
    assert result == {"name": "test"}

  def test_list_of_basemodel_schema(self):
    """Test validation with a list[BaseModel] schema."""
    json_text = '[{"name": "item1", "value": 1}, {"name": "item2", "value": 2}]'
    result = validate_schema(list[SampleModel], json_text)
    assert result == [
        {"name": "item1", "value": 1},
        {"name": "item2", "value": 2},
    ]

  def test_validate_schema_with_annotated_list_of_basemodel(self):
    """Test validate_schema with list[Annotated[Model, ...]]."""
    json_text = '[{"name": "test", "value": 42}]'
    result = validate_schema(
        list[Annotated[SampleModel, Field(description="A sample")]], json_text
    )
    assert result == [{"name": "test", "value": 42}]

  def test_validate_schema_with_annotated_basemodel(self):
    """Test validate_schema with Annotated[SampleModel, ...] validates and parses."""
    json_text = '{"name": "test", "value": "42"}'
    result = validate_schema(
        Annotated[SampleModel, Field(description="A sample")], json_text
    )
    assert result == {"name": "test", "value": 42}

  def test_list_of_str_schema(self):
    """Test validation with a list[str] schema."""
    json_text = '["a", "b", "c"]'
    result = validate_schema(list[str], json_text)
    assert result == ["a", "b", "c"]

  def test_dict_schema(self):
    """Test validation with a dict schema."""
    json_text = '{"key1": 1, "key2": 2}'
    result = validate_schema(dict[str, int], json_text)
    assert result == {"key1": 1, "key2": 2}

  def test_json_code_fence_is_stripped(self):
    """Test that a ```json fenced payload is unwrapped before validation."""
    json_text = '```json\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_uppercase_json_code_fence_is_stripped(self):
    """Test that an uppercase language tag is not left in the payload."""
    json_text = '```JSON\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_other_language_tag_code_fence_is_stripped(self):
    """Test that any language tag on the fence is unwrapped."""
    json_text = '```python\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_bare_code_fence_is_stripped(self):
    """Test that a fence without a language tag is unwrapped."""
    json_text = '```\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_code_fence_with_surrounding_whitespace_is_stripped(self):
    """Test that whitespace around the fence does not break unwrapping."""
    json_text = '  \n```json\n{"name": "test", "value": 42}\n```  \n'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_list_schema_code_fence_is_stripped(self):
    """Test that a fenced list[BaseModel] payload is unwrapped."""
    json_text = '```json\n[{"name": "item1", "value": 1}]\n```'
    result = validate_schema(list[SampleModel], json_text)
    assert result == [{"name": "item1", "value": 1}]

  def test_plain_json_is_unaffected(self):
    """Test that unfenced JSON is validated unchanged."""
    json_text = '{"name": "test", "value": 42}'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "test", "value": 42}

  def test_backticks_inside_value_are_preserved(self):
    """Test that triple backticks inside a valid JSON value are not stripped."""
    json_text = '{"name": "```", "value": 42}'
    result = validate_schema(SampleModel, json_text)
    assert result == {"name": "```", "value": 42}

  def test_unclosed_code_fence_with_whitespace_does_not_hang(self):
    """Test that an unclosed code fence with large whitespace runs does not ReDoS."""
    payload = "```json\n" + " " * 5000 + "x"
    if hasattr(signal, "SIGALRM"):
      old_handler = signal.signal(
          signal.SIGALRM,
          lambda s, f: pytest.fail("Test timed out - possible ReDoS"),
      )
      signal.alarm(2)
      try:
        start = time.perf_counter()
        result = _strip_json_code_fence(payload)
        elapsed = time.perf_counter() - start
      finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    else:
      start = time.perf_counter()
      result = _strip_json_code_fence(payload)
      elapsed = time.perf_counter() - start
    assert result == payload
    assert elapsed < 1.0

  def test_strip_json_code_fence_variations(self):
    """Test various markdown fence configurations."""
    assert _strip_json_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_code_fence('```   \n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_code_fence('```json  {"a": 1}```') == '{"a": 1}'
    assert _strip_json_code_fence('{"a": 1}') == '{"a": 1}'
    assert _strip_json_code_fence("```") == "```"
    assert _strip_json_code_fence("") == ""

  def test_validate_schema_dumps_native_types_in_json_mode(self):
    """validate_schema coerces Decimal, datetime, Enum to JSON primitives."""

    class Color(Enum):
      RED = 1

    class Payload(BaseModel):
      price: Decimal
      stamped_at: datetime
      color: Color

    json_text = (
        '{"price": "29.99", "stamped_at": "2026-01-02T03:04:05", "color": 1}'
    )
    result = validate_schema(Payload, json_text)
    assert result == {
        "price": "29.99",
        "stamped_at": "2026-01-02T03:04:05",
        "color": 1,
    }
    assert isinstance(result["price"], str)
    assert isinstance(result["stamped_at"], str)
    assert result["color"] == 1


class TestValidateNodeData:
  """Tests for validate_node_data function."""

  def test_none_schema_or_data_returns_data(self):
    """Bypasses validation if schema or data is None."""
    assert validate_node_data(None, "some_data") == "some_data"
    assert validate_node_data(SampleModel, None) is None

  def test_dict_or_types_schema_returns_data(self):
    """Bypasses validation if schema is dict or types.Schema."""
    assert validate_node_data({"key": int}, "some_data") == "some_data"
    # Mock types.Schema
    schema = types.Schema(type=types.Type.STRING)
    assert validate_node_data(schema, "some_data") == "some_data"

  def test_dict_or_types_schema_dumps_basemodel_in_json_mode(self):
    """dict or types.Schema schema dumps BaseModel outputs in JSON mode."""

    class Price(BaseModel):
      amount: Decimal
      note: str | None = None

    price = Price(amount=Decimal("29.99"))
    dict_result = validate_node_data({"type": "object"}, price)
    assert dict_result == {"amount": "29.99", "note": None}
    assert isinstance(dict_result["amount"], str)

    schema_result = validate_node_data(
        types.Schema(type=types.Type.OBJECT), price
    )
    assert schema_result == {"amount": "29.99", "note": None}
    assert isinstance(schema_result["amount"], str)

  def test_dict_schema_coerces_native_types_in_json_mode(self):
    """dict schema in validate_node_data coerces native types in JSON mode."""

    class Color(Enum):
      RED = 1

    data = {
        "price": Decimal("29.99"),
        "stamped_at": datetime(2026, 1, 2, 3, 4, 5),
        "color": Color.RED,
        "by_color": {Color.RED: Decimal("10.00")},
    }
    result = validate_node_data({"type": "object"}, data)
    assert result == {
        "price": "29.99",
        "stamped_at": "2026-01-02T03:04:05",
        "color": 1,
        "by_color": {"1": "10.00"},
    }
    assert isinstance(result["price"], str)
    assert isinstance(result["stamped_at"], str)
    assert result["color"] == 1
    json.dumps(result)

  def test_content_schema_returns_data(self):
    """Bypasses validation if target schema is types.Content or subclass."""
    result = validate_node_data(
        types.Content, types.Content(role="user", parts=[])
    )
    assert result == {"role": "user", "parts": []}

  def test_plain_basemodel_schema_validates_raw_dict(self):
    """Validates raw dict data against BaseModel schema."""
    result = validate_node_data(SampleModel, {"name": "test", "value": 42})
    assert result == {"name": "test", "value": 42}

  def test_content_data_and_preserve_content(self):
    """Validates wrapped content and wraps result back into Content."""
    data = types.Content(
        role="user",
        parts=[types.Part(text='{"name": "test", "value": 42}')],
    )
    result = validate_node_data(SampleModel, data, preserve_content=True)
    assert isinstance(result, types.Content)
    assert result.role == "user"
    assert len(result.parts) == 1
    assert result.parts[0].text == '{"name": "test", "value": 42}'

  def test_content_data_no_preserve_content(self):
    """Validates wrapped content and returns unwrapped dictionary."""
    data = types.Content(
        role="user",
        parts=[types.Part(text='{"name": "test", "value": 42}')],
    )
    result = validate_node_data(SampleModel, data, preserve_content=False)
    assert isinstance(result, dict)
    assert result == {"name": "test", "value": 42}

  def test_raw_json_string_validated_against_basemodel_schema(self):
    """Raw JSON string fails validation against BaseModel schema (not auto-parsed)."""
    with pytest.raises(ValidationError):
      validate_node_data(SampleModel, '{"name": "test", "value": 42}')

  def test_raw_string_not_parsed_with_str_schema(self):
    """Bypasses JSON parsing if schema is str."""
    result = validate_node_data(str, "hello")
    assert result == "hello"

  def test_json_mode_serializers_are_applied_for_decimal(self):
    """when_used='json' serializers run so validated node data is JSON-safe."""
    JsonDecimal = Annotated[
        Decimal, PlainSerializer(float, return_type=float, when_used="json")
    ]

    class Price(BaseModel):
      amount: JsonDecimal

    class Payload(BaseModel):
      price: Price

    result = validate_node_data(Payload, {"price": {"amount": "29.99"}})
    assert result == {"price": {"amount": 29.99}}
    assert isinstance(result["price"]["amount"], float)
    assert json.dumps(result) == '{"price": {"amount": 29.99}}'

  def test_datetime_and_enum_fields_are_json_serializable(self):
    """Python-mode types that json.dumps rejects become JSON-safe values."""

    class Color(Enum):
      RED = 1

    class Payload(BaseModel):
      stamped_at: datetime
      color: Color

    result = validate_node_data(
        Payload,
        {"stamped_at": "2026-01-02T03:04:05", "color": 1},
    )
    assert result["color"] == 1
    assert isinstance(result["stamped_at"], str)
    json.dumps(result)

  def test_base_node_output_validation_is_json_serializable(self):
    """BaseNode output_schema validation returns JSON-serializable dicts."""
    JsonDecimal = Annotated[
        Decimal, PlainSerializer(float, return_type=float, when_used="json")
    ]

    class Price(BaseModel):
      amount: JsonDecimal

    class Payload(BaseModel):
      price: Price

    node = BaseNode(name="pricing", output_schema=Payload)
    result = node._validate_output_data({"price": {"amount": "29.99"}})
    assert result == {"price": {"amount": 29.99}}
    json.dumps(result)

  def test_validate_node_data_dumps_native_types_in_json_mode(self):
    """validate_node_data coerces native types (datetime, Decimal, Enum) in JSON mode."""

    class Color(Enum):
      RED = 1

    class Payload(BaseModel):
      price: Decimal
      stamped_at: datetime
      color: Color

    data = {
        "price": Decimal("29.99"),
        "stamped_at": datetime(2026, 1, 2, 3, 4, 5),
        "color": Color.RED,
    }
    result = validate_node_data(Payload, data)
    assert result == {
        "price": "29.99",
        "stamped_at": "2026-01-02T03:04:05",
        "color": 1,
    }
    assert isinstance(result["price"], str)
    assert isinstance(result["stamped_at"], str)
    assert result["color"] == 1
    json.dumps(result)

  def test_validate_node_data_content_with_decimal_preserve_content(self):
    """validate_node_data re-wraps validated Decimal in Content without raising TypeError."""
    content = types.Content(parts=[types.Part(text="29.99")])
    result = validate_node_data(Decimal, content, preserve_content=True)
    assert isinstance(result, types.Content)
    assert result.parts[0].text == "29.99"

  def test_validate_node_data_raises_on_serialization_error(self):
    """Serialization errors are raised instead of falling back to un-coerced values."""

    def failing_serializer(v: str) -> str:
      raise ValueError("cannot serialize")

    BadType = Annotated[
        str, PlainSerializer(failing_serializer, when_used="json")
    ]

    class Payload(BaseModel):
      x: BadType

    with pytest.raises(PydanticSerializationError):
      validate_node_data(Payload, {"x": "test"})

  def test_dict_or_types_schema_raises_on_serialization_error(self):
    """dict and types.Schema in validate_node_data raise on serialization failure."""
    with pytest.raises(PydanticSerializationError):
      validate_node_data({"type": "object"}, {"bad": object()})

    with pytest.raises(PydanticSerializationError):
      validate_node_data(
          types.Schema(type=types.Type.OBJECT), {"bad": object()}
      )


class TestSchemaToJsonSchema:
  """Tests for schema_to_json_schema function."""

  def test_dict_schema_is_returned_unchanged(self):
    """A raw dict is already JSON Schema, so it must not be re-derived."""
    raw = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert schema_to_json_schema(raw) is raw

  def test_basemodel_schema_describes_its_fields(self):
    result = schema_to_json_schema(SampleModel)
    assert result["type"] == "object"
    assert result["properties"]["name"]["type"] == "string"
    assert result["properties"]["value"]["type"] == "integer"
    # Neither field has a default, so both are required.
    assert sorted(result["required"]) == ["name", "value"]

  def test_builtin_generic_schema_becomes_an_array(self):
    result = schema_to_json_schema(list[str])
    assert result == {"type": "array", "items": {"type": "string"}}

  def test_list_of_basemodel_schema_becomes_an_array_of_objects(self):
    result = schema_to_json_schema(list[SampleModel])
    assert result["type"] == "array"
    # The item schema is emitted by reference into $defs rather than inline.
    ref = result["items"]["$ref"].rsplit("/", 1)[-1]
    assert result["$defs"][ref]["properties"]["name"]["type"] == "string"


class TestLowercaseSchemaTypes:
  """Tests for lowercase_schema_types function."""

  def test_properties_and_items_are_lowercased(self):
    schema = {
        "type": "OBJECT",
        "properties": {
            "name": {"type": "STRING"},
            "age": {"type": "INTEGER"},
            "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }
    lowercase_schema_types(schema)
    assert schema["type"] == "object"
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["age"]["type"] == "integer"
    assert schema["properties"]["tags"]["type"] == "array"
    assert schema["properties"]["tags"]["items"]["type"] == "string"

  def test_union_branches_are_lowercased_under_either_spelling(self):
    schema = {"anyOf": [{"type": "STRING"}], "any_of": [{"type": "NUMBER"}]}
    lowercase_schema_types(schema)
    assert schema["anyOf"][0]["type"] == "string"
    assert schema["any_of"][0]["type"] == "number"

  def test_a_list_valued_type_is_lowercased_entry_by_entry(self):
    schema = {"type": ["STRING", "NULL"]}
    lowercase_schema_types(schema)
    assert schema["type"] == ["string", "null"]

  def test_referenced_definitions_are_lowercased(self):
    schema = {
        "$defs": {"Item": {"type": "OBJECT"}},
        "definitions": {"LegacyItem": {"type": "STRING"}},
    }
    lowercase_schema_types(schema)
    assert schema["$defs"]["Item"]["type"] == "object"
    assert schema["definitions"]["LegacyItem"]["type"] == "string"

  def test_a_list_of_schemas_is_accepted(self):
    schemas = [{"type": "STRING"}, {"type": "BOOLEAN"}]
    lowercase_schema_types(schemas)
    assert [schema["type"] for schema in schemas] == ["string", "boolean"]

  def test_non_schema_values_keep_their_type_key(self):
    """A ``type`` key inside a default value is data, not a schema keyword."""
    schema = {"type": "OBJECT", "default": {"type": "NOT_A_SCHEMA"}}
    lowercase_schema_types(schema)
    assert schema["type"] == "object"
    assert schema["default"]["type"] == "NOT_A_SCHEMA"

  def test_a_genai_schema_dump_is_lowercased(self):
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={"name": types.Schema(type=types.Type.STRING)},
    ).model_dump(exclude_none=True, mode="json")
    lowercase_schema_types(schema)
    assert schema["type"] == "object"
    assert schema["properties"]["name"]["type"] == "string"


class TestPreprocessArgs:
  """Tests for preprocess_args function."""

  def test_preprocess_args_converts_pydantic_model(self):
    def model_fn(data: SampleModel, note: Optional[SampleModel] = None):
      pass

    sig = inspect.signature(model_fn)
    hints = get_type_hints_cached(model_fn)
    raw_args = {
        "data": {"name": "custom", "value": 42},
        "note": {"name": "default", "value": 10},
    }
    coerced = preprocess_args(raw_args, sig, hints)

    assert isinstance(coerced["data"], SampleModel)
    assert coerced["data"].value == 42
    assert coerced["data"].name == "custom"
    assert isinstance(coerced["note"], SampleModel)
    assert coerced["note"].value == 10

  def test_preprocess_args_converts_list_of_models(self):
    def list_fn(items: list[SampleModel]):
      pass

    sig = inspect.signature(list_fn)
    hints = get_type_hints_cached(list_fn)
    raw_args = {
        "items": [{"name": "one", "value": 1}, {"name": "two", "value": 2}]
    }
    coerced = preprocess_args(raw_args, sig, hints)

    assert len(coerced["items"]) == 2
    assert isinstance(coerced["items"][0], SampleModel)
    assert coerced["items"][0].value == 1
    assert isinstance(coerced["items"][1], SampleModel)
    assert coerced["items"][1].value == 2

  def test_preprocess_args_optional_and_union(self):
    def union_fn(
        user: SampleModel | None = None,
        item: Optional[OtherModel] = None,
        count: int = 10,
    ):
      pass

    sig = inspect.signature(union_fn)
    hints = get_type_hints_cached(union_fn)
    dict_args = {
        "user": {"name": "Alice", "value": 42},
        "item": {"tag": "custom_tag"},
        "count": 20,
    }
    coerced = preprocess_args(dict_args, sig, hints)
    assert isinstance(coerced["user"], SampleModel)
    assert coerced["user"].name == "Alice"
    assert isinstance(coerced["item"], OtherModel)
    assert coerced["item"].tag == "custom_tag"
    assert coerced["count"] == 20

    existing = SampleModel(name="existing", value=99)
    existing_other = OtherModel(tag="existing_tag")
    raw_args = {"user": existing, "item": existing_other, "count": 20}
    coerced_existing = preprocess_args(raw_args, sig, hints)
    assert coerced_existing["user"] is existing
    assert coerced_existing["item"] is existing_other

    invalid_args = {"user": "not_a_dict", "item": 123}
    coerced_invalid = preprocess_args(invalid_args, sig, hints)
    assert coerced_invalid["user"] == "not_a_dict"
    assert coerced_invalid["item"] == 123

  def test_preprocess_args_optional_list_of_models(self):
    def optional_list_fn(
        items: Optional[list[SampleModel]] = None,
        pipe_items: list[OtherModel] | None = None,
    ):
      pass

    sig = inspect.signature(optional_list_fn)
    hints = get_type_hints_cached(optional_list_fn)
    raw_args = {
        "items": [{"name": "one", "value": 1}],
        "pipe_items": [{"tag": "tagged"}],
    }
    coerced = preprocess_args(raw_args, sig, hints)
    assert isinstance(coerced["items"], list)
    assert len(coerced["items"]) == 1
    assert isinstance(coerced["items"][0], SampleModel)
    assert coerced["items"][0].name == "one"
    assert coerced["items"][0].value == 1

    assert isinstance(coerced["pipe_items"], list)
    assert len(coerced["pipe_items"]) == 1
    assert isinstance(coerced["pipe_items"][0], OtherModel)
    assert coerced["pipe_items"][0].tag == "tagged"

  def test_preprocess_args_partial_converts_pydantic_model(self):
    def model_fn(x: int, data: SampleModel) -> int:
      return x + data.value

    partial_fn = functools.partial(model_fn, x=10)
    sig = inspect.signature(partial_fn)
    hints = get_type_hints_cached(partial_fn)
    raw_args = {"data": {"name": "partial", "value": 32}}
    coerced = preprocess_args(raw_args, sig, hints)

    assert isinstance(coerced["data"], SampleModel)
    assert coerced["data"].value == 32
    assert coerced["data"].name == "partial"

  def test_preprocess_args_signature_none_returns_copy(self):
    raw_args = {"a": 1, "b": "val"}
    result = preprocess_args(raw_args, None)
    assert result == raw_args
    assert result is not raw_args


class TestLlmAgentOutputKeyAndAgentToolSchemaValidation:
  """Tests covering breaking change behavior for llm_agent output_key and agent_tool."""

  def test_llm_agent_output_key_state_coerces_json_mode_types(self):
    """LlmAgent output_key state receives JSON-mode coerced types (Decimal, datetime, Enum)."""
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event import Event

    class Color(Enum):
      RED = 1

    class Payload(BaseModel):
      price: Decimal
      stamped_at: datetime
      color: Color

    agent = LlmAgent(
        name="test_agent", output_key="result", output_schema=Payload
    )
    json_text = (
        '{"price": "29.99", "stamped_at": "2026-01-02T03:04:05", "color": 1}'
    )
    event = Event(
        author="test_agent",
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=json_text)]
        ),
    )
    agent._LlmAgent__maybe_save_output_to_state(event)

    result = event.actions.state_delta["result"]
    assert result == {
        "price": "29.99",
        "stamped_at": "2026-01-02T03:04:05",
        "color": 1,
    }
    assert isinstance(result["price"], str)
    assert isinstance(result["stamped_at"], str)
    assert result["color"] == 1

  @pytest.mark.asyncio
  async def test_agent_tool_result_coerces_json_mode_types(self):
    """AgentTool result coerces Decimal/datetime to JSON primitives."""
    from unittest.mock import patch

    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.events.event import Event
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.tools.tool_context import ToolContext

    class Color(Enum):
      RED = 1

    class Payload(BaseModel):
      price: Decimal
      stamped_at: datetime
      color: Color

    inner = LlmAgent(name="inner", output_schema=Payload)
    tool = AgentTool(agent=inner)
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="app", user_id="user"
    )
    ctx = ToolContext(
        invocation_context=InvocationContext(
            invocation_id="inv",
            agent=inner,
            session=session,
            session_service=session_service,
        )
    )
    json_text = (
        '{"price": "29.99", "stamped_at": "2026-01-02T03:04:05", "color": 1}'
    )

    async def fake_run_async(*args, **kwargs):
      yield Event(
          author="inner",
          content=types.Content(
              role="model", parts=[types.Part.from_text(text=json_text)]
          ),
      )

    with patch.object(Runner, "run_async", side_effect=fake_run_async):
      tool_res = await tool.run_async(args={"request": "req"}, tool_context=ctx)

    assert tool_res == {
        "price": "29.99",
        "stamped_at": "2026-01-02T03:04:05",
        "color": 1,
    }
    assert isinstance(tool_res["price"], str)
    assert isinstance(tool_res["stamped_at"], str)
    assert tool_res["color"] == 1
