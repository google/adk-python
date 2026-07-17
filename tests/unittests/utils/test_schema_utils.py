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

from google.adk.utils._schema_utils import get_list_inner_type
from google.adk.utils._schema_utils import is_basemodel_schema
from google.adk.utils._schema_utils import is_list_of_basemodel
from google.adk.utils._schema_utils import validate_node_data
from google.adk.utils._schema_utils import validate_schema
from google.genai import types
from pydantic import BaseModel
from pydantic import ValidationError
import pytest


class SampleModel(BaseModel):
  """Sample model for testing."""

  name: str
  value: int


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


class TestGetListInnerType:
  """Tests for get_list_inner_type function."""

  def test_list_of_basemodel_returns_inner_type(self):
    """Test that list[BaseModel] returns the inner type."""
    assert get_list_inner_type(list[SampleModel]) is SampleModel

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
    assert result == {'name': 'test', 'value': 42}

  def test_basemodel_schema_excludes_none(self):
    """Test that None values are excluded from the result."""

    class ModelWithOptional(BaseModel):
      name: str
      optional_field: str | None = None

    json_text = '{"name": "test", "optional_field": null}'
    result = validate_schema(ModelWithOptional, json_text)
    assert result == {'name': 'test'}

  def test_list_of_basemodel_schema(self):
    """Test validation with a list[BaseModel] schema."""
    json_text = '[{"name": "item1", "value": 1}, {"name": "item2", "value": 2}]'
    result = validate_schema(list[SampleModel], json_text)
    assert result == [
        {'name': 'item1', 'value': 1},
        {'name': 'item2', 'value': 2},
    ]

  def test_list_of_str_schema(self):
    """Test validation with a list[str] schema."""
    json_text = '["a", "b", "c"]'
    result = validate_schema(list[str], json_text)
    assert result == ['a', 'b', 'c']

  def test_dict_schema(self):
    """Test validation with a dict schema."""
    json_text = '{"key1": 1, "key2": 2}'
    result = validate_schema(dict[str, int], json_text)
    assert result == {'key1': 1, 'key2': 2}

  def test_json_code_fence_is_stripped(self):
    """Test that a ```json fenced payload is unwrapped before validation."""
    json_text = '```json\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_uppercase_json_code_fence_is_stripped(self):
    """Test that an uppercase language tag is not left in the payload."""
    json_text = '```JSON\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_other_language_tag_code_fence_is_stripped(self):
    """Test that any language tag on the fence is unwrapped."""
    json_text = '```python\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_bare_code_fence_is_stripped(self):
    """Test that a fence without a language tag is unwrapped."""
    json_text = '```\n{"name": "test", "value": 42}\n```'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_code_fence_with_surrounding_whitespace_is_stripped(self):
    """Test that whitespace around the fence does not break unwrapping."""
    json_text = '  \n```json\n{"name": "test", "value": 42}\n```  \n'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_list_schema_code_fence_is_stripped(self):
    """Test that a fenced list[BaseModel] payload is unwrapped."""
    json_text = '```json\n[{"name": "item1", "value": 1}]\n```'
    result = validate_schema(list[SampleModel], json_text)
    assert result == [{'name': 'item1', 'value': 1}]

  def test_plain_json_is_unaffected(self):
    """Test that unfenced JSON is validated unchanged."""
    json_text = '{"name": "test", "value": 42}'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': 'test', 'value': 42}

  def test_backticks_inside_value_are_preserved(self):
    """Test that triple backticks inside a valid JSON value are not stripped."""
    json_text = '{"name": "```", "value": 42}'
    result = validate_schema(SampleModel, json_text)
    assert result == {'name': '```', 'value': 42}


class TestValidateNodeData:
  """Tests for validate_node_data function."""

  def test_none_schema_or_data_returns_data(self):
    """Bypasses validation if schema or data is None."""
    assert validate_node_data(None, 'some_data') == 'some_data'
    assert validate_node_data(SampleModel, None) is None

  def test_dict_or_types_schema_returns_data(self):
    """Bypasses validation if schema is dict or types.Schema."""
    assert validate_node_data({'key': int}, 'some_data') == 'some_data'
    # Mock types.Schema
    schema = types.Schema(type=types.Type.STRING)
    assert validate_node_data(schema, 'some_data') == 'some_data'

  def test_content_schema_returns_data(self):
    """Bypasses validation if target schema is types.Content or subclass."""
    result = validate_node_data(
        types.Content, types.Content(role='user', parts=[])
    )
    assert result == {'role': 'user', 'parts': []}

  def test_plain_basemodel_schema_validates_raw_dict(self):
    """Validates raw dict data against BaseModel schema."""
    result = validate_node_data(SampleModel, {'name': 'test', 'value': 42})
    assert result == {'name': 'test', 'value': 42}

  def test_content_data_and_preserve_content(self):
    """Validates wrapped content and wraps result back into Content."""
    data = types.Content(
        role='user',
        parts=[types.Part(text='{"name": "test", "value": 42}')],
    )
    result = validate_node_data(SampleModel, data, preserve_content=True)
    assert isinstance(result, types.Content)
    assert result.role == 'user'
    assert len(result.parts) == 1
    assert result.parts[0].text == '{"name": "test", "value": 42}'

  def test_content_data_no_preserve_content(self):
    """Validates wrapped content and returns unwrapped dictionary."""
    data = types.Content(
        role='user',
        parts=[types.Part(text='{"name": "test", "value": 42}')],
    )
    result = validate_node_data(SampleModel, data, preserve_content=False)
    assert isinstance(result, dict)
    assert result == {'name': 'test', 'value': 42}

  def test_raw_json_string_validated_against_basemodel_schema(self):
    """Raw JSON string fails validation against BaseModel schema (not auto-parsed)."""
    with pytest.raises(ValidationError):
      validate_node_data(SampleModel, '{"name": "test", "value": 42}')

  def test_raw_string_not_parsed_with_str_schema(self):
    """Bypasses JSON parsing if schema is str."""
    result = validate_node_data(str, 'hello')
    assert result == 'hello'
