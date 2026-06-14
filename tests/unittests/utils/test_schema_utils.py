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
from google.adk.utils._schema_utils import validate_schema
from pydantic import BaseModel


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
