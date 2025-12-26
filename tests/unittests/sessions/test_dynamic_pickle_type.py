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

import pickle
from unittest import mock

from google.adk.sessions.schemas.v0 import DynamicPickleType
import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects import mysql


@pytest.fixture
def pickle_type():
  """Fixture for DynamicPickleType instance."""
  return DynamicPickleType()


def test_load_dialect_impl_mysql(pickle_type):
  """Test that MySQL dialect uses LONGBLOB."""
  # Mock the MySQL dialect
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  # Mock the return value of type_descriptor
  mock_longblob_type = mock.Mock()
  mock_dialect.type_descriptor.return_value = mock_longblob_type

  impl = pickle_type.load_dialect_impl(mock_dialect)

  # Verify type_descriptor was called once with mysql.LONGBLOB
  mock_dialect.type_descriptor.assert_called_once_with(mysql.LONGBLOB)
  # Verify the return value is what we expect
  assert impl == mock_longblob_type


def test_load_dialect_impl_spanner(pickle_type):
  """Test that Spanner dialect uses SpannerPickleType."""
  # Mock the spanner dialect
  mock_dialect = mock.Mock()
  mock_dialect.name = "spanner+spanner"

  with mock.patch(
      "google.cloud.sqlalchemy_spanner.sqlalchemy_spanner.SpannerPickleType"
  ) as mock_spanner_type:
    pickle_type.load_dialect_impl(mock_dialect)
    mock_dialect.type_descriptor.assert_called_once_with(mock_spanner_type)


def test_load_dialect_impl_default(pickle_type):
  """Test that other dialects use default PickleType."""
  engine = create_engine("sqlite:///:memory:")
  dialect = engine.dialect
  impl = pickle_type.load_dialect_impl(dialect)
  # Should return the default impl (PickleType)
  assert impl == pickle_type.impl


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_bind_param_pickle_dialects(pickle_type, dialect_name):
  """Test that MySQL and Spanner dialects pickle the value."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  test_data = {"key": "value", "nested": [1, 2, 3]}
  result = pickle_type.process_bind_param(test_data, mock_dialect)

  # Should be pickled bytes
  assert isinstance(result, bytes)
  # Should be able to unpickle back to original
  assert pickle.loads(result) == test_data


def test_process_bind_param_default(pickle_type):
  """Test that other dialects return value as-is."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "sqlite"

  test_data = {"key": "value"}
  result = pickle_type.process_bind_param(test_data, mock_dialect)

  # Should return value unchanged (SQLAlchemy's PickleType handles it)
  assert result == test_data


def test_process_bind_param_none(pickle_type):
  """Test that None values are handled correctly."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  result = pickle_type.process_bind_param(None, mock_dialect)
  assert result is None


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_result_value_pickle_dialects(pickle_type, dialect_name):
  """Test that MySQL and Spanner dialects unpickle the value."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  test_data = {"key": "value", "nested": [1, 2, 3]}
  pickled_data = pickle.dumps(test_data)

  result = pickle_type.process_result_value(pickled_data, mock_dialect)

  # Should be unpickled back to original
  assert result == test_data


def test_process_result_value_default(pickle_type):
  """Test that other dialects return value as-is."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "sqlite"

  test_data = {"key": "value"}
  result = pickle_type.process_result_value(test_data, mock_dialect)

  # Should return value unchanged (SQLAlchemy's PickleType handles it)
  assert result == test_data


def test_process_result_value_none(pickle_type):
  """Test that None values are handled correctly."""
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  result = pickle_type.process_result_value(None, mock_dialect)
  assert result is None


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_roundtrip_pickle_dialects(pickle_type, dialect_name):
  """Test full roundtrip for MySQL and Spanner: bind -> result."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  original_data = {
      "string": "test",
      "number": 42,
      "list": [1, 2, 3],
      "nested": {"a": 1, "b": 2},
  }

  # Simulate bind (Python -> DB)
  bound_value = pickle_type.process_bind_param(original_data, mock_dialect)
  assert isinstance(bound_value, bytes)

  # Simulate result (DB -> Python)
  result_value = pickle_type.process_result_value(bound_value, mock_dialect)
  assert result_value == original_data
