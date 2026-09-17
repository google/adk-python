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

"""Tests for the shared SQLAlchemy column types."""

from __future__ import annotations

import datetime
import json
from unittest import mock

from google.adk.sessions.schemas.shared import DynamicJSON
from google.adk.sessions.schemas.shared import PreciseTimestamp
import pytest
from sqlalchemy import Text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql


def _dialect(name: str) -> mock.Mock:
  """Builds a stand-in dialect whose only relevant trait is its name."""
  dialect = mock.Mock()
  dialect.name = name
  return dialect


@pytest.fixture
def dynamic_json():
  return DynamicJSON()


@pytest.fixture
def precise_timestamp():
  return PreciseTimestamp()


@pytest.mark.parametrize(
    "dialect_name, expected_type",
    [
        ("postgresql", postgresql.JSONB),
        ("mysql", mysql.LONGTEXT),
        ("sqlite", Text),
    ],
)
def test_dynamic_json_load_dialect_impl(
    dynamic_json, dialect_name, expected_type
):
  """Each dialect gets the widest JSON-capable column type it supports."""
  dialect = _dialect(dialect_name)

  impl = dynamic_json.load_dialect_impl(dialect)

  dialect.type_descriptor.assert_called_once()
  # The dialect is handed an instance, so compare its type rather than the
  # class object.
  (requested_type,), _ = dialect.type_descriptor.call_args
  assert type(requested_type) is expected_type
  assert impl == dialect.type_descriptor.return_value


def test_dynamic_json_serializes_to_json_text_for_non_postgresql(dynamic_json):
  """Dialects without a JSON column store a JSON string and read it back."""
  dialect = _dialect("sqlite")
  value = {"key": "value", "nested": [1, 2, {"deep": True}]}

  bound = dynamic_json.process_bind_param(value, dialect)

  assert isinstance(bound, str)
  assert json.loads(bound) == value
  assert dynamic_json.process_result_value(bound, dialect) == value


def test_dynamic_json_passes_values_through_for_postgresql(dynamic_json):
  """JSONB accepts and returns Python objects, so no conversion happens."""
  dialect = _dialect("postgresql")
  value = {"key": "value"}

  assert dynamic_json.process_bind_param(value, dialect) is value
  assert dynamic_json.process_result_value(value, dialect) is value


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_dynamic_json_keeps_none_as_sql_null(dynamic_json, dialect_name):
  """None must stay NULL rather than becoming the JSON string 'null'."""
  dialect = _dialect(dialect_name)

  assert dynamic_json.process_bind_param(None, dialect) is None
  assert dynamic_json.process_result_value(None, dialect) is None


def test_precise_timestamp_load_dialect_impl_mysql_keeps_microseconds(
    precise_timestamp,
):
  """MySQL needs an explicit fractional-seconds precision of 6."""
  dialect = _dialect("mysql")

  impl = precise_timestamp.load_dialect_impl(dialect)

  assert impl == dialect.type_descriptor.return_value
  (requested_type,), _ = dialect.type_descriptor.call_args
  assert isinstance(requested_type, mysql.DATETIME)
  assert requested_type.fsp == 6


def test_precise_timestamp_load_dialect_impl_defaults_to_datetime(
    precise_timestamp,
):
  """Other dialects keep the plain DateTime implementation."""
  dialect = _dialect("sqlite")

  assert precise_timestamp.load_dialect_impl(dialect) is precise_timestamp.impl
  dialect.type_descriptor.assert_not_called()


@pytest.mark.parametrize(
    "raw_value",
    [1767322475.123456, 1767322475],
    ids=["float", "int"],
)
def test_precise_timestamp_result_processor_reads_epoch_as_utc(
    precise_timestamp, raw_value
):
  """A numeric column value is a Unix epoch and must come back as UTC."""
  process = precise_timestamp.result_processor(_dialect("sqlite"), None)

  result = process(raw_value)

  assert result == datetime.datetime.fromtimestamp(
      raw_value, datetime.timezone.utc
  )
  assert result.tzinfo is datetime.timezone.utc


def test_precise_timestamp_result_processor_keeps_none(precise_timestamp):
  """A NULL column stays None instead of becoming the epoch."""
  process = precise_timestamp.result_processor(_dialect("sqlite"), None)

  assert process(None) is None


def test_precise_timestamp_result_processor_delegates_non_numeric_values(
    precise_timestamp,
):
  """Values the driver hands back untouched go through the DateTime impl."""
  expected = datetime.datetime(2026, 1, 2, 3, 4, 5, 123456)
  impl = mock.Mock()
  impl.result_processor.return_value = lambda value: expected
  precise_timestamp.impl = impl

  process = precise_timestamp.result_processor(_dialect("mysql"), None)

  assert process("2026-01-02 03:04:05.123456") == expected


def test_precise_timestamp_uses_datetime2_on_mssql():
  """SQL Server's legacy DATETIME rounds to ~3.33ms, which destroys the
  microsecond update marker used by the optimistic-concurrency check."""
  from sqlalchemy.dialects import mssql as mssql_dialect

  ts = PreciseTimestamp()
  impl = ts.load_dialect_impl(mssql_dialect.dialect())
  assert isinstance(impl, mssql_dialect.DATETIME2)
  assert impl.precision == 6
