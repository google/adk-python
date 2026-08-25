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

from __future__ import annotations

import pickle
from unittest import mock

from google.adk.events.event_actions import EventActions
from google.adk.sessions.schemas.v0 import Base
from google.adk.sessions.schemas.v0 import DynamicPickleType
from google.adk.sessions.schemas.v0 import StorageEvent
import pytest
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy import text
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import sessionmaker

_EXECUTED_PAYLOAD_TAGS: list[str] = []


def _record_payload_execution(tag: str) -> str:
  """Stands in for the arbitrary callable a crafted blob would reach."""
  _EXECUTED_PAYLOAD_TAGS.append(tag)
  return tag


class _CraftedActionsBlob:
  """Pickles into a call of a global that the actions allowlist omits."""

  def __reduce__(self):
    return (_record_payload_execution, ("executed",))


@pytest.fixture
def pickle_type():
  """Fixture for DynamicPickleType instance."""
  return DynamicPickleType()


@pytest.fixture
def crafted_blob():
  """Fixture for a pickled blob that runs code when loaded unrestricted."""
  _EXECUTED_PAYLOAD_TAGS.clear()
  yield pickle.dumps(_CraftedActionsBlob())
  _EXECUTED_PAYLOAD_TAGS.clear()


def test_load_dialect_impl_mysql(pickle_type):
  """Test that MySQL dialect uses LONGBLOB."""
  # Mock the MySQL dialect
  mock_dialect = mock.Mock()
  mock_dialect.name = "mysql"

  # Mock the return value of type_descriptor
  mock_longblob_type = mock.Mock()
  mock_dialect.type_descriptor.return_value = mock_longblob_type

  impl = pickle_type.load_dialect_impl(mock_dialect)

  # SQLAlchemy dialect descriptors operate on type instances, not classes.
  mock_dialect.type_descriptor.assert_called_once()
  assert isinstance(
      mock_dialect.type_descriptor.call_args.args[0], mysql.LONGBLOB
  )
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
    mock_spanner_type.assert_called_once_with()
    mock_dialect.type_descriptor.assert_called_once_with(
        mock_spanner_type.return_value
    )


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


@pytest.mark.parametrize(
    "dialect_name",
    [
        pytest.param("mysql", id="mysql"),
        pytest.param("spanner+spanner", id="spanner"),
    ],
)
def test_process_result_value_rejects_disallowed_global(
    pickle_type, dialect_name, crafted_blob
):
  """MySQL and Spanner blobs may only name globals on the allowlist."""
  mock_dialect = mock.Mock()
  mock_dialect.name = dialect_name

  with pytest.raises(pickle.UnpicklingError):
    pickle_type.process_result_value(crafted_blob, mock_dialect)

  assert _EXECUTED_PAYLOAD_TAGS == []


def test_reading_event_rejects_actions_blob_with_disallowed_global(
    crafted_blob,
):
  """Dialects handled by SQLAlchemy's PickleType are restricted too."""
  engine = create_engine("sqlite://")
  Base.metadata.create_all(engine)
  with engine.begin() as conn:
    conn.execute(
        text(
            "INSERT INTO events (id, app_name, user_id, session_id,"
            " invocation_id, author, actions, timestamp) VALUES ('event1',"
            " 'app1', 'user1', 'session1', 'invoke1', 'user', :actions,"
            " '2026-01-01 00:00:00')"
        ),
        {"actions": crafted_blob},
    )

  with sessionmaker(bind=engine)() as sql_session:
    with pytest.raises(pickle.UnpicklingError):
      sql_session.execute(select(StorageEvent)).scalars().first()

  assert _EXECUTED_PAYLOAD_TAGS == []


def test_reading_event_still_loads_stored_actions():
  """Allowlisted action payloads keep round-tripping through the database."""
  engine = create_engine("sqlite://")
  Base.metadata.create_all(engine)
  Session = sessionmaker(bind=engine)
  with Session() as sql_session:
    sql_session.add(
        StorageEvent(
            id="event1",
            app_name="app1",
            user_id="user1",
            session_id="session1",
            invocation_id="invoke1",
            author="user",
            actions=EventActions(state_delta={"key": "value"}),
        )
    )
    sql_session.commit()

  with Session() as sql_session:
    stored = sql_session.execute(select(StorageEvent)).scalars().first()
    assert stored is not None
    assert stored.actions.state_delta == {"key": "value"}
