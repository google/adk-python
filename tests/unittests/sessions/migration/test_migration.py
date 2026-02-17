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
"""Tests for migration scripts."""

from __future__ import annotations

from datetime import datetime
from datetime import timezone

from google.adk.events.event_actions import EventActions
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.migration import migrate_from_sqlalchemy_pickle as mfsp
from google.adk.sessions.schemas import v0
from google.adk.sessions.schemas import v1
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


class TestToSyncUrl:
  """Tests for the to_sync_url function."""

  @pytest.mark.parametrize(
      "input_url,expected_url",
      [
          # PostgreSQL async drivers
          (
              "postgresql+asyncpg://localhost/mydb",
              "postgresql://localhost/mydb",
          ),
          (
              "postgresql+asyncpg://user:pass@localhost:5432/mydb",
              "postgresql://user:pass@localhost:5432/mydb",
          ),
          # PostgreSQL sync drivers (should still strip)
          (
              "postgresql+psycopg2://localhost/mydb",
              "postgresql://localhost/mydb",
          ),
          # MySQL async drivers
          (
              "mysql+aiomysql://localhost/mydb",
              "mysql://localhost/mydb",
          ),
          (
              "mysql+asyncmy://user:pass@localhost:3306/mydb",
              "mysql://user:pass@localhost:3306/mydb",
          ),
          # SQLite async driver
          (
              "sqlite+aiosqlite:///path/to/db.sqlite",
              "sqlite:///path/to/db.sqlite",
          ),
          (
              "sqlite+aiosqlite:///:memory:",
              "sqlite:///:memory:",
          ),
          # URLs without driver specification (unchanged)
          (
              "postgresql://localhost/mydb",
              "postgresql://localhost/mydb",
          ),
          (
              "mysql://localhost/mydb",
              "mysql://localhost/mydb",
          ),
          (
              "sqlite:///path/to/db.sqlite",
              "sqlite:///path/to/db.sqlite",
          ),
          # Edge cases
          (
              "sqlite:///:memory:",
              "sqlite:///:memory:",
          ),
          # Complex URL with query parameters
          (
              "postgresql+asyncpg://user:pass@host/db?ssl=require",
              "postgresql://user:pass@host/db?ssl=require",
          ),
      ],
  )
  def test_to_sync_url(self, input_url, expected_url):
    """Test that async driver specifications are correctly removed."""
    assert _schema_check_utils.to_sync_url(input_url) == expected_url

  def test_to_sync_url_no_scheme_separator(self):
    """Test that URLs without :// are returned unchanged."""
    # This is an invalid URL but the function should handle it gracefully
    assert _schema_check_utils.to_sync_url("not-a-url") == "not-a-url"

  def test_to_sync_url_empty_string(self):
    """Test that empty string is returned unchanged."""
    assert _schema_check_utils.to_sync_url("") == ""


def test_migrate_from_sqlalchemy_pickle(tmp_path):
  """Tests for migrate_from_sqlalchemy_pickle."""
  source_db_path = tmp_path / "source_pickle.db"
  dest_db_path = tmp_path / "dest_json.db"
  source_db_url = f"sqlite:///{source_db_path}"
  dest_db_url = f"sqlite:///{dest_db_path}"

  # Set up source DB with old pickle schema
  source_engine = create_engine(source_db_url)
  v0.Base.metadata.create_all(source_engine)
  SourceSession = sessionmaker(bind=source_engine)
  source_session = SourceSession()

  # Populate source data
  now = datetime.now(timezone.utc)
  app_state = v0.StorageAppState(
      app_name="app1", state={"akey": 1}, update_time=now
  )
  user_state = v0.StorageUserState(
      app_name="app1", user_id="user1", state={"ukey": 2}, update_time=now
  )
  session = v0.StorageSession(
      app_name="app1",
      user_id="user1",
      id="session1",
      state={"skey": 3},
      create_time=now,
      update_time=now,
  )
  event = v0.StorageEvent(
      id="event1",
      app_name="app1",
      user_id="user1",
      session_id="session1",
      invocation_id="invoke1",
      author="user",
      actions=EventActions(state_delta={"skey": 4}),
      timestamp=now,
  )
  source_session.add_all([app_state, user_state, session, event])
  source_session.commit()
  source_session.close()

  mfsp.migrate(source_db_url, dest_db_url)

  # Verify destination DB
  dest_engine = create_engine(dest_db_url)
  DestSession = sessionmaker(bind=dest_engine)
  dest_session = DestSession()

  metadata = dest_session.query(v1.StorageMetadata).first()
  assert metadata is not None
  assert metadata.key == _schema_check_utils.SCHEMA_VERSION_KEY
  assert metadata.value == _schema_check_utils.SCHEMA_VERSION_1_JSON

  app_state_res = dest_session.query(v1.StorageAppState).first()
  assert app_state_res is not None
  assert app_state_res.app_name == "app1"
  assert app_state_res.state == {"akey": 1}

  user_state_res = dest_session.query(v1.StorageUserState).first()
  assert user_state_res is not None
  assert user_state_res.user_id == "user1"
  assert user_state_res.state == {"ukey": 2}

  session_res = dest_session.query(v1.StorageSession).first()
  assert session_res is not None
  assert session_res.id == "session1"
  assert session_res.state == {"skey": 3}

  event_res = dest_session.query(v1.StorageEvent).first()
  assert event_res is not None
  assert event_res.id == "event1"
  assert "state_delta" in event_res.event_data["actions"]
  assert event_res.event_data["actions"]["state_delta"] == {"skey": 4}

  dest_session.close()


def test_migrate_from_sqlalchemy_pickle_with_async_driver_urls(tmp_path):
  """Tests that migration works with async driver URLs (fixes issue #4176).

  Users often provide async driver URLs (e.g., postgresql+asyncpg://) since
  that's what ADK requires at runtime. The migration tool should handle these
  by automatically converting them to sync URLs.
  """
  source_db_path = tmp_path / "source_pickle_async.db"
  dest_db_path = tmp_path / "dest_json_async.db"
  # Use async driver URLs like users would typically provide
  source_db_url = f"sqlite+aiosqlite:///{source_db_path}"
  dest_db_url = f"sqlite+aiosqlite:///{dest_db_path}"

  # Set up source DB with old pickle schema using sync URL
  sync_source_url = f"sqlite:///{source_db_path}"
  source_engine = create_engine(sync_source_url)
  v0.Base.metadata.create_all(source_engine)
  SourceSession = sessionmaker(bind=source_engine)
  source_session = SourceSession()

  # Populate source data
  now = datetime.now(timezone.utc)
  app_state = v0.StorageAppState(
      app_name="async_app", state={"key": "value"}, update_time=now
  )
  session = v0.StorageSession(
      app_name="async_app",
      user_id="async_user",
      id="async_session",
      state={},
      create_time=now,
      update_time=now,
  )
  source_session.add_all([app_state, session])
  source_session.commit()
  source_session.close()

  # This should NOT raise an error about async drivers (the fix for #4176)
  mfsp.migrate(source_db_url, dest_db_url)

  # Verify destination DB
  sync_dest_url = f"sqlite:///{dest_db_path}"
  dest_engine = create_engine(sync_dest_url)
  DestSession = sessionmaker(bind=dest_engine)
  dest_session = DestSession()

  metadata = dest_session.query(v1.StorageMetadata).first()
  assert metadata is not None
  assert metadata.key == _schema_check_utils.SCHEMA_VERSION_KEY
  assert metadata.value == _schema_check_utils.SCHEMA_VERSION_1_JSON

  app_state_res = dest_session.query(v1.StorageAppState).first()
  assert app_state_res is not None
  assert app_state_res.app_name == "async_app"
  assert app_state_res.state == {"key": "value"}

  session_res = dest_session.query(v1.StorageSession).first()
  assert session_res is not None
  assert session_res.id == "async_session"

  dest_session.close()


def _assert_update_timestamp_tz_is_utc_timestamp(schema_module) -> None:
  engine = create_engine("sqlite:///:memory:")
  schema_module.Base.metadata.create_all(engine)
  SessionLocal = sessionmaker(bind=engine)

  update_time = datetime(2026, 1, 1, 0, 0, 0)
  storage_session = schema_module.StorageSession(
      app_name="app",
      user_id="user",
      id="sid",
      state={},
      create_time=update_time,
      update_time=update_time,
  )

  with SessionLocal() as db:
    db.add(storage_session)
    db.commit()

    fetched = db.get(schema_module.StorageSession, ("app", "user", "sid"))
    assert fetched is not None
    assert isinstance(fetched.update_timestamp_tz, float)
    assert (
        fetched.update_timestamp_tz
        == update_time.replace(tzinfo=timezone.utc).timestamp()
    )


def test_v1_storage_session_update_timestamp_tz() -> None:
  _assert_update_timestamp_tz_is_utc_timestamp(v1)


def test_v0_storage_session_update_timestamp_tz() -> None:
  _assert_update_timestamp_tz_is_utc_timestamp(v0)
