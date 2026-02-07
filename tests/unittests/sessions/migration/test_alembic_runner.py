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

"""Unit tests for AlembicMigrationRunner."""

from google.adk.sessions.alembic_runner import AlembicMigrationRunner
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.schemas.v1 import Base as BaseV1
import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text


@pytest.fixture
def db_url(tmp_path):
  """Provide a fresh SQLite database URL for each test."""
  db_path = tmp_path / "test.db"
  return f"sqlite:///{db_path}"


@pytest.fixture
def runner(db_url):
  """Provide an AlembicMigrationRunner for a fresh database."""
  return AlembicMigrationRunner(db_url)


class TestCheckNeedsMigration:

  def test_new_database_needs_migration(self, runner):
    """A fresh database with no tables should need migration."""
    assert runner.check_needs_migration() is True

  def test_migrated_database_does_not_need_migration(self, runner):
    """After running migrations, database should not need migration."""
    runner.run_migrations()
    assert runner.check_needs_migration() is False

  def test_downgraded_database_needs_migration(self, runner):
    """After downgrading, database should need migration again."""
    runner.run_migrations()
    runner.downgrade("base")
    assert runner.check_needs_migration() is True


class TestRunMigrations:

  def test_creates_all_v1_tables(self, runner, db_url):
    """Migration should create all V1 schema tables."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
    finally:
      engine.dispose()

    expected = {
        "adk_internal_metadata",
        "sessions",
        "events",
        "app_states",
        "user_states",
        "alembic_version",
    }
    assert expected.issubset(tables)

  def test_sets_schema_version_metadata(self, runner, db_url):
    """Migration should insert schema_version = '1' into metadata."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT value FROM adk_internal_metadata"
                " WHERE key = 'schema_version'"
            )
        ).scalar_one()
    finally:
      engine.dispose()

    assert result == _schema_check_utils.SCHEMA_VERSION_1_JSON

  def test_sessions_table_has_correct_columns(self, runner, db_url):
    """Sessions table should have the expected V1 columns."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        columns = {c["name"] for c in inspector.get_columns("sessions")}
    finally:
      engine.dispose()

    expected = {
        "app_name",
        "user_id",
        "id",
        "state",
        "create_time",
        "update_time",
    }
    assert expected == columns

  def test_events_table_has_correct_columns(self, runner, db_url):
    """Events table should have V1 columns (event_data, not actions)."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        columns = {c["name"] for c in inspector.get_columns("events")}
    finally:
      engine.dispose()

    expected = {
        "id",
        "app_name",
        "user_id",
        "session_id",
        "invocation_id",
        "event_data",
        "timestamp",
    }
    assert expected == columns
    assert "actions" not in columns

  def test_events_foreign_key_to_sessions(self, runner, db_url):
    """Events table should have a FK to sessions with CASCADE delete."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        fks = inspector.get_foreign_keys("events")
    finally:
      engine.dispose()

    assert len(fks) == 1
    fk = fks[0]
    assert fk["referred_table"] == "sessions"
    assert set(fk["constrained_columns"]) == {
        "app_name",
        "user_id",
        "session_id",
    }
    assert set(fk["referred_columns"]) == {"app_name", "user_id", "id"}


class TestIdempotency:

  def test_upgrade_twice_is_safe(self, runner):
    """Running upgrade when already at head should be a no-op."""
    runner.run_migrations()
    assert runner.check_needs_migration() is False

    # Second upgrade should not raise
    runner.run_migrations()
    assert runner.check_needs_migration() is False

  def test_upgrade_downgrade_upgrade_cycle(self, runner, db_url):
    """Database should survive a full upgrade→downgrade→upgrade cycle."""
    runner.run_migrations()
    assert runner.check_needs_migration() is False

    runner.downgrade("base")
    assert runner.check_needs_migration() is True

    runner.run_migrations()
    assert runner.check_needs_migration() is False

    # Verify tables still exist and are correct
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
    finally:
      engine.dispose()

    assert "sessions" in tables
    assert "events" in tables
    assert "adk_internal_metadata" in tables


class TestGetCurrentRevision:

  def test_returns_none_for_new_database(self, runner):
    """New database should have no current revision."""
    assert runner.get_current_revision() is None

  def test_returns_revision_after_migration(self, runner):
    """After migration, current revision should match head."""
    runner.run_migrations()
    current = runner.get_current_revision()
    assert current is not None
    assert current == "001_baseline_v1"


class TestDowngrade:

  def test_downgrade_removes_tables(self, runner, db_url):
    """Downgrading to base should remove all ADK tables."""
    runner.run_migrations()
    runner.downgrade("base")

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())
    finally:
      engine.dispose()

    assert "sessions" not in tables
    assert "events" not in tables
    assert "adk_internal_metadata" not in tables

  def test_downgrade_minus_one(self, runner):
    """Downgrade -1 from head should go to base (only one migration)."""
    runner.run_migrations()
    assert runner.get_current_revision() == "001_baseline_v1"

    runner.downgrade("-1")
    assert runner.get_current_revision() is None


class TestStamp:

  def test_stamp_sets_revision_without_running_migration(self, db_url):
    """Stamp should set alembic_version without creating tables."""
    runner = AlembicMigrationRunner(db_url)
    runner.stamp("001_baseline_v1")

    assert runner.get_current_revision() == "001_baseline_v1"

    # Tables should NOT exist (stamp doesn't run migrations)
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        assert not inspector.has_table("sessions")
    finally:
      engine.dispose()


class TestBootstrapExistingDatabase:

  def test_bootstrap_v1_database(self, db_url):
    """Bootstrapping a V1 database should stamp the baseline revision."""
    # Manually create V1 schema (simulating existing deployment)
    engine = create_engine(db_url)
    try:
      BaseV1.metadata.create_all(engine)
      with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO adk_internal_metadata (key, value)"
                " VALUES ('schema_version', '1')"
            )
        )
    finally:
      engine.dispose()

    runner = AlembicMigrationRunner(db_url)
    runner.bootstrap_existing_database()

    assert runner.get_current_revision() == "001_baseline_v1"
    assert runner.check_needs_migration() is False

  def test_bootstrap_v0_database_migrates_in_place(self, tmp_path):
    """Bootstrapping a V0 database should migrate it to V1 in-place."""
    import pickle

    from google.adk.sessions.schemas import v0

    db_path = tmp_path / "v0.db"
    db_url = f"sqlite:///{db_path}"

    # Create V0 schema and insert test data
    engine = create_engine(db_url)
    try:
      v0.Base.metadata.create_all(engine)
      with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id,"
                " state, create_time, update_time)"
                " VALUES ('app1', 'user1', 's1', '{}',"
                " CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO events (id, app_name, user_id,"
                " session_id, invocation_id, author,"
                " actions, timestamp)"
                " VALUES (:id, :app, :user, :sid,"
                " :inv, :author, :actions,"
                " CURRENT_TIMESTAMP)"
            ),
            {
                "id": "e1",
                "app": "app1",
                "user": "user1",
                "sid": "s1",
                "inv": "inv1",
                "author": "agent",
                "actions": pickle.dumps({}),
            },
        )
    finally:
      engine.dispose()

    runner = AlembicMigrationRunner(db_url)
    runner.bootstrap_existing_database()

    assert runner.get_current_revision() == "001_baseline_v1"
    assert runner.check_needs_migration() is False

    # Verify V1 schema: event_data column exists, actions does not
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        event_cols = {c["name"] for c in inspector.get_columns("events")}
        assert "event_data" in event_cols
        assert "actions" not in event_cols

        # Verify metadata table was created
        assert inspector.has_table("adk_internal_metadata")

        # Verify schema version
        version = conn.execute(
            text(
                "SELECT value FROM adk_internal_metadata"
                " WHERE key = 'schema_version'"
            )
        ).scalar_one()
        assert version == "1"

        # Verify event data was migrated
        event_data_raw = conn.execute(
            text("SELECT event_data FROM events WHERE id = 'e1'")
        ).scalar_one()
        assert event_data_raw is not None
    finally:
      engine.dispose()


class TestAsyncUrlConversion:

  def test_async_url_is_converted_to_sync(self, tmp_path):
    """Runner should handle async SQLAlchemy URLs transparently."""
    db_path = tmp_path / "async_test.db"
    async_url = f"sqlite+aiosqlite:///{db_path}"

    runner = AlembicMigrationRunner(async_url)
    # Should not raise — async driver stripped internally
    assert runner.check_needs_migration() is True

    runner.run_migrations()
    assert runner.check_needs_migration() is False
