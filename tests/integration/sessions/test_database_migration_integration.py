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

"""Integration tests for Alembic database migrations.

These tests run against real database engines (PostgreSQL, MySQL, SQLite)
to verify that migrations work correctly across dialects.

Usage:
  # SQLite only (no external dependencies)
  pytest tests/integration/sessions/test_database_migration_integration.py

  # With PostgreSQL and MySQL (start containers first)
  docker compose -f tests/integration/sessions/docker-compose.yml up -d
  TEST_POSTGRES_URL=postgresql://testuser:testpass@localhost:5432/test_adk \
  TEST_MYSQL_URL=mysql://testuser:testpass@localhost:3306/test_adk \
  pytest tests/integration/sessions/test_database_migration_integration.py
"""

import os

from google.adk.sessions.alembic_runner import AlembicMigrationRunner
from google.adk.sessions.migration._schema_check_utils import SCHEMA_VERSION_1_JSON
import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
_MYSQL_URL = os.environ.get("TEST_MYSQL_URL")


def _sqlite_url(tmp_path, name="test.db"):
  return f"sqlite:///{tmp_path / name}"


def _clean_database(db_url):
  """Drop all tables so tests start from a clean state."""
  engine = create_engine(db_url)
  try:
    with engine.begin() as conn:
      inspector = inspect(conn)
      table_names = inspector.get_table_names()
      if table_names:
        # Disable FK checks for MySQL during cleanup
        dialect = engine.dialect.name
        if dialect == "mysql":
          conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        for table in table_names:
          conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        if dialect == "mysql":
          conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
  finally:
    engine.dispose()


@pytest.fixture(
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param(
            "postgres",
            id="postgres",
            marks=pytest.mark.skipif(
                _POSTGRES_URL is None,
                reason="TEST_POSTGRES_URL not set",
            ),
        ),
        pytest.param(
            "mysql",
            id="mysql",
            marks=pytest.mark.skipif(
                _MYSQL_URL is None,
                reason="TEST_MYSQL_URL not set",
            ),
        ),
    ]
)
def db_url(request, tmp_path):
  """Provide a clean database URL for each test."""
  if request.param == "sqlite":
    url = _sqlite_url(tmp_path)
  elif request.param == "postgres":
    url = _POSTGRES_URL
  elif request.param == "mysql":
    url = _MYSQL_URL
  else:
    raise ValueError(f"Unknown db param: {request.param}")

  # Clean before test
  if request.param != "sqlite":
    _clean_database(url)

  yield url

  # Clean after test
  if request.param != "sqlite":
    _clean_database(url)


@pytest.fixture
def runner(db_url):
  return AlembicMigrationRunner(db_url)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpgradeDowngradeUpgradeCycle:
  """Verify the full upgrade -> downgrade -> upgrade cycle."""

  def test_full_cycle(self, runner, db_url):
    """Database survives upgrade -> downgrade -> upgrade."""
    # 1. Upgrade to head
    runner.run_migrations()
    assert runner.check_needs_migration() is False
    assert runner.get_current_revision() == "001_baseline_v1"

    # 2. Downgrade to base
    runner.downgrade("base")
    assert runner.check_needs_migration() is True
    assert runner.get_current_revision() is None

    # 3. Upgrade again
    runner.run_migrations()
    assert runner.check_needs_migration() is False
    assert runner.get_current_revision() == "001_baseline_v1"

  def test_tables_correct_after_cycle(self, runner, db_url):
    """All V1 tables exist and are correct after a full cycle."""
    runner.run_migrations()
    runner.downgrade("base")
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        tables = set(inspector.get_table_names())

        assert "adk_internal_metadata" in tables
        assert "sessions" in tables
        assert "events" in tables
        assert "app_states" in tables
        assert "user_states" in tables

        # Verify events has V1 columns
        event_cols = {c["name"] for c in inspector.get_columns("events")}
        assert "event_data" in event_cols
        assert "actions" not in event_cols
    finally:
      engine.dispose()


class TestUpgradeIdempotency:
  """Verify running upgrade multiple times is safe."""

  def test_double_upgrade(self, runner):
    """Running upgrade twice should not fail or change state."""
    runner.run_migrations()
    rev_after_first = runner.get_current_revision()

    runner.run_migrations()
    rev_after_second = runner.get_current_revision()

    assert rev_after_first == rev_after_second
    assert runner.check_needs_migration() is False

  def test_triple_upgrade(self, runner):
    """Running upgrade three times should be equally safe."""
    for _ in range(3):
      runner.run_migrations()
    assert runner.check_needs_migration() is False


class TestSchemaVersionMetadata:
  """Verify ADK schema_version metadata is set correctly."""

  def test_schema_version_set_after_upgrade(self, runner, db_url):
    """schema_version should be '1' after V1 migration."""
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

    assert result == SCHEMA_VERSION_1_JSON

  def test_schema_version_restored_after_cycle(self, runner, db_url):
    """schema_version should be correct after upgrade/downgrade/upgrade."""
    runner.run_migrations()
    runner.downgrade("base")
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

    assert result == SCHEMA_VERSION_1_JSON


class TestDataPreservation:
  """Verify data is preserved through migrations."""

  def test_data_survives_idempotent_upgrade(self, runner, db_url):
    """Data inserted after migration should survive a second upgrade."""
    runner.run_migrations()

    # Insert test data
    engine = create_engine(db_url)
    try:
      with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, create_time,"
                " update_time) VALUES (:app, :user, :sid, CURRENT_TIMESTAMP,"
                " CURRENT_TIMESTAMP)"
            ),
            {"app": "test_app", "user": "test_user", "sid": "session_1"},
        )
    finally:
      engine.dispose()

    # Run upgrade again (should be no-op)
    runner.run_migrations()

    # Verify data still exists
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM sessions")).scalar_one()
    finally:
      engine.dispose()

    assert count == 1


class TestForeignKeyConstraints:
  """Verify FK constraints work correctly across dialects."""

  def test_cascade_delete(self, runner, db_url):
    """Deleting a session should cascade-delete its events."""
    runner.run_migrations()

    engine = create_engine(db_url)
    try:
      with engine.begin() as conn:
        # Enable FK enforcement for SQLite
        if engine.dialect.name == "sqlite":
          conn.execute(text("PRAGMA foreign_keys = ON"))

        conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, create_time,"
                " update_time) VALUES (:app, :user, :sid,"
                " CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
            ),
            {"app": "app1", "user": "user1", "sid": "s1"},
        )
        conn.execute(
            text(
                "INSERT INTO events (id, app_name, user_id, session_id,"
                " invocation_id, timestamp) VALUES (:eid, :app, :user,"
                " :sid, :inv, CURRENT_TIMESTAMP)"
            ),
            {
                "eid": "e1",
                "app": "app1",
                "user": "user1",
                "sid": "s1",
                "inv": "inv1",
            },
        )

        # Delete session — event should cascade
        conn.execute(
            text(
                "DELETE FROM sessions WHERE app_name = :app"
                " AND user_id = :user AND id = :sid"
            ),
            {"app": "app1", "user": "user1", "sid": "s1"},
        )

        event_count = conn.execute(
            text("SELECT COUNT(*) FROM events")
        ).scalar_one()

    finally:
      engine.dispose()

    assert event_count == 0


class TestBootstrapExistingDatabase:
  """Verify bootstrapping works across dialects."""

  def test_bootstrap_then_upgrade_is_noop(self, runner, db_url):
    """After bootstrap, upgrade should detect no pending migrations."""
    # Create schema manually (simulating existing deployment)
    from google.adk.sessions.schemas.v1 import Base as BaseV1

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

    # Bootstrap Alembic
    runner.bootstrap_existing_database()
    assert runner.check_needs_migration() is False

    # Upgrade should be a no-op
    runner.run_migrations()
    assert runner.check_needs_migration() is False
