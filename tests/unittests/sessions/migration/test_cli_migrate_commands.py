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

"""Tests for ``adk migrate`` CLI commands (upgrade, downgrade, check, stamp)."""

import os
import pickle

from click.testing import CliRunner
from google.adk.cli import cli_tools_click
from google.adk.sessions.alembic_runner import AlembicMigrationRunner
import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text


@pytest.fixture
def db_url(tmp_path):
  """Provide a fresh SQLite database URL for each test."""
  db_path = tmp_path / "test_cli.db"
  return f"sqlite:///{db_path}"


@pytest.fixture
def cli():
  """Provide a Click CliRunner."""
  return CliRunner()


def _create_v1_tables(db_url):
  """Create V1 schema tables without Alembic tracking."""
  engine = create_engine(db_url)
  try:
    with engine.begin() as conn:
      conn.execute(
          text(
              "CREATE TABLE adk_internal_metadata ("
              "  key VARCHAR(128) NOT NULL PRIMARY KEY,"
              "  value VARCHAR(256) NOT NULL"
              ")"
          )
      )
      conn.execute(
          text(
              "INSERT INTO adk_internal_metadata (key, value)"
              " VALUES ('schema_version', '1')"
          )
      )
      conn.execute(
          text(
              "CREATE TABLE sessions ("
              "  app_name VARCHAR NOT NULL,"
              "  user_id VARCHAR NOT NULL,"
              "  id VARCHAR NOT NULL,"
              "  state TEXT,"
              "  create_time DATETIME,"
              "  update_time DATETIME,"
              "  PRIMARY KEY (app_name, user_id, id)"
              ")"
          )
      )
      conn.execute(
          text(
              "CREATE TABLE events ("
              "  id VARCHAR NOT NULL,"
              "  app_name VARCHAR NOT NULL,"
              "  user_id VARCHAR NOT NULL,"
              "  session_id VARCHAR NOT NULL,"
              "  invocation_id VARCHAR,"
              "  event_data TEXT,"
              "  timestamp DATETIME,"
              "  PRIMARY KEY (id, app_name, user_id, session_id),"
              "  FOREIGN KEY (app_name, user_id, session_id)"
              "    REFERENCES sessions (app_name, user_id, id)"
              "    ON DELETE CASCADE"
              ")"
          )
      )
  finally:
    engine.dispose()


def _create_v0_tables(db_url):
  """Create V0 schema tables (pickle-based) without Alembic tracking."""
  engine = create_engine(db_url)
  try:
    with engine.begin() as conn:
      conn.execute(
          text(
              "CREATE TABLE sessions ("
              "  app_name VARCHAR NOT NULL,"
              "  user_id VARCHAR NOT NULL,"
              "  id VARCHAR NOT NULL,"
              "  state TEXT NOT NULL,"
              "  create_time DATETIME,"
              "  update_time DATETIME,"
              "  PRIMARY KEY (app_name, user_id, id)"
              ")"
          )
      )
      conn.execute(
          text(
              "CREATE TABLE events ("
              "  id VARCHAR NOT NULL,"
              "  app_name VARCHAR NOT NULL,"
              "  user_id VARCHAR NOT NULL,"
              "  session_id VARCHAR NOT NULL,"
              "  invocation_id VARCHAR,"
              "  author VARCHAR,"
              "  branch VARCHAR,"
              "  actions BLOB NOT NULL,"
              "  long_running_tool_ids_json TEXT,"
              "  content TEXT,"
              "  grounding_metadata TEXT,"
              "  custom_metadata TEXT,"
              "  usage_metadata TEXT,"
              "  citation_metadata TEXT,"
              "  partial BOOLEAN,"
              "  turn_complete BOOLEAN,"
              "  error_code TEXT,"
              "  error_message TEXT,"
              "  interrupted BOOLEAN,"
              "  input_transcription TEXT,"
              "  output_transcription TEXT,"
              "  timestamp DATETIME,"
              "  PRIMARY KEY (id, app_name, user_id, session_id),"
              "  FOREIGN KEY (app_name, user_id, session_id)"
              "    REFERENCES sessions (app_name, user_id, id)"
              "    ON DELETE CASCADE"
              ")"
          )
      )
  finally:
    engine.dispose()


def _has_alembic_version(db_url):
  """Check if alembic_version table exists and has a revision."""
  engine = create_engine(db_url)
  try:
    with engine.connect() as conn:
      inspector = inspect(conn)
      if not inspector.has_table("alembic_version"):
        return False
      row = conn.execute(
          text("SELECT version_num FROM alembic_version")
      ).fetchone()
      return row is not None
  finally:
    engine.dispose()


# ── upgrade command ──────────────────────────────────────────────


class TestUpgrade:

  def test_fresh_database(self, cli, db_url):
    """upgrade on empty DB creates tables and stamps Alembic."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "completed successfully" in result.output

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        assert inspector.has_table("sessions")
        assert inspector.has_table("events")
        assert inspector.has_table("alembic_version")
    finally:
      engine.dispose()

  def test_existing_v1_database_bootstraps(self, cli, db_url):
    """upgrade on existing V1 DB (no Alembic) bootstraps then is up-to-date."""
    _create_v1_tables(db_url)

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "Bootstrapping" in result.output
    assert _has_alembic_version(db_url)

  def test_already_up_to_date(self, cli, db_url):
    """upgrade on already-migrated DB is a no-op."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "already up-to-date" in result.output

  def test_existing_v0_database_auto_migrates(self, cli, db_url):
    """upgrade on V0 DB bootstraps (V0→V1 in-place) then is up-to-date."""
    _create_v0_tables(db_url)

    # Insert a V0 event so bootstrap has something to migrate.
    engine = create_engine(db_url)
    try:
      with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state)"
                " VALUES ('app', 'user', 's1', '{}')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO events"
                " (id, app_name, user_id, session_id, actions)"
                " VALUES ('e1', 'app', 'user', 's1', :actions)"
            ),
            {"actions": pickle.dumps({})},
        )
    finally:
      engine.dispose()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "Bootstrapping" in result.output
    assert _has_alembic_version(db_url)

    # Verify V0 columns are gone
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("events")}
        assert "event_data" in cols
        assert "actions" not in cols
    finally:
      engine.dispose()

  def test_invalid_url_exits_1(self, cli):
    """upgrade with invalid URL exits with code 1."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", "invalid://not-a-db"],
    )
    assert result.exit_code == 1

  def test_idempotent_double_upgrade(self, cli, db_url):
    """Running upgrade twice is safe."""
    for _ in range(2):
      result = cli.invoke(
          cli_tools_click.main,
          ["migrate", "upgrade", "--db_url", db_url],
      )
    assert result.exit_code == 0, result.output
    assert "already up-to-date" in result.output


# ── downgrade command ────────────────────────────────────────────


class TestDowngrade:

  def test_downgrade_one_step(self, cli, db_url):
    """downgrade -1 after upgrade removes tables."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "downgrade", "--db_url", db_url, "--revision", "-1"],
    )
    assert result.exit_code == 0, result.output
    assert "completed successfully" in result.output

    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        inspector = inspect(conn)
        assert not inspector.has_table("sessions")
        assert not inspector.has_table("events")
    finally:
      engine.dispose()

  def test_downgrade_to_base(self, cli, db_url):
    """downgrade to 'base' removes all migration state."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "downgrade", "--db_url", db_url, "--revision", "base"],
    )
    assert result.exit_code == 0, result.output

  def test_downgrade_default_revision(self, cli, db_url):
    """downgrade without --revision defaults to -1."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "downgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "'-1'" in result.output

  def test_downgrade_empty_db_exits_1(self, cli, db_url):
    """downgrade on empty DB fails."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "downgrade", "--db_url", db_url],
    )
    assert result.exit_code == 1


# ── check command ────────────────────────────────────────────────


class TestCheck:

  def test_up_to_date_exits_0(self, cli, db_url):
    """check on migrated DB exits 0."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "up-to-date" in result.output

  def test_pending_migrations_exits_1(self, cli, db_url):
    """check on fresh DB exits 1 (migrations pending)."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 1
    assert "pending" in result.output.lower()

  def test_after_downgrade_exits_1(self, cli, db_url):
    """check after downgrade exits 1."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()
    runner.downgrade("base")

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 1


# ── stamp command ────────────────────────────────────────────────


class TestStamp:

  def test_stamp_v1_database(self, cli, db_url):
    """stamp on V1 DB sets Alembic tracking."""
    _create_v1_tables(db_url)

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "stamp", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "bootstrapped" in result.output.lower()
    assert _has_alembic_version(db_url)

    # After stamp, check should be up-to-date
    runner = AlembicMigrationRunner(db_url)
    assert runner.check_needs_migration() is False

  def test_stamp_v0_database_auto_migrates(self, cli, db_url):
    """stamp on V0 DB performs in-place migration and stamps."""
    _create_v0_tables(db_url)
    engine = create_engine(db_url)
    try:
      with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO sessions (app_name, user_id, id, state)"
                " VALUES ('app', 'user', 's1', '{}')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO events"
                " (id, app_name, user_id, session_id, actions)"
                " VALUES ('e1', 'app', 'user', 's1', :actions)"
            ),
            {"actions": pickle.dumps({})},
        )
    finally:
      engine.dispose()

    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "stamp", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert _has_alembic_version(db_url)

    # Verify V1 schema
    engine = create_engine(db_url)
    try:
      with engine.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("events")}
        assert "event_data" in cols
        assert "actions" not in cols
    finally:
      engine.dispose()

  def test_stamp_fresh_database(self, cli, db_url):
    """stamp on empty DB stamps the baseline."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "stamp", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert _has_alembic_version(db_url)


# ── upgrade → check → downgrade → check cycle ───────────────────


class TestUpgradeDowngradeCycle:

  def test_full_cycle(self, cli, db_url):
    """upgrade → check (0) → downgrade → check (1) → upgrade → check (0)."""
    # Step 1: upgrade
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output

    # Step 2: check → up-to-date
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output

    # Step 3: downgrade
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "downgrade", "--db_url", db_url, "--revision", "base"],
    )
    assert result.exit_code == 0, result.output

    # Step 4: check → pending
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 1

    # Step 5: upgrade again
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "upgrade", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output
    assert "completed successfully" in result.output

    # Step 6: check → up-to-date
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "check", "--db_url", db_url],
    )
    assert result.exit_code == 0, result.output


# ── generate command ──────────────────────────────────────────────


class TestGenerate:

  def test_generate_empty_template(self, db_url, tmp_path):
    """generate without autogenerate produces an empty migration script."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    output_dir = str(tmp_path / "versions")
    os.makedirs(output_dir)
    path = runner.generate_revision(
        "empty_template", autogenerate=False, output_dir=output_dir
    )
    assert os.path.isfile(path)
    content = open(path).read()
    assert "empty_template" in content
    assert "def upgrade" in content
    assert "def downgrade" in content

  def test_generate_autogenerate(self, db_url, tmp_path):
    """generate with autogenerate produces a migration script."""
    runner = AlembicMigrationRunner(db_url)
    runner.run_migrations()

    output_dir = str(tmp_path / "versions")
    os.makedirs(output_dir)
    path = runner.generate_revision(
        "auto_test", autogenerate=True, output_dir=output_dir
    )
    assert os.path.isfile(path)
    content = open(path).read()
    assert "auto_test" in content

  def test_cli_generate_missing_message_exits_nonzero(self, cli, db_url):
    """CLI generate requires --message."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", "generate", "--db_url", db_url],
    )
    assert result.exit_code != 0


# ── missing --db_url ─────────────────────────────────────────────


class TestMissingRequiredArgs:

  @pytest.mark.parametrize(
      "subcommand",
      [
          "upgrade",
          "downgrade",
          "check",
          "stamp",
          "generate",
      ],
  )
  def test_missing_db_url_exits_nonzero(self, cli, subcommand):
    """All migrate subcommands require --db_url."""
    result = cli.invoke(
        cli_tools_click.main,
        ["migrate", subcommand],
    )
    assert result.exit_code != 0
    assert "db_url" in result.output.lower() or "error" in result.output.lower()
