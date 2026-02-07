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

"""Alembic migration runner for ADK database schema management.

Provides programmatic access to Alembic migrations without requiring
an alembic.ini file. Migrations are bundled inside the package and
located relative to the source tree.
"""

from __future__ import annotations

from datetime import timezone
import json
import logging
import os
import pathlib
import pickle
from typing import Optional

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

from .migration._schema_check_utils import get_db_schema_version_from_connection
from .migration._schema_check_utils import LATEST_SCHEMA_VERSION
from .migration._schema_check_utils import SCHEMA_VERSION_0_PICKLE
from .migration._schema_check_utils import SCHEMA_VERSION_1_JSON
from .migration._schema_check_utils import to_sync_url

logger = logging.getLogger("google_adk." + __name__)

_MIGRATION_DIR = pathlib.Path(__file__).parent / "migration"
_VERSIONS_DIR = _MIGRATION_DIR / "versions"

# Revision IDs matching the migration scripts in versions/
_BASELINE_V1_REVISION = "001_baseline_v1"


class AlembicMigrationRunner:
  """Runs Alembic migrations programmatically.

  This class wraps Alembic's command API to provide a simple interface
  for checking, running, and rolling back database migrations. It uses
  synchronous SQLAlchemy engines because Alembic's migration machinery
  is synchronous.

  Attributes:
    db_url: The database URL (async drivers are converted to sync).
  """

  def __init__(
      self,
      db_url: str,
      log: Optional[logging.Logger] = None,
  ) -> None:
    self._db_url = to_sync_url(db_url)
    self._log = log or logger

  def _make_alembic_config(self) -> AlembicConfig:
    """Build a programmatic Alembic Config pointing at our migrations."""
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", str(_MIGRATION_DIR))
    cfg.set_main_option("sqlalchemy.url", self._db_url)
    # version_locations lets Alembic find our migration scripts
    cfg.set_main_option("version_locations", str(_VERSIONS_DIR))
    # Use OS path separator to avoid deprecation warning
    cfg.set_main_option("path_separator", "os")
    return cfg

  def _get_script_directory(self) -> ScriptDirectory:
    """Return the Alembic ScriptDirectory for our migrations."""
    return ScriptDirectory.from_config(self._make_alembic_config())

  def _get_head_revision(self) -> Optional[str]:
    """Return the head revision ID from the migration scripts."""
    script = self._get_script_directory()
    head = script.get_current_head()
    return head

  def get_current_revision(self) -> Optional[str]:
    """Return the current Alembic revision stamped in the database.

    Returns None if alembic_version table does not exist or is empty.
    """
    engine = create_engine(self._db_url)
    try:
      with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        return context.get_current_revision()
    finally:
      engine.dispose()

  def check_needs_migration(self) -> bool:
    """Check whether the database needs migration.

    Returns True if the current revision differs from head, or if
    the database has no alembic_version table yet.
    """
    head = self._get_head_revision()
    current = self.get_current_revision()
    self._log.debug("Migration check: current=%s, head=%s", current, head)
    return current != head

  def run_migrations(self) -> None:
    """Run all pending migrations up to head.

    Raises:
      Exception: If migration execution fails.
    """
    cfg = self._make_alembic_config()
    engine = create_engine(self._db_url)
    try:
      with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic_command.upgrade(cfg, "head")
      self._log.info("Migrations completed successfully.")
    finally:
      engine.dispose()

  def downgrade(self, revision: str = "-1") -> None:
    """Downgrade the database by the given revision spec.

    Args:
      revision: Alembic revision target. Use "-1" to roll back one
        step, or a specific revision ID.

    Raises:
      Exception: If downgrade execution fails.
    """
    cfg = self._make_alembic_config()
    engine = create_engine(self._db_url)
    try:
      with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic_command.downgrade(cfg, revision)
      self._log.info("Downgrade to '%s' completed successfully.", revision)
    finally:
      engine.dispose()

  def stamp(self, revision: str) -> None:
    """Stamp the database with a specific revision without running migrations.

    This is used to bootstrap Alembic for existing databases that
    already have the schema but no alembic_version tracking.

    Args:
      revision: The revision ID to stamp.
    """
    cfg = self._make_alembic_config()
    engine = create_engine(self._db_url)
    try:
      with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        alembic_command.stamp(cfg, revision)
      self._log.info("Database stamped with revision '%s'.", revision)
    finally:
      engine.dispose()

  def generate_revision(
      self,
      message: str,
      *,
      autogenerate: bool = True,
      output_dir: Optional[str] = None,
  ) -> str:
    """Generate a new Alembic migration script.

    Compares the current SQLAlchemy models (target metadata) against
    the live database and produces an ``upgrade()``/``downgrade()``
    migration script.

    Args:
      message: A short description used in the filename and docstring
        (e.g. ``"add_session_tags"``).
      autogenerate: If True (default), Alembic inspects the database
        and auto-generates migration operations.  If False, an empty
        migration template is created.
      output_dir: Optional directory to write the generated script
        into.  Defaults to ``sessions/migration/versions/``.

    Returns:
      The path to the generated migration script.

    Raises:
      Exception: If revision generation fails.
    """
    cfg = self._make_alembic_config()
    revision_kwargs = {
        "message": message,
        "autogenerate": autogenerate,
    }
    if output_dir is not None:
      revision_kwargs["version_path"] = output_dir
      # Register the output dir so Alembic accepts it as valid.
      sep = os.pathsep
      cfg.set_main_option(
          "version_locations",
          f"{output_dir}{sep}{_VERSIONS_DIR}",
      )
    if autogenerate:
      engine = create_engine(self._db_url)
      try:
        with engine.begin() as conn:
          cfg.attributes["connection"] = conn
          script = alembic_command.revision(cfg, **revision_kwargs)
      finally:
        engine.dispose()
    else:
      script = alembic_command.revision(cfg, **revision_kwargs)
    path = script.path
    self._log.info("Generated migration script: %s", path)
    return path

  def bootstrap_existing_database(self) -> None:
    """Bootstrap Alembic tracking for an existing database.

    Detects the current ADK schema version and handles it:

    - **V1 databases**: stamps with ``001_baseline_v1``.
    - **V0 databases**: runs an in-place V0→V1 transformation
      (pickle → JSON events, schema restructure), then stamps.

    The V0→V1 transformation is performed outside of Alembic's
    migration chain so the chain stays linear.  The existing
    copy-based ``adk migrate session`` command remains available
    as an alternative for users who prefer a two-database approach.

    Raises:
      RuntimeError: If the database has an unrecognized schema
        version.
    """
    engine = create_engine(self._db_url)
    try:
      with engine.connect() as conn:
        schema_version = get_db_schema_version_from_connection(conn)
    finally:
      engine.dispose()

    if schema_version == SCHEMA_VERSION_1_JSON:
      self._log.info("Detected V1 schema. Stamping Alembic baseline.")
      self.stamp(_BASELINE_V1_REVISION)
    elif schema_version == SCHEMA_VERSION_0_PICKLE:
      self._log.info(
          "Detected V0 (pickle) schema. Running in-place V0→V1 migration."
      )
      self._migrate_v0_to_v1_in_place()
      self.stamp(_BASELINE_V1_REVISION)
      self._log.info("V0→V1 migration complete. Alembic baseline stamped.")
    elif schema_version == LATEST_SCHEMA_VERSION:
      self._log.info("Database at latest schema version. Stamping baseline.")
      self.stamp(_BASELINE_V1_REVISION)
    else:
      raise RuntimeError(f"Unrecognized schema version: {schema_version}")

  # ------------------------------------------------------------------
  # V0 → V1 in-place migration helpers
  # ------------------------------------------------------------------

  def _migrate_v0_to_v1_in_place(self) -> None:
    """Transform a V0 database to V1 schema in-place.

    Steps:
      1. Create ``adk_internal_metadata`` table.
      2. Add ``event_data`` column to ``events``.
      3. Convert each V0 event row into a V1 JSON blob.
      4. Drop the V0-only columns.
      5. Set ``schema_version = '1'`` in metadata.
    """
    engine = create_engine(self._db_url)
    try:
      with engine.begin() as conn:
        dialect = engine.dialect.name
        inspector = sa_inspect(conn)

        # 1. Create metadata table
        if not inspector.has_table("adk_internal_metadata"):
          conn.execute(
              text(
                  "CREATE TABLE adk_internal_metadata ("
                  "  key VARCHAR(128) NOT NULL PRIMARY KEY,"
                  "  value VARCHAR(256) NOT NULL"
                  ")"
              )
          )

        # 2. Add event_data column
        event_cols = {c["name"] for c in inspector.get_columns("events")}
        if "event_data" not in event_cols:
          if dialect == "postgresql":
            conn.execute(text("ALTER TABLE events ADD COLUMN event_data JSONB"))
          elif dialect == "mysql":
            conn.execute(
                text("ALTER TABLE events ADD COLUMN event_data LONGTEXT")
            )
          else:
            conn.execute(text("ALTER TABLE events ADD COLUMN event_data TEXT"))

        # 3. Migrate each event row
        rows = conn.execute(text("SELECT * FROM events"))
        for row in rows:
          event_data = self._v0_row_to_event_data(row._mapping)
          conn.execute(
              text(
                  "UPDATE events SET event_data = :data"
                  " WHERE id = :id AND app_name = :app"
                  " AND user_id = :user AND session_id = :sid"
              ),
              {
                  "data": json.dumps(event_data),
                  "id": row._mapping["id"],
                  "app": row._mapping["app_name"],
                  "user": row._mapping["user_id"],
                  "sid": row._mapping["session_id"],
              },
          )

        # 4. Drop V0-only columns
        v0_only = [
            "author",
            "actions",
            "long_running_tool_ids_json",
            "branch",
            "content",
            "grounding_metadata",
            "custom_metadata",
            "usage_metadata",
            "citation_metadata",
            "partial",
            "turn_complete",
            "error_code",
            "error_message",
            "interrupted",
            "input_transcription",
            "output_transcription",
        ]
        if dialect == "sqlite":
          # SQLite requires table recreation for column drops.
          # Build a new table with only V1 columns.
          keep_cols = [c for c in event_cols if c not in v0_only]
          keep_cols_with_data = sorted(set(keep_cols) | {"event_data"})
          cols_str = ", ".join(keep_cols_with_data)
          conn.execute(
              text(
                  f"CREATE TABLE events_v1_tmp AS SELECT {cols_str} FROM events"
              )
          )
          conn.execute(text("DROP TABLE events"))
          conn.execute(text("ALTER TABLE events_v1_tmp RENAME TO events"))
        else:
          for col in v0_only:
            if col in event_cols:
              conn.execute(text(f"ALTER TABLE events DROP COLUMN {col}"))

        # 5. Set schema version
        conn.execute(
            text(
                "INSERT INTO adk_internal_metadata (key, value)"
                " VALUES ('schema_version', '1')"
            )
        )

        self._log.info("In-place V0→V1 migration completed successfully.")
    finally:
      engine.dispose()

  @staticmethod
  def _v0_row_to_event_data(row) -> dict:
    """Convert a V0 event row mapping to a V1 event_data dict."""
    data = {}
    data["id"] = row["id"]
    data["invocation_id"] = row.get("invocation_id") or ""
    data["author"] = row.get("author") or "agent"

    branch = row.get("branch")
    if branch:
      data["branch"] = branch

    # Unpickle actions
    actions_raw = row.get("actions")
    if actions_raw is not None:
      try:
        if isinstance(actions_raw, bytes):
          obj = pickle.loads(actions_raw)  # noqa: S301
        else:
          obj = actions_raw
        if hasattr(obj, "model_dump"):
          actions_dict = obj.model_dump(mode="json", exclude_none=True)
        elif isinstance(obj, dict):
          actions_dict = obj
        else:
          actions_dict = {}
        if actions_dict:
          data["actions"] = actions_dict
      except Exception as e:
        logger.warning("Failed to unpickle actions: %s", e)

    # Timestamp
    ts = row.get("timestamp")
    if ts is not None:
      if isinstance(ts, str):
        from datetime import datetime

        try:
          ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
          try:
            ts = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
          except ValueError:
            ts = None
      if ts is not None and hasattr(ts, "replace"):
        ts = ts.replace(tzinfo=timezone.utc)
        data["timestamp"] = ts.timestamp()

    # long_running_tool_ids
    lrt_json = row.get("long_running_tool_ids_json")
    if lrt_json:
      try:
        ids = json.loads(lrt_json)
        if ids:
          data["long_running_tool_ids"] = ids
      except json.JSONDecodeError:
        pass

    # Scalar fields
    for field in ("partial", "turn_complete", "interrupted"):
      val = row.get(field)
      if val is not None:
        data[field] = val

    for field in ("error_code", "error_message"):
      val = row.get(field)
      if val:
        data[field] = val

    # JSON fields
    for field in (
        "content",
        "grounding_metadata",
        "custom_metadata",
        "usage_metadata",
        "citation_metadata",
        "input_transcription",
        "output_transcription",
    ):
      val = row.get(field)
      if val is None:
        continue
      if isinstance(val, str):
        try:
          parsed = json.loads(val)
          if parsed:
            data[field] = parsed
        except json.JSONDecodeError:
          pass
      elif isinstance(val, dict) and val:
        data[field] = val

    return data
