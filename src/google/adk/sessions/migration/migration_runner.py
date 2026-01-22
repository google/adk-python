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

"""Migration runner to upgrade schemas to the latest version."""
from __future__ import annotations

import logging
import os
import tempfile

from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.migration import migrate_from_sqlalchemy_pickle

logger = logging.getLogger("google_adk." + __name__)

# Migration map where key is start_version and value is
# (end_version, migration_function).
# Each key is a schema version, and its value is a tuple containing:
# (the schema version AFTER this migration step, the migration function to run).
# The migration function should accept (source_db_url, dest_db_url) as
# arguments.
MIGRATIONS = {
    _schema_check_utils.SCHEMA_VERSION_0_PICKLE: (
        _schema_check_utils.SCHEMA_VERSION_1_JSON,
        migrate_from_sqlalchemy_pickle.migrate,
    ),
}
# The most recent schema version. The migration process stops once this version
# is reached.
LATEST_VERSION = _schema_check_utils.LATEST_SCHEMA_VERSION


def upgrade(source_db_url: str, dest_db_url: str):
  """Migrates a database from its current version to the latest version.

  If the source database schema is older than the latest version, this
  function applies migration scripts sequentially until the schema reaches the
  LATEST_VERSION.

  If multiple migration steps are required, intermediate results are stored in
  temporary SQLite database files. This means a multistep migration
  between other database types (e.g. PostgreSQL to PostgreSQL) will use
  SQLite for intermediate steps.

  In-place migration (source_db_url == dest_db_url) is not supported,
  as migrations always read from a source and write to a destination.

  Args:
    source_db_url: The SQLAlchemy URL of the database to migrate from.
    dest_db_url: The SQLAlchemy URL of the database to migrate to. This must be
      different from source_db_url.

  Raises:
    RuntimeError: If source_db_url and dest_db_url are the same, or if no
      migration path is found.
  """
  if source_db_url == dest_db_url:
    raise RuntimeError(
        "In-place migration is not supported. "
        "Please provide a different URL for dest_db_url."
    )

  current_version = _schema_check_utils.get_db_schema_version(source_db_url)
  if current_version == LATEST_VERSION:
    logger.info(
        f"Database {source_db_url} is already at latest version"
        f" {LATEST_VERSION}. No migration needed."
    )
    return

  # Build the list of migration steps required to reach LATEST_VERSION.
  migrations_to_run = []
  ver = current_version
  while ver in MIGRATIONS and ver != LATEST_VERSION:
    migrations_to_run.append(MIGRATIONS[ver])
    ver = MIGRATIONS[ver][0]

  if not migrations_to_run:
    raise RuntimeError(
        "Could not find migration path for schema version"
        f" {current_version} to {LATEST_VERSION}."
    )

  temp_files = []
  in_url = source_db_url
  try:
    for i, (end_version, migrate_func) in enumerate(migrations_to_run):
      is_last_step = i == len(migrations_to_run) - 1

      if is_last_step:
        out_url = dest_db_url
      else:
        # For intermediate steps, create a temporary SQLite DB to store the
        # result.
        fd, temp_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        out_url = f"sqlite:///{temp_path}"
        temp_files.append(temp_path)
        logger.debug("Created temp db %s for step %d", out_url, i + 1)

      logger.info(
          f"Migrating from {in_url} to {out_url} (schema v{end_version})..."
      )
      migrate_func(in_url, out_url)
      logger.info("Finished migration step to schema %s.", end_version)
      # The output of this step becomes the input for the next step.
      in_url = out_url
  finally:
    # Ensure temporary files are cleaned up even if migration fails.
    for path in temp_files:
      try:
        os.remove(path)
        logger.debug("Removed temp db %s", path)
      except OSError as e:
        logger.warning("Failed to remove temp db file %s: %s", path, e)
