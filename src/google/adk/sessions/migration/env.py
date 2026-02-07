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

"""Alembic environment configuration for ADK database migrations.

This module is invoked by Alembic's migration machinery. It configures
the migration context with the target metadata (V1 schema) and handles
both offline (SQL generation) and online (live connection) modes.

The connection is always provided programmatically by
AlembicMigrationRunner via config.attributes['connection'].
"""

from __future__ import annotations

from alembic import context
from google.adk.sessions.schemas.v1 import Base as BaseV1

config = context.config

target_metadata = BaseV1.metadata


def run_migrations_offline() -> None:
  """Run migrations in 'offline' mode (SQL script generation).

  In this mode, Alembic generates SQL statements without connecting
  to a database. The URL must be set in the Alembic config.
  """
  url = config.get_main_option("sqlalchemy.url")
  context.configure(
      url=url,
      target_metadata=target_metadata,
      literal_binds=True,
      dialect_opts={"paramstyle": "named"},
  )

  with context.begin_transaction():
    context.run_migrations()


def run_migrations_online() -> None:
  """Run migrations in 'online' mode.

  Connection is provided by AlembicMigrationRunner via
  config.attributes['connection'].
  """
  connectable = config.attributes.get("connection")

  if connectable is None:
    raise RuntimeError(
        "No connection provided. Use AlembicMigrationRunner "
        "to run migrations programmatically."
    )

  context.configure(
      connection=connectable,
      target_metadata=target_metadata,
  )

  with context.begin_transaction():
    context.run_migrations()


if context.is_offline_mode():
  run_migrations_offline()
else:
  run_migrations_online()
