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

"""Baseline V1 schema (JSON-based).

Revision ID: 001_baseline_v1
Revises: None
Create Date: 2026-02-05

Database Schema Version: v1
Compatible ADK Versions: >=1.22.0

This migration creates the V1 database schema used by
DatabaseSessionService. It is the baseline for new deployments
and for existing V1 databases being bootstrapped into Alembic.
"""

from __future__ import annotations

from alembic import op
from google.adk.sessions.schemas.shared import DynamicJSON
from google.adk.sessions.schemas.shared import PreciseTimestamp
import sqlalchemy as sa

revision = "001_baseline_v1"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
  """Create V1 schema tables."""
  op.create_table(
      "adk_internal_metadata",
      sa.Column("key", sa.String(128), nullable=False),
      sa.Column("value", sa.String(256), nullable=False),
      sa.PrimaryKeyConstraint("key"),
  )

  op.create_table(
      "sessions",
      sa.Column("app_name", sa.String(128), nullable=False),
      sa.Column("user_id", sa.String(128), nullable=False),
      sa.Column("id", sa.String(128), nullable=False),
      sa.Column("state", DynamicJSON(), nullable=True),
      sa.Column(
          "create_time",
          PreciseTimestamp(),
          nullable=False,
          server_default=sa.func.now(),
      ),
      sa.Column(
          "update_time",
          PreciseTimestamp(),
          nullable=False,
          server_default=sa.func.now(),
      ),
      sa.PrimaryKeyConstraint("app_name", "user_id", "id"),
  )

  op.create_table(
      "events",
      sa.Column("id", sa.String(128), nullable=False),
      sa.Column("app_name", sa.String(128), nullable=False),
      sa.Column("user_id", sa.String(128), nullable=False),
      sa.Column("session_id", sa.String(128), nullable=False),
      sa.Column("invocation_id", sa.String(256), nullable=False),
      sa.Column("event_data", DynamicJSON(), nullable=True),
      sa.Column(
          "timestamp",
          PreciseTimestamp(),
          nullable=False,
          server_default=sa.func.now(),
      ),
      sa.PrimaryKeyConstraint("id", "app_name", "user_id", "session_id"),
      sa.ForeignKeyConstraint(
          ["app_name", "user_id", "session_id"],
          ["sessions.app_name", "sessions.user_id", "sessions.id"],
          ondelete="CASCADE",
      ),
  )

  op.create_table(
      "app_states",
      sa.Column("app_name", sa.String(128), nullable=False),
      sa.Column("state", DynamicJSON(), nullable=True),
      sa.Column(
          "update_time",
          PreciseTimestamp(),
          nullable=False,
          server_default=sa.func.now(),
      ),
      sa.PrimaryKeyConstraint("app_name"),
  )

  op.create_table(
      "user_states",
      sa.Column("app_name", sa.String(128), nullable=False),
      sa.Column("user_id", sa.String(128), nullable=False),
      sa.Column("state", DynamicJSON(), nullable=True),
      sa.Column(
          "update_time",
          PreciseTimestamp(),
          nullable=False,
          server_default=sa.func.now(),
      ),
      sa.PrimaryKeyConstraint("app_name", "user_id"),
  )

  op.execute(
      sa.text(
          "INSERT INTO adk_internal_metadata (key, value)"
          " VALUES ('schema_version', '1')"
      )
  )


def downgrade() -> None:
  """Drop all V1 schema tables."""
  op.drop_table("user_states")
  op.drop_table("app_states")
  op.drop_table("events")
  op.drop_table("sessions")
  op.drop_table("adk_internal_metadata")
