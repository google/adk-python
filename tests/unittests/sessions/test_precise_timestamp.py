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

"""Tests for PreciseTimestamp dialect-specific type resolution.

These tests verify that PreciseTimestamp maps to the correct database-specific
column types, particularly ensuring PostgreSQL uses TIMESTAMP WITH TIME ZONE
to prevent asyncpg 'offset-naive and offset-aware datetimes' errors.
"""

from google.adk.sessions.schemas.shared import PreciseTimestamp
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from sqlalchemy.types import DateTime


class TestPreciseTimestampDialectImpl:
  """Tests that PreciseTimestamp.load_dialect_impl returns the correct type."""

  def test_postgresql_returns_timestamp_with_timezone(self):
    """PostgreSQL must use TIMESTAMP WITH TIME ZONE to accept timezone-aware
    datetimes from asyncpg without raising DataError."""
    ts = PreciseTimestamp()
    dialect = postgresql.dialect()
    impl = ts.load_dialect_impl(dialect)
    assert isinstance(impl, postgresql.TIMESTAMP)
    assert impl.timezone is True

  def test_postgresql_not_timestamp_without_timezone(self):
    """Regression test: PostgreSQL must NOT use TIMESTAMP WITHOUT TIME ZONE.

    Without timezone=True, asyncpg raises:
      DataError: can't subtract offset-naive and offset-aware datetimes
    when inserting datetime objects with tzinfo=UTC.
    """
    ts = PreciseTimestamp()
    dialect = postgresql.dialect()
    impl = ts.load_dialect_impl(dialect)
    # Ensure it's not the default TIMESTAMP (which has timezone=False)
    naive_timestamp = postgresql.TIMESTAMP()
    assert naive_timestamp.timezone is False  # baseline: default is False
    assert impl.timezone is not naive_timestamp.timezone

  def test_mysql_returns_datetime_with_fsp6(self):
    """MySQL must use DATETIME(fsp=6) for microsecond precision."""
    ts = PreciseTimestamp()
    dialect = mysql.dialect()
    impl = ts.load_dialect_impl(dialect)
    assert isinstance(impl, mysql.DATETIME)
    assert impl.fsp == 6

  def test_sqlite_returns_default_datetime(self):
    """SQLite falls back to the default DateTime implementation."""
    ts = PreciseTimestamp()
    dialect = sqlite.dialect()
    impl = ts.load_dialect_impl(dialect)
    assert isinstance(impl, DateTime)
