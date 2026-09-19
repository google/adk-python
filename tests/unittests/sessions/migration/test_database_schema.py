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

from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.migration import _schema_check_utils
from google.adk.sessions.schemas import v0
from google.adk.sessions.schemas import v1
import pytest
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateIndex


async def create_v0_db(db_path):
  db_url = f'sqlite+aiosqlite:///{db_path}'
  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.run_sync(v0.Base.metadata.create_all)
  await engine.dispose()


# Use async context managers so DatabaseSessionService always closes.


@pytest.mark.asyncio
async def test_new_db_uses_latest_schema(tmp_path):
  db_path = tmp_path / 'new_db.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'
  async with DatabaseSessionService(db_url) as session_service:
    assert session_service._db_schema_version is None
    await session_service.create_session(app_name='my_app', user_id='test_user')
    assert (
        session_service._db_schema_version
        == _schema_check_utils.LATEST_SCHEMA_VERSION
    )

  # Verify metadata table
  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    has_metadata_table = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).has_table('adk_internal_metadata')
    )
    assert has_metadata_table

    def get_schema_version(sync_conn):
      inspector = inspect(sync_conn)
      key_col = inspector.dialect.identifier_preparer.quote('key')
      return sync_conn.execute(
          text(
              f'SELECT value FROM adk_internal_metadata WHERE {key_col} = :key'
          ),
          {'key': _schema_check_utils.SCHEMA_VERSION_KEY},
      ).scalar_one_or_none()

    schema_version = await conn.run_sync(get_schema_version)
    assert schema_version == _schema_check_utils.LATEST_SCHEMA_VERSION

    # Verify events table columns for v1
    event_cols = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_columns('events')
    )
    event_col_names = {c['name'] for c in event_cols}
    assert 'event_data' in event_col_names
    assert 'actions' not in event_col_names

    event_indexes = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_indexes('events')
    )
    assert any(
        index['name'] == 'idx_events_app_user_session_ts_id'
        and index['column_names']
        == ['app_name', 'user_id', 'session_id', 'timestamp', 'id']
        for index in event_indexes
    )
    xinfo = await conn.run_sync(
        lambda sync_conn: sync_conn.execute(
            text("PRAGMA index_xinfo('idx_events_app_user_session_ts_id')")
        ).fetchall()
    )
    col_desc = {row[2]: row[3] for row in xinfo if row[2]}
    assert col_desc['timestamp'] == 1
    assert col_desc['id'] == 1
  await engine.dispose()


@pytest.mark.asyncio
async def test_existing_v0_db_uses_v0_schema(tmp_path):
  db_path = tmp_path / 'v0_db.db'
  await create_v0_db(db_path)
  db_url = f'sqlite+aiosqlite:///{db_path}'
  async with DatabaseSessionService(db_url) as session_service:
    assert session_service._db_schema_version is None
    await session_service.create_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    assert (
        session_service._db_schema_version
        == _schema_check_utils.SCHEMA_VERSION_0_PICKLE
    )

    session = await session_service.get_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    assert session.id == 's1'

  # Verify schema tables
  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    has_metadata_table = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).has_table('adk_internal_metadata')
    )
    assert not has_metadata_table

    # Verify events table columns for v0
    event_cols = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_columns('events')
    )
    event_col_names = {c['name'] for c in event_cols}
    assert 'event_data' not in event_col_names
    assert 'actions' in event_col_names
  await engine.dispose()


@pytest.mark.asyncio
async def test_existing_latest_db_uses_latest_schema(tmp_path):
  db_path = tmp_path / 'new_db.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'

  # Create session service which creates db with latest schema
  async with DatabaseSessionService(db_url) as session_service1:
    await session_service1.create_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    assert (
        session_service1._db_schema_version
        == _schema_check_utils.LATEST_SCHEMA_VERSION
    )

    # Create another session service on same db and check it detects latest schema
    async with DatabaseSessionService(db_url) as session_service2:
      await session_service2.create_session(
          app_name='my_app', user_id='test_user2', session_id='s2'
      )
      assert (
          session_service2._db_schema_version
          == _schema_check_utils.LATEST_SCHEMA_VERSION
      )
      s2 = await session_service2.get_session(
          app_name='my_app', user_id='test_user2', session_id='s2'
      )
      assert s2.id == 's2'

      s1 = await session_service2.get_session(
          app_name='my_app', user_id='test_user', session_id='s1'
      )
      assert s1.id == 's1'

      list_sessions_response = await session_service2.list_sessions(
          app_name='my_app'
      )
      assert len(list_sessions_response.sessions) == 2

  # Verify schema tables
  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    has_metadata_table = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).has_table('adk_internal_metadata')
    )
    assert has_metadata_table

    # Verify events table columns for v1
    event_cols = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_columns('events')
    )
    event_col_names = {c['name'] for c in event_cols}
    assert 'event_data' in event_col_names
    assert 'actions' not in event_col_names
  await engine.dispose()


@pytest.mark.asyncio
async def test_prepare_tables_recreates_missing_latest_events_index(tmp_path):
  db_path = tmp_path / 'missing_latest_index.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'

  async with DatabaseSessionService(db_url) as session_service:
    await session_service.create_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )

  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.execute(text('DROP INDEX idx_events_app_user_session_ts_id'))
  await engine.dispose()

  async with DatabaseSessionService(db_url) as session_service:
    session = await session_service.get_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    assert session.id == 's1'

  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    event_indexes = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_indexes('events')
    )
  await engine.dispose()

  assert any(
      index['name'] == 'idx_events_app_user_session_ts_id'
      and index['column_names']
      == ['app_name', 'user_id', 'session_id', 'timestamp', 'id']
      for index in event_indexes
  )


@pytest.mark.asyncio
async def test_prepare_tables_recreates_missing_v0_events_index(tmp_path):
  db_path = tmp_path / 'missing_v0_index.db'
  await create_v0_db(db_path)
  db_url = f'sqlite+aiosqlite:///{db_path}'

  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.execute(text('DROP INDEX idx_events_app_user_session_ts_id'))
  await engine.dispose()

  async with DatabaseSessionService(db_url) as session_service:
    await session_service.create_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    session = await session_service.get_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )
    assert session.id == 's1'

  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    event_indexes = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_indexes('events')
    )
  await engine.dispose()

  assert any(
      index['name'] == 'idx_events_app_user_session_ts_id'
      and index['column_names']
      == ['app_name', 'user_id', 'session_id', 'timestamp', 'id']
      for index in event_indexes
  )


@pytest.mark.asyncio
async def test_prepare_tables_adds_new_events_index_to_existing_db(tmp_path):
  db_path = tmp_path / 'legacy_index.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'

  # First create the db properly with DatabaseSessionService
  async with DatabaseSessionService(db_url) as session_service:
    await session_service.create_session(
        app_name='my_app', user_id='test_user', session_id='s1'
    )

  # Simulate an existing database that had the old 4-column index
  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.execute(text('DROP INDEX idx_events_app_user_session_ts_id'))
    await conn.execute(
        text(
            'CREATE INDEX idx_events_app_user_session_ts ON events ('
            'app_name, user_id, session_id, timestamp DESC)'
        )
    )
  await engine.dispose()

  # Connecting with DatabaseSessionService should create the new composite index
  async with DatabaseSessionService(db_url) as session_service:
    await session_service.create_session(
        app_name='my_app', user_id='test_user', session_id='s2'
    )

  engine = create_async_engine(db_url)
  async with engine.connect() as conn:
    event_indexes = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_indexes('events')
    )
  await engine.dispose()

  index_names = {idx['name'] for idx in event_indexes}
  assert 'idx_events_app_user_session_ts' not in index_names
  assert 'idx_events_app_user_session_ts_id' in index_names
  composite_idx = next(
      idx
      for idx in event_indexes
      if idx['name'] == 'idx_events_app_user_session_ts_id'
  )
  assert composite_idx['column_names'] == [
      'app_name',
      'user_id',
      'session_id',
      'timestamp',
      'id',
  ]

  # Verify column sort directions in SQLite: 1 indicates DESC, 0 indicates ASC
  async with engine.connect() as conn:
    xinfo = await conn.run_sync(
        lambda sync_conn: sync_conn.execute(
            text("PRAGMA index_xinfo('idx_events_app_user_session_ts_id')")
        ).fetchall()
    )
  col_desc = {row[2]: row[3] for row in xinfo if row[2]}
  assert col_desc['timestamp'] == 1
  assert col_desc['id'] == 1


@pytest.mark.parametrize('dialect_name', ['mysql', 'postgresql', 'sqlite'])
def test_storage_event_composite_index_preserves_desc_across_dialects(
    dialect_name,
):
  """Ensures composite index compiles both timestamp and id as DESC across dialects."""
  dialects = {
      'mysql': mysql.dialect(),
      'postgresql': postgresql.dialect(),
      'sqlite': sqlite.dialect(),
  }
  dialect = dialects[dialect_name]
  for schema_module in (v0, v1):
    idx = next(
        i
        for i in schema_module.StorageEvent.__table__.indexes
        if i.name == 'idx_events_app_user_session_ts_id'
    )
    ddl = str(CreateIndex(idx).compile(dialect=dialect))
    assert 'timestamp DESC' in ddl
    assert 'id DESC' in ddl


def _run_sqlite_ddl(db_path, statements):
  """Creates a local SQLite file and applies the given DDL statements."""
  engine = create_engine(f'sqlite:///{db_path}')
  try:
    with engine.begin() as conn:
      for statement in statements:
        conn.execute(text(statement))
  finally:
    engine.dispose()


_V0_EVENTS_TABLE_DDL = (
    'CREATE TABLE events (id VARCHAR(128) PRIMARY KEY, actions BLOB)'
)
_V1_EVENTS_TABLE_DDL = (
    'CREATE TABLE events (id VARCHAR(128) PRIMARY KEY, event_data TEXT)'
)
_METADATA_TABLE_DDL = (
    'CREATE TABLE adk_internal_metadata ("key" VARCHAR(128) PRIMARY KEY,'
    ' value VARCHAR(128))'
)


def test_get_db_schema_version_empty_db_defaults_to_latest(tmp_path):
  """A database with neither marker is treated as brand new."""
  db_path = tmp_path / 'empty.db'
  _run_sqlite_ddl(db_path, ['CREATE TABLE unrelated (id INTEGER PRIMARY KEY)'])

  assert (
      _schema_check_utils.get_db_schema_version(f'sqlite:///{db_path}')
      == _schema_check_utils.LATEST_SCHEMA_VERSION
  )


def test_get_db_schema_version_legacy_events_table_detects_v0(tmp_path):
  """An events table with `actions` and no `event_data` is the pickle schema."""
  db_path = tmp_path / 'legacy.db'
  _run_sqlite_ddl(db_path, [_V0_EVENTS_TABLE_DDL])

  assert (
      _schema_check_utils.get_db_schema_version(f'sqlite:///{db_path}')
      == _schema_check_utils.SCHEMA_VERSION_0_PICKLE
  )


@pytest.mark.parametrize(
    'events_ddl',
    [
        _V1_EVENTS_TABLE_DDL,
        # A table carrying both columns still has the JSON column, so it is
        # not the pickle-only schema.
        (
            'CREATE TABLE events (id VARCHAR(128) PRIMARY KEY, actions BLOB,'
            ' event_data TEXT)'
        ),
    ],
)
def test_get_db_schema_version_events_table_with_event_data_is_not_v0(
    tmp_path, events_ddl
):
  """Only the `actions`-without-`event_data` shape counts as the v0 schema."""
  db_path = tmp_path / 'json_events.db'
  _run_sqlite_ddl(db_path, [events_ddl])

  assert (
      _schema_check_utils.get_db_schema_version(f'sqlite:///{db_path}')
      == _schema_check_utils.LATEST_SCHEMA_VERSION
  )


def test_get_db_schema_version_metadata_row_wins_over_table_shape(tmp_path):
  """The recorded version is authoritative even when the tables disagree."""
  db_path = tmp_path / 'metadata_wins.db'
  # v1-shaped events table, but the metadata table still records v0.
  _run_sqlite_ddl(
      db_path,
      [
          _V1_EVENTS_TABLE_DDL,
          _METADATA_TABLE_DDL,
          'INSERT INTO adk_internal_metadata ("key", value) VALUES'
          f" ('{_schema_check_utils.SCHEMA_VERSION_KEY}',"
          f" '{_schema_check_utils.SCHEMA_VERSION_0_PICKLE}')",
      ],
  )

  assert (
      _schema_check_utils.get_db_schema_version(f'sqlite:///{db_path}')
      == _schema_check_utils.SCHEMA_VERSION_0_PICKLE
  )


def test_get_db_schema_version_metadata_without_version_row_raises(tmp_path):
  """A metadata table missing the version row means a malformed database."""
  db_path = tmp_path / 'malformed.db'
  _run_sqlite_ddl(db_path, [_V0_EVENTS_TABLE_DDL, _METADATA_TABLE_DDL])

  with pytest.raises(ValueError, match='Schema version not found'):
    _schema_check_utils.get_db_schema_version(f'sqlite:///{db_path}')


def test_get_db_schema_version_accepts_async_driver_url(tmp_path):
  """An async driver URL is downgraded to its sync form before connecting."""
  db_path = tmp_path / 'async_url.db'
  _run_sqlite_ddl(db_path, [_V0_EVENTS_TABLE_DDL])

  assert (
      _schema_check_utils.get_db_schema_version(
          f'sqlite+aiosqlite:///{db_path}'
      )
      == _schema_check_utils.SCHEMA_VERSION_0_PICKLE
  )


def test_get_db_schema_version_from_connection_uses_open_connection(tmp_path):
  """The connection variant reports the same version without a new engine."""
  db_path = tmp_path / 'from_connection.db'
  _run_sqlite_ddl(db_path, [_V0_EVENTS_TABLE_DDL])

  engine = create_engine(f'sqlite:///{db_path}')
  try:
    with engine.connect() as connection:
      version = _schema_check_utils.get_db_schema_version_from_connection(
          connection
      )
  finally:
    engine.dispose()

  assert version == _schema_check_utils.SCHEMA_VERSION_0_PICKLE
