# Copyright 2025 Google LLC
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
import pytest
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def create_v0_db(db_path):
  db_url = f'sqlite+aiosqlite:///{db_path}'
  engine = create_async_engine(db_url)
  async with engine.begin() as conn:
    await conn.run_sync(v0.Base.metadata.create_all)
  await engine.dispose()


@pytest.mark.asyncio
async def test_new_db_uses_latest_schema(tmp_path):
  db_path = tmp_path / 'new_db.db'
  db_url = f'sqlite+aiosqlite:///{db_path}'
  session_service = DatabaseSessionService(db_url)
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
    schema_version = await conn.run_sync(
        lambda sync_conn: sync_conn.execute(
            text('SELECT value FROM adk_internal_metadata WHERE key = :key'),
            {'key': _schema_check_utils.SCHEMA_VERSION_KEY},
        ).scalar_one_or_none()
    )
    assert schema_version == _schema_check_utils.LATEST_SCHEMA_VERSION

    # Verify events table columns for v1
    event_cols = await conn.run_sync(
        lambda sync_conn: inspect(sync_conn).get_columns('events')
    )
    event_col_names = {c['name'] for c in event_cols}
    assert 'event_data' in event_col_names
    assert 'actions' not in event_col_names
  await engine.dispose()


@pytest.mark.asyncio
async def test_existing_v0_db_uses_v0_schema(tmp_path):
  db_path = tmp_path / 'v0_db.db'
  await create_v0_db(db_path)
  db_url = f'sqlite+aiosqlite:///{db_path}'
  session_service = DatabaseSessionService(db_url)

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
  session_service1 = DatabaseSessionService(db_url)
  await session_service1.create_session(
      app_name='my_app', user_id='test_user', session_id='s1'
  )
  assert (
      session_service1._db_schema_version
      == _schema_check_utils.LATEST_SCHEMA_VERSION
  )

  # Create another session service on same db and check it detects latest schema
  session_service2 = DatabaseSessionService(db_url)
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
