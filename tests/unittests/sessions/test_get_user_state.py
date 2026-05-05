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

"""Tests for BaseSessionService.get_user_state across concrete implementations."""

import enum

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
import pytest

_APP = 'test-app'
_OTHER_APP = 'other-app'
_USER = 'user-42'
_OTHER_USER = 'user-99'


class SessionServiceType(enum.Enum):
  IN_MEMORY = 'IN_MEMORY'
  DATABASE = 'DATABASE'
  SQLITE = 'SQLITE'


def _make_service(
    service_type: SessionServiceType, tmp_path=None
) -> BaseSessionService:
  if service_type == SessionServiceType.DATABASE:
    return DatabaseSessionService('sqlite+aiosqlite:///:memory:')
  if service_type == SessionServiceType.SQLITE:
    return SqliteSessionService(str(tmp_path / 'sqlite.db'))
  return InMemorySessionService()


@pytest.fixture(
    params=[
        SessionServiceType.IN_MEMORY,
        SessionServiceType.DATABASE,
        SessionServiceType.SQLITE,
    ]
)
async def session_service(request, tmp_path):
  """Provides a session service and closes database backends on teardown."""
  service = _make_service(request.param, tmp_path)
  yield service
  if isinstance(service, DatabaseSessionService):
    await service.close()


async def test_get_user_state_returns_empty_dict_when_no_state_exists(
    session_service,
):
  """Returns {} when (app_name, user_id) has never had state written."""
  state = await session_service.get_user_state(app_name=_APP, user_id=_USER)
  assert state == {}


async def test_get_user_state_returns_state_written_via_append_event(
    session_service,
):
  """State written with the user: prefix is returned without the prefix."""
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(
              state_delta={'user:profile': {'name': 'Alice'}, 'session_key': 1}
          ),
      ),
  )

  state = await session_service.get_user_state(app_name=_APP, user_id=_USER)

  assert state == {'profile': {'name': 'Alice'}}
  assert 'session_key' not in state


async def test_get_user_state_is_not_visible_across_users(session_service):
  """User state is scoped to (app_name, user_id) — other users see {}."""
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:secret': 'only-for-user-42'}),
      ),
  )

  other_state = await session_service.get_user_state(
      app_name=_APP, user_id=_OTHER_USER
  )
  assert other_state == {}


async def test_get_user_state_is_not_visible_across_apps(session_service):
  """User state is scoped to (app_name, user_id) — other apps see {}."""
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:data': 'only-app-a'}),
      ),
  )

  other_state = await session_service.get_user_state(
      app_name=_OTHER_APP, user_id=_USER
  )
  assert other_state == {}


async def test_get_user_state_available_before_session_is_created(
    session_service,
):
  """Core use case: user state is readable without an active session_id."""
  first_session = await session_service.create_session(
      app_name=_APP, user_id=_USER
  )
  await session_service.append_event(
      first_session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:ctx': {'v': 1}}),
      ),
  )

  # Simulate a brand-new session_id (not yet created) — get_user_state must
  # still return the persisted user state.
  state = await session_service.get_user_state(app_name=_APP, user_id=_USER)
  assert state == {'ctx': {'v': 1}}


async def test_get_user_state_reflects_latest_write(session_service):
  """Subsequent writes overwrite earlier values under the same key."""
  session = await session_service.create_session(app_name=_APP, user_id=_USER)
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:counter': 1}),
      ),
  )
  await session_service.append_event(
      session,
      Event(
          author='system',
          actions=EventActions(state_delta={'user:counter': 2}),
      ),
  )

  state = await session_service.get_user_state(app_name=_APP, user_id=_USER)
  assert state['counter'] == 2


async def test_vertex_ai_session_service_raises_not_implemented():
  """VertexAiSessionService raises NotImplementedError for get_user_state."""
  service = VertexAiSessionService(project='proj', location='us-central1')
  with pytest.raises(NotImplementedError):
    await service.get_user_state(app_name=_APP, user_id=_USER)
