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

"""Unit tests for AgentCoreSessionService."""

from __future__ import annotations

from google.adk.errors.already_exists_error import AlreadyExistsError
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.integrations.agentcore._agentcore_session_service import AgentCoreSessionService
from google.adk.integrations.agentcore._config import AgentCoreSessionServiceConfig
from google.adk.sessions.base_session_service import GetSessionConfig
from google.genai import types
import pytest

from ._fake_agentcore import FakeAgentCoreClient


@pytest.fixture
def fake_client():
  return FakeAgentCoreClient()


@pytest.fixture
def session_service(fake_client):
  return AgentCoreSessionService(
      config=AgentCoreSessionServiceConfig(memory_id='mem-1'),
      client=fake_client,
  )


def test_memory_id_is_required():
  with pytest.raises(ValueError, match='memory_id is required'):
    AgentCoreSessionService()


def _user_event(text: str, **kwargs) -> Event:
  return Event(
      author='user',
      content=types.Content(role='user', parts=[types.Part(text=text)]),
      **kwargs,
  )


@pytest.mark.asyncio
async def test_create_session(session_service):
  session = await session_service.create_session(
      app_name='app1',
      user_id='user1',
      state={'key1': 'val1', 'temp:scratch': 'nope'},
  )

  assert session.app_name == 'app1'
  assert session.user_id == 'user1'
  assert session.state['key1'] == 'val1'
  assert session.state['temp:scratch'] == 'nope'
  assert session.id is not None

  fetched = await session_service.get_session(
      app_name='app1', user_id='user1', session_id=session.id
  )
  assert fetched is not None
  assert fetched.state['key1'] == 'val1'
  assert 'temp:scratch' not in fetched.state


@pytest.mark.asyncio
async def test_create_session_already_exists(session_service):
  await session_service.create_session(
      app_name='app1', user_id='user1', session_id='sess_123'
  )

  with pytest.raises(AlreadyExistsError):
    await session_service.create_session(
        app_name='app1', user_id='user1', session_id='sess_123'
    )


@pytest.mark.asyncio
async def test_get_session_not_found(session_service):
  fetched = await session_service.get_session(
      app_name='app1', user_id='user1', session_id='missing'
  )
  assert fetched is None


@pytest.mark.asyncio
async def test_append_event_round_trips_blob_and_text(
    session_service, fake_client
):
  session = await session_service.create_session(
      app_name='app1', user_id='user1', session_id='s1'
  )
  event = _user_event('hello there')
  await session_service.append_event(session, event)

  fetched = await session_service.get_session(
      app_name='app1', user_id='user1', session_id='s1'
  )
  assert fetched is not None
  assert len(fetched.events) == 1
  assert fetched.events[0].author == 'user'
  assert fetched.events[0].content.parts[0].text == 'hello there'

  stored = fake_client._events[('app1:user1', 's1')]
  payloads = stored[-1]['payload']
  assert payloads[0]['conversational']['role'] == 'USER'
  assert payloads[0]['conversational']['content']['text'] == 'hello there'
  assert 'blob' in payloads[1]

  assistant = Event(
      author='simple_agent',
      content=types.Content(
          role='model', parts=[types.Part(text='hi from the model')]
      ),
  )
  await session_service.append_event(fetched, assistant)
  assistant_payload = fake_client._events[('app1:user1', 's1')][-1]['payload']
  assert assistant_payload[0]['conversational']['role'] == 'ASSISTANT'


@pytest.mark.asyncio
async def test_append_event_tool_role(session_service, fake_client):
  session = await session_service.create_session(
      app_name='app1', user_id='user1'
  )
  event = Event(
      author='tool',
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name='lookup', response={'ok': True}
                  )
              )
          ],
      ),
  )
  # No text → conversational payload is omitted; role helper still maps TOOL.
  await session_service.append_event(session, event)
  assert session_service._conversational_role(event) == 'TOOL'


@pytest.mark.asyncio
async def test_partial_events_are_not_written(session_service, fake_client):
  session = await session_service.create_session(
      app_name='app1', user_id='user1', session_id='s1'
  )
  await session_service.append_event(
      session, Event(author='user', partial=True)
  )

  fetched = await session_service.get_session(
      app_name='app1', user_id='user1', session_id='s1'
  )
  assert fetched is not None
  assert fetched.events == []
  # Bootstrap only.
  assert len(fake_client._events[('app1:user1', 's1')]) == 1


@pytest.mark.asyncio
async def test_get_session_with_event_filter(session_service):
  session = await session_service.create_session(
      app_name='app1', user_id='user1'
  )
  for i in range(5):
    await session_service.append_event(
        session, Event(author=f'user_{i}', timestamp=float(100 + i))
    )

  fetched = await session_service.get_session(
      app_name='app1',
      user_id='user1',
      session_id=session.id,
      config=GetSessionConfig(num_recent_events=2),
  )
  assert fetched is not None
  assert len(fetched.events) == 2
  assert fetched.events[-1].author == 'user_4'

  fetched_zero = await session_service.get_session(
      app_name='app1',
      user_id='user1',
      session_id=session.id,
      config=GetSessionConfig(num_recent_events=0),
  )
  assert fetched_zero is not None
  assert fetched_zero.events == []

  fetched_after = await session_service.get_session(
      app_name='app1',
      user_id='user1',
      session_id=session.id,
      config=GetSessionConfig(after_timestamp=103.0),
  )
  assert fetched_after is not None
  assert [e.author for e in fetched_after.events] == ['user_3', 'user_4']


@pytest.mark.asyncio
async def test_append_event_and_state_delta(session_service):
  session = await session_service.create_session(app_name='app1', user_id='u1')
  event = Event(
      author='agent',
      actions=EventActions(
          state_delta={
              'count': 1,
              'user:score': 100,
              'app:status': 'active',
              'temp:scratch': 'gone',
          }
      ),
  )
  await session_service.append_event(session, event)

  fetched = await session_service.get_session(
      app_name='app1', user_id='u1', session_id=session.id
  )
  assert fetched is not None
  assert len(fetched.events) == 1
  assert fetched.state['count'] == 1
  assert fetched.state['user:score'] == 100
  assert fetched.state['app:status'] == 'active'
  assert 'temp:scratch' not in fetched.state


@pytest.mark.asyncio
async def test_list_sessions(session_service):
  await session_service.create_session(
      app_name='app1', user_id='u1', session_id='s1'
  )
  await session_service.create_session(
      app_name='app1', user_id='u1', session_id='s2'
  )
  await session_service.create_session(
      app_name='app1', user_id='u2', session_id='s3'
  )
  await session_service.create_session(
      app_name='app2', user_id='u1', session_id='s1'
  )

  resp_u1 = await session_service.list_sessions(app_name='app1', user_id='u1')
  assert {s.id for s in resp_u1.sessions} == {'s1', 's2'}
  assert all(s.events == [] for s in resp_u1.sessions)

  resp_all = await session_service.list_sessions(app_name='app1')
  assert {s.id for s in resp_all.sessions} == {'s1', 's2', 's3'}
  assert {s.user_id for s in resp_all.sessions} == {'u1', 'u2'}


@pytest.mark.asyncio
async def test_delete_session(session_service):
  await session_service.create_session(
      app_name='app1', user_id='u1', session_id='to_delete'
  )
  session = await session_service.get_session(
      app_name='app1', user_id='u1', session_id='to_delete'
  )
  assert session is not None
  await session_service.append_event(session, _user_event('bye'))

  await session_service.delete_session(
      app_name='app1', user_id='u1', session_id='to_delete'
  )
  fetched = await session_service.get_session(
      app_name='app1', user_id='u1', session_id='to_delete'
  )
  assert fetched is None


@pytest.mark.asyncio
async def test_apps_do_not_share_actor_namespace(session_service):
  await session_service.create_session(
      app_name='app1', user_id='u1', session_id='shared-id'
  )
  other = await session_service.get_session(
      app_name='app2', user_id='u1', session_id='shared-id'
  )
  assert other is None
