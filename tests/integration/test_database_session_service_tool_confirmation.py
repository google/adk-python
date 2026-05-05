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

"""Integration tests for DatabaseSessionService with tool confirmation support."""

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI'], indirect=True)
async def test_database_session_service_save_retrieve_tool_confirmation():
  """Test that DatabaseSessionService correctly saves and retrieves events with requested_tool_confirmations."""
  async with DatabaseSessionService('sqlite+aiosqlite:///:memory:') as session_service:
    # Create a session
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )

    # Create an event with requested_tool_confirmations
    tool_confirmation = ToolConfirmation(
        confirmed=True, hint='Approve this', payload={'amount': 100}
    )
    event_actions = EventActions(
        requested_tool_confirmations={'func_call_456': tool_confirmation}
    )

    event = Event(
        id='event_2',
        invocation_id='inv_2',
        author='agent',
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text='Test message')],
        ),
        actions=event_actions,
    )

    # Append event to session
    await session_service.append_event(session, event)

    # Retrieve the session
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )

    # Verify the event was saved and retrieved correctly
    assert len(retrieved_session.events) == 1
    retrieved_event = retrieved_session.events[0]

    # Verify EventActions structure
    assert isinstance(retrieved_event.actions, EventActions)
    assert 'func_call_456' in retrieved_event.actions.requested_tool_confirmations

    # Verify ToolConfirmation object was correctly reconstructed
    retrieved_confirmation = (
        retrieved_event.actions.requested_tool_confirmations['func_call_456']
    )
    assert isinstance(retrieved_confirmation, ToolConfirmation)
    assert retrieved_confirmation.confirmed is True
    assert retrieved_confirmation.hint == 'Approve this'
    assert retrieved_confirmation.payload == {'amount': 100}


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI'], indirect=True)
async def test_database_session_service_multiple_tool_confirmations():
  """Test that DatabaseSessionService handles multiple tool confirmations in a single event."""
  async with DatabaseSessionService('sqlite+aiosqlite:///:memory:') as session_service:
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )

    # Create an event with multiple requested_tool_confirmations
    tool_confirmation_1 = ToolConfirmation(
        confirmed=False, hint='First action', payload={'step': 1}
    )
    tool_confirmation_2 = ToolConfirmation(
        confirmed=False, hint='Second action', payload={'step': 2}
    )
    event_actions = EventActions(
        requested_tool_confirmations={
            'func_call_1': tool_confirmation_1,
            'func_call_2': tool_confirmation_2,
        }
    )

    event = Event(
        id='event_3',
        invocation_id='inv_3',
        author='agent',
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text='Multiple confirmations')],
        ),
        actions=event_actions,
    )

    await session_service.append_event(session, event)

    # Retrieve and verify
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )
    assert len(retrieved_session.events) == 1
    retrieved_event = retrieved_session.events[0]

    assert isinstance(retrieved_event.actions, EventActions)
    assert len(retrieved_event.actions.requested_tool_confirmations) == 2

    # Verify both confirmations
    conf_1 = retrieved_event.actions.requested_tool_confirmations['func_call_1']
    assert isinstance(conf_1, ToolConfirmation)
    assert conf_1.confirmed is False
    assert conf_1.hint == 'First action'
    assert conf_1.payload == {'step': 1}

    conf_2 = retrieved_event.actions.requested_tool_confirmations['func_call_2']
    assert isinstance(conf_2, ToolConfirmation)
    assert conf_2.confirmed is False
    assert conf_2.hint == 'Second action'
    assert conf_2.payload == {'step': 2}


@pytest.mark.asyncio
@pytest.mark.parametrize('llm_backend', ['GOOGLE_AI'], indirect=True)
async def test_database_session_service_empty_tool_confirmations():
  """Test that DatabaseSessionService handles events without tool confirmations."""
  async with DatabaseSessionService('sqlite+aiosqlite:///:memory:') as session_service:
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )

    # Create an event without requested_tool_confirmations
    event = Event(
        id='event_4',
        invocation_id='inv_4',
        author='agent',
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text='No confirmations')],
        ),
        actions=EventActions(),
    )

    await session_service.append_event(session, event)

    # Retrieve and verify
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )
    assert len(retrieved_session.events) == 1
    retrieved_event = retrieved_session.events[0]

    assert isinstance(retrieved_event.actions, EventActions)
    assert not retrieved_event.actions.requested_tool_confirmations
