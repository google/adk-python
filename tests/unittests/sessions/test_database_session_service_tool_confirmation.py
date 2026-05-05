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

"""Tests for DatabaseSessionService with tool confirmation support."""

import copy
from unittest import mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows.functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.schemas.v0 import StorageEvent
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import FunctionCall
from google.genai.types import FunctionResponse
from google.genai.types import GenerateContentResponse
from google.genai.types import Part
import pytest

from .. import testing_utils


def get_database_urls():
  """Returns a list of database URLs to test with.

  For unit tests, we only use SQLite in-memory database to keep tests
  isolated and fast. Integration tests with real databases should be
  in a separate test file.
  """
  return [
      ('sqlite+aiosqlite:///:memory:', 'sqlite'),
  ]


@pytest.mark.asyncio
async def test_storage_event_serialize_deserialize_tool_confirmation():
  """Test that StorageEvent correctly serializes and deserializes events with requested_tool_confirmations."""
  # Create a session for testing
  session = Session(
      app_name='test_app',
      user_id='test_user',
      id='test_session',
      state={},
      events=[],
      last_update_time=0.0,
  )

  # Create an event with requested_tool_confirmations
  tool_confirmation = ToolConfirmation(
      confirmed=False, hint='Please approve this action', payload={'key': 'value'}
  )
  event_actions = EventActions(
      requested_tool_confirmations={'function_call_123': tool_confirmation}
  )

  event = Event(
      id='event_1',
      invocation_id='inv_1',
      author='agent',
      content=types.Content(
          role='model',
          parts=[types.Part.from_text(text='Test message')],
      ),
      actions=event_actions,
  )

  # Test serialization: Event -> StorageEvent
  storage_event = StorageEvent.from_event(session, event)

  # Verify that actions was serialized as a dict
  assert isinstance(storage_event.actions, dict)
  assert 'requested_tool_confirmations' in storage_event.actions
  assert 'function_call_123' in storage_event.actions['requested_tool_confirmations']

  # Verify the tool confirmation data is preserved
  stored_confirmation = storage_event.actions['requested_tool_confirmations'][
      'function_call_123'
  ]
  assert isinstance(stored_confirmation, dict)
  assert stored_confirmation['confirmed'] is False
  assert stored_confirmation['hint'] == 'Please approve this action'
  assert stored_confirmation['payload'] == {'key': 'value'}

  # Test deserialization: StorageEvent -> Event
  deserialized_event = storage_event.to_event()

  # Verify the event structure
  assert deserialized_event.id == event.id
  assert deserialized_event.invocation_id == event.invocation_id
  assert deserialized_event.author == event.author

  # Verify EventActions was correctly reconstructed
  assert isinstance(deserialized_event.actions, EventActions)
  assert 'function_call_123' in deserialized_event.actions.requested_tool_confirmations

  # Verify ToolConfirmation objects were correctly reconstructed
  deserialized_confirmation = (
      deserialized_event.actions.requested_tool_confirmations['function_call_123']
  )
  assert isinstance(deserialized_confirmation, ToolConfirmation)
  assert deserialized_confirmation.confirmed is False
  assert deserialized_confirmation.hint == 'Please approve this action'
  assert deserialized_confirmation.payload == {'key': 'value'}


@pytest.mark.asyncio
async def test_pickle_limitation_with_tool_confirmation():
  """Test that demonstrates the pickle serialization limitation with ToolConfirmation.

  This test simulates the old behavior where EventActions with ToolConfirmation
  objects were pickled directly without using model_dump(). The problem is that
  when pickling Pydantic models nested inside EventActions, the deserialized
  objects may not be properly reconstructed as Pydantic models, losing validation
  and type safety.

  The current implementation fixes this by:
  1. Using model_dump(mode='python') before pickling (converts to dict)
  2. Using EventActions.model_validate() after unpickling (reconstructs Pydantic models)
  """
  import pickle

  # Create an event with requested_tool_confirmations
  tool_confirmation = ToolConfirmation(
      confirmed=False, hint='Test hint', payload={'test': 'data'}
  )
  event_actions = EventActions(
      requested_tool_confirmations={'func_call_1': tool_confirmation}
  )

  # Simulate the OLD behavior: pickle EventActions directly (without model_dump)
  # This is what would happen in the old implementation
  pickled_data = pickle.dumps(event_actions)
  unpickled_actions = pickle.loads(pickled_data)

  # Verify that the unpickled object is still an EventActions instance
  assert isinstance(unpickled_actions, EventActions)
  assert 'func_call_1' in unpickled_actions.requested_tool_confirmations

  # THE PROBLEM: The ToolConfirmation object may not be properly reconstructed
  # In some cases, pickle might deserialize it, but it may lose Pydantic validation
  unpickled_confirmation = (
      unpickled_actions.requested_tool_confirmations['func_call_1']
  )

  # This might work in some cases, but the object may not be a proper ToolConfirmation
  # instance with full Pydantic validation
  # The issue is that pickle doesn't guarantee proper Pydantic model reconstruction
  if isinstance(unpickled_confirmation, ToolConfirmation):
    # If it works, verify the data
    assert unpickled_confirmation.confirmed is False
    assert unpickled_confirmation.hint == 'Test hint'
    assert unpickled_confirmation.payload == {'test': 'data'}
  else:
    # This demonstrates the problem: the object might not be a ToolConfirmation
    # It could be a dict or a generic object without Pydantic validation
    pytest.fail(
        'ToolConfirmation was not properly reconstructed after pickle/unpickle. '
        'This demonstrates the limitation that required the fix.'
    )

  # NOW demonstrate the CORRECT approach (current implementation):
  # 1. Convert to dict using model_dump before pickling
  actions_dict = event_actions.model_dump(mode='python')
  pickled_dict = pickle.dumps(actions_dict)
  unpickled_dict = pickle.loads(pickled_dict)

  # 2. Reconstruct using model_validate (ensures proper Pydantic model creation)
  reconstructed_actions = EventActions.model_validate(unpickled_dict)

  # Verify proper reconstruction
  assert isinstance(reconstructed_actions, EventActions)
  assert 'func_call_1' in reconstructed_actions.requested_tool_confirmations

  reconstructed_confirmation = (
      reconstructed_actions.requested_tool_confirmations['func_call_1']
  )

  # This ALWAYS works because we use model_validate
  assert isinstance(reconstructed_confirmation, ToolConfirmation)
  assert reconstructed_confirmation.confirmed is False
  assert reconstructed_confirmation.hint == 'Test hint'
  assert reconstructed_confirmation.payload == {'test': 'data'}

  # Verify Pydantic validation still works (this might fail with direct pickle)
  # Try to create a new ToolConfirmation with invalid data to test validation
  try:
    # This should work because it's a proper Pydantic model
    invalid_confirmation = ToolConfirmation(
        confirmed='not a boolean', hint='test'  # type: ignore
    )
    pytest.fail('Pydantic validation should have failed')
  except Exception:
    # Expected: Pydantic validation should catch the type error
    pass


@pytest.mark.parametrize('db_url,db_name', get_database_urls())
@pytest.mark.asyncio
async def test_database_session_service_save_retrieve_tool_confirmation(
    db_url, db_name
):
  """Test that DatabaseSessionService correctly saves and retrieves events with requested_tool_confirmations."""
  async with DatabaseSessionService(db_url) as session_service:
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
            role='model', parts=[types.Part.from_text(text='Requesting confirmation')]
        ),
        actions=event_actions,
    )

    # Append the event to the session
    appended_event = await session_service.append_event(session, event)
    assert appended_event.id == event.id

    # Retrieve the session
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )

    assert retrieved_session is not None
    assert len(retrieved_session.events) == 1

    # Verify the retrieved event
    retrieved_event = retrieved_session.events[0]
    assert retrieved_event.id == event.id
    assert isinstance(retrieved_event.actions, EventActions)

    # Verify requested_tool_confirmations was preserved
    assert 'func_call_456' in retrieved_event.actions.requested_tool_confirmations
    retrieved_confirmation = (
        retrieved_event.actions.requested_tool_confirmations['func_call_456']
    )
    assert isinstance(retrieved_confirmation, ToolConfirmation)
    assert retrieved_confirmation.confirmed is True
    assert retrieved_confirmation.hint == 'Approve this'
    assert retrieved_confirmation.payload == {'amount': 100}


@pytest.mark.parametrize('db_url,db_name', get_database_urls())
@pytest.mark.asyncio
async def test_database_session_service_multiple_tool_confirmations(
    db_url, db_name
):
  """Test that DatabaseSessionService handles multiple tool confirmations in one event."""
  async with DatabaseSessionService(db_url) as session_service:
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )

    # Create event with multiple tool confirmations
    tool_confirmation_1 = ToolConfirmation(confirmed=False, hint='First action')
    tool_confirmation_2 = ToolConfirmation(
        confirmed=True, hint='Second action', payload={'value': 42}
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
            role='model', parts=[types.Part.from_text(text='Multiple confirmations')]
        ),
        actions=event_actions,
    )

    await session_service.append_event(session, event)

    # Retrieve and verify
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )

    retrieved_event = retrieved_session.events[0]
    assert len(retrieved_event.actions.requested_tool_confirmations) == 2

    conf1 = retrieved_event.actions.requested_tool_confirmations['func_call_1']
    assert isinstance(conf1, ToolConfirmation)
    assert conf1.confirmed is False
    assert conf1.hint == 'First action'

    conf2 = retrieved_event.actions.requested_tool_confirmations['func_call_2']
    assert isinstance(conf2, ToolConfirmation)
    assert conf2.confirmed is True
    assert conf2.hint == 'Second action'
    assert conf2.payload == {'value': 42}


@pytest.mark.parametrize('db_url,db_name', get_database_urls())
@pytest.mark.asyncio
async def test_database_session_service_empty_tool_confirmations(
    db_url, db_name
):
  """Test that DatabaseSessionService handles events without tool confirmations correctly."""
  async with DatabaseSessionService(db_url) as session_service:
    session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )

    # Create event without tool confirmations
    event = Event(
        id='event_4',
        invocation_id='inv_4',
        author='user',
        content=types.Content(
            role='user', parts=[types.Part.from_text(text='Regular message')]
        ),
        actions=EventActions(),
    )

    await session_service.append_event(session, event)

    # Retrieve and verify
    retrieved_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session.id
    )

    retrieved_event = retrieved_session.events[0]
    assert isinstance(retrieved_event.actions, EventActions)
    assert len(retrieved_event.actions.requested_tool_confirmations) == 0


def _test_function(tool_context: ToolContext) -> dict[str, str]:
  """Test function that requires confirmation."""
  return {"result": f"confirmed={tool_context.tool_confirmation.confirmed}"}


def _create_llm_response_from_tools(
    tools: list[FunctionTool],
) -> GenerateContentResponse:
  """Creates a mock LLM response containing a function call."""
  parts = [
      Part(function_call=FunctionCall(name=tool.name, args={}))
      for tool in tools
  ]
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=parts)
  )


def _create_llm_response_from_text(text: str) -> GenerateContentResponse:
  """Creates a mock LLM response containing text."""
  return testing_utils.LlmResponse(
      content=testing_utils.ModelContent(parts=[Part(text=text)])
  )


HINT_TEXT = (
    "Please approve or reject the tool call _test_function() by"
    " responding with a FunctionResponse with an"
    " expected ToolConfirmation payload."
)


@pytest.mark.parametrize('db_url,db_name', get_database_urls())
@pytest.mark.asyncio
async def test_tool_confirmation_flow_with_database_session_service(
    db_url, db_name
):
  """Test the complete tool confirmation flow using DatabaseSessionService."""
  async with DatabaseSessionService(db_url) as session_service:
    # Create tools with confirmation requirement
    tools = [FunctionTool(func=_test_function, require_confirmation=True)]

    # Create mock LLM responses
    llm_responses = [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test llm response after tool call"),
    ]
    mock_model = testing_utils.MockModel(responses=llm_responses)

    # Create agent
    agent = LlmAgent(name="test_agent", model=mock_model, tools=tools)

    # Create runner with DatabaseSessionService
    runner = Runner(
        app_name='test_app',
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=session_service,
        memory_service=InMemoryMemoryService(),
    )

    # Create a session first
    test_session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )
    session_id = test_session.id

    # First invocation: user query triggers tool call that requires confirmation
    user_query = testing_utils.UserContent("test user query")
    events = []
    async for event in runner.run_async(
        user_id='test_user', session_id=session_id, new_message=user_query
    ):
      events.append(event)

    # Verify that confirmation was requested
    assert len(events) >= 3
    # Find the request confirmation event
    request_confirmation_event = None
    for event in events:
      if (
          event.content
          and event.content.parts
          and event.content.parts[0].function_call
          and event.content.parts[0].function_call.name
          == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      ):
        request_confirmation_event = event
        break

    assert request_confirmation_event is not None
    ask_for_confirmation_function_call_id = (
        request_confirmation_event.content.parts[0].function_call.id
    )
    invocation_id = request_confirmation_event.invocation_id

    # Get the session to verify events were persisted (using the same session_id)
    session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session_id
    )
    assert session is not None
    assert len(session.events) > 0

    # Verify that the request confirmation event was persisted correctly
    persisted_confirmation_event = None
    for event in session.events:
      if (
          event.content
          and event.content.parts
          and event.content.parts[0].function_call
          and event.content.parts[0].function_call.name
          == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      ):
        persisted_confirmation_event = event
        break

    assert persisted_confirmation_event is not None
    assert (
        persisted_confirmation_event.content.parts[0].function_call.id
        == ask_for_confirmation_function_call_id
    )

    # Second invocation: user provides confirmation
    user_confirmation = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=ask_for_confirmation_function_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": True},
            )
        )
    )

    # Run with the confirmation
    final_events = []
    async for event in runner.run_async(
        user_id='test_user', session_id=session_id, new_message=user_confirmation
    ):
      final_events.append(event)

    # Verify the tool was executed after confirmation
    assert len(final_events) > 0
    tool_response_found = False
    for event in final_events:
      if (
          event.content
          and event.content.parts
          and event.content.parts[0].function_response
          and event.content.parts[0].function_response.name == tools[0].name
      ):
        tool_response_found = True
        assert event.content.parts[0].function_response.response == {
            "result": "confirmed=True"
        }
        break

    assert tool_response_found, "Tool response not found in final events"

    # Verify all events were persisted
    final_session = await session_service.get_session(
        app_name='test_app', user_id='test_user', session_id=session_id
    )
    assert final_session is not None
    assert len(final_session.events) > len(session.events)


@pytest.mark.parametrize('db_url,db_name', get_database_urls())
@pytest.mark.asyncio
async def test_tool_confirmation_rejected_with_database_session_service(
    db_url, db_name
):
  """Test tool confirmation rejection flow using DatabaseSessionService."""
  async with DatabaseSessionService(db_url) as session_service:
    tools = [FunctionTool(func=_test_function, require_confirmation=True)]
    llm_responses = [
        _create_llm_response_from_tools(tools),
        _create_llm_response_from_text("test response"),
    ]
    mock_model = testing_utils.MockModel(responses=llm_responses)
    agent = LlmAgent(name="test_agent", model=mock_model, tools=tools)

    runner = Runner(
        app_name='test_app',
        agent=agent,
        artifact_service=InMemoryArtifactService(),
        session_service=session_service,
        memory_service=InMemoryMemoryService(),
    )

    # Create a session first
    test_session = await session_service.create_session(
        app_name='test_app', user_id='test_user', state={}
    )
    session_id = test_session.id

    # First invocation
    user_query = testing_utils.UserContent("test query")
    events = []
    async for event in runner.run_async(
        user_id='test_user', session_id=session_id, new_message=user_query
    ):
      events.append(event)

    # Find confirmation request
    request_confirmation_event = None
    for event in events:
      if (
          event.content
          and event.content.parts
          and event.content.parts[0].function_call
          and event.content.parts[0].function_call.name
          == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      ):
        request_confirmation_event = event
        break

    assert request_confirmation_event is not None
    confirmation_call_id = (
        request_confirmation_event.content.parts[0].function_call.id
    )

    # User rejects the confirmation
    user_rejection = testing_utils.UserContent(
        Part(
            function_response=FunctionResponse(
                id=confirmation_call_id,
                name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                response={"confirmed": False},
            )
        )
    )

    final_events = []
    async for event in runner.run_async(
        user_id='test_user', session_id=session_id, new_message=user_rejection
    ):
      final_events.append(event)

    # Verify rejection was handled
    rejection_found = False
    for event in final_events:
      if (
          event.content
          and event.content.parts
          and event.content.parts[0].function_response
          and event.content.parts[0].function_response.name == tools[0].name
      ):
        rejection_found = True
        assert event.content.parts[0].function_response.response == {
            "error": "This tool call is rejected."
        }
        break

    assert rejection_found, "Rejection response not found"

