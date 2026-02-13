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

from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.base_agent import BaseAgentState
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
from google.genai.types import Content
from google.genai.types import FunctionCall
from google.genai.types import Part
import pytest

from .. import testing_utils


class TestInvocationContext:
  """Test suite for InvocationContext."""

  @pytest.fixture
  def mock_events(self):
    """Create mock events for testing."""
    event1 = Mock(spec=Event)
    event1.invocation_id = 'inv_1'
    event1.branch = 'agent_1'

    event2 = Mock(spec=Event)
    event2.invocation_id = 'inv_1'
    event2.branch = 'agent_2'

    event3 = Mock(spec=Event)
    event3.invocation_id = 'inv_2'
    event3.branch = 'agent_1'

    event4 = Mock(spec=Event)
    event4.invocation_id = 'inv_2'
    event4.branch = 'agent_2'

    return [event1, event2, event3, event4]

  @pytest.fixture
  def mock_invocation_context(self, mock_events):
    """Create a mock invocation context for testing."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        branch='agent_1',
        session=Mock(spec=Session, events=mock_events),
    )
    return ctx

  def test_get_events_returns_all_events_by_default(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events returns all events when no filters are applied."""
    events = mock_invocation_context._get_events()
    assert events == mock_events

  def test_get_events_filters_by_current_invocation(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current invocation."""
    event1, event2, _, _ = mock_events
    events = mock_invocation_context._get_events(current_invocation=True)
    assert events == [event1, event2]

  def test_get_events_filters_by_current_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current branch."""
    event1, _, event3, _ = mock_events
    events = mock_invocation_context._get_events(current_branch=True)
    assert events == [event1, event3]

  def test_get_events_filters_by_invocation_and_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events filters by invocation and branch."""
    event1, _, _, _ = mock_events
    events = mock_invocation_context._get_events(
        current_invocation=True,
        current_branch=True,
    )
    assert events == [event1]

  def test_get_events_with_no_events_in_session(self, mock_invocation_context):
    """Tests get_events when the session has no events."""
    mock_invocation_context.session.events = []
    events = mock_invocation_context._get_events()
    assert not events

  def test_get_events_with_no_matching_events(self, mock_invocation_context):
    """Tests get_events when no events match the filters."""
    mock_invocation_context.invocation_id = 'inv_3'
    mock_invocation_context.branch = 'branch_C'

    # Filter by invocation
    events = mock_invocation_context._get_events(current_invocation=True)
    assert not events

    # Filter by branch
    events = mock_invocation_context._get_events(current_branch=True)
    assert not events

    # Filter by both
    events = mock_invocation_context._get_events(
        current_invocation=True,
        current_branch=True,
    )
    assert not events


class TestInvocationContextWithAppResumablity:
  """Test suite for InvocationContext regarding app resumability."""

  @pytest.fixture
  def long_running_function_call(self) -> FunctionCall:
    """A long running function call."""
    return FunctionCall(
        id='tool_call_id_1',
        name='long_running_function_call',
        args={},
    )

  @pytest.fixture
  def event_to_pause(self, long_running_function_call) -> Event:
    """An event with a long running function call."""
    return Event(
        invocation_id='inv_1',
        author='agent',
        content=testing_utils.ModelContent(
            [Part(function_call=long_running_function_call)]
        ),
        long_running_tool_ids=[long_running_function_call.id],
    )

  def _create_test_invocation_context(
      self, resumability_config
  ) -> InvocationContext:
    """Create a mock invocation context for testing."""
    ctx = InvocationContext(
        session_service=Mock(spec=BaseSessionService),
        agent=Mock(spec=BaseAgent),
        invocation_id='inv_1',
        session=Mock(spec=Session),
        resumability_config=resumability_config,
    )
    return ctx

  def test_should_pause_invocation_with_resumable_app(self, event_to_pause):
    """Tests should_pause_invocation with a resumable app."""
    mock_invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )

    assert mock_invocation_context.should_pause_invocation(event_to_pause)

  def test_should_not_pause_invocation_with_non_resumable_app(
      self, event_to_pause
  ):
    """Tests should_pause_invocation with a non-resumable app."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )

    assert not invocation_context.should_pause_invocation(event_to_pause)

  def test_should_not_pause_invocation_with_no_long_running_tool_ids(
      self, event_to_pause
  ):
    """Tests should_pause_invocation with no long running tools."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    nonpausable_event = event_to_pause.model_copy(
        update={'long_running_tool_ids': []}
    )

    assert not invocation_context.should_pause_invocation(nonpausable_event)

  def test_should_not_pause_invocation_with_no_function_calls(
      self, event_to_pause
  ):
    """Tests should_pause_invocation with a non-model event."""
    mock_invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    nonpausable_event = event_to_pause.model_copy(
        update={'content': testing_utils.UserContent('test text part')}
    )

    assert not mock_invocation_context.should_pause_invocation(
        nonpausable_event
    )

  def test_is_resumable_true(self):
    """Tests that is_resumable is True when resumability is enabled."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    assert invocation_context.is_resumable

  def test_is_resumable_false(self):
    """Tests that is_resumable is False when resumability is disabled."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )
    assert not invocation_context.is_resumable

  def test_is_resumable_no_config(self):
    """Tests that is_resumable is False when no resumability config is set."""
    invocation_context = self._create_test_invocation_context(None)
    assert not invocation_context.is_resumable

  def test_populate_invocation_agent_states_not_resumable(self):
    """Tests that populate_invocation_agent_states does nothing if not resumable."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=False)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=True, agent_state=None),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_populate_invocation_agent_states_end_of_agent(self):
    """Tests that populate_invocation_agent_states handles end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=True, agent_state=None),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert invocation_context.end_of_agents == {'agent1': True}

  def test_populate_invocation_agent_states_with_agent_state(self):
    """Tests that populate_invocation_agent_states handles agent_state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(
            end_of_agent=False,
            agent_state=BaseAgentState().model_dump(mode='json'),
        ),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert invocation_context.agent_states == {'agent1': {}}
    assert invocation_context.end_of_agents == {'agent1': False}

  def test_populate_invocation_agent_states_with_agent_state_and_end_of_agent(
      self,
  ):
    """Tests that populate_invocation_agent_states handles agent_state and end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(
            end_of_agent=True,
            agent_state=BaseAgentState().model_dump(mode='json'),
        ),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    # When both agent_state and end_of_agent are set, agent_state should be
    # cleared, as end_of_agent is of a higher priority.
    assert not invocation_context.agent_states
    assert invocation_context.end_of_agents == {'agent1': True}

  def test_populate_invocation_agent_states_with_content_no_state(self):
    """Tests that populate_invocation_agent_states creates default state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=False, agent_state=None),
        content=Content(role='model', parts=[Part(text='hi')]),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert invocation_context.agent_states == {'agent1': BaseAgentState()}
    assert invocation_context.end_of_agents == {'agent1': False}

  def test_populate_invocation_agent_states_user_message_event(self):
    """Tests that populate_invocation_agent_states ignores user message events for default state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='user',
        actions=EventActions(end_of_agent=False, agent_state=None),
        content=Content(role='user', parts=[Part(text='hi')]),
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_populate_invocation_agent_states_no_content(self):
    """Tests that populate_invocation_agent_states ignores events with no content if no state."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    event = Event(
        invocation_id='inv_1',
        author='agent1',
        actions=EventActions(end_of_agent=None, agent_state=None),
        content=None,
    )
    invocation_context.session.events = [event]
    invocation_context.populate_invocation_agent_states()
    assert not invocation_context.agent_states
    assert not invocation_context.end_of_agents

  def test_set_agent_state_with_end_of_agent_true(self):
    """Tests that set_agent_state clears agent_state and sets end_of_agent to True."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent_states['agent1'] = {}
    invocation_context.end_of_agents['agent1'] = False

    # Set state with end_of_agent=True, which should clear the existing
    # agent_state.
    invocation_context.set_agent_state('agent1', end_of_agent=True)
    assert 'agent1' not in invocation_context.agent_states
    assert invocation_context.end_of_agents['agent1']

  def test_set_agent_state_with_agent_state(self):
    """Tests that set_agent_state sets agent_state and sets end_of_agent to False."""
    agent_state = BaseAgentState()
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.end_of_agents['agent1'] = True

    # Set state with agent_state=agent_state, which should set the agent_state
    # and reset the end_of_agent flag to False.
    invocation_context.set_agent_state('agent1', agent_state=agent_state)
    assert invocation_context.agent_states['agent1'] == agent_state.model_dump(
        mode='json'
    )
    assert invocation_context.end_of_agents['agent1'] is False

  def test_reset_agent_state(self):
    """Tests that set_agent_state clears agent_state and end_of_agent."""
    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent_states['agent1'] = {}
    invocation_context.end_of_agents['agent1'] = True

    # Reset state, which should clear the agent_state and end_of_agent flag.
    invocation_context.set_agent_state('agent1')
    assert 'agent1' not in invocation_context.agent_states
    assert 'agent1' not in invocation_context.end_of_agents

  def test_reset_sub_agent_states(self):
    """Tests that reset_sub_agent_states resets sub-agent states."""
    sub_sub_agent_1 = BaseAgent(name='sub_sub_agent_1')
    sub_agent_1 = BaseAgent(name='sub_agent_1', sub_agents=[sub_sub_agent_1])
    sub_agent_2 = BaseAgent(name='sub_agent_2')
    root_agent = BaseAgent(
        name='root_agent', sub_agents=[sub_agent_1, sub_agent_2]
    )

    invocation_context = self._create_test_invocation_context(
        ResumabilityConfig(is_resumable=True)
    )
    invocation_context.agent = root_agent
    invocation_context.set_agent_state(
        'sub_agent_1', agent_state=BaseAgentState()
    )
    invocation_context.set_agent_state('sub_agent_2', end_of_agent=True)
    invocation_context.set_agent_state(
        'sub_sub_agent_1', agent_state=BaseAgentState()
    )

    assert 'sub_agent_1' in invocation_context.agent_states
    assert 'sub_agent_2' in invocation_context.end_of_agents
    assert 'sub_sub_agent_1' in invocation_context.agent_states

    invocation_context.reset_sub_agent_states('root_agent')

    assert 'sub_agent_1' not in invocation_context.agent_states
    assert 'sub_agent_1' not in invocation_context.end_of_agents
    assert 'sub_agent_2' not in invocation_context.agent_states
    assert 'sub_agent_2' not in invocation_context.end_of_agents
    assert 'sub_sub_agent_1' not in invocation_context.agent_states
    assert 'sub_sub_agent_1' not in invocation_context.end_of_agents


class TestFindMatchingFunctionCall:
  """Test suite for find_matching_function_call."""

  @pytest.fixture
  def test_invocation_context(self):
    """Create a mock invocation context for testing."""

    def _create_invocation_context(events):
      return InvocationContext(
          session_service=Mock(spec=BaseSessionService),
          agent=Mock(spec=BaseAgent, name='agent'),
          invocation_id='inv_1',
          session=Mock(spec=Session, events=events),
      )

    return _create_invocation_context

  def test_find_matching_function_call_found(self, test_invocation_context):
    """Tests that a matching function call is found."""
    fc = Part.from_function_call(name='some_tool', args={})
    fc.function_call.id = 'test_function_call_id'
    fc_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=testing_utils.ModelContent([fc]),
    )
    fr = Part.from_function_response(
        name='some_tool', response={'result': 'ok'}
    )
    fr.function_response.id = 'test_function_call_id'
    fr_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=Content(role='user', parts=[fr]),
    )
    invocation_context = test_invocation_context([fc_event, fr_event])
    matching_fc_event = invocation_context._find_matching_function_call(
        fr_event
    )
    assert testing_utils.simplify_content(
        matching_fc_event.content
    ) == testing_utils.simplify_content(fc_event.content)

  def test_find_matching_function_call_not_found(self, test_invocation_context):
    """Tests that no matching function call is returned if id doesn't match."""
    fc = Part.from_function_call(name='some_tool', args={})
    fc.function_call.id = 'another_function_call_id'
    fc_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=testing_utils.ModelContent([fc]),
    )
    fr = Part.from_function_response(
        name='some_tool', response={'result': 'ok'}
    )
    fr.function_response.id = 'test_function_call_id'
    fr_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=Content(role='user', parts=[fr]),
    )
    invocation_context = test_invocation_context([fc_event, fr_event])
    match = invocation_context._find_matching_function_call(fr_event)
    assert match is None

  def test_find_matching_function_call_no_call_events(
      self, test_invocation_context
  ):
    """Tests that no matching function call is returned if there are no call events."""
    fr = Part.from_function_response(
        name='some_tool', response={'result': 'ok'}
    )
    fr.function_response.id = 'test_function_call_id'
    fr_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=Content(role='user', parts=[fr]),
    )
    invocation_context = test_invocation_context([fr_event])
    match = invocation_context._find_matching_function_call(fr_event)
    assert match is None

  def test_find_matching_function_call_no_response_in_event(
      self, test_invocation_context
  ):
    """Tests result is None if function_response_event has no function response."""
    fr_event_no_fr = Event(
        author='agent',
        content=Content(role='user', parts=[Part(text='user message')]),
    )
    fc = Part.from_function_call(name='some_tool', args={})
    fc.function_call.id = 'test_function_call_id'
    fc_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=testing_utils.ModelContent([fc]),
    )
    fr = Part.from_function_response(
        name='some_tool', response={'result': 'ok'}
    )
    fr.function_response.id = 'test_function_call_id'
    fr_event = Event(
        invocation_id='inv_1',
        author='agent',
        content=Content(role='user', parts=[Part(text='user message')]),
    )
    invocation_context = test_invocation_context([fc_event, fr_event])
    match = invocation_context._find_matching_function_call(fr_event_no_fr)
    assert match is None
