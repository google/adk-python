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

from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
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
    events = mock_invocation_context.get_events()
    assert events == mock_events

  def test_get_events_filters_by_current_invocation(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current invocation."""
    event1, event2, _, _ = mock_events
    events = mock_invocation_context.get_events(current_invocation=True)
    assert events == [event1, event2]

  def test_get_events_filters_by_current_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events correctly filters by the current branch."""
    event1, _, event3, _ = mock_events
    events = mock_invocation_context.get_events(current_branch=True)
    assert events == [event1, event3]

  def test_get_events_filters_by_invocation_and_branch(
      self, mock_invocation_context, mock_events
  ):
    """Tests that get_events filters by invocation and branch."""
    event1, _, _, _ = mock_events
    events = mock_invocation_context.get_events(
        current_invocation=True,
        current_branch=True,
    )
    assert events == [event1]

  def test_get_events_with_no_events_in_session(self, mock_invocation_context):
    """Tests get_events when the session has no events."""
    mock_invocation_context.session.events = []
    events = mock_invocation_context.get_events()
    assert not events

  def test_get_events_with_no_matching_events(self, mock_invocation_context):
    """Tests get_events when no events match the filters."""
    mock_invocation_context.invocation_id = 'inv_3'
    mock_invocation_context.branch = 'branch_C'

    # Filter by invocation
    events = mock_invocation_context.get_events(current_invocation=True)
    assert not events

    # Filter by branch
    events = mock_invocation_context.get_events(current_branch=True)
    assert not events

    # Filter by both
    events = mock_invocation_context.get_events(
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
