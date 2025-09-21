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
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.session import Session
import pytest


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
