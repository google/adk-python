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

"""Tests for the interactions processor."""

from unittest.mock import MagicMock

from google.adk.events.event import Event
from google.adk.flows.llm_flows import interactions_processor
from google.genai import types
import pytest


class TestInteractionsRequestProcessor:
  """Tests for InteractionsRequestProcessor."""

  def test_find_previous_interaction_id_empty_events(self):
    """Test that None is returned when there are no events."""
    processor = interactions_processor.InteractionsRequestProcessor()
    invocation_context = MagicMock()
    invocation_context.session.events = []
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result is None

  def test_find_previous_interaction_id_user_only_events(self):
    """Test that None is returned when only user events exist."""
    processor = interactions_processor.InteractionsRequestProcessor()
    events = [
        Event(
            invocation_id="inv1",
            author="user",
            content=types.UserContent("Hello"),
        ),
        Event(
            invocation_id="inv2",
            author="user",
            content=types.UserContent("World"),
        ),
    ]
    invocation_context = MagicMock()
    invocation_context.session.events = events
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result is None

  def test_find_previous_interaction_id_no_interaction_id(self):
    """Test that None is returned when model events have no interaction_id."""
    processor = interactions_processor.InteractionsRequestProcessor()
    events = [
        Event(
            invocation_id="inv1",
            author="user",
            content=types.UserContent("Hello"),
        ),
        Event(
            invocation_id="inv2",
            author="test_agent",
            content=types.ModelContent("Response without interaction_id"),
        ),
    ]
    invocation_context = MagicMock()
    invocation_context.session.events = events
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result is None

  def test_find_previous_interaction_id_from_model_event(self):
    """Test that interaction_id is returned from model event."""
    processor = interactions_processor.InteractionsRequestProcessor()
    events = [
        Event(
            invocation_id="inv1",
            author="user",
            content=types.UserContent("Hello"),
        ),
        Event(
            invocation_id="inv2",
            author="test_agent",
            content=types.ModelContent("Response"),
            interaction_id="interaction_123",
        ),
    ]
    invocation_context = MagicMock()
    invocation_context.session.events = events
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result == "interaction_123"

  def test_find_previous_interaction_id_returns_most_recent(self):
    """Test that the most recent interaction_id is returned."""
    processor = interactions_processor.InteractionsRequestProcessor()
    events = [
        Event(
            invocation_id="inv1",
            author="user",
            content=types.UserContent("Hello"),
        ),
        Event(
            invocation_id="inv2",
            author="test_agent",
            content=types.ModelContent("First response"),
            interaction_id="interaction_first",
        ),
        Event(
            invocation_id="inv3",
            author="user",
            content=types.UserContent("Second message"),
        ),
        Event(
            invocation_id="inv4",
            author="test_agent",
            content=types.ModelContent("Second response"),
            interaction_id="interaction_second",
        ),
    ]
    invocation_context = MagicMock()
    invocation_context.session.events = events
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result == "interaction_second"

  def test_find_previous_interaction_id_skips_user_events(self):
    """Test that user events with interaction_id are skipped."""
    processor = interactions_processor.InteractionsRequestProcessor()
    events = [
        Event(
            invocation_id="inv1",
            author="test_agent",
            content=types.ModelContent("Model response"),
            interaction_id="interaction_model",
        ),
        Event(
            invocation_id="inv2",
            author="user",
            content=types.UserContent("User message"),
            interaction_id="interaction_user",  # This should be skipped
        ),
    ]
    invocation_context = MagicMock()
    invocation_context.session.events = events
    invocation_context.branch = None
    invocation_context.agent.name = "test_agent"

    result = processor._find_previous_interaction_id(invocation_context)
    assert result == "interaction_model"

  def test_is_event_in_branch_no_branch(self):
    """Test branch filtering with no current branch."""
    processor = interactions_processor.InteractionsRequestProcessor()

    # Event without branch should be included when no current branch
    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
    )
    assert processor._is_event_in_branch(None, event) is True

    # Event with branch should be excluded when no current branch
    event_with_branch = Event(
        invocation_id="inv2",
        author="test",
        content=types.ModelContent("test"),
        branch="some_branch",
    )
    assert processor._is_event_in_branch(None, event_with_branch) is False

  def test_is_event_in_branch_same_branch(self):
    """Test that events in the same branch are included."""
    processor = interactions_processor.InteractionsRequestProcessor()

    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
        branch="root.child",
    )
    assert processor._is_event_in_branch("root.child", event) is True

  def test_is_event_in_branch_different_branch(self):
    """Test that events in different branches are excluded."""
    processor = interactions_processor.InteractionsRequestProcessor()

    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
        branch="root.other",
    )
    assert processor._is_event_in_branch("root.child", event) is False

  def test_is_event_in_branch_root_events_included(self):
    """Test that root events (no branch) are included in child branches."""
    processor = interactions_processor.InteractionsRequestProcessor()

    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
    )
    assert processor._is_event_in_branch("root.child", event) is True
