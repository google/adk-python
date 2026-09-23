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

import logging
from unittest.mock import MagicMock

from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows.context import _contents as contents
from google.adk.flows.llm_flows.context import _interactions as interactions_processor
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.genai import types


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
    # Event without branch should be included when no current branch
    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
    )
    assert interactions_processor._is_event_in_branch(None, event) is True

    # Event with branch should be excluded when no current branch
    event_with_branch = Event(
        invocation_id="inv2",
        author="test",
        content=types.ModelContent("test"),
        branch="some_branch",
    )
    assert (
        interactions_processor._is_event_in_branch(None, event_with_branch)
        is False
    )

  def test_is_event_in_branch_same_branch(self):
    """Test that events in the same branch are included."""
    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
        branch="root.child",
    )
    assert (
        interactions_processor._is_event_in_branch("root.child", event) is True
    )

  def test_is_event_in_branch_different_branch(self):
    """Test that events in different branches are excluded."""
    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
        branch="root.other",
    )
    assert (
        interactions_processor._is_event_in_branch("root.child", event) is False
    )

  def test_is_event_in_branch_root_events_included(self):
    """Test that root events (no branch) are included in child branches."""
    event = Event(
        invocation_id="inv1",
        author="test",
        content=types.ModelContent("test"),
    )
    assert (
        interactions_processor._is_event_in_branch("root.child", event) is True
    )


def test_single_flow_extracts_interaction_state_before_contents():
  """Chained requests expose their interaction ID to content assembly."""
  flow = SingleFlow()

  interactions_index = flow.request_processors.index(
      interactions_processor.request_processor
  )
  contents_index = flow.request_processors.index(contents.request_processor)

  assert interactions_index < contents_index


def _evt(author: str, interaction_id: str | None, branch: str | None) -> Event:
  return Event(author=author, interaction_id=interaction_id, branch=branch)


def test_find_previous_interaction_id_returns_latest_for_agent():
  events = [
      _evt("my_agent", "int_1", None),
      _evt("user", None, None),
      _evt("my_agent", "int_2", None),
      _evt("other_agent", "int_3", None),
  ]

  result = interactions_processor._find_previous_interaction_state(
      events, agent_name="my_agent", current_branch=None
  )

  assert result[0] == "int_2"


def test_find_previous_interaction_id_respects_branch():
  events = [
      _evt("my_agent", "int_main", None),
      _evt("my_agent", "int_other_branch", "branch_b"),
  ]

  result = interactions_processor._find_previous_interaction_state(
      events, agent_name="my_agent", current_branch="branch_a"
  )

  assert result[0] == "int_main"


def test_find_previous_interaction_id_none_when_absent():
  events = [_evt("user", None, None)]

  result = interactions_processor._find_previous_interaction_state(
      events, agent_name="my_agent", current_branch=None
  )

  assert result[0] is None


def test_find_previous_interaction_state_returns_both_ids():
  events = [
      Event(author="my_agent", interaction_id="int_1", environment_id="env_1"),
      Event(author="user"),
      Event(author="my_agent", interaction_id="int_2", environment_id="env_2"),
  ]

  state = interactions_processor._find_previous_interaction_state(
      events, agent_name="my_agent", current_branch=None
  )

  assert state == ("int_2", "env_2")


def _tier_context(
    *,
    use_interactions_api: bool,
    service_tier: str | None,
    invocation_id: str = "inv-1",
) -> MagicMock:
  """Build an invocation context for the service-tier tests."""
  invocation_context = MagicMock()
  invocation_context.invocation_id = invocation_id
  invocation_context.branch = None
  invocation_context.session.events = []
  invocation_context.agent.name = "test_agent"
  invocation_context.agent.canonical_model = Gemini(
      model="gemini-2.5-flash", use_interactions_api=use_interactions_api
  )
  invocation_context.run_config = RunConfig(service_tier=service_tier)
  return invocation_context


async def _run(processor, invocation_context, llm_request) -> None:
  """Drain the processor, which yields no events."""
  assert not [
      event
      async for event in processor.run_async(invocation_context, llm_request)
  ]


class TestServiceTierFromRunConfig:
  """Tests for carrying RunConfig.service_tier onto the LlmRequest.

  The tier is a per-request field on the interactions API, so the run picks it
  rather than the model being configured for it once and for all.
  """

  async def test_copies_tier_onto_the_request(self):
    """A tier on the run reaches the model via the request."""
    processor = interactions_processor.InteractionsRequestProcessor()
    invocation_context = _tier_context(
        use_interactions_api=True, service_tier="deferred"
    )
    llm_request = LlmRequest()

    await _run(processor, invocation_context, llm_request)

    assert llm_request.service_tier == "deferred"

  async def test_no_tier_leaves_the_request_unset(self):
    """A run that asks for no tier leaves the model's default in place."""
    processor = interactions_processor.InteractionsRequestProcessor()
    invocation_context = _tier_context(
        use_interactions_api=True, service_tier=None
    )
    llm_request = LlmRequest()

    await _run(processor, invocation_context, llm_request)

    assert llm_request.service_tier is None

  async def test_does_not_copy_for_a_non_interactions_model(self):
    """Only the interactions path has a serving tier."""
    processor = interactions_processor.InteractionsRequestProcessor()
    invocation_context = _tier_context(
        use_interactions_api=False, service_tier="deferred"
    )
    llm_request = LlmRequest()

    await _run(processor, invocation_context, llm_request)

    assert llm_request.service_tier is None


class TestUnusableServiceTierWarning:
  """Tests for the warning when a run's tier cannot be applied."""

  async def test_warns_once_per_run_not_once_per_turn(self, caplog):
    """The processor runs per model call, so the warning must not repeat.

    Args:
      caplog: pytest fixture capturing log records.
    """
    caplog.set_level(logging.WARNING)
    processor = interactions_processor.InteractionsRequestProcessor()

    for _ in range(3):
      invocation_context = _tier_context(
          use_interactions_api=False,
          service_tier="deferred",
          invocation_id="inv-same",
      )
      await _run(processor, invocation_context, LlmRequest())

    warnings = [r for r in caplog.records if "service_tier" in r.getMessage()]
    assert len(warnings) == 1

  async def test_warns_again_for_a_different_run(self, caplog):
    """A separate invocation is worth telling the caller about again.

    Args:
      caplog: pytest fixture capturing log records.
    """
    caplog.set_level(logging.WARNING)
    processor = interactions_processor.InteractionsRequestProcessor()

    for invocation_id in ("inv-1", "inv-2"):
      invocation_context = _tier_context(
          use_interactions_api=False,
          service_tier="deferred",
          invocation_id=invocation_id,
      )
      await _run(processor, invocation_context, LlmRequest())

    warnings = [r for r in caplog.records if "service_tier" in r.getMessage()]
    assert len(warnings) == 2

  async def test_does_not_warn_without_a_tier(self, caplog):
    """No tier asked for, nothing to warn about.

    Args:
      caplog: pytest fixture capturing log records.
    """
    caplog.set_level(logging.WARNING)
    processor = interactions_processor.InteractionsRequestProcessor()
    invocation_context = _tier_context(
        use_interactions_api=False, service_tier=None
    )

    await _run(processor, invocation_context, LlmRequest())

    assert not [r for r in caplog.records if "service_tier" in r.getMessage()]
