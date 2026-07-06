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

"""Unit tests for source_filter parameter in _get_contents / _get_current_turn_contents."""

from google.adk.events.event import Event
from google.adk.flows.llm_flows import contents
from google.genai import types
import pytest


def _user_event(text: str, invocation_id: str = 'inv') -> Event:
  return Event(
      invocation_id=invocation_id,
      author='user',
      content=types.Content(role='user', parts=[types.Part(text=text)]),
  )


def _model_event(text: str, author: str, invocation_id: str = 'inv') -> Event:
  return Event(
      invocation_id=invocation_id,
      author=author,
      content=types.Content(role='model', parts=[types.Part(text=text)]),
  )


def _function_response_event(
    name: str, response: dict, invocation_id: str = 'inv'
) -> Event:
  return Event(
      invocation_id=invocation_id,
      author='user',
      content=types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(name=name, response=response)
          ],
      ),
  )


def _function_call_event(
    name: str, args: dict, author: str, invocation_id: str = 'inv'
) -> Event:
  return Event(
      invocation_id=invocation_id,
      author=author,
      content=types.Content(
          role='model',
          parts=[types.Part.from_function_call(name=name, args=args)],
      ),
  )


# ---------------------------------------------------------------------------
# Regression: source_filter=None is a no-op
# ---------------------------------------------------------------------------


def test_source_filter_none_is_no_op():
  """source_filter=None should produce identical output to omitting the param."""
  events = [
      _user_event('hello'),
      _model_event('hi there', author='agent_a'),
      _model_event('peer reply', author='agent_b'),
  ]
  without = contents._get_contents(None, events, agent_name='agent_a')
  with_none = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=None
  )
  assert without == with_none


# ---------------------------------------------------------------------------
# user only
# ---------------------------------------------------------------------------


def test_source_filter_user_keeps_user_drops_model_and_others():
  """['user'] keeps user messages, drops this agent's model turns and peers."""
  events = [
      _user_event('user msg 1'),
      _model_event('self reply', author='agent_a'),
      _model_event('peer reply', author='agent_b'),
      _user_event('user msg 2'),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'user msg 1' in texts
  assert 'user msg 2' in texts
  assert 'self reply' not in texts
  assert 'peer reply' not in texts
  # No narrative "For context:" wrapper — other agent was dropped entirely
  assert not any('For context:' in t for t in texts)


# ---------------------------------------------------------------------------
# self only
# ---------------------------------------------------------------------------


def test_source_filter_self_keeps_model_drops_user_and_others():
  """['self'] keeps this agent's model turns, drops user messages and peers."""
  events = [
      _user_event('user msg'),
      _model_event('self turn 1', author='agent_a'),
      _model_event('peer reply', author='agent_b'),
      _model_event('self turn 2', author='agent_a'),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['self']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'self turn 1' in texts
  assert 'self turn 2' in texts
  assert 'user msg' not in texts
  assert 'peer reply' not in texts


# ---------------------------------------------------------------------------
# user + self
# ---------------------------------------------------------------------------


def test_source_filter_user_and_self_drops_other_agents():
  """['user', 'self'] keeps user + this agent's turns, drops all other agents."""
  events = [
      _user_event('hi'),
      _model_event('my answer', author='agent_a'),
      _model_event('agent_b reply', author='agent_b'),
      _model_event('agent_c reply', author='agent_c'),
      _user_event('follow up'),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user', 'self']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'hi' in texts
  assert 'follow up' in texts
  assert 'my answer' in texts
  assert 'agent_b reply' not in texts
  assert 'agent_c reply' not in texts
  assert not any('For context:' in t for t in texts)


# ---------------------------------------------------------------------------
# specific agent name
# ---------------------------------------------------------------------------


def test_source_filter_specific_agent_name():
  """['agent_b'] keeps only agent_b's entries, drops user, self, and agent_c."""
  events = [
      _user_event('user msg'),
      _model_event('self reply', author='agent_a'),
      _model_event('b says hi', author='agent_b'),
      _model_event('c says bye', author='agent_c'),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['agent_b']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert any('b says hi' in t for t in texts)
  assert 'user msg' not in texts
  assert 'self reply' not in texts
  assert 'c says bye' not in texts


def test_source_filter_user_and_specific_agent():
  """['user', 'agent_b'] keeps user + agent_b, drops self and agent_c."""
  events = [
      _user_event('user msg'),
      _model_event('self reply', author='agent_a'),
      _model_event('b says hi', author='agent_b'),
      _model_event('c says bye', author='agent_c'),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user', 'agent_b']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'user msg' in texts
  assert any('b says hi' in t for t in texts)
  assert 'self reply' not in texts
  assert 'c says bye' not in texts


# ---------------------------------------------------------------------------
# Function responses are never filtered
# ---------------------------------------------------------------------------


def test_source_filter_self_keeps_fc_call_and_response_together():
  """FC call and response are both tied to 'self': including 'self' keeps both."""
  events = [
      _user_event('user msg'),
      _function_call_event('my_tool', {'x': 1}, author='agent_a'),
      _function_response_event('my_tool', {'result': 'ok'}),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['self']
  )
  # FC call (role=model) and FC response (role=user) both belong to 'self'
  roles = [c.role for c in result]
  assert 'model' in roles  # function call kept
  assert 'user' in roles  # function response kept (no orphan)


def test_source_filter_without_self_drops_fc_call_and_response_together():
  """Dropping 'self' drops both sides of the FC/FR pair to avoid orphaned responses."""
  events = [
      _user_event('plain user message'),
      _function_call_event('tool', {}, author='agent_a'),
      _function_response_event('tool', {'v': 1}),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'plain user message' in texts
  # Both FC call and FC response are dropped — no orphaned function_response part
  assert not any(c.role == 'model' for c in result)
  assert not any(
      p.function_response is not None for c in result for p in c.parts or []
  )


# ---------------------------------------------------------------------------
# Interaction with _get_current_turn_contents
# ---------------------------------------------------------------------------


def test_source_filter_propagates_to_current_turn():
  """source_filter is respected when include_contents='none' path is taken.

  Simulates the start of a new invocation where only the user message has
  arrived; _get_current_turn_contents identifies it as the turn boundary.
  With source_filter=['user'], prior self/peer history is excluded and only
  the current user message survives.
  """
  events = [
      _user_event('turn 1', invocation_id='inv1'),
      _model_event('self turn 1', author='agent_a', invocation_id='inv1'),
      _model_event('peer old', author='agent_b', invocation_id='inv1'),
      # New invocation: only the user message has arrived so far
      _user_event('turn 2', invocation_id='inv2'),
  ]
  result = contents._get_current_turn_contents(
      None, events, agent_name='agent_a', source_filter=['user']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  # Only the current-turn user message survives; prior history is excluded
  assert 'turn 2' in texts
  assert 'turn 1' not in texts
  assert 'self turn 1' not in texts
  assert 'peer old' not in texts


def test_source_filter_drops_other_agent_fc_response_when_call_author_filtered():
  """When agent_b is not in filter, its FC call AND its 'user'-authored response are both dropped.

  Without this fix, the response would survive as
  '[agent_b] tool returned X' text with no visible call — misleading context.
  """
  events = [
      _user_event('user msg'),
      # agent_b makes a function call (role=model, is_other_reply=True)
      Event(
          invocation_id='inv',
          author='agent_b',
          content=types.Content(
              role='model',
              parts=[types.Part.from_function_call(name='search', args={})],
          ),
      ),
      # 'user'-authored response to agent_b's call (is_other_reply=True via fc_author_by_id)
      _function_response_event('search', {'results': 'found it'}),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user', 'self']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'user msg' in texts
  # Neither the call nor the response from agent_b should appear
  assert not any('search' in t for t in texts)
  assert not any('found it' in t for t in texts)


def test_source_filter_keeps_other_agent_fc_response_when_call_author_included():
  """When agent_b IS in filter, its FC response is kept and converted to context text."""
  events = [
      _user_event('user msg'),
      Event(
          invocation_id='inv',
          author='agent_b',
          content=types.Content(
              role='model',
              parts=[types.Part.from_function_call(name='lookup', args={})],
          ),
      ),
      _function_response_event('lookup', {'value': 42}),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user', 'agent_b']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'user msg' in texts
  # agent_b's call and response both present (as narrative text)
  assert any('lookup' in t for t in texts)


def test_source_filter_self_matches_current_agent_in_live_mode():
  """In live mode, the current agent's events are classified as other_reply.

  source_filter=['self'] must still keep them by mapping event.author==agent_name
  to the 'self' reserved name, not by literal string comparison.
  """
  live_session_id = 'live-123'
  events = [
      Event(
          invocation_id='inv',
          author='user',
          live_session_id=live_session_id,
          content=types.Content(
              role='user', parts=[types.Part(text='user prompt')]
          ),
      ),
      # In live mode, current agent's own turn has is_other_reply=True
      Event(
          invocation_id='inv',
          author='agent_a',
          live_session_id=live_session_id,
          content=types.Content(
              role='model', parts=[types.Part(text='my own reply')]
          ),
      ),
      Event(
          invocation_id='inv',
          author='agent_b',
          live_session_id=live_session_id,
          content=types.Content(
              role='model', parts=[types.Part(text='peer reply')]
          ),
      ),
  ]
  result = contents._get_contents(
      None, events, agent_name='agent_a', source_filter=['user', 'self']
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'user prompt' in texts
  # Current agent's own turn must survive even though is_other_reply=True in live mode
  assert any('my own reply' in t for t in texts)
  # Peer agent must be filtered
  assert not any('peer reply' in t for t in texts)


def test_source_filter_all_sources_is_same_as_none():
  """Filtering with all relevant source names present is equivalent to no filter."""
  agent_name = 'agent_a'
  events = [
      _user_event('hello'),
      _model_event('self reply', author=agent_name),
      _model_event('peer reply', author='agent_b'),
  ]
  no_filter = contents._get_contents(None, events, agent_name=agent_name)
  all_sources = contents._get_contents(
      None,
      events,
      agent_name=agent_name,
      source_filter=['user', 'self', 'agent_b'],
  )
  assert no_filter == all_sources


# ---------------------------------------------------------------------------
# include_contents='current' (stop_at_user_only=True)
# ---------------------------------------------------------------------------


def test_current_includes_user_and_all_sibling_agents():
  """stop_at_user_only=True anchors at user msg, giving full invocation context."""
  events = [
      _user_event('hello'),
      _model_event('agent_a reply', author='agent_a'),
      _model_event('agent_b reply', author='agent_b'),
  ]
  result = contents._get_current_turn_contents(
      None, events, agent_name='agent_c', stop_at_user_only=True
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'hello' in texts
  assert any('agent_a reply' in t for t in texts)
  assert any('agent_b reply' in t for t in texts)


def test_current_vs_none_differ_when_agent_precedes_current():
  """'none' (default) anchors at last agent; 'current' anchors at user message."""
  events = [
      _user_event('hello'),
      _model_event('agent_a reply', author='agent_a'),
      _model_event('agent_b reply', author='agent_b'),
  ]
  # 'none' mode: stops at agent_b (last boundary)
  result_none = contents._get_current_turn_contents(
      None, events, agent_name='agent_c', stop_at_user_only=False
  )
  texts_none = [p.text for c in result_none for p in c.parts if p.text]
  assert 'hello' not in texts_none
  assert any('agent_b reply' in t for t in texts_none)

  # 'current' mode: stops at user message, includes everything
  result_current = contents._get_current_turn_contents(
      None, events, agent_name='agent_c', stop_at_user_only=True
  )
  texts_current = [p.text for c in result_current for p in c.parts if p.text]
  assert 'hello' in texts_current
  assert any('agent_a reply' in t for t in texts_current)
  assert any('agent_b reply' in t for t in texts_current)


def test_current_with_source_filter_user_gives_user_message_not_empty():
  """stop_at_user_only=True + source_filter=['user'] → user message, not empty.

  This is the footgun fixed: include_contents='none' + include_sources=['user']
  returns empty because the boundary lands on an agent event that then gets
  filtered. 'current' mode anchors at the user message, so filtering works.
  """
  events = [
      _user_event('original request'),
      _model_event('agent_a reply', author='agent_a'),
  ]
  result = contents._get_current_turn_contents(
      None,
      events,
      agent_name='agent_b',
      stop_at_user_only=True,
      source_filter=['user'],
  )
  texts = [p.text for c in result for p in c.parts if p.text]
  assert 'original request' in texts
  assert not any('agent_a reply' in t for t in texts)
  # Crucially: result is not empty
  assert len(result) > 0


def test_current_no_user_event_returns_empty():
  """When no user event exists, 'current' mode returns [] (same as 'none')."""
  events = [
      _model_event('agent_a reply', author='agent_a'),
      _model_event('agent_b reply', author='agent_b'),
  ]
  result = contents._get_current_turn_contents(
      None, events, agent_name='agent_c', stop_at_user_only=True
  )
  assert result == []
