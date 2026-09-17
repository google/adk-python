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

"""Unit tests for _rewind_utils helper module."""

from __future__ import annotations

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _rewind_utils
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
import pytest


async def test_compute_state_delta_reverts_state_to_rewind_point():
  """State delta restores modified keys and removes keys added after rewind point."""
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      state={"count": 5, "new_key": "val", "app:stay": "app_val"},
      events=[
          Event(
              invocation_id="inv1",
              actions=EventActions(state_delta={"count": 1}),
          ),
          Event(
              invocation_id="inv2",
              actions=EventActions(state_delta={"count": 5, "new_key": "val"}),
          ),
      ],
  )

  delta = await _rewind_utils.compute_state_delta_for_rewind(session, 1)

  assert delta["count"] == 1
  assert delta["new_key"] is None
  assert "app:stay" not in delta


async def test_compute_artifact_delta_returns_empty_when_no_artifact_service():
  """Without an artifact service, artifact delta computation returns empty dict."""
  session = Session(id="s1", app_name="app", user_id="u1", events=[])
  delta = await _rewind_utils.compute_artifact_delta_for_rewind(session, 0)
  assert delta == {}


async def test_rewind_session_raises_when_invocation_not_found():
  """Rewinding to an invocation id not present in session raises ValueError."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="app", user_id="u1", session_id="s1"
  )

  with pytest.raises(ValueError, match="Invocation ID not found"):
    await _rewind_utils.rewind_session(
        session_service=session_service,
        session=session,
        rewind_before_invocation_id="inv_missing",
    )


async def test_rewind_session_appends_rewind_event_with_deltas():
  """Successful rewind appends a user rewind event carrying state and artifact deltas."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="app", user_id="u1", session_id="s1"
  )
  event1 = Event(
      invocation_id="inv1",
      author="user",
      actions=EventActions(state_delta={"k": "v1"}),
  )
  event2 = Event(
      invocation_id="inv2",
      author="user",
      actions=EventActions(state_delta={"k": "v2"}),
  )
  await session_service.append_event(session, event1)
  await session_service.append_event(session, event2)

  await _rewind_utils.rewind_session(
      session_service=session_service,
      session=session,
      rewind_before_invocation_id="inv2",
  )

  rewound_session = await session_service.get_session(
      app_name="app", user_id="u1", session_id="s1"
  )
  assert len(rewound_session.events) == 3
  last_event = rewound_session.events[-1]
  assert last_event.actions.rewind_before_invocation_id == "inv2"
  assert last_event.actions.state_delta == {"k": "v1"}


async def test_rewind_session_uses_custom_delta_callbacks():
  """Custom compute_state_delta and compute_artifact_delta are called during rewind."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="app", user_id="u1", session_id="s1"
  )
  event = Event(
      invocation_id="inv1",
      author="user",
  )
  await session_service.append_event(session, event)

  custom_state_called = False
  custom_artifact_called = False

  async def mock_compute_state(s, idx):
    nonlocal custom_state_called
    custom_state_called = True
    return {"custom_key": "custom_val"}

  async def mock_compute_artifact(s, idx):
    nonlocal custom_artifact_called
    custom_artifact_called = True
    return {"art.txt": 2}

  await _rewind_utils.rewind_session(
      session_service=session_service,
      session=session,
      rewind_before_invocation_id="inv1",
      compute_state_delta=mock_compute_state,
      compute_artifact_delta=mock_compute_artifact,
  )

  assert custom_state_called
  assert custom_artifact_called
  last_event = session.events[-1]
  assert last_event.actions.state_delta == {"custom_key": "custom_val"}
  assert last_event.actions.artifact_delta == {"art.txt": 2}
