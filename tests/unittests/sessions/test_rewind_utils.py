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

from unittest import mock

from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.errors import InvocationNotFoundError
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.errors.not_found_error import NotFoundError
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import _rewind_utils
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types
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


async def test_compute_state_delta_for_rewind_ignores_temp_keys():
  """compute_state_delta_for_rewind must exclude temp:-prefixed state keys."""
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      state={"k1": "v2", "temp:scratch": "ephemeral", "app:global": 1},
      events=[
          Event(
              invocation_id="inv1",
              actions=EventActions(state_delta={"k1": "v1"}),
          ),
          Event(
              invocation_id="inv2",
              actions=EventActions(
                  state_delta={"k1": "v2", "temp:scratch": "ephemeral"}
              ),
          ),
      ],
  )
  delta = await _rewind_utils.compute_state_delta_for_rewind(session, 1)
  assert delta == {"k1": "v1"}
  assert "temp:scratch" not in delta


async def test_compute_artifact_delta_returns_empty_when_no_artifact_service():
  """Without an artifact service, artifact delta computation returns empty dict."""
  session = Session(id="s1", app_name="app", user_id="u1", events=[])
  delta = await _rewind_utils.compute_artifact_delta_for_rewind(session, 0)
  assert delta == {}


async def test_compute_artifact_delta_records_version_returned_by_service():
  """The artifact delta holds the version save_artifact returned."""
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              actions=EventActions(artifact_delta={"report.txt": 0}),
          ),
          Event(
              invocation_id="inv2",
              actions=EventActions(artifact_delta={"report.txt": 1}),
          ),
      ],
  )
  artifact_service = mock.AsyncMock()
  artifact_service.load_artifact.return_value = types.Part.from_text(
      text="first"
  )
  artifact_service.save_artifact.return_value = 7

  delta = await _rewind_utils.compute_artifact_delta_for_rewind(
      session, 1, artifact_service=artifact_service, app_name="app"
  )

  assert delta == {"report.txt": 7}


async def test_compute_artifact_delta_requires_app_name():
  """Rewinding an artifact with no app name anywhere raises InputValidationError."""
  session = Session(
      id="s1",
      app_name="",
      user_id="u1",
      events=[
          Event(invocation_id="inv1"),
          Event(
              invocation_id="inv2",
              actions=EventActions(artifact_delta={"f.txt": 0}),
          ),
      ],
  )

  artifact_service = mock.create_autospec(BaseArtifactService, instance=True)

  with pytest.raises(InputValidationError, match="app_name"):
    await _rewind_utils.compute_artifact_delta_for_rewind(
        session, 1, artifact_service=artifact_service
    )
  artifact_service.save_artifact.assert_not_called()


async def test_compute_artifact_delta_falls_back_to_the_session_app_name():
  """An omitted app_name is taken from the session the artifacts belong to."""
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(invocation_id="inv1"),
          Event(
              invocation_id="inv2",
              actions=EventActions(artifact_delta={"f.txt": 0}),
          ),
      ],
  )

  artifact_service = mock.create_autospec(BaseArtifactService, instance=True)
  artifact_service.save_artifact.return_value = 1

  delta = await _rewind_utils.compute_artifact_delta_for_rewind(
      session, 1, artifact_service=artifact_service
  )

  assert delta == {"f.txt": 1}
  assert artifact_service.save_artifact.await_args.kwargs["app_name"] == "app"


async def test_rewind_session_raises_when_invocation_not_found():
  """Rewinding to an invocation id not present in session raises InvocationNotFoundError."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="app", user_id="u1", session_id="s1"
  )

  with pytest.raises(
      InvocationNotFoundError, match="Invocation ID not found"
  ) as exc_info:
    await _rewind_utils.rewind_session(
        session_service=session_service,
        session=session,
        rewind_before_invocation_id="inv_missing",
    )
  assert isinstance(exc_info.value, NotFoundError)
  assert isinstance(exc_info.value, ValueError)


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


def _session_for_state_rewind() -> Session:
  return Session(
      id="s1",
      app_name="app",
      user_id="u1",
      state={
          "app:first_in_dict": "global",
          "user:second_in_dict": "pref",
          "temp:third_in_dict": "scratch",
          "added_after_rewind": "should_be_cleared",
      },
      events=[
          Event(
              invocation_id="inv1",
              actions=EventActions(
                  state_delta={
                      "app:first_in_dict": "global",
                      "restored_key": "original",
                      "cleared_before_rewind": "temp_val",
                  }
              ),
          ),
          Event(
              invocation_id="inv1",
              actions=EventActions(state_delta={"cleared_before_rewind": None}),
          ),
          Event(
              invocation_id="inv2",
              actions=EventActions(
                  state_delta={
                      "added_after_rewind": "should_be_cleared",
                      "cleared_before_rewind": "re_added",
                  }
              ),
          ),
      ],
  )


async def test_compute_state_delta_restores_keys_and_skips_prefixed():
  """Keys added after the rewind point are cleared, prefixed ones untouched."""
  delta = await _rewind_utils.compute_state_delta_for_rewind(
      _session_for_state_rewind(), 2
  )
  assert delta == {"restored_key": "original", "added_after_rewind": None}


async def test_rewind_session_rewinds_to_first_event_of_invocation():
  """Rewinding before a multi-event invocation undoes all of its events."""
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="app", user_id="u1", session_id="s1"
  )
  for invocation_id, step in [
      ("inv1", "from_inv1"),
      ("inv2", "from_inv2_first_event"),
      ("inv2", "from_inv2_second_event"),
  ]:
    await session_service.append_event(
        session,
        Event(
            invocation_id=invocation_id,
            author="user",
            actions=EventActions(state_delta={"step": step}),
        ),
    )

  await _rewind_utils.rewind_session(
      session_service=session_service,
      session=session,
      rewind_before_invocation_id="inv2",
  )

  assert session.state["step"] == "from_inv1"


_DOC_V0 = types.Part.from_text(text="doc v0")
_UNCHANGED = types.Part.from_text(text="unchanged")


async def _rewind_artifacts() -> tuple[dict[str, int], InMemoryArtifactService]:
  """Rewinds a session whose second invocation changed several artifacts."""
  artifact_service = InMemoryArtifactService()
  for filename, artifact in [
      ("doc.txt", _DOC_V0),
      ("unchanged.txt", _UNCHANGED),
      ("user:profile.txt", types.Part.from_text(text="user scoped")),
      ("doc.txt", types.Part.from_text(text="doc v1")),
      ("later.txt", types.Part.from_text(text="created in inv2")),
  ]:
    await artifact_service.save_artifact(
        app_name="app",
        user_id="u1",
        session_id="s1",
        filename=filename,
        artifact=artifact,
    )
  session = Session(
      id="s1",
      app_name="app",
      user_id="u1",
      events=[
          Event(
              invocation_id="inv1",
              actions=EventActions(
                  artifact_delta={
                      "doc.txt": 0,
                      "unchanged.txt": 0,
                      "missing_in_store.txt": 0,
                      "user:profile.txt": 0,
                  }
              ),
          ),
          Event(
              invocation_id="inv2",
              actions=EventActions(
                  artifact_delta={
                      "doc.txt": 1,
                      "later.txt": 0,
                      "missing_in_store.txt": 1,
                      "user:profile.txt": 0,
                  }
              ),
          ),
      ],
  )
  delta = await _rewind_utils.compute_artifact_delta_for_rewind(
      session, 1, artifact_service=artifact_service, app_name="app"
  )
  return delta, artifact_service


async def _load(
    artifact_service: InMemoryArtifactService, filename: str
) -> types.Part | None:
  return await artifact_service.load_artifact(
      app_name="app", user_id="u1", session_id="s1", filename=filename
  )


async def test_compute_artifact_delta_rewinds_only_changed_session_artifacts():
  """Unchanged and user-scoped artifacts are skipped; a missing one tombstones."""
  delta, artifact_service = await _rewind_artifacts()

  assert delta == {"doc.txt": 2, "later.txt": 1, "missing_in_store.txt": 0}
  assert await _load(artifact_service, "missing_in_store.txt") is None
  assert await _load(artifact_service, "unchanged.txt") == _UNCHANGED
