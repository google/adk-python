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

"""Private helper module for session and artifact rewind in ADK."""

from __future__ import annotations

import logging
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types

from ..errors._invocation_not_found_error import InvocationNotFoundError
from ..errors.input_validation_error import InputValidationError
from ..events.event import Event
from ..events.event_actions import EventActions
from ..platform import uuid as platform_uuid
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..sessions.state import State

if TYPE_CHECKING:
  from ..artifacts.base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)

_NON_SESSION_STATE_PREFIXES = (
    State.APP_PREFIX,
    State.USER_PREFIX,
    State.TEMP_PREFIX,
)


def _is_session_scoped_key(key: str) -> bool:
  """Returns whether a state key is scoped to the session being rewound."""
  return not key.startswith(_NON_SESSION_STATE_PREFIXES)


async def compute_state_delta_for_rewind(
    session: Session, rewind_event_index: int
) -> dict[str, Any]:
  """Computes the state delta that reverses changes made after a rewind point.

  Restores keys present at the rewind point and clears session-scoped keys that
  an event at or after the rewind point added, leaving other keys untouched.

  Args:
    session: The session to rewind.
    rewind_event_index: Index in `session.events` of the first event to undo.

  Returns:
    A state delta that restores keys from `state_at_rewind_point` that differ
    from `session.state` and sets keys written only at or after the rewind point
    to None.
  """
  state_at_rewind_point: dict[str, Any] = {}
  for event in session.events[:rewind_event_index]:
    for key, value in (event.actions.state_delta or {}).items():
      if not _is_session_scoped_key(key):
        continue
      if value is None:
        state_at_rewind_point.pop(key, None)
      else:
        state_at_rewind_point[key] = value

  current_state = session.state
  rewind_state_delta: dict[str, Any] = {}

  # 1. Add/update keys in rewind_state_delta to match state_at_rewind_point.
  for key, value_at_rewind in state_at_rewind_point.items():
    if key not in current_state or current_state[key] != value_at_rewind:
      rewind_state_delta[key] = value_at_rewind

  # 2. Clear session-scoped keys that an undone event wrote and that had no
  #    value at the rewind point. Iterating over the undone events rather than
  #    `current_state` keeps initial `create_session` state untouched, except
  #    for an initial key an undone event overwrote: its initial value is not
  #    recorded in any event, so it is cleared too.
  for event in session.events[rewind_event_index:]:
    for key in event.actions.state_delta or {}:
      if not _is_session_scoped_key(key):
        continue
      if key not in state_at_rewind_point and key in current_state:
        rewind_state_delta[key] = None

  return rewind_state_delta


async def compute_artifact_delta_for_rewind(
    session: Session,
    rewind_event_index: int,
    *,
    artifact_service: Optional[BaseArtifactService] = None,
    app_name: Optional[str] = None,
) -> dict[str, int]:
  """Computes the artifact delta to reverse changes."""
  if not artifact_service:
    return {}

  # Imported here so that importing sessions does not load artifacts.
  from ..artifacts import artifact_util

  versions_at_rewind_point: dict[str, int] = {}
  for i in range(rewind_event_index):
    event = session.events[i]
    if event.actions.artifact_delta:
      versions_at_rewind_point.update(event.actions.artifact_delta)

  current_versions: dict[str, int] = {}
  for event in session.events:
    if event.actions.artifact_delta:
      current_versions.update(event.actions.artifact_delta)

  rewind_artifact_delta = {}
  effective_app_name = app_name or session.app_name
  for filename, vn in current_versions.items():
    if filename.startswith("user:"):
      # User artifacts are not restored on rewind.
      continue
    vt = versions_at_rewind_point.get(filename)
    if vt == vn:
      continue
    if not effective_app_name:
      raise InputValidationError("app_name must not be empty.")

    artifact: types.Part
    if vt is None:
      # Artifact did not exist at rewind point. Mark it as inaccessible.
      artifact = artifact_util._new_rewind_tombstone()
    else:
      # Artifact version changed after rewind point. Restore to version at
      # rewind point by loading the actual data via the artifact service.
      loaded_artifact = await artifact_service.load_artifact(
          app_name=effective_app_name,
          user_id=session.user_id,
          session_id=session.id,
          filename=filename,
          version=vt,
      )
      if loaded_artifact is None:
        logger.warning(
            "Artifact %s version %d not found during rewind for"
            " session %s. Replacing with empty data.",
            filename,
            vt,
            session.id,
        )
        artifact = artifact_util._new_rewind_tombstone()
      else:
        artifact = loaded_artifact
    rewind_artifact_delta[filename] = await artifact_service.save_artifact(
        app_name=effective_app_name,
        user_id=session.user_id,
        session_id=session.id,
        filename=filename,
        artifact=artifact,
    )

  return rewind_artifact_delta


async def rewind_session(
    *,
    session_service: BaseSessionService,
    session: Session,
    rewind_before_invocation_id: str,
    artifact_service: Optional[BaseArtifactService] = None,
    app_name: Optional[str] = None,
    compute_state_delta: Optional[
        Callable[[Session, int], Awaitable[dict[str, Any]]]
    ] = None,
    compute_artifact_delta: Optional[
        Callable[[Session, int], Awaitable[dict[str, int]]]
    ] = None,
) -> None:
  """Rewinds the session to before the specified invocation."""
  rewind_event_index = -1
  for i, event in enumerate(session.events):
    if event.invocation_id == rewind_before_invocation_id:
      rewind_event_index = i
      break

  if rewind_event_index == -1:
    raise InvocationNotFoundError(
        f"Invocation ID not found: {rewind_before_invocation_id}"
    )

  # Compute state delta to reverse changes
  if compute_state_delta is not None:
    state_delta = await compute_state_delta(session, rewind_event_index)
  else:
    state_delta = await compute_state_delta_for_rewind(
        session, rewind_event_index
    )

  # Compute artifact delta to reverse changes
  if compute_artifact_delta is not None:
    artifact_delta = await compute_artifact_delta(session, rewind_event_index)
  else:
    artifact_delta = await compute_artifact_delta_for_rewind(
        session,
        rewind_event_index,
        artifact_service=artifact_service,
        app_name=app_name,
    )

  # Create rewind event
  rewind_event = Event(
      invocation_id=f"e-{platform_uuid.new_uuid()}",
      author="user",
      actions=EventActions(
          rewind_before_invocation_id=rewind_before_invocation_id,
          state_delta=state_delta,
          artifact_delta=artifact_delta,
      ),
  )

  logger.info("Rewinding session to invocation: %s", rewind_event)
  await session_service.append_event(session=session, event=rewind_event)
