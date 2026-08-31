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

from ..events.event import Event
from ..events.event_actions import EventActions
from ..platform import uuid as platform_uuid
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session

if TYPE_CHECKING:
  from ..artifacts.base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)


async def compute_state_delta_for_rewind(
    session: Session, rewind_event_index: int
) -> dict[str, Any]:
  """Computes the state delta to reverse changes."""
  state_at_rewind_point: dict[str, Any] = {}
  for i in range(rewind_event_index):
    if session.events[i].actions.state_delta:
      for k, v in session.events[i].actions.state_delta.items():
        if k.startswith("app:") or k.startswith("user:"):
          continue
        if v is None:
          state_at_rewind_point.pop(k, None)
        else:
          state_at_rewind_point[k] = v

  current_state = session.state
  rewind_state_delta = {}

  # 1. Add/update keys in rewind_state_delta to match state_at_rewind_point.
  for key, value_at_rewind in state_at_rewind_point.items():
    if key not in current_state or current_state[key] != value_at_rewind:
      rewind_state_delta[key] = value_at_rewind

  # 2. Set keys to None in rewind_state_delta if they are in current_state
  #    but not in state_at_rewind_point. These keys were added after the
  #    rewind point and need to be removed.
  for key in current_state:
    if key.startswith("app:") or key.startswith("user:"):
      continue
    if key not in state_at_rewind_point:
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
  for filename, vn in current_versions.items():
    if filename.startswith("user:"):
      # User artifacts are not restored on rewind.
      continue
    vt = versions_at_rewind_point.get(filename)
    if vt == vn:
      continue

    rewind_artifact_delta[filename] = vn + 1
    artifact: types.Part
    if vt is None:
      # Artifact did not exist at rewind point. Mark it as inaccessible.
      artifact = types.Part(
          inline_data=types.Blob(mime_type="application/octet-stream", data=b"")
      )
    else:
      # Artifact version changed after rewind point. Restore to version at
      # rewind point by loading the actual data via the artifact service.
      loaded_artifact = await artifact_service.load_artifact(
          app_name=app_name,
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
        artifact = types.Part(
            inline_data=types.Blob(
                mime_type="application/octet-stream", data=b""
            )
        )
      else:
        artifact = loaded_artifact
    await artifact_service.save_artifact(
        app_name=app_name,
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
    raise ValueError(f"Invocation ID not found: {rewind_before_invocation_id}")

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
