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

"""SSE framing for the events streamed by ``/run_sse``.

``/run_sse`` owns a wire format that ADK clients parse: internal metadata is
dropped, credential secrets are redacted, and an event carrying an
``artifactDelta`` is split into two frames. Applications that keep a ``Runner``
outside the HTTP handler (or put their own transport in front of it) still need
to emit exactly that format, so the framing lives here instead of inside the
route's generator. ``run_agent_sse`` calls these functions directly, which keeps
the public helper and the built-in server from drifting apart.

The module is private per the repo's visibility rules; ``serialize_event`` and
``format_event_frames`` are exported from ``google.adk.cli``.
"""

from __future__ import annotations

import pydantic_core

from ..auth.auth_credential import _redact_credential_secrets
from ..events._internal_metadata import public_event
from ..events.event import Event


def serialize_event(event: Event) -> str:
  """Serializes an event to the JSON text carried by a ``data:`` frame.

  This applies the same transformations as ``/run_sse``: internal metadata is
  removed via ``public_event`` and credential secrets are redacted.

  Args:
    event: The event to serialize. It is not modified.

  Returns:
    The JSON text of the event, without the SSE ``data:`` prefix or trailing
    blank line.
  """
  return pydantic_core.to_json(
      _redact_credential_secrets(
          public_event(event).model_dump(
              exclude_none=True, by_alias=True, mode="json"
          )
      )
  ).decode("utf-8")


def format_event_frames(
    event: Event, *, split_artifact_delta: bool = True
) -> list[str]:
  """Formats the SSE frames that ``/run_sse`` writes for one runner event.

  Args:
    event: The event to format. It is not modified.
    split_artifact_delta: Whether an event that has both content and an
      ``artifactDelta`` is split into a content frame and a content-less
      ``artifactDelta`` frame. ADK Web renders artifacts from both content and
      actions, so the split keeps an artifact from rendering twice. Pass
      ``False`` to emit the event as a single frame, as ``/run_sse`` does for
      the resume request that answers a long-running function call.

  Returns:
    One or two ``data: <json>\\n\\n`` strings, ready to be written to an SSE
    stream in order.
  """
  events_to_stream = [event]
  if (
      split_artifact_delta
      and event.actions.artifact_delta
      and event.content
      and event.content.parts
  ):
    content_event = event.model_copy(deep=True)
    content_event.actions.artifact_delta = {}
    artifact_event = event.model_copy(deep=True)
    artifact_event.content = None
    events_to_stream = [content_event, artifact_event]
  return [f"data: {serialize_event(e)}\n\n" for e in events_to_stream]
