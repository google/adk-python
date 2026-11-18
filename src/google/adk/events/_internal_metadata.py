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

"""``Event.custom_metadata`` keys that only ADK code may write."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from ..sessions.session import Session
  from .event import Event

logger = logging.getLogger("google_adk." + __name__)

INTERNAL_METADATA_PREFIX = "__adk_internal_"
"""Prefix of ``custom_metadata`` keys that callers cannot set.

Keys with this prefix are dropped from ``RunConfig.custom_metadata``, both where
it is merged into events and where it is copied into the invocation context,
and from events that are restored into a session or received from a remote A2A
agent. Stored events keep them, but they are removed from API responses, saved
session files, CLI output and A2A messages.
"""

RESTORED_EVENT_KEY = INTERNAL_METADATA_PREFIX + "restored_event"
"""Set on events that ADK restored into a session from outside it."""


def _is_internal_key(key: Any) -> bool:
  return isinstance(key, str) and key.startswith(INTERNAL_METADATA_PREFIX)


def _drop_internal_keys(metadata: Mapping[str, Any]) -> dict[str, Any]:
  return {
      key: value for key, value in metadata.items() if not _is_internal_key(key)
  }


def without_internal_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
  """Returns ``metadata`` without keys that have the internal prefix.

  Use it where callers supply metadata. Dropped keys are logged at debug level,
  because a caller set a key it cannot set.

  Args:
    metadata: A ``custom_metadata`` value. It is not modified.

  Returns:
    A new dict without internal keys. A value that is not a mapping, such as
    ``None`` or a test stub, is returned unchanged.
  """
  if not isinstance(metadata, Mapping):
    return metadata
  kept = _drop_internal_keys(metadata)
  if len(kept) != len(metadata):
    logger.debug(
        "Dropping ADK-internal custom_metadata keys: %s",
        sorted(metadata.keys() - kept.keys()),
    )
  return kept


def public_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
  """Returns ``metadata`` as callers see it, without internal keys.

  Use it where events leave ADK. Removal there is routine, so nothing is
  logged.

  Args:
    metadata: A ``custom_metadata`` value. It is not modified.

  Returns:
    A new dict without internal keys, or ``None`` if nothing is left. A value
    that is not a mapping, such as a test stub, is returned unchanged.
  """
  if not isinstance(metadata, Mapping):
    return metadata
  return _drop_internal_keys(metadata) or None


def internal_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
  """Returns only the keys of ``metadata`` that have the internal prefix."""
  if not isinstance(metadata, Mapping):
    return {}
  return {k: v for k, v in metadata.items() if _is_internal_key(k)}


def mark_restored(event: Event) -> Event:
  """Marks an event restored from outside the session, in place.

  Internal metadata carried by the event is removed first, so a caller cannot
  supply its own.

  Args:
    event: The event about to be appended to a new session.

  Returns:
    The same event.
  """
  event.custom_metadata = {
      **(without_internal_metadata(event.custom_metadata) or {}),
      RESTORED_EVENT_KEY: True,
  }
  return event


def public_event(event: Event) -> Event:
  """Returns the event as callers see it, without internal metadata.

  Args:
    event: A stored event. It is not modified.

  Returns:
    The same event if it has no internal keys. Otherwise a copy whose
    ``custom_metadata`` drops them, or is ``None`` if nothing else is left.
  """
  metadata = event.custom_metadata
  if not isinstance(metadata, Mapping) or not any(
      _is_internal_key(key) for key in metadata
  ):
    return event
  return event.model_copy(update={"custom_metadata": public_metadata(metadata)})


def public_session(session: Session) -> Session:
  """Returns the session as callers see it, without internal metadata.

  Args:
    session: A stored session. It is not modified.

  Returns:
    The same session if no event changes. Otherwise a copy whose events are
    passed through ``public_event``.
  """
  events = [public_event(event) for event in session.events]
  if all(new is old for new, old in zip(events, session.events)):
    return session
  return session.model_copy(update={"events": events})
