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
"""Utility functions for session service."""

from __future__ import annotations

import logging
from typing import Any
from typing import cast
from typing import Iterable
from typing import TypeVar

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from ..events.event import Event
from ..events.event_actions import _make_json_serializable
from .state import State

logger = logging.getLogger("google_adk." + __name__)

M = TypeVar("M", bound=BaseModel)

_reported_lossy_storage: set[str] = set()


def decode_model(data: object | None, model_cls: type[M]) -> M | None:
  """Decodes a pydantic model object from a JSON dictionary."""
  # Guard against primitive non-dict values (e.g. a legacy/corrupted "null" string
  # persisted in place of SQL NULL). Passing those to model_validate would
  # raise a ValidationError and break session replay in get_session().
  # We allow dicts and other objects (like PydanticNamespace in tests).
  if data is None or isinstance(
      data, (str, int, float, bool, list, set, tuple, bytes)
  ):
    return None
  return model_cls.model_validate(data)


def extract_state_delta(
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
  """Extracts app, user, and session state deltas from a state dictionary."""
  deltas: dict[str, dict[str, Any]] = {
      "app": {},
      "user": {},
      "session": {},
  }
  if state:
    for key in state.keys():
      if key.startswith(State.APP_PREFIX):
        deltas["app"][key.removeprefix(State.APP_PREFIX)] = state[key]
      elif key.startswith(State.USER_PREFIX):
        deltas["user"][key.removeprefix(State.USER_PREFIX)] = state[key]
      elif not key.startswith(State.TEMP_PREFIX):
        deltas["session"][key] = state[key]
  return deltas


def make_json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
  """Coerces a state dictionary into a JSON-serializable form.

  Rich types such as datetimes and Pydantic models are serialized faithfully.
  Only when that fails is the dictionary re-serialized with the offending values
  (e.g. callables) replaced by their string representation, and a warning
  logged, so that a lossy write is diagnosable rather than silent.
  """
  try:
    return cast(dict[str, Any], to_jsonable_python(state))
  except Exception:  # pylint: disable=broad-except
    logger.warning(
        "Failed to serialize session state; some values are not"
        " JSON-serializable (e.g. callables) and will be replaced with a"
        " string representation in the persisted state.",
        exc_info=True,
    )
    return cast(dict[str, Any], _make_json_serializable(state))


def event_fields_not_stored(stored_fields: Iterable[str]) -> list[str]:
  """Returns the event fields lost by a backend that keeps only `stored_fields`.

  A backend that writes an event as a fixed set of named fields has nowhere to
  put a field added to `Event` afterwards. Callers pass what their layout
  keeps, which is frozen, and the loss is derived from `Event`, so it stays
  accurate as `Event` grows rather than going stale.

  Args:
    stored_fields: Names of the event fields the backend's layout can hold.
  """
  return sorted(set(Event.model_fields) - set(stored_fields))


def warn_event_fields_not_stored(
    stored_fields: Iterable[str], *, cause: str, remedy: str
) -> None:
  """Reports, once per cause, the event fields a backend cannot store.

  Nothing downstream can tell a field the backend dropped from one that was
  never set, so an agent reading one back misbehaves with no error anywhere.

  Args:
    stored_fields: Names of the event fields the backend's layout can hold.
    cause: Why the backend is storing an event this way, as a sentence opener.
    remedy: What the reader can do to stop losing the fields.
  """
  dropped = event_fields_not_stored(stored_fields)
  if not dropped or cause in _reported_lossy_storage:
    return
  _reported_lossy_storage.add(cause)
  logger.warning(
      "%s. %d event fields are dropped on write and read back as unset: %s. %s",
      cause,
      len(dropped),
      ", ".join(dropped),
      remedy,
  )


def extract_json_safe_state_delta(
    state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
  """Extracts state deltas coerced into a JSON-serializable form.

  Services that persist state to a JSON column must use this rather than
  `extract_state_delta`: a value that cannot be serialized is replaced with its
  string representation instead of failing the whole write.
  """
  return cast(
      dict[str, dict[str, Any]],
      make_json_safe_state(extract_state_delta(state)),
  )
