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

from __future__ import annotations

from urllib.parse import quote
from urllib.parse import unquote

ADK_METADATA_KEY_PREFIX = "adk_"
ADK_CONTEXT_ID_PREFIX = "ADK"
ADK_CONTEXT_ID_SEPARATOR = "/"


def _get_adk_metadata_key(key: str) -> str:
  """Gets the A2A event metadata key for the given key.

  Args:
    key: The metadata key to prefix.

  Returns:
    The prefixed metadata key.

  Raises:
    ValueError: If key is empty or None.
  """
  if not key:
    raise ValueError("Metadata key cannot be empty or None")
  return f"{ADK_METADATA_KEY_PREFIX}{key}"


def _to_a2a_context_id(app_name: str, user_id: str, session_id: str) -> str:
  """Converts app name, user id and session id to an A2A context id.

  Args:
    app_name: The app name.
    user_id: The user id.
    session_id: The session id.

  Returns:
    The A2A context id.

  Raises:
    ValueError: If any of the input parameters are empty or None.
  """
  if not all([app_name, user_id, session_id]):
    raise ValueError(
        "All parameters (app_name, user_id, session_id) must be non-empty"
    )
  return ADK_CONTEXT_ID_SEPARATOR.join([
      ADK_CONTEXT_ID_PREFIX,
      quote(app_name, safe=""),
      quote(user_id, safe=""),
      quote(session_id, safe=""),
  ])


def _from_a2a_context_id(
    context_id: str | None,
) -> tuple[str, str, str] | tuple[None, None, None]:
  """Converts an A2A context id to app name, user id and session id.

  Reverses ``_to_a2a_context_id``. The context id is expected to be
  ``ADK/<app_name>/<user_id>/<session_id>`` with each field percent-encoded, so
  that ids containing the separator still round-trip. Returns (None, None, None)
  if context_id is None or does not match this format.

  Args:
    context_id: The A2A context id.

  Returns:
    The app name, user id and session id, or (None, None, None) if invalid.
  """
  if not context_id:
    return None, None, None

  parts = context_id.split(ADK_CONTEXT_ID_SEPARATOR)
  if len(parts) != 4:
    return None, None, None

  prefix, app_name, user_id, session_id = parts
  if prefix != ADK_CONTEXT_ID_PREFIX:
    return None, None, None

  app_name = unquote(app_name)
  user_id = unquote(user_id)
  session_id = unquote(session_id)
  if app_name and user_id and session_id:
    return app_name, user_id, session_id

  return None, None, None
