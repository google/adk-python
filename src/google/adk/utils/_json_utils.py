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

import json
from typing import Any


def safe_json_loads(
    text: str | bytes | bytearray, context: str | None = None
) -> Any:
  """Parses a JSON string or bytes, raising ValueError on malformed input.

  Wraps ``json.loads`` with a consistent error type so callers don't need
  to handle ``json.JSONDecodeError`` directly. Prefer this helper for callers
  that want a uniform ``ValueError``. Some callsites (e.g. ``_schema_utils.py``)
  intentionally keep ``json.loads`` to distinguish decode failure from
  validation failure.

  Args:
    text: The JSON string, bytes, or bytearray to parse.
    context: Optional human-readable label for the source of ``text``
      (e.g. ``"session state"``), included in the error message to aid
      debugging.

  Returns:
    The parsed Python object.

  Raises:
    ValueError: If ``text`` is not valid JSON.
  """
  try:
    return json.loads(text)
  except json.JSONDecodeError as exc:
    suffix = f' in {context}' if context else ''
    raise ValueError(f'Invalid JSON{suffix}: {exc}') from exc
