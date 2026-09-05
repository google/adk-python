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

"""Serialization utilities for JSON-safe data conversion."""

from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python


def _convert_key(key: Any) -> str:
  if isinstance(key, str):
    return key
  if isinstance(key, bool):
    return 'true' if key else 'false'
  if isinstance(key, tuple):
    return ','.join(_convert_key(element) for element in key)
  try:
    converted = to_jsonable_python(key, serialize_unknown=True)
    if isinstance(converted, bool):
      return 'true' if converted else 'false'
    # pydantic-core unwraps enum keys before key inference and converts
    # tuples to lists; comma-join the unwrapped sequence so e.g. an enum
    # with a tuple value renders like the equivalent plain tuple key would.
    if isinstance(converted, (tuple, list)):
      return ','.join(_convert_key(element) for element in converted)
  except Exception:  # pylint: disable=broad-except
    return repr(key)
  return converted if isinstance(converted, str) else str(converted)


def safe_serialize(obj: Any, *, seen_ids: frozenset[int] = frozenset()) -> Any:
  """Recursively sanitizes an object for JSON serialization, falling back to repr()."""
  if isinstance(obj, (dict, list, tuple, set, frozenset)):
    if id(obj) in seen_ids:
      return repr(obj)
    seen_ids = seen_ids | {id(obj)}

  if isinstance(obj, dict):
    try:
      return {
          _convert_key(k): safe_serialize(v, seen_ids=seen_ids)
          for k, v in obj.items()
      }
    except Exception:  # pylint: disable=broad-except
      return repr(obj)
  elif isinstance(obj, (list, tuple, set, frozenset)):
    try:
      return [safe_serialize(v, seen_ids=seen_ids) for v in obj]
    except Exception:  # pylint: disable=broad-except
      return repr(obj)

  try:
    return to_jsonable_python(obj, serialize_unknown=True)
  except Exception:  # pylint: disable=broad-except
    return repr(obj)
