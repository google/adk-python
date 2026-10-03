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

"""Shared serialization helpers used by telemetry modules."""

from __future__ import annotations

from collections.abc import Mapping
import json

from google.genai import types
from opentelemetry.util.types import AnyValue
from pydantic import BaseModel

from ..utils._credential_names import is_credential_arg_name
from ..utils._credential_names import is_credential_type
from ..utils._credential_names import MAX_REDACT_DEPTH
from ..utils._credential_names import MAX_REDACT_NODES
from ..utils._credential_names import REDACTED

_ELIDED = "<elided>"


def _redact_credentials(
    obj: object, depth: int, budget: list[int], active: set[int]
) -> object:
  """Returns ``obj`` as JSON-ready data with credential material masked.

  `model_dump()` renders every field, including the ones `AuthCredential` and
  friends declare `Field(repr=False)`, so a credential that reaches a span
  attribute through this serializer would be exported in the clear even though
  `repr()` masks it. Masking is by declared credential type and by field name,
  from `..utils._credential_names`, so the rule here is the same rule
  `AutoTracingPlugin` applies to captured arguments.

  A masked value keeps its key, so a trace still shows that a credential was
  present without showing what it was.

  The walk is bounded by depth and by a node budget. A subtree it refuses to
  walk is elided rather than emitted, so hitting a bound can never uncover a
  secret.

  Args:
    obj: The value to redact.
    depth: Current nesting depth.
    budget: Single-element remaining-node budget, mutated as containers are
      visited.
    active: Ids of the containers on the path to `obj`, so a cycle is caught
      rather than rebuilt into a finite tree.

  Returns:
    A JSON-ready copy of `obj` with credentials replaced by a marker.

  Raises:
    ValueError: If `obj` contains a reference cycle. `json.dumps` would raise
      the same error on it, and `safe_json_serialize` reports it the same way.
  """
  if is_credential_type(type(obj)):
    return REDACTED
  if isinstance(obj, (str, int, float, bool, type(None))):
    return obj
  if depth >= MAX_REDACT_DEPTH or budget[0] <= 0:
    return _ELIDED
  budget[0] -= 1
  if id(obj) in active:
    raise ValueError("Circular reference detected")
  active.add(id(obj))
  try:
    if isinstance(obj, BaseModel):
      return _redact_credentials(
          obj.model_dump(mode="json"), depth + 1, budget, active
      )
    if isinstance(obj, Mapping):
      return {
          str(key): (
              REDACTED
              if is_credential_arg_name(str(key))
              else _redact_credentials(value, depth + 1, budget, active)
          )
          for key, value in obj.items()
      }
    if isinstance(obj, (list, tuple)):
      return [
          _redact_credentials(item, depth + 1, budget, active) for item in obj
      ]
  finally:
    active.discard(id(obj))
  # Anything else is left to `json.dumps`'s `default` hook below.
  return obj


def safe_json_serialize(obj: object) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Handles Pydantic `BaseModel` instances (common as tool return types) by
  calling `model_dump(mode="json")` before JSON encoding, and masks credential
  material on the way (see `_redact_credentials`).

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or `<not serializable>` if the object
    cannot be serialized.
  """

  def _default(o: object) -> object:
    if is_credential_type(type(o)):
      return REDACTED
    if isinstance(o, BaseModel):
      return _redact_credentials(
          o.model_dump(mode="json"), 0, [MAX_REDACT_NODES], set()
      )
    return "<not serializable>"

  try:
    return json.dumps(
        _redact_credentials(obj, 0, [MAX_REDACT_NODES], set()),
        ensure_ascii=False,
        default=_default,
    )
  except (TypeError, ValueError, OverflowError, RecursionError):
    return "<not serializable>"


def serialize_content(content: types.ContentUnion | None) -> AnyValue:
  """Serialize a `types.ContentUnion` value into an OTel-friendly form.

  - `None` is preserved.
  - Pydantic models are dumped via `model_dump()`.
  - Strings are returned as-is.
  - Lists are recursively serialized.
  - Anything else falls back to `safe_json_serialize`.
  """
  if content is None:
    return None
  if isinstance(content, BaseModel):
    return content.model_dump()
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return [serialize_content(part) for part in content]
  return safe_json_serialize(content)
