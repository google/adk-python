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

import json

from google.genai import types
from opentelemetry.util.types import AnyValue
from pydantic import BaseModel


def safe_json_serialize(obj: object) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Handles Pydantic `BaseModel` instances (common as tool return types) by
  calling `model_dump(mode="json")` before JSON encoding.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or `<not serializable>` if the object
    cannot be serialized.
  """

  def _default(o: object) -> object:
    if isinstance(o, BaseModel):
      return o.model_dump(mode="json")
    return "<not serializable>"

  try:
    return json.dumps(obj, ensure_ascii=False, default=_default)
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
