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

"""Shared JSON-schema helpers for the OpenAI labs models."""

from __future__ import annotations

from typing import Any


def enforce_strict_openai_schema(schema: dict[str, Any]) -> None:
  """Recursively transforms a JSON schema for strict structured outputs."""
  if not isinstance(schema, dict):
    return
  if '$ref' in schema:
    for key in list(schema.keys()):
      if key != '$ref':
        del schema[key]
    return
  if schema.get('type') == 'object' and 'properties' in schema:
    schema['additionalProperties'] = False
    schema['required'] = sorted(schema['properties'].keys())
  for defn in schema.get('$defs', {}).values():
    enforce_strict_openai_schema(defn)
  for prop in schema.get('properties', {}).values():
    enforce_strict_openai_schema(prop)
  for key in ('anyOf', 'oneOf', 'allOf'):
    for item in schema.get(key, []):
      enforce_strict_openai_schema(item)
  if 'items' in schema and isinstance(schema['items'], dict):
    enforce_strict_openai_schema(schema['items'])
