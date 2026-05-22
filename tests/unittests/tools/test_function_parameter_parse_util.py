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
#

from __future__ import annotations

from typing import Any

from google.adk.tools import _function_parameter_parse_util


def test_normalize_strips_prefixItems() -> None:
  schema: dict[str, Any] = {
      "type": "array",
      "prefixItems": [{"type": "string"}, {"type": "number"}],
      "minItems": 2,
      "maxItems": 2,
      "unevaluatedItems": False,
  }
  normalized = (
      _function_parameter_parse_util._normalize_tuple_schema_for_genai_schema(
          schema
      )
  )
  assert "prefixItems" not in normalized
  assert "unevaluatedItems" not in normalized
  assert normalized["items"] == {
      "anyOf": [{"type": "string"}, {"type": "number"}]
  }


def test_normalize_strips_unevaluatedItems() -> None:
  schema: dict[str, Any] = {
      "type": "object",
      "properties": {
          "field1": {"type": "string"},
      },
      "unevaluatedItems": False,
  }
  normalized = (
      _function_parameter_parse_util._normalize_tuple_schema_for_genai_schema(
          schema
      )
  )
  assert "unevaluatedItems" not in normalized
  assert normalized["properties"] == {"field1": {"type": "string"}}


def test_normalize_handles_items_false() -> None:
  schema: dict[str, Any] = {
      "type": "array",
      "prefixItems": [{"type": "string"}],
      "items": False,
  }
  normalized = (
      _function_parameter_parse_util._normalize_tuple_schema_for_genai_schema(
          schema
      )
  )
  assert "items" in normalized
  assert normalized["items"] == {"type": "string"}
  assert normalized.get("items") is not False


def test_normalize_handles_nested_schemas() -> None:
  schema: dict[str, Any] = {
      "type": "object",
      "properties": {
          "field1": {
              "type": "array",
              "prefixItems": [{"type": "string"}],
              "unevaluatedItems": False,
          }
      },
  }
  normalized = (
      _function_parameter_parse_util._normalize_tuple_schema_for_genai_schema(
          schema
      )
  )
  assert "unevaluatedItems" not in normalized["properties"]["field1"]
  assert "prefixItems" not in normalized["properties"]["field1"]
  assert normalized["properties"]["field1"]["items"] == {"type": "string"}
