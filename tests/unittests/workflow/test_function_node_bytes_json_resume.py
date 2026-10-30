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

"""Regression: bytes node_input after JSON Event round-trip (Fixes #7852)."""

from __future__ import annotations

import base64

from google.adk.events.event import Event
from google.adk.workflow._function_node import FunctionNode


def test_function_node_coerces_base64_str_to_bytes_after_event_json_roundtrip():
  """Database/Sqlite session stores Event.output as JSON; bytes become base64.

  Event.output is typed Any, so val_json_bytes does not restore bytes on load.
  FunctionNode must base64-decode when the parameter annotation is bytes.
  """

  def consume(node_input: bytes):
    return node_input

  node = FunctionNode(func=consume, name="consume")
  original = b"\x00\x01"
  event = Event(author="produce", invocation_id="inv", output=original)
  restored = Event.model_validate(event.model_dump(mode="json", exclude_none=True))
  assert isinstance(restored.output, str)
  assert restored.output == base64.b64encode(original).decode("ascii")

  coerced = node._coerce_param("node_input", restored.output, bytes)
  assert coerced == original
  assert isinstance(coerced, bytes)


def test_function_node_bytes_coerce_falls_back_for_non_base64_str():
  def consume(node_input: bytes):
    return node_input

  node = FunctionNode(func=consume, name="consume")
  # Not valid base64 (wrong padding / alphabet) → TypeAdapter encodes utf-8
  out = node._coerce_param("node_input", "hello", bytes)
  assert out == b"hello"
