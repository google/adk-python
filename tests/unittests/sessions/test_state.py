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

"""Unit tests for State class in google.adk.sessions.state."""

from __future__ import annotations

from google.adk.sessions.state import State
import pytest


def test_delitem_removes_key_from_value():
  """Deleting a key removes it from the live view and makes lookup fail."""
  state = State(value={"key1": "value1", "key2": "value2"}, delta={})

  del state["key1"]

  assert "key1" not in state
  assert state.get("key1") is None
  with pytest.raises(KeyError):
    _ = state["key1"]


def test_delitem_records_tombstone_in_delta():
  """Deleting a key records a None tombstone in the pending delta."""
  state = State(value={"key1": "value1"}, delta={})

  del state["key1"]

  assert state.has_delta()
  assert state._delta["key1"] is None


def test_delitem_missing_key_raises_key_error():
  """Deleting a key that is not in state raises KeyError."""
  state = State(value={}, delta={})

  with pytest.raises(KeyError):
    del state["nonexistent"]

  assert not state.has_delta()


def test_to_dict_omits_tombstoned_keys():
  """to_dict reflects live state with tombstoned keys excluded."""
  state = State(value={"key1": "value1", "key2": "value2"}, delta={})

  del state["key1"]

  assert state.to_dict() == {"key2": "value2"}


def test_pop_removes_key_and_returns_value():
  """Popping a key returns its value, removes it, and records a tombstone."""
  state = State(value={"key1": "value1"}, delta={})

  val = state.pop("key1")

  assert val == "value1"
  assert "key1" not in state
  assert state.has_delta()
  assert state._delta["key1"] is None


def test_pop_default_when_key_missing():
  """Popping a missing key with a default returns the default."""
  state = State(value={}, delta={})

  val = state.pop("missing", "default_val")

  assert val == "default_val"


def test_pop_missing_key_raises_key_error():
  """Popping a missing key without default raises KeyError."""
  state = State(value={}, delta={})

  with pytest.raises(KeyError):
    state.pop("missing")
