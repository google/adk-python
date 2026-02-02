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

"""Tests for CheckpointableAgentState."""

from typing import Any
from typing import Dict

from google.adk.durable.checkpointable_state import CheckpointableAgentState
from google.adk.durable.checkpointable_state import SimpleCheckpointableState
import pytest


class TestSimpleCheckpointableState:
  """Tests for SimpleCheckpointableState."""

  def test_default_state(self):
    """Test default state initialization."""
    state = SimpleCheckpointableState()
    assert state.data == {}

  def test_state_with_data(self):
    """Test state with initial data."""
    state = SimpleCheckpointableState(data={"key": "value", "count": 5})
    assert state.data["key"] == "value"
    assert state.data["count"] == 5

  def test_to_checkpoint_dict(self):
    """Test serialization to checkpoint dict."""
    state = SimpleCheckpointableState(data={"items": [1, 2, 3], "name": "test"})
    checkpoint = state.to_checkpoint_dict()

    assert checkpoint == {"data": {"items": [1, 2, 3], "name": "test"}}

  def test_from_checkpoint_dict(self):
    """Test deserialization from checkpoint dict."""
    checkpoint = {"data": {"counter": 10, "results": ["a", "b"]}}
    state = SimpleCheckpointableState.from_checkpoint_dict(checkpoint)

    assert state.data["counter"] == 10
    assert state.data["results"] == ["a", "b"]

  def test_roundtrip(self):
    """Test roundtrip serialization/deserialization."""
    original = SimpleCheckpointableState(
        data={
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3],
            "string": "hello",
        }
    )

    checkpoint = original.to_checkpoint_dict()
    restored = SimpleCheckpointableState.from_checkpoint_dict(checkpoint)

    assert restored.data == original.data

  def test_empty_checkpoint_dict(self):
    """Test deserialization from empty checkpoint dict."""
    state = SimpleCheckpointableState.from_checkpoint_dict({})
    assert state.data == {}


class CustomState(CheckpointableAgentState):
  """Custom state implementation for testing."""

  counter: int = 0
  items: list[str] = []
  metadata: dict[str, Any] = {}

  def __init__(self, **data):
    super().__init__(**data)
    if "items" not in data:
      self.items = []
    if "metadata" not in data:
      self.metadata = {}

  def to_checkpoint_dict(self) -> Dict[str, Any]:
    return {
        "counter": self.counter,
        "items": self.items.copy(),
        "metadata": self.metadata.copy(),
    }

  @classmethod
  def from_checkpoint_dict(cls, data: Dict[str, Any]) -> "CustomState":
    return cls(
        counter=data.get("counter", 0),
        items=data.get("items", []),
        metadata=data.get("metadata", {}),
    )


class TestCustomCheckpointableState:
  """Tests for custom CheckpointableAgentState implementations."""

  def test_custom_state_default(self):
    """Test custom state with default values."""
    state = CustomState()
    assert state.counter == 0
    assert state.items == []
    assert state.metadata == {}

  def test_custom_state_with_values(self):
    """Test custom state with initial values."""
    state = CustomState(
        counter=5,
        items=["a", "b"],
        metadata={"key": "value"},
    )
    assert state.counter == 5
    assert state.items == ["a", "b"]
    assert state.metadata == {"key": "value"}

  def test_custom_state_to_checkpoint(self):
    """Test custom state serialization."""
    state = CustomState(counter=10, items=["x", "y", "z"])
    checkpoint = state.to_checkpoint_dict()

    assert checkpoint["counter"] == 10
    assert checkpoint["items"] == ["x", "y", "z"]
    assert checkpoint["metadata"] == {}

  def test_custom_state_from_checkpoint(self):
    """Test custom state deserialization."""
    checkpoint = {
        "counter": 42,
        "items": ["item1", "item2"],
        "metadata": {"created_by": "test"},
    }
    state = CustomState.from_checkpoint_dict(checkpoint)

    assert state.counter == 42
    assert state.items == ["item1", "item2"]
    assert state.metadata == {"created_by": "test"}

  def test_custom_state_roundtrip(self):
    """Test custom state roundtrip."""
    original = CustomState(
        counter=100,
        items=["first", "second", "third"],
        metadata={"version": 1, "tags": ["test", "demo"]},
    )

    checkpoint = original.to_checkpoint_dict()
    restored = CustomState.from_checkpoint_dict(checkpoint)

    assert restored.counter == original.counter
    assert restored.items == original.items
    assert restored.metadata == original.metadata

  def test_custom_state_isolation(self):
    """Test that checkpoint data is isolated from original."""
    state = CustomState(items=["a", "b"])
    checkpoint = state.to_checkpoint_dict()

    # Modify checkpoint
    checkpoint["items"].append("c")

    # Original should be unchanged
    assert state.items == ["a", "b"]
