"""Tests for checkpoint utilities."""

import json

from google.adk.checkpoints.utils import compute_checkpoint_summary
from google.adk.checkpoints.utils import compute_state_diff
from google.adk.checkpoints.utils import deserialize_checkpoint_state
from google.adk.checkpoints.utils import PydanticJSONEncoder
from google.adk.checkpoints.utils import serialize_state_for_checkpoint
from pydantic import BaseModel
import pytest


def test_compute_state_diff():
  """Test state diff computation."""
  state1 = {"a": 1, "b": 2, "c": 3}
  state2 = {"a": 1, "b": 99, "d": 4}

  diff = compute_state_diff(state1, state2)

  # Check added keys
  assert "d" in diff["added"]
  assert diff["added"]["d"] == 4

  # Check removed keys
  assert "c" in diff["removed"]
  assert diff["removed"]["c"] == 3

  # Check changed keys
  assert "b" in diff["changed"]
  assert diff["changed"]["b"]["old"] == 2
  assert diff["changed"]["b"]["new"] == 99

  # Check unchanged keys
  assert "a" in diff["unchanged"]
  assert diff["unchanged"]["a"] == 1


def test_compute_state_diff_empty_states():
  """Test diff with empty states."""
  diff = compute_state_diff({}, {})
  assert len(diff["added"]) == 0
  assert len(diff["removed"]) == 0
  assert len(diff["changed"]) == 0
  assert len(diff["unchanged"]) == 0


def test_compute_state_diff_one_empty():
  """Test diff with one empty state."""
  state1 = {}
  state2 = {"a": 1, "b": 2}

  diff = compute_state_diff(state1, state2)
  assert "a" in diff["added"]
  assert "b" in diff["added"]
  assert len(diff["removed"]) == 0
  assert len(diff["changed"]) == 0


def test_compute_state_diff_nested_values():
  """Test diff with nested values."""
  state1 = {"config": {"debug": True, "level": 1}}
  state2 = {"config": {"debug": False, "level": 1}}

  diff = compute_state_diff(state1, state2)

  # Nested dicts are compared by equality
  assert "config" in diff["changed"]
  assert diff["changed"]["config"]["old"] == {"debug": True, "level": 1}
  assert diff["changed"]["config"]["new"] == {"debug": False, "level": 1}


def test_serialize_state_for_checkpoint():
  """Test state serialization (currently just deep copy)."""
  state = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}

  serialized = serialize_state_for_checkpoint(state)

  # Should be equal
  assert serialized == state

  # Should be a deep copy (not same object)
  assert serialized is not state
  assert serialized["b"] is not state["b"]
  assert serialized["c"] is not state["c"]


def test_deserialize_checkpoint_state():
  """Test state deserialization (currently just deep copy)."""
  data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}

  deserialized = deserialize_checkpoint_state(data)

  # Should be equal
  assert deserialized == data

  # Should be a deep copy
  assert deserialized is not data
  assert deserialized["b"] is not data["b"]
  assert deserialized["c"] is not data["c"]


def test_compute_checkpoint_summary():
  """Test checkpoint summary computation."""
  state = {
      "int_val": 42,
      "str_val": "hello",
      "list_val": [1, 2, 3],
      "dict_val": {"nested": "value"},
      "bool_val": True,
  }

  summary = compute_checkpoint_summary(state)

  # Check total keys
  assert summary["total_keys"] == 5

  # Check key types
  assert (
      summary["key_types"]["int"] >= 1
  )  # int_val and bool_val (bool is int subclass in Python)
  assert summary["key_types"]["str"] == 1
  assert summary["key_types"]["list"] == 1
  assert summary["key_types"]["dict"] == 1

  # Check total size (should be > 0)
  assert summary["total_size_bytes"] > 0


def test_compute_checkpoint_summary_empty():
  """Test summary with empty state."""
  summary = compute_checkpoint_summary({})

  assert summary["total_keys"] == 0
  assert len(summary["key_types"]) == 0
  assert summary["total_size_bytes"] == 0


def test_roundtrip_serialization():
  """Test serialize -> deserialize roundtrip."""
  original_state = {
      "user_id": "12345",
      "session_data": {"count": 42, "items": ["a", "b", "c"]},
      "metadata": {"timestamp": 1234567890, "version": "1.0"},
  }

  # Serialize
  serialized = serialize_state_for_checkpoint(original_state)

  # Deserialize
  restored = deserialize_checkpoint_state(serialized)

  # Should be equal to original
  assert restored == original_state

  # Should be independent copies
  assert restored is not original_state
  restored["session_data"]["count"] = 999
  assert original_state["session_data"]["count"] == 42  # Original unchanged


def test_compute_checkpoint_summary_getsizeof_fallback():
  """compute_checkpoint_summary falls back to str() when getsizeof raises."""

  class NoSizeOf:
    """Object whose __sizeof__ raises TypeError."""

    def __sizeof__(self):
      raise TypeError("unsupported")

  obj = NoSizeOf()
  state = {"tricky": obj}
  # Should not raise; falls back to sys.getsizeof(str(obj))
  summary = compute_checkpoint_summary(state)
  assert summary["total_keys"] == 1
  assert summary["total_size_bytes"] > 0


def test_pydantic_json_encoder_serializes_model():
  """PydanticJSONEncoder converts BaseModel instances to dicts."""

  class UserProfile(BaseModel):
    name: str
    age: int

  data = {"user": UserProfile(name="Alice", age=30), "plain": "text"}
  result = json.loads(json.dumps(data, cls=PydanticJSONEncoder))

  assert result["user"] == {"name": "Alice", "age": 30}
  assert result["plain"] == "text"


def test_pydantic_json_encoder_falls_back_for_non_model():
  """PydanticJSONEncoder raises TypeError for non-serializable non-model objects."""
  with pytest.raises(TypeError):
    json.dumps({"bad": object()}, cls=PydanticJSONEncoder)
