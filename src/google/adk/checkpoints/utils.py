"""Utilities for checkpoint management (diff, serialize, etc.)."""

from __future__ import annotations

import copy
import json
import sys
from typing import Any
from typing import Dict
from typing import Set

from pydantic import BaseModel


class PydanticJSONEncoder(json.JSONEncoder):
  """JSON encoder that automatically handles Pydantic models.

  This encoder allows json.dumps() to work transparently with Pydantic models
  without requiring special serialization methods.

  Example:
      ```python
      import json
      state_json = json.dumps(data, cls=PydanticJSONEncoder, indent=2)
      ```
  """

  def default(self, obj: Any) -> Any:
    """Convert Pydantic models to dicts automatically."""
    if isinstance(obj, BaseModel):
      return obj.model_dump()
    return super().default(obj)


def compute_state_diff(
    state1: Dict[str, Any],
    state2: Dict[str, Any],
) -> Dict[str, Any]:
  """Compute diff between two state dictionaries.

  Args:
      state1: First state (older)
      state2: Second state (newer)

  Returns:
      Dict with:
      - added: Keys added in state2
      - removed: Keys removed in state2
      - changed: Keys that exist in both but with different values
      - unchanged: Keys with same values in both states

  Example:
      ```python
      state1 = {"a": 1, "b": 2, "c": 3}
      state2 = {"a": 1, "b": 99, "d": 4}

      diff = compute_state_diff(state1, state2)
      # {
      #   "added": {"d": 4},
      #   "removed": {"c": 3},
      #   "changed": {"b": {"old": 2, "new": 99}},
      #   "unchanged": {"a": 1}
      # }
      ```
  """
  keys1: Set[str] = set(state1.keys())
  keys2: Set[str] = set(state2.keys())

  added_keys = keys2 - keys1
  removed_keys = keys1 - keys2
  common_keys = keys1 & keys2

  result: Dict[str, Any] = {
      "added": {},
      "removed": {},
      "changed": {},
      "unchanged": {},
  }

  # Added keys
  for key in added_keys:
    result["added"][key] = state2[key]

  # Removed keys
  for key in removed_keys:
    result["removed"][key] = state1[key]

  # Common keys - check if changed
  for key in common_keys:
    if state1[key] != state2[key]:
      result["changed"][key] = {
          "old": state1[key],
          "new": state2[key],
      }
    else:
      result["unchanged"][key] = state1[key]

  return result


def serialize_state_for_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
  """Serialize state dict for checkpoint storage.

  Currently a passthrough, but can be extended to handle:
  - Large object compression
  - Binary data encoding
  - Type preservation
  - Circular reference handling

  Args:
      state: State dictionary to serialize

  Returns:
      Serialized state (currently just a copy)
  """
  return copy.deepcopy(state)


def deserialize_checkpoint_state(data: Dict[str, Any]) -> Dict[str, Any]:
  """Deserialize checkpoint data back to state dict.

  Currently a passthrough, but can be extended to handle:
  - Decompression
  - Binary data decoding
  - Type reconstruction

  Args:
      data: Serialized checkpoint data

  Returns:
      Deserialized state dict
  """
  return copy.deepcopy(data)


def compute_checkpoint_summary(state: Dict[str, Any]) -> Dict[str, Any]:
  """Compute summary statistics for a checkpoint state.

  Useful for checkpoint listings and debugging.

  Args:
      state: State dictionary

  Returns:
      Summary dict with:
      - total_keys: Number of keys in state
      - key_types: Count of values by type
      - total_size_bytes: Approximate size in bytes

  Example:
      ```python
      state = {"a": 1, "b": "hello", "c": [1, 2, 3]}
      summary = compute_checkpoint_summary(state)
      # {
      #   "total_keys": 3,
      #   "key_types": {"int": 1, "str": 1, "list": 1},
      #   "total_size_bytes": 150
      # }
      ```
  """
  total_keys = len(state)
  key_types: Dict[str, int] = {}
  total_size = 0

  for key, value in state.items():
    # Count types
    type_name = type(value).__name__
    key_types[type_name] = key_types.get(type_name, 0) + 1

    # Estimate size
    try:
      total_size += sys.getsizeof(value)
    except TypeError:
      # Some objects don't support getsizeof
      total_size += sys.getsizeof(str(value))

  return {
      "total_keys": total_keys,
      "key_types": key_types,
      "total_size_bytes": total_size,
  }
