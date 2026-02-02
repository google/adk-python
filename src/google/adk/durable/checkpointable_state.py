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

"""Abstract base class for checkpointable agent state."""

from __future__ import annotations

import abc
from typing import Any
from typing import Dict

from pydantic import BaseModel
from pydantic import ConfigDict


class CheckpointableAgentState(BaseModel, abc.ABC):
  """Abstract base class for agent state that can be checkpointed.

  Agents that need to preserve custom state across checkpoints should inherit
  from this class and implement the serialization methods.

  Example:
    ```python
    class MyAgentState(CheckpointableAgentState):
        counter: int = 0
        processed_items: list[str] = []

        def to_checkpoint_dict(self) -> dict[str, Any]:
            return {
                "counter": self.counter,
                "processed_items": self.processed_items,
            }

        @classmethod
        def from_checkpoint_dict(cls, data: dict[str, Any]) -> "MyAgentState":
            return cls(
                counter=data.get("counter", 0),
                processed_items=data.get("processed_items", []),
            )
    ```
  """

  model_config = ConfigDict(
      extra="allow",
  )

  @abc.abstractmethod
  def to_checkpoint_dict(self) -> Dict[str, Any]:
    """Serialize the state to a dictionary for checkpointing.

    Returns:
      A dictionary containing all state that should be persisted.
      The dictionary must be JSON-serializable.
    """

  @classmethod
  @abc.abstractmethod
  def from_checkpoint_dict(
      cls, data: Dict[str, Any]
  ) -> "CheckpointableAgentState":
    """Deserialize the state from a checkpoint dictionary.

    Args:
      data: The dictionary previously returned by to_checkpoint_dict().

    Returns:
      A new instance of the state class with restored values.
    """


class SimpleCheckpointableState(CheckpointableAgentState):
  """A simple implementation of CheckpointableAgentState using a dict.

  This class provides a basic implementation that stores arbitrary key-value
  pairs. Use this when you don't need custom serialization logic.

  Example:
    ```python
    state = SimpleCheckpointableState()
    state.data["counter"] = 5
    state.data["results"] = ["a", "b", "c"]

    # Checkpoint
    checkpoint = state.to_checkpoint_dict()

    # Restore
    restored = SimpleCheckpointableState.from_checkpoint_dict(checkpoint)
    assert restored.data["counter"] == 5
    ```
  """

  data: Dict[str, Any] = {}

  def to_checkpoint_dict(self) -> Dict[str, Any]:
    """Serialize the state to a dictionary."""
    return {"data": self.data.copy()}

  @classmethod
  def from_checkpoint_dict(
      cls, data: Dict[str, Any]
  ) -> "SimpleCheckpointableState":
    """Deserialize the state from a checkpoint dictionary."""
    return cls(data=data.get("data", {}))
