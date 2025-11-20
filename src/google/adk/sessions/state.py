# Copyright 2025 Google LLC
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

from __future__ import annotations

from typing import Any


class State:
  """A state dict that maintains the current value and the pending-commit delta."""

  APP_PREFIX = "app:"
  USER_PREFIX = "user:"
  TEMP_PREFIX = "temp:"

  def __init__(self, value: dict[str, Any], delta: dict[str, Any]):
    """
    Args:
      value: The current value of the state dict.
      delta: The delta change to the current value that hasn't been committed.
    """
    self._value = value
    self._delta = delta

  def __getitem__(self, key: str) -> Any:
    """Returns the value of the state dict for the given key."""
    if key in self._delta:
      return self._delta[key]
    return self._value[key]

  def __setitem__(self, key: str, value: Any):
    """Sets the value of the state dict for the given key."""
    # TODO: make new change only store in delta, so that self._value is only
    #   updated at the storage commit time.
    self._value[key] = value
    self._delta[key] = value

  def __contains__(self, key: str) -> bool:
    """Whether the state dict contains the given key."""
    return key in self._value or key in self._delta

  def setdefault(self, key: str, default: Any = None) -> Any:
    """Gets the value of a key, or sets it to a default if the key doesn't exist."""
    if key in self:
      return self[key]
    else:
      self[key] = default
      return default

  def has_delta(self) -> bool:
    """Whether the state has pending delta."""
    return bool(self._delta)

  def get(self, key: str, default: Any = None) -> Any:
    """Returns the value of the state dict for the given key."""
    if key not in self:
      return default
    return self[key]

  def update(self, delta: dict[str, Any]):
    """Updates the state dict with the given delta."""
    self._value.update(delta)
    self._delta.update(delta)

  def to_dict(self) -> dict[str, Any]:
    """Returns the state dict."""
    result = {}
    result.update(self._value)
    result.update(self._delta)
    return result
