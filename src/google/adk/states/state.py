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

from typing import Any
from pydantic_core import core_schema
from pydantic.json_schema import JsonSchemaValue
from pydantic._internal._schema_generation_shared import GetCoreSchemaHandler, GetJsonSchemaHandler


class StateValue:
  """A wrapper for state values of any JSON-serializable type."""

  def __init__(self, value: Any, mutable: bool = True):
    self._value = value
    self._mutable = mutable

  @property
  def value(self):
    return self._value

  @property
  def mutable(self):
    return self._mutable

  def __repr__(self):
    return repr(self._value)

  def __eq__(self, other):
    if isinstance(other, StateValue):
      return self._value == other._value
    return self._value == other

  def __str__(self):
    return str(self._value)

  # Pydantic v2: full schema override
  @classmethod
  def __get_pydantic_core_schema__(
      cls,
      _source_type: Any,
      _handler: GetCoreSchemaHandler,
  ) -> core_schema.CoreSchema:
    def _serialize(instance: Any, info: core_schema.SerializationInfo) -> Any:
      if info.mode == "json":
        return instance.value  # serialize just the wrapped value
      return instance

    return core_schema.json_or_python_schema(
        python_schema=core_schema.any_schema(),
        json_schema=core_schema.any_schema(),
        serialization=core_schema.plain_serializer_function_ser_schema(
            _serialize, info_arg=True
        ),
    )

  @classmethod
  def __get_pydantic_json_schema__(
      cls, _core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
  ) -> JsonSchemaValue:
    return handler(core_schema.any_schema())


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
    for k, v in value.items():
      value[k] = self._wrap(v)

    for k, v in delta.items():
      delta[k] = self._wrap(v)

    self._value = value
    self._delta = delta

  def _wrap(self, v: Any) -> StateValue:
    """Returns the value wrapped in StateValue if not already."""
    return v if isinstance(v, StateValue) else StateValue(v)

  def __getitem__(self, key: str) -> Any:
    """Returns the value of the state dict for the given key."""
    if key in self._delta:
      ret = self._delta[key]
    else:
      ret = self._value[key]
    return ret.value if isinstance(ret, StateValue) else ret

  def __setitem__(self, key: str, value: Any, force: bool = False):
    """
    Sets the value of the state dict for the given key.

    Args:
        key (str): The key to set.
        value (Any): The value to associate with the key.
        force (bool): If True, override immutability.
    """
    # TODO: make new change only store in delta, so that self._value is only
    #   updated at the storage commit time.

    existing_val = self._delta.get(key, self._value.get(key))

    # Keep current mutability if key already exists and new value isn't wrapped
    if existing_val and not isinstance(value, StateValue):
      new_val = StateValue(value, mutable=existing_val.mutable)
    else:
      new_val = self._wrap(value)

    if (
        not force
        and existing_val
        and isinstance(existing_val, StateValue)
        and not existing_val.mutable
    ):
      print(f"Cannot modify immutable key: {key}")
      return

    self._value[key] = new_val
    self._delta[key] = new_val

  def set_immutable(self, key: str, value: Any):
    """Sets the value of the state dict for the given key even if immutable."""
    self.__setitem__(key, value, force=True)

  def __contains__(self, key: str) -> bool:
    """Whether the state dict contains the given key."""
    return key in self._value or key in self._delta

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
    for key, value in delta.items():
      self[key] = value

  def to_dict(self) -> dict[str, Any]:
    """Returns the state dict."""
    result = {}
    for d in [self._value, self._delta]:
      for k, v in d.items():
        result[k] = v.value if isinstance(v, StateValue) else v
    return result
