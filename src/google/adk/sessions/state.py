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

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from pydantic import BaseModel


class StateSchemaError(TypeError):
  """Raised when a state mutation violates the declared state_schema."""


def _validate_state_entry(
    schema: type[BaseModel],
    key: str,
    value: Any,
) -> None:
  """Validates a single state key-value pair against a Pydantic schema.

  Raises StateSchemaError if the key is not in the schema or the value
  does not match the field's type annotation.  Prefixed keys (any key
  containing ``:``) bypass validation.
  """
  if ":" in key:
    return

  fields = schema.model_fields
  if key not in fields:
    raise StateSchemaError(
        f"Key '{key}' is not declared in state schema "
        f"'{schema.__name__}'. Declared fields: {sorted(fields.keys())}"
    )

  from pydantic import TypeAdapter
  from pydantic import ValidationError as PydanticValidationError

  try:
    TypeAdapter(fields[key].annotation).validate_python(value)
  except PydanticValidationError as e:
    raise StateSchemaError(
        f"Value for '{key}' does not match type "
        f"'{fields[key].annotation}' in '{schema.__name__}': {e}"
    ) from e


class State:
  """A state dict that maintains the current value and the pending-commit delta."""

  APP_PREFIX = "app:"
  USER_PREFIX = "user:"
  TEMP_PREFIX = "temp:"

  def __init__(
      self,
      value: dict[str, Any],
      delta: dict[str, Any],
      schema: type[BaseModel] | None = None,
  ):
    """
    Args:
      value: The current value of the state dict.
      delta: The delta change to the current value that hasn't been committed.
      schema: Optional Pydantic model declaring the expected state keys and
        types.  When set, mutations are validated against this schema.
    """
    self._value = value
    self._delta = delta
    self._schema = schema

  def __getitem__(self, key: str) -> Any:
    """Returns the value of the state dict for the given key."""
    if key in self._delta:
      return self._delta[key]
    return self._value[key]

  def __setitem__(self, key: str, value: Any) -> None:
    """Sets the value of the state dict for the given key."""
    if self._schema is not None:
      _validate_state_entry(self._schema, key, value)
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

  def update(self, delta: dict[str, Any]) -> None:
    """Updates the state dict with the given delta."""
    if self._schema is not None:
      for key, value in delta.items():
        _validate_state_entry(self._schema, key, value)
    self._value.update(delta)
    self._delta.update(delta)

  def to_dict(self) -> dict[str, Any]:
    """Returns the state dict."""
    result = {}
    result.update(self._value)
    result.update(self._delta)
    return result
