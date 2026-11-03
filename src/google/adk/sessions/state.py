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

from collections.abc import Iterator
import copy
import functools
import logging
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from pydantic import BaseModel
  from pydantic import TypeAdapter

logger = logging.getLogger("google_adk." + __name__)


class StateSchemaError(TypeError):
  """Raised when a state mutation violates the declared state_schema."""


@functools.lru_cache(maxsize=256)
def _get_type_adapter(schema: type[BaseModel], key: str) -> TypeAdapter[Any]:
  """Returns a cached adapter preserving a field's annotation and metadata."""
  from pydantic import TypeAdapter

  return TypeAdapter(schema.model_fields[key].rebuild_annotation())


def _validate_state_entry(
    schema: type[BaseModel],
    key: str,
    value: Any,
) -> None:
  """Validates a single state key-value pair against a Pydantic schema.

  Raises StateSchemaError if the key is not in the schema or the value
  does not match the field's type annotation or constraints. Prefixed keys
  (any key containing ``:``) bypass validation: besides the app:, user: and
  temp: scopes, ADK keeps its own state under ``<owner>:<key>`` names.
  """
  if ":" in key:
    return

  fields = schema.model_fields
  if key not in fields:
    raise StateSchemaError(
        f"Key '{key}' is not declared in state schema "
        f"'{schema.__name__}'. Declared fields: {sorted(fields.keys())}"
    )

  from pydantic import ValidationError as PydanticValidationError

  try:
    _get_type_adapter(schema, key).validate_python(value)
  except PydanticValidationError as e:
    raise StateSchemaError(
        f"Value for '{key}' does not satisfy field '{key}' "
        f"({fields[key].rebuild_annotation()}) in '{schema.__name__}': {e}"
    ) from e


def _equals_its_json_form(value: Any) -> bool:
  """Whether value converts to JSON data equal to it."""
  from pydantic_core import to_jsonable_python

  try:
    return bool(to_jsonable_python(value, inf_nan_mode="null") == value)
  except Exception:  # pylint: disable=broad-exception-caught
    return False


class State:
  """A state dict that maintains the current value and the pending-commit delta.

  A write into a dict or list read from the state reaches the delta unvalidated:
  only an assignment is checked against the schema, so `state['tags'].append(1)`
  is recorded even where `tags` is declared `list[str]`. A container with no
  JSON form equal to it, as when it holds a protobuf message or a datetime, is
  left out of the delta, along with writes into it this state recorded earlier.
  """

  APP_PREFIX = "app:"
  USER_PREFIX = "user:"
  TEMP_PREFIX = "temp:"

  def __init__(
      self,
      value: dict[str, Any],
      delta: dict[str, Any],
      schema: type[BaseModel] | None = None,
  ) -> None:
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
    self._snapshots: dict[str, tuple[Any, Any]] = {}
    # TODO: recorded keys are per State, so a container another context over
    # the same event re-records or makes non-JSON is not withdrawn.
    self._recorded_nested_keys: set[str] = set()

  def __getitem__(self, key: str) -> Any:
    """Returns the value of the state dict for the given key."""
    if key in self._delta:
      value = self._delta[key]
    else:
      value = self._value[key]
    if isinstance(value, (dict, list)) and key not in self._snapshots:
      self._take_snapshot(key, value)
    return value

  def __setitem__(self, key: str, value: Any) -> None:
    """Sets the value of the state dict for the given key."""
    if self._schema is not None and isinstance(self._schema, type):
      _validate_state_entry(self._schema, key, value)
    # TODO: make new change only store in delta, so that self._value is only
    #   updated at the storage commit time.
    self._value[key] = value
    self._delta[key] = value
    self._snapshots.pop(key, None)
    self._recorded_nested_keys.discard(key)

  def _take_snapshot(self, key: str, value: Any) -> None:
    """Keeps the container a read handed out beside a copy to compare it with.

    A copy that does not compare equal to what it was copied from can never
    report the container unchanged, so such a key is left untracked rather
    than recorded by every read.
    """
    self._snapshots.pop(key, None)
    try:
      snapshot = copy.deepcopy(value)
      if bool(value != snapshot):
        return
    except Exception:  # pylint: disable=broad-exception-caught
      return
    self._snapshots[key] = (value, snapshot)

  def _record_nested_writes(self) -> None:
    """Records each dict or list handed out by a read that has since changed.

    A write into a container read from the state changes the stored object in
    place without assigning its key, so it is found by comparing the container
    with the copy taken when it was read. Every container a read handed out is
    compared on every call, so a caller that reads the delta repeatedly pays
    that comparison each time. A changed container with no JSON form equal to
    it is not recorded, and one this state recorded earlier leaves the delta.
    """
    for key, (handed_out, snapshot) in list(self._snapshots.items()):
      if key in self._delta:
        current = self._delta[key]
      elif key in self._value:
        current = self._value[key]
      else:
        del self._snapshots[key]
        self._recorded_nested_keys.discard(key)
        continue
      # Another context may have assigned the key since the read, and its
      # value wins; only the container this state handed out is tracked.
      if current is not handed_out:
        del self._snapshots[key]
        self._recorded_nested_keys.discard(key)
        continue
      try:
        changed = bool(current != snapshot)
      except Exception:  # pylint: disable=broad-exception-caught
        changed = True
      if changed:
        if not _equals_its_json_form(current):
          del self._snapshots[key]
          if key in self._recorded_nested_keys:
            self._recorded_nested_keys.remove(key)
            self._delta.pop(key, None)
            logger.debug(
                "Nested write to state key %r withdrawn from the delta: its"
                " container has no JSON form equal to it.",
                key,
            )
          else:
            logger.debug(
                "Nested write to state key %r not recorded: its container has"
                " no JSON form equal to it.",
                key,
            )
          continue
        if key not in self._delta:
          self._recorded_nested_keys.add(key)
        self._value[key] = current
        self._delta[key] = current
        self._take_snapshot(key, current)

  def __contains__(self, key: object) -> bool:
    """Whether the state dict contains the given key."""
    return key in self._value or key in self._delta

  def __iter__(self) -> Iterator[str]:
    """Iterates over the keys in the state dict."""
    return iter(self.to_dict())

  def setdefault(self, key: str, default: Any = None) -> Any:
    """Gets the value of a key, or sets it to a default if the key doesn't exist."""
    if key in self:
      return self[key]
    else:
      self[key] = default
      return default

  def has_delta(self) -> bool:
    """Whether the state has pending delta.

    Not a pure query: a write made into a dict or list read from the state is
    recorded into the delta here, and a container this state recorded leaves
    the delta once it has no JSON form equal to it, so a True answer can turn
    False with no assignment in between.
    """
    self._record_nested_writes()
    return bool(self._delta)

  def get(self, key: str, default: Any = None) -> Any:
    """Returns the value of the state dict for the given key."""
    if key not in self:
      return default
    return self[key]

  def update(self, delta: dict[str, Any]) -> None:
    """Updates the state dict with the given delta."""
    if self._schema is not None and isinstance(self._schema, type):
      for key, value in delta.items():
        _validate_state_entry(self._schema, key, value)
    self._value.update(delta)
    self._delta.update(delta)
    for key in delta:
      self._snapshots.pop(key, None)
      self._recorded_nested_keys.discard(key)

  def to_dict(self) -> dict[str, Any]:
    """Returns the state dict."""
    result: dict[str, Any] = {}
    result.update(self._value)
    result.update(self._delta)
    return result
