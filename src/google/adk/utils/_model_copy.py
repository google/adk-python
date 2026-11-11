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

"""Deep copies of pydantic models that equal pydantic's own, made faster.

Lists, dicts, sets and pydantic models are copied directly; every other value
is handed to `copy.deepcopy` with the same memo. The one difference is a model
that contains itself, which copies as a cycle.
"""

from __future__ import annotations

import copy
from typing import Any
from typing import TypeVar

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

_ModelT = TypeVar('_ModelT', bound=BaseModel)

_ATOMIC_TYPES = frozenset({type(None), bool, int, float, str, bytes})
_MISSING = object()
_BASE_DEEPCOPY = BaseModel.__deepcopy__
_object_setattr = object.__setattr__


def deep_copy_model(
    model: _ModelT, memo: dict[int, Any] | None = None
) -> _ModelT:
  """Returns what `copy.deepcopy(model, memo)` returns with pydantic's copier.

  Assign it as a model class's `__deepcopy__` to speed up `copy.deepcopy` and
  `model_copy(deep=True)` for that class. Nested models whose class uses this
  function or pydantic's own `__deepcopy__` are copied the same way. A model
  that contains itself copies as a cycle, where pydantic's copier made a second
  object sharing the copy's fields.

  Args:
    model: The model to copy.
    memo: The `copy.deepcopy` memo. Copies of shared objects are recorded in it,
      so objects shared within the copied tree stay shared in the copy.

  Returns:
    A deep copy of `model`.
  """
  if memo is None:
    memo = {}
  keep_alive = memo.get(id(memo))
  if keep_alive is None:
    keep_alive = memo[id(memo)] = []
  return _deep_copy_model(model, memo, keep_alive)


def _deep_copy(value: Any, memo: dict[int, Any], keep_alive: list[Any]) -> Any:
  """Returns what `copy.deepcopy(value, memo)` returns."""
  cls = type(value)
  if cls in _ATOMIC_TYPES:
    return value
  copied = memo.get(id(value), _MISSING)
  if copied is not _MISSING:
    return copied
  if cls is list:
    copied_list: list[Any] = []
    memo[id(value)] = copied_list
    keep_alive.append(value)
    for item in value:
      copied_list.append(
          item
          if type(item) in _ATOMIC_TYPES
          else _deep_copy(item, memo, keep_alive)
      )
    return copied_list
  if cls is dict:
    copied_dict: dict[Any, Any] = {}
    memo[id(value)] = copied_dict
    keep_alive.append(value)
    for key, item in value.items():
      if type(key) not in _ATOMIC_TYPES:
        key = _deep_copy(key, memo, keep_alive)
      copied_dict[key] = (
          item
          if type(item) in _ATOMIC_TYPES
          else _deep_copy(item, memo, keep_alive)
      )
    return copied_dict
  if cls is set:
    copied_set = {
        item
        if type(item) in _ATOMIC_TYPES
        else _deep_copy(item, memo, keep_alive)
        for item in value
    }
    memo[id(value)] = copied_set
    keep_alive.append(value)
    return copied_set
  deepcopy_method = getattr(cls, '__deepcopy__', None)
  if deepcopy_method is _BASE_DEEPCOPY or deepcopy_method is deep_copy_model:
    return _deep_copy_model(value, memo, keep_alive)
  return copy.deepcopy(value, memo)


def _deep_copy_model(
    model: _ModelT, memo: dict[int, Any], keep_alive: list[Any]
) -> _ModelT:
  """Copies `model` the way `BaseModel.__deepcopy__` does."""
  cls = type(model)
  clone = cls.__new__(cls)
  # Recorded first, so a cycle back to `model` resolves to the clone.
  memo[id(model)] = clone
  keep_alive.append(model)
  fields = model.__dict__
  copied_fields = fields.copy()
  memo[id(fields)] = copied_fields
  keep_alive.append(fields)
  for name, value in fields.items():
    if value is not None and type(value) not in _ATOMIC_TYPES:
      copied_fields[name] = _deep_copy(value, memo, keep_alive)
  _object_setattr(clone, '__dict__', copied_fields)
  extra = model.__pydantic_extra__
  _object_setattr(
      clone,
      '__pydantic_extra__',
      None if extra is None else _deep_copy(extra, memo, keep_alive),
  )
  _object_setattr(
      clone, '__pydantic_fields_set__', model.__pydantic_fields_set__.copy()
  )
  private = getattr(model, '__pydantic_private__', None)
  if private is not None:
    private = {
        k: _deep_copy(v, memo, keep_alive)
        for k, v in private.items()
        if v is not PydanticUndefined
    }
  _object_setattr(clone, '__pydantic_private__', private)
  return clone
