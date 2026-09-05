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

"""Unit tests for serialization utilities."""

from __future__ import annotations

import datetime
from enum import Enum

from google.adk.utils._serialization import safe_serialize
from pydantic import BaseModel
from pydantic_core import to_jsonable_python


class _Sample(BaseModel):
  x: int = 5
  label: str = 'hi'


class _DeferredModel(BaseModel):
  """A Pydantic model whose serializer was not built yet (deferred build)."""

  x: int = 1


_DeferredModel.__pydantic_serializer__ = object()


def test_plain_values_are_unchanged():
  value = {'a': 1, 'b': [1, 2], 'c': {'d': 'e'}, 'f': None, 'g': True}
  assert safe_serialize(value) == value


def test_datetime_is_preserved_not_discarded():
  dt = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
  assert safe_serialize(dt) == '2024-01-02T03:04:05Z'


def test_pydantic_model_is_serialized_to_dict():
  assert safe_serialize(_Sample()) == {'x': 5, 'label': 'hi'}


def test_nested_rich_types_are_serialized():
  dt = datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc)
  result = safe_serialize({'when': dt, 'model': _Sample(), 'n': [1]})
  assert result == {
      'when': '2024-05-06T00:00:00Z',
      'model': {'x': 5, 'label': 'hi'},
      'n': [1],
  }


def test_unserializable_value_is_replaced_with_repr():
  result = safe_serialize(lambda: 1)
  assert isinstance(result, str)
  assert 'function' in result


def test_unserializable_value_nested():
  result = safe_serialize({'cb': lambda: 1, 'ok': 2})
  assert result['ok'] == 2
  assert isinstance(result['cb'], str)


def test_deferred_model_at_root_is_replaced_with_repr():
  result = safe_serialize(_DeferredModel())
  assert isinstance(result, str)
  assert '_DeferredModel' in result


def test_deferred_model_is_replaced_with_repr():
  result = safe_serialize({'bad': _DeferredModel(), 'ok': 1})
  assert result['ok'] == 1
  assert isinstance(result['bad'], str)
  assert '_DeferredModel' in result['bad']


def test_deferred_model_nested_in_containers_is_replaced():
  result = safe_serialize({
      'items': [_DeferredModel(), 'ok'],
      'inner': {'deep': _DeferredModel()},
  })
  assert result['items'][1] == 'ok'
  assert isinstance(result['items'][0], str)
  assert isinstance(result['inner']['deep'], str)


def test_tuple_and_set_serialize_to_lists():
  assert safe_serialize((1, (2, 3))) == [1, [2, 3]]
  assert sorted(safe_serialize({1, 2})) == [1, 2]
  assert safe_serialize(frozenset({'a'})) == ['a']


def test_healthy_output_matches_to_jsonable_python():
  dt = datetime.datetime(2024, 5, 6, tzinfo=datetime.timezone.utc)

  class _EnumWithTupleValue(Enum):
    TUP = ('a', 'b')

  values = [
      {'a': 1, 'b': [1, 2], 'c': {'d': 'e'}, 'f': None, 'g': True},
      {'when': dt, 'model': _Sample(), 'n': [1]},
      {'cb': lambda: 1, 'ok': 2},
      [1, 'two', None],
      {'t': (1, 2), 's': {'x'}},
      {_EnumWithTupleValue.TUP: 1, (_EnumWithTupleValue.TUP,): 2},
  ]

  for value in values:
    assert safe_serialize(value) == to_jsonable_python(
        value, serialize_unknown=True
    )


def test_dict_with_non_string_keys():
  class _SampleEnum(Enum):
    VAL = 'test'

  data = {
      1: 'int',
      2: 'two',
      _SampleEnum.VAL: 'enum',
      False: 'bool',
      (1, 2): 'tuple',
  }
  result = safe_serialize(data)
  assert result['1'] == 'int'
  assert result['2'] == 'two'
  assert result['test'] == 'enum'
  assert result['false'] == 'bool'
  assert result['1,2'] == 'tuple'


def test_nested_dict_keys_match_healthy_serialization_when_sibling_fails():
  class _SampleEnum(Enum):
    VAL = 'test'

  nested = {
      1: 'int',
      _SampleEnum.VAL: 'enum',
      False: 'bool',
      (1, 2): 'tuple',
  }
  fallback = safe_serialize({'bad': lambda: 1, 'nested': nested})
  healthy = to_jsonable_python({'nested': nested}, serialize_unknown=True)
  assert fallback['nested'] == healthy['nested']


def test_container_with_raising_items_falls_back_to_repr():
  class _DictWithBoomItems(dict):

    def items(self):
      raise RuntimeError('boom')

  value = {'outer': _DictWithBoomItems({'a': 1})}
  result = safe_serialize(value)
  assert isinstance(result['outer'], str)
  assert 'a' in result['outer']


def test_circular_dict_is_replaced_with_repr():
  data = {'a': 1}
  data['self'] = data
  result = safe_serialize(data)
  assert result['a'] == 1
  assert isinstance(result['self'], str)


def test_circular_list_is_replaced_with_repr():
  data = [1, 2]
  data.append(data)
  result = safe_serialize(data)
  assert result[0] == 1
  assert result[1] == 2
  assert isinstance(result[2], str)


def test_exponential_back_references_terminate_quickly():
  data = {}
  data['left'] = data
  data['right'] = data
  result = safe_serialize(data)
  assert isinstance(result['left'], str)
  assert isinstance(result['right'], str)


def test_shared_sub_container_in_dag_is_not_treated_as_cycle():
  shared = {'nested': 42}
  data = {'first': shared, 'second': shared}
  assert safe_serialize(data) == {
      'first': {'nested': 42},
      'second': {'nested': 42},
  }
