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

"""Tests for JSON utility functions."""

from google.adk.utils._json_utils import safe_json_loads
import pytest


def test_parses_object():
  result = safe_json_loads('{"key": "value"}')
  assert result == {'key': 'value'}


def test_parses_array():
  result = safe_json_loads('[1, 2, 3]')
  assert result == [1, 2, 3]


def test_parses_nested_structure():
  result = safe_json_loads('{"a": {"b": [1, null, true]}}')
  assert result == {'a': {'b': [1, None, True]}}


def test_parses_string_value():
  result = safe_json_loads('"hello"')
  assert result == 'hello'


def test_parses_bytes():
  result = safe_json_loads(b'{"key": "value"}')
  assert result == {'key': 'value'}


def test_parses_bytearray():
  result = safe_json_loads(bytearray(b'{"key": "value"}'))
  assert result == {'key': 'value'}


def test_parses_number():
  result = safe_json_loads('42')
  assert result == 42


def test_parses_null():
  result = safe_json_loads('null')
  assert result is None


def test_malformed_raises_value_error():
  with pytest.raises(ValueError):
    safe_json_loads('{bad json}')


def test_empty_string_raises_value_error():
  with pytest.raises(ValueError):
    safe_json_loads('')


def test_error_message_includes_context():
  with pytest.raises(ValueError, match='session state'):
    safe_json_loads('{bad}', context='session state')


def test_error_message_without_context():
  with pytest.raises(ValueError, match='Invalid JSON'):
    safe_json_loads('{bad}')


def test_error_wraps_json_decode_error():
  with pytest.raises(ValueError) as exc_info:
    safe_json_loads('{bad}', context='test')
  assert exc_info.value.__cause__ is not None
  import json

  assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_context_none_no_suffix():
  with pytest.raises(ValueError) as exc_info:
    safe_json_loads('{bad}', context=None)
  msg = str(exc_info.value)
  assert msg.startswith('Invalid JSON:'), msg
  assert not msg.startswith('Invalid JSON in '), msg


def test_unicode_content():
  result = safe_json_loads('{"emoji": "🎉", "chinese": "你好"}')
  assert result == {'emoji': '🎉', 'chinese': '你好'}
