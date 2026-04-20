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

"""Tests for _safe_json_serialize in models/lite_llm.py.

Verifies that the function never raises exceptions, even for inputs that
cause json.dumps to raise ValueError (circular references) or
RecursionError (deeply nested structures).

Fixes https://github.com/google/adk-python/issues/5412
"""

import pytest

from google.adk.models.lite_llm import _safe_json_serialize


def test_circular_reference_returns_str_fallback():
    """json.dumps raises ValueError on circular references; should fall back to str()."""

    class Node:
        def __init__(self):
            self.ref = self

    obj = Node()
    result = _safe_json_serialize(obj)
    assert isinstance(result, str)
    # Should return str(obj) fallback rather than raising ValueError


def test_deeply_nested_structure_returns_str_fallback():
    """json.dumps raises RecursionError on deeply nested structures."""
    obj = current = {}
    for _ in range(10000):
        current["child"] = {}
        current = current["child"]

    result = _safe_json_serialize(obj)
    assert isinstance(result, str)
    # str(obj) itself can raise RecursionError, so expect the safe fallback
    assert "recursion" in result.lower() or result  # non-empty string


def test_normal_dict_serializes():
    """Normal dicts should serialize as JSON."""
    result = _safe_json_serialize({"key": "value", "num": 42})
    assert '"key"' in result
    assert '"value"' in result


def test_non_serializable_object_falls_back_to_str():
    """Objects without a JSON representation should fall back to str()."""
    obj = object()
    result = _safe_json_serialize(obj)
    assert isinstance(result, str)
    assert "object" in result.lower()
