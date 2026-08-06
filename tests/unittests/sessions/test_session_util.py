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

"""Tests for _session_util.

Verifies that session utilities correctly decode models and extract state deltas.
"""

from google.adk.sessions._session_util import decode_model
from google.adk.sessions._session_util import extract_state_delta
from google.genai import types
from pydantic import BaseModel
import pytest


class TestDecodeModel:
  """Tests for decode_model utility."""

  def test_returns_none_for_none_input(self):
    """decode_model returns None if the input data is None."""
    assert decode_model(None, types.Content) is None

  def test_decodes_dict_into_model_instance(self):
    """decode_model decodes a dictionary into the specified BaseModel subclass."""
    result = decode_model(
        {"role": "user", "parts": [{"text": "hello"}]}, types.Content
    )

    assert isinstance(result, types.Content)
    assert result.role == "user"
    assert result.parts[0].text == "hello"

  def test_returns_none_for_non_dict_value(self):
    """decode_model returns None for primitive values like 'null' string."""
    assert decode_model("null", types.Transcription) is None

  def test_raises_for_invalid_data(self):
    """decode_model raises an exception if the input data fails validation."""

    class _SampleModel(BaseModel):
      name: str
      value: int

    with pytest.raises(Exception):
      decode_model({"name": "foo"}, _SampleModel)


class TestExtractStateDelta:
  """Tests for extract_state_delta utility."""

  def test_returns_empty_deltas_for_empty_state(self):
    """extract_state_delta returns empty dicts for empty state input."""
    assert extract_state_delta({}) == {"app": {}, "user": {}, "session": {}}

  def test_returns_empty_deltas_for_none_state(self):
    """extract_state_delta returns empty dicts for None state input."""
    assert extract_state_delta(None) == {"app": {}, "user": {}, "session": {}}

  def test_routes_app_prefixed_keys_with_prefix_stripped(self):
    """extract_state_delta routes 'app:' prefixed keys to the 'app' bucket, stripping the prefix."""
    deltas = extract_state_delta({"app:theme": "dark"})

    assert deltas["app"] == {"theme": "dark"}
    assert deltas["user"] == {}
    assert deltas["session"] == {}

  def test_routes_user_prefixed_keys_with_prefix_stripped(self):
    """extract_state_delta routes 'user:' prefixed keys to the 'user' bucket, stripping the prefix."""
    deltas = extract_state_delta({"user:lang": "en"})

    assert deltas["user"] == {"lang": "en"}
    assert deltas["app"] == {}
    assert deltas["session"] == {}

  def test_routes_unprefixed_keys_to_session(self):
    """extract_state_delta routes unprefixed keys to the 'session' bucket."""
    deltas = extract_state_delta({"turn": 3})

    assert deltas["session"] == {"turn": 3}
    assert deltas["app"] == {}
    assert deltas["user"] == {}

  def test_skips_temp_prefixed_keys(self):
    """extract_state_delta ignores keys with 'temp:' prefix."""
    deltas = extract_state_delta({"temp:scratch": "ignore_me"})

    assert deltas == {"app": {}, "user": {}, "session": {}}

  def test_routes_mixed_keys_into_correct_buckets(self):
    """extract_state_delta correctly routes multiple keys of different prefixes to their respective buckets."""
    state = {
        "app:theme": "dark",
        "user:lang": "en",
        "temp:scratch": "ignore_me",
        "turn": 3,
    }

    deltas = extract_state_delta(state)

    assert deltas == {
        "app": {"theme": "dark"},
        "user": {"lang": "en"},
        "session": {"turn": 3},
    }
