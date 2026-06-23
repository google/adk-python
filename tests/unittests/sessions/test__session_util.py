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

from google.adk.sessions._session_util import decode_model
from google.adk.sessions._session_util import extract_state_delta
from pydantic import BaseModel
import pytest


class _SampleModel(BaseModel):
  name: str
  value: int


class TestDecodeModel:

  def test_returns_none_for_none_input(self):
    assert decode_model(None, _SampleModel) is None

  def test_decodes_dict_into_model_instance(self):
    result = decode_model({"name": "foo", "value": 7}, _SampleModel)

    assert result == _SampleModel(name="foo", value=7)

  def test_raises_for_invalid_data(self):
    with pytest.raises(Exception):
      decode_model({"name": "foo"}, _SampleModel)


class TestExtractStateDelta:

  def test_returns_empty_deltas_for_empty_state(self):
    assert extract_state_delta({}) == {"app": {}, "user": {}, "session": {}}

  def test_returns_empty_deltas_for_none_state(self):
    assert extract_state_delta(None) == {"app": {}, "user": {}, "session": {}}

  def test_routes_app_prefixed_keys_with_prefix_stripped(self):
    deltas = extract_state_delta({"app:theme": "dark"})

    assert deltas["app"] == {"theme": "dark"}
    assert deltas["user"] == {}
    assert deltas["session"] == {}

  def test_routes_user_prefixed_keys_with_prefix_stripped(self):
    deltas = extract_state_delta({"user:lang": "en"})

    assert deltas["user"] == {"lang": "en"}
    assert deltas["app"] == {}
    assert deltas["session"] == {}

  def test_routes_unprefixed_keys_to_session(self):
    deltas = extract_state_delta({"turn": 3})

    assert deltas["session"] == {"turn": 3}
    assert deltas["app"] == {}
    assert deltas["user"] == {}

  def test_skips_temp_prefixed_keys(self):
    deltas = extract_state_delta({"temp:scratch": "ignore_me"})

    assert deltas == {"app": {}, "user": {}, "session": {}}

  def test_routes_mixed_keys_into_correct_buckets(self):
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
