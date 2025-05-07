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

from google.adk.states import State
from google.adk.states import StateValue
import pytest
import json
from pydantic import BaseModel


@pytest.fixture
def empty_state():
    return State({}, {})


def test_set_raw_value_auto_wraps_mutable(empty_state):
    empty_state["theme"] = "dark"
    assert isinstance(empty_state._value["theme"], StateValue)
    assert isinstance(empty_state["theme"], str)
    assert empty_state["theme"] == "dark"


def test_set_immutable_value(empty_state):
    empty_state["booking_id"] = StateValue("ABC123", mutable=False)
    assert empty_state["booking_id"] == "ABC123"
    assert empty_state["booking_id"] == StateValue("ABC123")
    assert not empty_state._value["booking_id"].mutable


def test_set_immutable_with_override(empty_state):
    empty_state["booking_id"] = StateValue("ABC123", mutable=False)
    empty_state.set_immutable("booking_id", "NEW456")
    assert empty_state["booking_id"] == "NEW456"


def test_direct_assignment_to_immutable_does_not_override(empty_state):
    empty_state["booking_id"] = StateValue("ABC123", mutable=False)
    empty_state.set_immutable("booking_id", "NEW456")
    empty_state["booking_id"] = "NEW789"  # should be ignored
    assert empty_state["booking_id"] == "NEW456"


def test_update_mutable_key(empty_state):
    empty_state["theme"] = "dark"
    empty_state["theme"] = "light"
    assert empty_state["theme"] == "light"


def test_update_with_dict_and_immutables(empty_state):
    empty_state.update({
        "lang": "fr",  # Should wrap automatically as mutable
        "token": StateValue("xyz", mutable=False)
    })
    empty_state.update({
        "lang": "en",   # allowed (mutable)
        "token": "abc"  # ignored (immutable)
    })
    assert empty_state["lang"] == "en"
    assert empty_state["token"] == "xyz"
    assert isinstance(empty_state._value["lang"], StateValue)
    assert not empty_state._value["token"].mutable


def test_to_dict_outputs_raw_values(empty_state):
    empty_state["theme"] = "light"
    empty_state["booking_id"] = StateValue("NEW456", mutable=False)
    flat = empty_state.to_dict()
    assert isinstance(flat["theme"], str)
    assert flat["theme"] == "light"
    assert flat["booking_id"] == "NEW456"


def test_statevalue_in_pydantic_model():
    class Wrapper(BaseModel):
        item: StateValue

    data = {"item": StateValue([1, 2, 3])}
    model = Wrapper(**data)
    assert model.item.value == [1, 2, 3]

    serialized = model.model_dump_json()
    assert '"item":[1,2,3]' in serialized.replace(" ", "")

    raw_data = json.loads(serialized)
    wrapped = Wrapper(**raw_data)
    assert wrapped.item == StateValue([1, 2, 3])
