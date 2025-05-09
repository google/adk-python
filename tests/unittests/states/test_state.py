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
    assert '"item":{"value":[1,2,3],"mutable":true}' in serialized.replace(" ", "")

    raw_data = json.loads(serialized)
    wrapped = Wrapper(**raw_data)
    assert wrapped.item == StateValue([1, 2, 3])


def test_immutable_statevalue_in_pydantic_model():
    class Wrapper(BaseModel):
        item: StateValue

    data = {"item": StateValue([1, 2, 3], mutable=False)}
    model = Wrapper(**data)
    assert model.item.value == [1, 2, 3]

    serialized = model.model_dump_json()
    assert '"item":{"value":[1,2,3],"mutable":false}' in serialized.replace(" ", "")

    raw_data = json.loads(serialized)
    wrapped = Wrapper(**raw_data)
    assert wrapped.item == StateValue([1, 2, 3], mutable=False)


def test_statevalue_equality_and_str_repr():
    sv1 = StateValue("foo", mutable=False)
    sv2 = StateValue("foo")
    sv3 = StateValue("bar")

    assert sv1 == "foo"
    assert sv1 != sv2
    assert sv1 != sv3
    assert repr(sv2) == "'foo'"
    assert str(sv3) == "bar"


def test_statevalue_from_dict_variants():
    d1 = {"value": 10}
    d2 = {"value": 10, "mutable": False}
    d3 = {"random": "data"}

    assert StateValue.from_dict(d1).value == 10
    assert not StateValue.from_dict(d2).mutable
    assert StateValue.from_dict(d3).value == d3


def test_statevalue_from_value_json_str():
    sv = StateValue.from_value('{"value": "abc", "mutable": false}')
    assert sv.value == "abc"
    assert sv.mutable is False


def test_statevalue_from_value_non_json_str():
    sv = StateValue.from_value("hello")
    assert sv.value == "hello"
    assert sv.mutable


def test_statevalue_to_json_output():
    sv = StateValue({"k": "v"}, mutable=False)
    assert sv.to_json() == {"value": {"k": "v"}, "mutable": False}


def test_state_immutable_overwrite_chain(empty_state):
    empty_state["a"] = StateValue("first", mutable=False)
    empty_state.set_immutable("a", "second")
    empty_state.set_immutable("a", "third")
    assert empty_state["a"] == "third"
    empty_state["a"] = "fourth"  # Should be ignored
    assert empty_state["a"] == "third"


def test_state_nested_dict_values_are_wrapped(empty_state):
    empty_state["config"] = {"dark_mode": True}
    assert isinstance(empty_state._value["config"], StateValue)
    assert empty_state["config"] == {"dark_mode": True}


def test_state_to_dict_with_nested_values():
    s = State({}, {})
    s["a"] = StateValue({"x": 1}, mutable=False)
    flat = s.to_dict()
    assert flat == {"a": {"x": 1}}


def test_state_json_serialization_with_pydantic():
    class Model(BaseModel):
        state_value: StateValue

    m = Model(state_value={"value": {"foo": 1}, "mutable": False})
    dumped = m.model_dump_json().replace(" ", "")
    assert '"foo":1' in dumped


def test_statevalue_deserialize_from_model_json():
    class M(BaseModel):
        val: StateValue

    original = M(val={"value": "hi", "mutable": True})
    dumped = original.model_dump_json()
    reloaded = M.model_validate_json(dumped)
    assert reloaded.val.value == "hi"
    assert reloaded.val.mutable


def test_state_handles_invalid_json_string():
    val = StateValue.from_value('not a json')
    assert val.value == 'not a json'
    assert val.mutable


def test_state_multiple_types_handled():
    s = State({}, {})
    s["i"] = 123
    s["l"] = [1, 2]
    s["d"] = {"key": "val"}
    s["b"] = True
    s["n"] = None
    assert s.to_dict() == {
        "i": 123,
        "l": [1, 2],
        "d": {"key": "val"},
        "b": True,
        "n": None,
    }


def test_model_with_dict_of_statevalues_all_types():
    class ComplexModel(BaseModel):
        payload: dict[str, StateValue]

    input_data = {
        "payload": {
            "int_val": StateValue(42),
            "str_val": StateValue("hello"),
            "list_val": StateValue([1, 2, 3]),
            "dict_val": StateValue({"nested": "no", "nest": {"nested": "yes"}}),
            "bool_val": StateValue(True, mutable=False)
        }
    }

    model = ComplexModel(**input_data)
    dumped = model.model_dump_json()
    reloaded = ComplexModel.model_validate_json(dumped)

    assert isinstance(reloaded.payload["int_val"], StateValue)
    assert reloaded.payload["int_val"].value == 42

    assert reloaded.payload["str_val"].value == "hello"
    assert reloaded.payload["list_val"].value == [1, 2, 3]
    assert reloaded.payload["dict_val"].value == {"nested": "no", "nest": {"nested": "yes"}}
    assert reloaded.payload["bool_val"].value is True
    assert reloaded.payload["bool_val"].mutable is False
