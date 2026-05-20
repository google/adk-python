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

from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.request_input import RequestInput
from google.adk.workflow.utils._rehydration_utils import _ChildScanState
from google.adk.workflow.utils._rehydration_utils import _reconstruct_node_states
from google.adk.workflow.utils._rehydration_utils import _unwrap_response
from google.adk.workflow.utils._rehydration_utils import _validate_resume_response
from google.adk.workflow.utils._rehydration_utils import _wrap_response
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.genai import types
from pydantic import BaseModel
import pytest

# --- _wrap_response ---


class TestWrapResponse:

  def test_dict_returned_as_is(self):
    d = {"foo": "bar"}
    assert _wrap_response(d) is d

  def test_string_wrapped(self):
    assert _wrap_response("hello") == {"result": "hello"}

  def test_int_wrapped(self):
    assert _wrap_response(42) == {"result": 42}

  def test_none_wrapped(self):
    assert _wrap_response(None) == {"result": None}

  def test_list_wrapped(self):
    assert _wrap_response([1, 2]) == {"result": [1, 2]}


# --- _unwrap_response ---


class TestUnwrapResponse:

  def test_single_result_key_string(self):
    assert _unwrap_response({"result": "hello"}) == "hello"

  def test_single_result_key_int(self):
    assert _unwrap_response({"result": 42}) == 42

  def test_single_result_key_none(self):
    assert _unwrap_response({"result": None}) is None

  def test_dict_without_result_key_unchanged(self):
    d = {"foo": "bar"}
    assert _unwrap_response(d) == {"foo": "bar"}

  def test_dict_with_multiple_keys_unchanged(self):
    d = {"result": "x", "other": "y"}
    assert _unwrap_response(d) == {"result": "x", "other": "y"}

  def test_non_dict_unchanged(self):
    assert _unwrap_response("hello") == "hello"
    assert _unwrap_response(42) == 42
    assert _unwrap_response(None) is None

  def test_json_string_parsed_to_dict(self):
    """Web frontend sends {"result": '{"approved": false}'}."""
    assert _unwrap_response({"result": '{"approved": false}'}) == {
        "approved": False
    }

  def test_json_string_parsed_to_list(self):
    assert _unwrap_response({"result": "[1, 2, 3]"}) == [1, 2, 3]

  def test_json_string_parsed_to_number(self):
    assert _unwrap_response({"result": "42"}) == 42

  def test_json_string_parsed_to_bool(self):
    assert _unwrap_response({"result": "true"}) is True

  def test_non_json_string_stays_string(self):
    assert _unwrap_response({"result": "plain text"}) == "plain text"

  def test_roundtrip_wrap_unwrap_string(self):
    assert _unwrap_response(_wrap_response("hello")) == "hello"

  def test_roundtrip_wrap_unwrap_dict(self):
    """Dicts are not wrapped, so unwrap is a no-op."""
    d = {"foo": "bar"}
    assert _unwrap_response(_wrap_response(d)) == d


# --- _validate_resume_response ---


class TestValidateResumeResponse:

  def test_none_schema_returns_data(self):
    assert _validate_resume_response("hello", None) == "hello"

  def test_str_to_int_coercion(self):
    assert _validate_resume_response("42", {"type": "integer"}) == 42

  def test_str_to_float_coercion(self):
    assert _validate_resume_response("42.5", {"type": "number"}) == 42.5

  def test_str_to_bool_true(self):
    assert _validate_resume_response("true", {"type": "boolean"}) is True
    assert _validate_resume_response("1", {"type": "boolean"}) is True

  def test_str_to_bool_false(self):
    assert _validate_resume_response("false", {"type": "boolean"}) is False
    assert _validate_resume_response("0", {"type": "boolean"}) is False

  def test_invalid_coercion_raises_value_error(self):
    with pytest.raises(ValueError):
      _validate_resume_response("abc", {"type": "integer"})

  def test_object_schema_validates_dict_type(self):
    schema = {"type": "object"}
    assert _validate_resume_response({"name": "Alice"}, schema) == {
        "name": "Alice"
    }

    with pytest.raises(ValueError, match="Failed to coerce data to object"):
      _validate_resume_response("not a dict", schema)

  def test_array_schema_validates_list_type(self):
    schema = {"type": "array"}
    assert _validate_resume_response([1, 2], schema) == [1, 2]

    with pytest.raises(ValueError, match="Failed to coerce data to array"):
      _validate_resume_response("not a list", schema)

  def test_pydantic_type_validation(self):
    class User(BaseModel):
      name: str
      age: int

    assert _validate_resume_response(
        {"name": "Alice", "age": 30}, User
    ) == User(name="Alice", age=30)


# --- _reconstruct_node_states ---


class TestScanNodeEvents:

  def test_scan_empty_events(self):
    results = _reconstruct_node_states([], "/wf@1", invocation_id="test_id")
    assert results == {}

  def test_scan_direct_child_output(self):
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        output="node_a output",
        invocation_id="test_id",
    )
    results = _reconstruct_node_states(
        [event], "/wf@1", invocation_id="test_id", group_by_direct_child=True
    )

    assert "node_a@1" in results
    assert results["node_a@1"].output == "node_a output"
    assert results["node_a@1"].run_id == "1"

  def test_scan_message_as_output(self):
    content = types.Content(parts=[types.Part(text="hello")])
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        content=content,
        invocation_id="test_id",
    )
    event.node_info.message_as_output = True

    results = _reconstruct_node_states(
        [event], "/wf@1", invocation_id="test_id", group_by_direct_child=True
    )

    assert "node_a@1" in results
    assert results["node_a@1"].output == content

  def test_scan_descendant_interrupts(self):
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1/sub_node@1"),
        long_running_tool_ids={"interrupt-1"},
        invocation_id="test_id",
    )
    results = _reconstruct_node_states(
        [event], "/wf@1", invocation_id="test_id", group_by_direct_child=True
    )

    assert "node_a@1" in results
    assert "interrupt-1" in results["node_a@1"].interrupt_ids

  def test_scan_resolve_interrupts(self):
    event_int = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        long_running_tool_ids={"interrupt-1"},
        invocation_id="test_id",
    )
    event_fr = Event(
        author="user",
        content=types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="interrupt-1",
                        name="adk_request_input",
                        response={"result": "user answer"},
                    )
                )
            ]
        ),
        invocation_id="test_id",
    )

    # Act
    results = _reconstruct_node_states(
        [event_int, event_fr],
        "/wf@1",
        invocation_id="test_id",
        group_by_direct_child=True,
    )

    # Assert
    assert "node_a@1" in results
    assert "interrupt-1" in results["node_a@1"].resolved_ids
    assert (
        results["node_a@1"].resolved_responses["interrupt-1"] == "user answer"
    )

  def test_scan_matches_specific_node_path_without_child_grouping(self):
    """Scanning matches events for a specific node path when not grouping by direct child."""
    event = Event(
        node_info=NodeInfo(path="/wf@1/node_a@1"),
        output="node_a output",
        invocation_id="test_id",
    )

    # Act
    results = _reconstruct_node_states(
        [event],
        "/wf@1/node_a@1",
        invocation_id="test_id",
        group_by_direct_child=False,
    )

    # Assert
    assert "/wf@1/node_a@1" in results
    assert results["/wf@1/node_a@1"].output == "node_a output"

  def test_scan_validates_and_coerces_response_against_schema(self):
    """Scanning validates and coerces user response data against the provided schema."""

    class MySchema(BaseModel):
      count: int

    ri = RequestInput(
        interrupt_id="interrupt-1",
        response_schema=MySchema,
    )
    event_int = create_request_input_event(ri)
    event_int.node_info = NodeInfo(path="/wf@1/node_a@1")
    event_int.invocation_id = "test_id"

    event_fr = Event(
        author="user",
        content=types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="interrupt-1",
                        name="adk_request_input",
                        response={"result": '{"count": "42"}'},
                    )
                )
            ]
        ),
        invocation_id="test_id",
    )

    # Act
    results = _reconstruct_node_states(
        [event_int, event_fr],
        "/wf@1",
        invocation_id="test_id",
        group_by_direct_child=True,
    )

    # Assert
    assert "node_a@1" in results
    assert results["node_a@1"].resolved_responses["interrupt-1"] == {
        "count": 42
    }
