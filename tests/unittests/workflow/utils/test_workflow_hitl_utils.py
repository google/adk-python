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
from google.adk.workflow.utils._workflow_hitl_utils import create_auth_request_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_event
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import has_request_input_function_call
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
from google.genai import types

# --- create_request_input_event ---


class TestCreateRequestInputEvent:

  def test_basic_event(self):
    ri = RequestInput(
        interrupt_id="test-id",
        message="Please approve",
    )
    event = create_request_input_event(ri)

    assert event.long_running_tool_ids == {"test-id"}
    assert event.content is not None
    fc = event.content.parts[0].function_call
    assert fc.name == "adk_request_input"
    assert fc.id == "test-id"
    assert fc.args["message"] == "Please approve"

  def test_with_payload(self):
    ri = RequestInput(
        interrupt_id="id-1",
        payload={"key": "value"},
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    assert fc.args["payload"] == {"key": "value"}

  def test_with_response_schema(self):
    from pydantic import BaseModel

    class MySchema(BaseModel):
      approved: bool

    ri = RequestInput(
        interrupt_id="id-2",
        response_schema=MySchema,
    )
    event = create_request_input_event(ri)
    fc = event.content.parts[0].function_call
    schema = fc.args["response_schema"]
    assert "approved" in schema["properties"]
    assert schema["properties"]["approved"]["type"] == "boolean"


# --- has_request_input_function_call ---


class TestHasRequestInputFunctionCall:

  def test_true_for_request_input_event(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert has_request_input_function_call(event) is True

  def test_false_for_empty_event(self):
    assert has_request_input_function_call(Event()) is False

  def test_false_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(name="other_tool", args={})
                )
            ]
        )
    )
    assert has_request_input_function_call(event) is False


# --- create_request_input_response ---


class TestCreateRequestInputResponse:

  def test_creates_function_response_part(self):
    part = create_request_input_response("id-1", {"approved": True})
    assert part.function_response.id == "id-1"
    assert part.function_response.name == "adk_request_input"
    assert part.function_response.response == {"approved": True}


# --- get_request_input_interrupt_ids ---


class TestGetRequestInputInterruptIds:

  def test_extracts_ids(self):
    event = create_request_input_event(
        RequestInput(interrupt_id="id-1", message="test")
    )
    assert get_request_input_interrupt_ids(event) == ["id-1"]

  def test_empty_for_no_function_calls(self):
    assert get_request_input_interrupt_ids(Event()) == []

  def test_empty_for_non_request_input(self):
    from google.genai import types

    event = Event(
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="other_tool", args={}, id="id-1"
                    )
                )
            ]
        )
    )
    assert get_request_input_interrupt_ids(event) == []


# --- create_auth_request_event ---


class TestCreateAuthRequestEvent:

  def test_creates_credential_request(self):
    from fastapi.openapi.models import APIKey
    from fastapi.openapi.models import APIKeyIn
    from google.adk.auth.auth_credential import AuthCredential
    from google.adk.auth.auth_credential import AuthCredentialTypes
    from google.adk.auth.auth_tool import AuthConfig

    auth_config = AuthConfig(
        auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key="test_key",
        ),
        credential_key="test_cred",
    )
    event = create_auth_request_event(auth_config, "auth-id-1")

    assert event.long_running_tool_ids is not None
    fc = event.content.parts[0].function_call
    assert fc.name == REQUEST_CREDENTIAL_FUNCTION_CALL_NAME
    assert fc.id == "auth-id-1"
    assert "authConfig" in fc.args


#
