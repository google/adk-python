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

"""Tests that SSE streaming preserves function call IDs across partial/final events.

Regression test for https://github.com/google/adk-python/issues/4609
"""

from google.adk.events.event import Event
from google.adk.flows.llm_flows.base_llm_flow import _finalize_model_response_event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
import pytest


def _make_base_event() -> Event:
    return Event(
        id=Event.new_id(),
        invocation_id="test-inv",
        author="test-agent",
    )


def _make_llm_response(*, partial: bool, fc_name: str = "get_weather") -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=fc_name,
                        args={"location": "NYC"},
                    )
                )
            ],
        ),
        partial=partial,
    )


def _make_llm_request() -> LlmRequest:
    req = LlmRequest()
    req.tools_dict = {}
    return req


class TestStreamingFunctionCallIdConsistency:
    """Ensure partial and final events share the same function call ID."""

    def test_partial_and_final_share_same_id(self):
        """The core regression: partial event ID must equal final event ID."""
        llm_request = _make_llm_request()
        function_call_ids: dict[tuple[str, int], str] = {}

        # Simulate partial event
        partial_event = _finalize_model_response_event(
            llm_request,
            _make_llm_response(partial=True),
            _make_base_event(),
            function_call_ids,
        )
        partial_fc_id = partial_event.get_function_calls()[0].id
        assert partial_fc_id is not None
        assert partial_fc_id.startswith("adk-")

        # Simulate final event (new Event object, same streaming sequence)
        final_event = _finalize_model_response_event(
            llm_request,
            _make_llm_response(partial=False),
            _make_base_event(),
            function_call_ids,
        )
        final_fc_id = final_event.get_function_calls()[0].id

        assert final_fc_id == partial_fc_id

    def test_without_function_call_ids_dict_generates_different_ids(self):
        """Without the fix dict, each event gets a fresh ID (old behaviour)."""
        llm_request = _make_llm_request()

        partial_event = _finalize_model_response_event(
            llm_request,
            _make_llm_response(partial=True),
            _make_base_event(),
        )
        final_event = _finalize_model_response_event(
            llm_request,
            _make_llm_response(partial=False),
            _make_base_event(),
        )

        # Without the dict, IDs differ (demonstrating the old bug)
        assert (
            partial_event.get_function_calls()[0].id
            != final_event.get_function_calls()[0].id
        )

    def test_multiple_function_calls_preserve_ids(self):
        """Each function call in a multi-call response keeps its own stable ID."""
        llm_request = _make_llm_request()
        function_call_ids: dict[tuple[str, int], str] = {}

        def make_multi_fc_response(partial: bool) -> LlmResponse:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_weather",
                                args={"location": "NYC"},
                            )
                        ),
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_time",
                                args={"timezone": "EST"},
                            )
                        ),
                    ],
                ),
                partial=partial,
            )

        partial_event = _finalize_model_response_event(
            llm_request,
            make_multi_fc_response(partial=True),
            _make_base_event(),
            function_call_ids,
        )
        partial_ids = [fc.id for fc in partial_event.get_function_calls()]

        final_event = _finalize_model_response_event(
            llm_request,
            make_multi_fc_response(partial=False),
            _make_base_event(),
            function_call_ids,
        )
        final_ids = [fc.id for fc in final_event.get_function_calls()]

        assert partial_ids == final_ids
        # The two function calls should have different IDs from each other
        assert partial_ids[0] != partial_ids[1]

    def test_server_provided_id_is_preserved(self):
        """If the server already provides an ID, it should not be overwritten."""
        llm_request = _make_llm_request()
        function_call_ids: dict[tuple[str, int], str] = {}

        server_id = "server-provided-id-123"
        response = LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=server_id,
                            name="get_weather",
                            args={"location": "NYC"},
                        )
                    )
                ],
            ),
            partial=False,
        )

        event = _finalize_model_response_event(
            llm_request,
            response,
            _make_base_event(),
            function_call_ids,
        )

        assert event.get_function_calls()[0].id == server_id
