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

from __future__ import annotations

from unittest import mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.evaluation.request_intercepter_plugin import _LLM_REQUEST_ID_KEY
from google.adk.evaluation.request_intercepter_plugin import _RequestIntercepterPlugin
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


class TestRequestIntercepterPlugin:

  async def test_intercept_request_and_response(self):
    plugin = _RequestIntercepterPlugin(name="test_plugin")
    llm_request = LlmRequest(
        model="test_model",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="hello")],
            )
        ],
    )
    mock_invocation_context = mock.MagicMock()
    mock_invocation_context.session.state = {}
    callback_context = CallbackContext(mock_invocation_context)
    llm_response = LlmResponse()

    # Test before_model_callback
    await plugin.before_model_callback(
        callback_context=callback_context, llm_request=llm_request
    )
    assert _LLM_REQUEST_ID_KEY in callback_context.state
    request_id = callback_context.state[_LLM_REQUEST_ID_KEY]
    assert isinstance(request_id, str)

    # Test after_model_callback
    await plugin.after_model_callback(
        callback_context=callback_context, llm_response=llm_response
    )
    assert llm_response.custom_metadata is not None
    assert _LLM_REQUEST_ID_KEY in llm_response.custom_metadata
    assert llm_response.custom_metadata[_LLM_REQUEST_ID_KEY] == request_id

    # Test get_model_request
    retrieved_request = plugin.get_model_request(llm_response)
    assert retrieved_request == llm_request

  def test_get_model_request_not_found(self):
    plugin = _RequestIntercepterPlugin(name="test_plugin")
    llm_response = LlmResponse()
    assert plugin.get_model_request(llm_response) is None

    llm_response_with_metadata = LlmResponse(
        custom_metadata={_LLM_REQUEST_ID_KEY: "non_existent_id"}
    )
    assert plugin.get_model_request(llm_response_with_metadata) is None
