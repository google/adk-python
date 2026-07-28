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

from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins._reflect_retry_model_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins._reflect_retry_model_plugin import ReflectAndRetryModelPlugin
from google.adk.plugins._reflect_retry_model_plugin import RESERVED_TOOL_CALL_ERROR_TYPE
from google.adk.plugins._reflect_retry_model_plugin import TrackingScope
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


class TestReflectAndRetryModelPlugin(IsolatedAsyncioTestCase):
  """Tests for model error handling in the ReflectAndRetryModelPlugin."""

  async def test_plugin_initialization_default(self):
    """Test plugin initialization with default parameters for model errors."""
    plugin = ReflectAndRetryModelPlugin()

    self.assertEqual(plugin.name, "reflect_retry_model_plugin")
    self.assertEqual(plugin.max_retries, 3)
    self.assertIs(plugin.throw_exception_if_retry_exceeded, True)
    self.assertEqual(plugin.scope, TrackingScope.INVOCATION)
    self.assertEqual(
        plugin.on_model_errors,
        [types.FinishReason.MALFORMED_FUNCTION_CALL],
    )

  async def test_validate_model_errors_ensures_finish_reason_types(self):
    """Checks that input model errors must all be of type FinishReason."""
    valid_reasons = [
        types.FinishReason.MALFORMED_FUNCTION_CALL,
        types.FinishReason.SAFETY,
    ]
    plugin = ReflectAndRetryModelPlugin(on_model_errors=valid_reasons)
    self.assertEqual(plugin.on_model_errors, valid_reasons)

    with self.assertRaises(ValueError):
      ReflectAndRetryModelPlugin(
          on_model_errors=[
              types.FinishReason.MALFORMED_FUNCTION_CALL,
              "NOT_A_FINISH_REASON",
          ]
      )

  async def test_adk_handle_model_error_format(self):
    """Checks the function call / response format of the tool."""
    plugin = ReflectAndRetryModelPlugin()
    result = plugin.adk_handle_model_error(
        response_type=REFLECT_AND_RETRY_RESPONSE_TYPE,
        error_type="TEST_ERROR_TYPE",
        error_details="TEST_ERROR_DETAILS",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
        retry_count=1,
    )
    self.assertIsInstance(result, dict)
    self.assertIn("reflection_guidance", result)

  async def test_check_for_model_error_uses_input_model_errors(self):
    """Checks that _check_for_model_error correctly identifies errors in the configured on_model_errors list."""
    plugin = ReflectAndRetryModelPlugin(
        on_model_errors=[
            types.FinishReason.MALFORMED_FUNCTION_CALL,
            types.FinishReason.SAFETY,
        ]
    )

    response_safety = LlmResponse(
        error_code=types.FinishReason.SAFETY,
        finish_reason=types.FinishReason.SAFETY,
    )
    response_malformed = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    self.assertTrue(plugin._check_for_model_error(llm_response=response_safety))
    self.assertTrue(
        plugin._check_for_model_error(llm_response=response_malformed)
    )

    response_recitation = LlmResponse(
        error_code=types.FinishReason.RECITATION,
        finish_reason=types.FinishReason.RECITATION,
    )

    self.assertFalse(
        plugin._check_for_model_error(llm_response=response_recitation)
    )

  async def test_check_for_model_error_requires_error_code(self):
    """Checks that _check_for_model_error returns False if the response has no error code, even if the finish reason matches."""
    plugin = ReflectAndRetryModelPlugin()
    response = LlmResponse(
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )
    self.assertFalse(plugin._check_for_model_error(llm_response=response))

  async def test_get_model_name_from_context_success(self):
    """Checks that _get_model_name_from_context successfully retrieves the model name from a valid callback context with an LlmAgent."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin()
    model_name = plugin._get_model_name_from_context(
        callback_context=mock_callback_context
    )
    self.assertEqual(model_name, "TEST_MODEL_NAME")

  async def test_get_model_name_from_context_requires_llm_agent(self):
    """Checks that _get_model_name_from_context raises ValueError if the agent in context is not an LlmAgent."""
    mock_agent = Mock(spec=BaseAgent)

    mock_invocation_context = Mock(spec=InvocationContext)
    mock_invocation_context.agent = mock_agent

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin()
    with self.assertRaises(ValueError):
      plugin._get_model_name_from_context(
          callback_context=mock_callback_context
      )

  async def test_before_model_callback_adds_reflect_tool_to_llm_request(self):
    """Checks that before_model_callback adds adk_handle_model_error to llm_request.tools_dict."""
    mock_callback_context = Mock(spec=CallbackContext)
    llm_request = LlmRequest()

    plugin = ReflectAndRetryModelPlugin()
    response = await plugin.before_model_callback(
        callback_context=mock_callback_context,
        llm_request=llm_request,
    )

    self.assertIsNone(response)
    self.assertIn(
        ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
        llm_request.tools_dict,
    )
    tool = llm_request.tools_dict[
        ReflectAndRetryModelPlugin.adk_handle_model_error.__name__
    ]
    self.assertIsInstance(tool, FunctionTool)

  async def test_after_model_callback_retries_on_malformed_call(self):
    """Test that a retry tool call is returned on a malformed function call"""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(
        on_model_errors=[types.FinishReason.MALFORMED_FUNCTION_CALL]
    )

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR_MESSAGE",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    response = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )

    self.assertIsNotNone(response)
    self.assertIsNone(response.error_code)
    self.assertIsNotNone(response.content)
    self.assertEqual(len(response.content.parts), 1)
    part = response.content.parts[0]
    self.assertIsNotNone(part.function_call)
    self.assertEqual(
        part.function_call.name,
        ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
    )
    self.assertEqual(
        part.function_call.args["finish_reason"],
        types.FinishReason.MALFORMED_FUNCTION_CALL.value,
    )

  async def test_after_model_callback_can_perform_multiple_retries(self):
    """Checks that after_model_callback increments the retry count for consecutive model errors."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(max_retries=3)

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR_MESSAGE",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    response1 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertEqual(
        response1.content.parts[0].function_call.args["retry_count"], 1
    )

    response2 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertEqual(
        response2.content.parts[0].function_call.args["retry_count"], 2
    )

    response3 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertEqual(
        response3.content.parts[0].function_call.args["retry_count"], 3
    )

  async def test_after_model_callback_returns_response_when_retry_limit_reached(
      self,
  ):
    """Checks that after_model_callback returns the failed response when retry limit is reached and throw_exception_if_retry_exceeded is False."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR_MESSAGE",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    response1 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertIsNotNone(response1)

    response2 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertEqual(response2.error_code, llm_response.error_code)
    self.assertEqual(response2.error_message, llm_response.error_message)
    self.assertEqual(response2.finish_reason, llm_response.finish_reason)

  async def test_after_model_callback_throws_when_retry_limit_reached(self):
    """Checks that after_model_callback raises an Exception when retry limit is reached and throw_exception_if_retry_exceeded is True."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=True
    )

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR_MESSAGE",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    response1 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertIsNotNone(response1)

    with self.assertRaises(RuntimeError):
      await plugin.after_model_callback(
          callback_context=mock_callback_context,
          llm_response=llm_response,
      )

  async def test_after_model_callback_resets_retry_limit_upon_success(self):
    """Checks that a successful model response resets the failure counter for the model."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(max_retries=3)

    llm_response_error = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR_MESSAGE",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )
    llm_response_success = LlmResponse()

    response1 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response_error,
    )
    self.assertEqual(
        response1.content.parts[0].function_call.args["retry_count"], 1
    )

    response2 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response_error,
    )
    self.assertEqual(
        response2.content.parts[0].function_call.args["retry_count"], 2
    )

    response_success = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response_success,
    )
    self.assertIsNone(response_success)
    self.assertEqual(len(plugin._tracker._scoped_failure_counters), 0)

    response2 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response_error,
    )
    self.assertEqual(
        response2.content.parts[0].function_call.args["retry_count"], 1
    )

  async def test_after_model_callback_intercepts_reserved_tool_call(self):
    """Checks that after_model_callback intercepts direct calls to reserved tool."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(max_retries=3)

    # Simulate model response containing a call to adk_handle_model_error
    llm_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
                        args={
                            "response_type": REFLECT_AND_RETRY_RESPONSE_TYPE,
                            "error_type": "TEST_ERROR_TYPE",
                            "error_details": "TEST_ERROR_MESSAGE",
                            "finish_reason": (
                                types.FinishReason.MALFORMED_FUNCTION_CALL
                            ),
                            "retry_count": 1,
                        },
                    )
                )
            ],
        ),
    )

    response = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )

    self.assertIsNotNone(response)
    self.assertEqual(
        response.content.parts[0].function_call.name,
        ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
    )
    # Check that the arguments were overwritten by the plugin
    self.assertEqual(
        response.content.parts[0].function_call.args["error_type"],
        RESERVED_TOOL_CALL_ERROR_TYPE,
    )
    self.assertEqual(
        response.content.parts[0].function_call.args["retry_count"], 1
    )

  async def test_after_model_callback_returns_error_response_when_reserved_tool_call_limit_reached(
      self,
  ):
    """Checks that after_model_callback returns an error response (blocking execution) when reserved tool call limit is reached and throw_exception_if_retry_exceeded is False."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_invocation_context = Mock()
    mock_invocation_context.agent = mock_agent
    mock_invocation_context.invocation_id = "TEST_INVOCATION_ID"

    mock_callback_context = Mock(spec=CallbackContext)
    mock_callback_context.get_invocation_context.return_value = (
        mock_invocation_context
    )

    plugin = ReflectAndRetryModelPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )

    # Simulate model response containing a call to adk_handle_model_error
    llm_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name=ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
                        args={
                            "response_type": REFLECT_AND_RETRY_RESPONSE_TYPE,
                            "error_type": "TEST_ERROR_TYPE",
                            "error_details": "TEST_ERROR_MESSAGE",
                            "finish_reason": (
                                types.FinishReason.MALFORMED_FUNCTION_CALL
                            ),
                            "retry_count": 1,
                        },
                    )
                )
            ],
        ),
    )

    # First call (1st failure) -> should retry (returns tool call)
    response1 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertIsNotNone(response1)
    self.assertEqual(
        response1.content.parts[0].function_call.name,
        ReflectAndRetryModelPlugin.adk_handle_model_error.__name__,
    )

    # Second call (2nd failure) -> limit exceeded -> should return error response (no tool call)
    response2 = await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=llm_response,
    )
    self.assertIsNotNone(response2)
    self.assertEqual(response2.error_code, RESERVED_TOOL_CALL_ERROR_TYPE)
    self.assertIsNone(response2.content)

  async def test_different_models_have_separate_retry_counters(self):
    """Checks that different models maintain separate retry counters within the same invocation."""
    mock_agent_gemini = Mock(spec=LlmAgent)
    mock_agent_gemini.canonical_model = Mock()
    mock_agent_gemini.canonical_model.model = "gemini-2.5-pro"

    mock_agent_claude = Mock(spec=LlmAgent)
    mock_agent_claude.canonical_model = Mock()
    mock_agent_claude.canonical_model.model = "claude-3-5-sonnet"

    mock_ctx_gemini = Mock(spec=CallbackContext)
    mock_inv_gemini = Mock()
    mock_inv_gemini.agent = mock_agent_gemini
    mock_inv_gemini.invocation_id = "INVOCATION_SAME"
    mock_ctx_gemini.get_invocation_context.return_value = mock_inv_gemini

    mock_ctx_claude = Mock(spec=CallbackContext)
    mock_inv_claude = Mock()
    mock_inv_claude.agent = mock_agent_claude
    mock_inv_claude.invocation_id = "INVOCATION_SAME"
    mock_ctx_claude.get_invocation_context.return_value = mock_inv_claude

    plugin = ReflectAndRetryModelPlugin(max_retries=5)
    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    # First failure on Gemini -> count is 1
    resp_gemini_1 = await plugin.after_model_callback(
        callback_context=mock_ctx_gemini, llm_response=llm_response
    )
    self.assertEqual(
        resp_gemini_1.content.parts[0].function_call.args["retry_count"], 1
    )

    # First failure on Claude -> count should start fresh at 1 (separate model counter!)
    resp_claude_1 = await plugin.after_model_callback(
        callback_context=mock_ctx_claude, llm_response=llm_response
    )
    self.assertEqual(
        resp_claude_1.content.parts[0].function_call.args["retry_count"], 1
    )

    # Second failure on Gemini -> count increments to 2 for Gemini
    resp_gemini_2 = await plugin.after_model_callback(
        callback_context=mock_ctx_gemini, llm_response=llm_response
    )
    self.assertEqual(
        resp_gemini_2.content.parts[0].function_call.args["retry_count"], 2
    )

  async def test_invocation_tracking_scope_for_models(self):
    """Checks that TrackingScope.INVOCATION isolates failure counts between different invocations for models."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_inv_1 = Mock()
    mock_inv_1.agent = mock_agent
    mock_inv_1.invocation_id = "INVOCATION_1"

    mock_inv_2 = Mock()
    mock_inv_2.agent = mock_agent
    mock_inv_2.invocation_id = "INVOCATION_2"

    mock_ctx_1 = Mock(spec=CallbackContext)
    mock_ctx_1.get_invocation_context.return_value = mock_inv_1

    mock_ctx_2 = Mock(spec=CallbackContext)
    mock_ctx_2.get_invocation_context.return_value = mock_inv_2

    plugin = ReflectAndRetryModelPlugin(
        max_retries=5, tracking_scope=TrackingScope.INVOCATION
    )

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    # First failure on invocation 1 -> count is 1
    resp1 = await plugin.after_model_callback(
        callback_context=mock_ctx_1, llm_response=llm_response
    )
    self.assertEqual(
        resp1.content.parts[0].function_call.args["retry_count"], 1
    )

    # First failure on invocation 2 -> count is ALSO 1 (isolated scope)
    resp2 = await plugin.after_model_callback(
        callback_context=mock_ctx_2, llm_response=llm_response
    )
    self.assertEqual(
        resp2.content.parts[0].function_call.args["retry_count"], 1
    )

    # Second failure on invocation 1 -> increments invocation 1's counter to 2
    resp3 = await plugin.after_model_callback(
        callback_context=mock_ctx_1, llm_response=llm_response
    )
    self.assertEqual(
        resp3.content.parts[0].function_call.args["retry_count"], 2
    )

  async def test_global_tracking_scope_for_models(self):
    """Checks that TrackingScope.GLOBAL shares failure counts across different invocations for models."""
    mock_agent = Mock(spec=LlmAgent)
    mock_agent.canonical_model = Mock()
    mock_agent.canonical_model.model = "TEST_MODEL_NAME"

    mock_inv_1 = Mock()
    mock_inv_1.agent = mock_agent
    mock_inv_1.invocation_id = "INVOCATION_1"

    mock_inv_2 = Mock()
    mock_inv_2.agent = mock_agent
    mock_inv_2.invocation_id = "INVOCATION_2"

    mock_ctx_1 = Mock(spec=CallbackContext)
    mock_ctx_1.get_invocation_context.return_value = mock_inv_1

    mock_ctx_2 = Mock(spec=CallbackContext)
    mock_ctx_2.get_invocation_context.return_value = mock_inv_2

    plugin = ReflectAndRetryModelPlugin(
        max_retries=5, tracking_scope=TrackingScope.GLOBAL
    )

    llm_response = LlmResponse(
        error_code=types.FinishReason.MALFORMED_FUNCTION_CALL,
        error_message="TEST_ERROR",
        finish_reason=types.FinishReason.MALFORMED_FUNCTION_CALL,
    )

    # First failure on invocation 1
    resp1 = await plugin.after_model_callback(
        callback_context=mock_ctx_1, llm_response=llm_response
    )
    self.assertEqual(
        resp1.content.parts[0].function_call.args["retry_count"], 1
    )

    # Second failure on invocation 2 should increment to 2 (shared global scope)
    resp2 = await plugin.after_model_callback(
        callback_context=mock_ctx_2, llm_response=llm_response
    )
    self.assertEqual(
        resp2.content.parts[0].function_call.args["retry_count"], 2
    )
