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

import asyncio
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import functions
from google.adk.plugins.reflect_retry_tool_plugin import GLOBAL_SCOPE_KEY
from google.adk.plugins.reflect_retry_tool_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.plugins.reflect_retry_tool_plugin import TrackingScope
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .. import testing_utils


class MockTool(BaseTool):
  """Mock tool for testing purposes."""

  def __init__(self, name: str = "mock_tool"):
    self.name = name
    self.description = f"Mock tool named {name}"

  async def run(self, **kwargs) -> Any:
    return "mock result"


class CustomErrorExtractionPlugin(ReflectAndRetryToolPlugin):
  """Custom plugin for testing error extraction from tool responses."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.error_conditions = {}

  def set_error_condition(self, condition_func):
    """Set a custom error condition function for testing."""
    self.error_condition = condition_func

  async def extract_error_from_result(
      self, *, tool, tool_args, tool_context, result
  ):
    """Extract error based on custom conditions set for testing."""
    if hasattr(self, "error_condition"):
      return self.error_condition(result)
    return None


def _answer(events: list[Any], name: str) -> types.Content:
  """Returns the user's answer to the paused call to `name` in `events`."""
  call = next(
      fc
      for event in events
      for fc in event.get_function_calls()
      if fc.name == name
  )
  return types.UserContent(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=call.id, name=name, response={"status": "done"}
              )
          )
      ]
  )


# Inheriting from IsolatedAsyncioTestCase ensures consistent behavior, because
# pytest-asyncio's own event-loop scoping varies across versions.
class TestReflectAndRetryToolPlugin(IsolatedAsyncioTestCase):
  """Comprehensive tests for ReflectAndRetryToolPlugin focusing on behavior."""

  def get_plugin(self):
    """Create a default plugin instance for testing."""
    return ReflectAndRetryToolPlugin()

  def get_custom_plugin(self):
    """Create a plugin with custom parameters."""
    return ReflectAndRetryToolPlugin(
        name="custom_plugin",
        max_retries=5,
        throw_exception_if_retry_exceeded=False,
    )

  def get_mock_tool(self):
    """Create a mock tool for testing."""
    return MockTool("test_tool_id")

  def get_mock_tool_context(self):
    """Create a mock tool context."""
    tool_context = Mock(spec=ToolContext)
    tool_context.actions = EventActions()
    return tool_context

  def get_custom_error_plugin(self):
    """Create a custom error extraction plugin for testing."""
    return CustomErrorExtractionPlugin(max_retries=3)

  def get_sample_tool_args(self):
    """Sample tool arguments for testing."""
    return {"param1": "value1", "param2": 42, "param3": True}

  async def test_plugin_initialization_default(self):
    """Test plugin initialization with default parameters."""
    plugin = self.get_plugin()

    self.assertEqual(plugin.name, "reflect_retry_tool_plugin")
    self.assertEqual(plugin.max_retries, 3)
    self.assertIs(plugin.throw_exception_if_retry_exceeded, True)

  async def test_plugin_initialization_custom(self):
    """Test plugin initialization with custom parameters."""
    plugin = ReflectAndRetryToolPlugin(
        name="custom_name",
        max_retries=10,
        throw_exception_if_retry_exceeded=False,
    )

    self.assertEqual(plugin.name, "custom_name")
    self.assertEqual(plugin.max_retries, 10)
    self.assertIsNot(plugin.throw_exception_if_retry_exceeded, True)

  async def test_after_tool_callback_successful_call(self):
    """Test after_tool_callback with successful tool call."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    result = {"success": True, "data": "test_data"}

    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=result,
    )

    # Should return None for successful calls
    self.assertIsNone(callback_result)

  async def test_after_tool_callback_ignore_retry_response(self):
    """Test that retry responses are ignored in after_tool_callback."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    retry_result = {"response_type": REFLECT_AND_RETRY_RESPONSE_TYPE}

    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=retry_result,
    )

    # Retry responses should be ignored
    self.assertIsNone(callback_result)

  async def test_on_tool_error_callback_max_retries_zero(self):
    """Test error callback when max_retries is 0.

    This should return None so that the exception is rethrown
    """
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=0)
    error = ValueError("Test error")

    with self.assertRaises(ValueError) as cm:
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Should re-raise the original exception when max_retries is 0
    self.assertIs(cm.exception, error)

  async def test_on_tool_error_callback_max_retries_zero_without_exception(
      self,
  ):
    """Test error callback when max_retries is 0 and exception is disabled."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=False
    )
    error = ValueError("Test error")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Should return a retry exceeded message instead of raising
    self.assertIsNotNone(result)
    self.assertEqual(result["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result["error_type"], "ValueError")
    self.assertEqual(result["retry_count"], 0)
    self.assertIn(
        "the retry limit has been exceeded", result["reflection_guidance"]
    )

  async def test_on_tool_error_callback_max_retries_zero_with_dict_error(self):
    """Test error callback when max_retries is 0 and error is a dict."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = CustomErrorExtractionPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=True
    )
    dict_error = {"status": "error", "message": "Custom dict error"}
    plugin.set_error_condition(lambda result: dict_error)

    with self.assertRaises(Exception) as cm:
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          result={"some": "result"},
      )

    # Should raise an Exception wrapping the dict
    self.assertNotIsInstance(cm.exception, TypeError)
    self.assertIn("Custom dict error", str(cm.exception))

  async def test_on_tool_error_callback_first_failure(self):
    """Test first tool failure creates reflection response."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Test error message")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertIsNotNone(result)
    self.assertEqual(result["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result["error_type"], "ValueError")
    self.assertEqual(result["error_details"], "Test error message")
    self.assertEqual(result["retry_count"], 1)
    self.assertIn("test_tool_id", result["reflection_guidance"])
    self.assertIn("Test error message", result["reflection_guidance"])

  async def test_retry_behavior_with_consecutive_failures(self):
    """Test the retry behavior with consecutive failures."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = RuntimeError("Runtime error")

    # First failure
    result1 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(result1["retry_count"], 1)

    # Second failure - should have different retry count based on plugin logic
    result2 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    # The plugin's internal logic determines the exact retry count
    self.assertIsNotNone(result2)
    self.assertEqual(result2["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result2["retry_count"], 2)

  async def test_different_tools_behavior(self):
    """Test behavior when using different tools."""
    plugin = self.get_plugin()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    tool1 = MockTool("tool1")
    tool2 = MockTool("tool2")
    error = ValueError("Test error")

    # First failure on tool1
    result1 = await plugin.on_tool_error_callback(
        tool=tool1,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(result1["retry_count"], 1)

    # Failure on tool2
    result2 = await plugin.on_tool_error_callback(
        tool=tool2,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    # Since tool is different, retry count should start over.
    self.assertIsNotNone(result2)
    self.assertEqual(result2["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result2["retry_count"], 1)

  async def test_max_retries_exceeded_with_exception(self):
    """Test that original exception is raised when max retries exceeded."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=True
    )
    error = ConnectionError("Connection failed")

    # First call should succeed and return a retry response
    await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Second call should exceed max_retries and raise
    with self.assertRaises(ConnectionError) as cm:
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Verify exception properties
    self.assertIs(cm.exception, error)

  async def test_max_retries_exceeded_with_dict_error(self):
    """Test that Exception is raised when max retries exceeded with dict error."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = CustomErrorExtractionPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=True
    )
    dict_error = {"status": "error", "message": "Custom dict error"}
    plugin.set_error_condition(lambda result: dict_error)

    # First call should fail and return a retry response
    result1 = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result={"some": "result"},
    )
    self.assertIsNotNone(result1)
    self.assertEqual(result1["retry_count"], 1)

    # Second call should exceed max_retries and raise
    with self.assertRaises(Exception) as cm:
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          result={"some": "result"},
      )

    # Verify exception properties
    self.assertNotIsInstance(cm.exception, TypeError)
    self.assertIn("Custom dict error", str(cm.exception))

  async def test_max_retries_exceeded_without_exception(self):
    """Test max retries exceeded returns failure message when exception is disabled."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=2, throw_exception_if_retry_exceeded=False
    )
    error = TimeoutError("Timeout occurred")

    # Call until we exceed the retry limit
    result = None
    for _ in range(3):
      result = await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    # Should get a retry exceeded message on the last call
    self.assertIsNotNone(result)
    self.assertEqual(result["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result["error_type"], "TimeoutError")
    self.assertIn(
        "the retry limit has been exceeded", result["reflection_guidance"]
    )
    self.assertIn("Do not attempt to use the", result["reflection_guidance"])

  async def test_successful_call_resets_retry_behavior(self):
    """Test that successful calls reset the retry behavior."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Test error")

    # First failure
    result1 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(result1["retry_count"], 1)

    # Successful call
    await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result={"success": True},
    )

    # Next failure should start fresh
    result2 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(result2["retry_count"], 1)  # Should restart from 1

  async def test_none_result_handling(self):
    """Test handling of None results in after_tool_callback."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()

    # None result should be handled gracefully
    callback_result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=None,
    )

    self.assertIsNone(callback_result)

  async def test_empty_tool_args_handling(self):
    """Test handling of empty tool arguments."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    empty_args = {}
    error = ValueError("Test error")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=empty_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertIsNotNone(result)
    # Empty args should be represented in the response
    self.assertIn("{}", result["reflection_guidance"])

  async def test_retry_count_progression(self):
    """Test that retry counts progress correctly for the same tool."""
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=5)
    error = ValueError("Test error")
    tool = MockTool("single_tool")

    for i in range(1, 4):
      result = await plugin.on_tool_error_callback(
          tool=tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
      self.assertEqual(result["retry_count"], i)

  async def test_max_retries_parameter_behavior(self):
    """Test that max_retries parameter affects behavior correctly."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    # Test with very low max_retries
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )
    error = ValueError("Test error")

    # First call is fine
    await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Second call exceeds limit
    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    # Should hit max retries quickly with max_retries=1
    self.assertIn(
        "the retry limit has been exceeded.", result["reflection_guidance"]
    )

  async def test_default_extract_error_returns_none(self):
    """Test that default extract_error_from_result returns None."""
    plugin = self.get_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    result = {"status": "success", "data": "some data"}

    error = await plugin.extract_error_from_result(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=result,
    )
    self.assertIsNone(error)

  async def test_custom_error_detection_and_success_handling(self):
    """Test custom error detection, success handling, and retry progression."""
    custom_error_plugin = self.get_custom_error_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    custom_error_plugin.set_error_condition(
        lambda result: result if result.get("status") == "error" else None
    )

    # Test error detection
    error_result = {"status": "error", "message": "Something went wrong"}
    callback_result = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=error_result,
    )
    self.assertIsNotNone(callback_result)
    self.assertEqual(
        callback_result["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE
    )
    self.assertEqual(callback_result["retry_count"], 1)

    # Test success handling
    success_result = {"status": "success", "data": "operation completed"}
    callback_result = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=success_result,
    )
    self.assertIsNone(callback_result)

  async def test_retry_state_management(self):
    """Test retry state management with custom errors and mixed error types."""
    custom_error_plugin = self.get_custom_error_plugin()
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    custom_error_plugin.set_error_condition(
        lambda result: result if result.get("failed") else None
    )

    # Custom error followed by exception
    custom_error = {"failed": True, "reason": "Network timeout"}
    result1 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=custom_error,
    )
    self.assertEqual(result1["retry_count"], 1)

    # Exception should increment retry count
    exception = ValueError("Invalid parameter")
    result2 = await custom_error_plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=exception,
    )
    self.assertEqual(result2["retry_count"], 2)

    # Success should reset
    success = {"result": "success"}
    result3 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=success,
    )
    self.assertIsNone(result3)

    # Next error should start fresh
    result4 = await custom_error_plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=custom_error,
    )
    self.assertEqual(result4["retry_count"], 1)

  async def test_hallucinating_tool_name(self):
    """Test that hallucinating tool name is handled correctly."""
    wrong_function_call = types.Part.from_function_call(
        name="increase_by_one", args={"x": 1}
    )
    correct_function_call = types.Part.from_function_call(
        name="increase", args={"x": 1}
    )
    responses: list[types.Content] = [
        wrong_function_call,
        correct_function_call,
        "response1",
    ]
    mock_model = testing_utils.MockModel.create(responses=responses)

    function_called = 0

    def increase(x: int) -> int:
      nonlocal function_called
      function_called += 1
      return x + 1

    agent = LlmAgent(name="root_agent", model=mock_model, tools=[increase])
    runner = testing_utils.TestInMemoryRunner(
        agent=agent, plugins=[self.get_plugin()]
    )

    events = await runner.run_async_with_new_session("test")
    # Filter out agent_state events (no content).
    events = [e for e in events if e.content is not None]

    # Assert that the first event is a function call with the wrong name
    assert events[0].content.parts[0].function_call.name == "increase_by_one"

    # Assert that the second event is a function response with the
    # reflection_guidance
    assert (
        events[1].content.parts[0].function_response.response["error_type"]
        == "ValueError"
    )
    assert (
        events[1].content.parts[0].function_response.response["retry_count"]
        == 1
    )
    assert (
        "Wrong Function Name"
        in events[1]
        .content.parts[0]
        .function_response.response["reflection_guidance"]
    )

    # Assert that the third event is a function call with the correct name
    assert events[2].content.parts[0].function_call.name == "increase"
    self.assertEqual(function_called, 1)

  async def test_negative_max_retries_rejected(self):
    """Test that a negative retry budget is rejected at construction."""
    with self.assertRaises(ValueError) as cm:
      ReflectAndRetryToolPlugin(max_retries=-1)

    self.assertIn("non-negative", str(cm.exception))

  async def test_reflection_response_does_not_reset_the_retry_count(self):
    """Test that feeding a reflection response back does not clear failures.

    The plugin's own reflection guidance is delivered to the model as the
    tool result, so it comes back through after_tool_callback. Treating it
    as a success would reset the counter and make the retry budget
    unenforceable.
    """
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(max_retries=3)
    error = ValueError("Test error")

    reflection = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(reflection["retry_count"], 1)

    passthrough = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=reflection,
    )
    self.assertIsNone(passthrough)

    next_failure = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(next_failure["retry_count"], 2)

  async def test_cancelled_error_ignored(self):
    """Test that cancelled tool calls do not increment retry count or raise."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    plugin = ReflectAndRetryToolPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=True
    )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=asyncio.CancelledError(),
    )
    self.assertIsNone(result)

  async def _run_flaky_agent(
      self,
      plugin: ReflectAndRetryToolPlugin,
      *,
      num_invocations: int,
      fail_every_call: bool,
  ) -> None:
    """Runs invocations whose tool fails, so the plugin records failures."""
    calls = 0

    def flaky_tool() -> dict[str, int]:
      """A tool that fails on its first call, or always."""
      nonlocal calls
      calls += 1
      if fail_every_call or calls % 2 == 1:
        raise ValueError("tool failed")
      return {"calls": calls}

    function_call = types.Part.from_function_call(name="flaky_tool", args={})
    done = types.Part.from_text(text="done")
    mock_model = testing_utils.MockModel.create(
        responses=[function_call, function_call, done] * num_invocations
    )
    agent = LlmAgent(name="root_agent", model=mock_model, tools=[flaky_tool])
    runner = testing_utils.TestInMemoryRunner(agent=agent, plugins=[plugin])
    for _ in range(num_invocations):
      await runner.run_async_with_new_session("go")

  async def test_invocation_scope_counts_are_dropped_after_each_run(
      self,
  ) -> None:
    """Invocation-scoped failure counts do not outlive their invocation."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    await self._run_flaky_agent(
        plugin, num_invocations=3, fail_every_call=False
    )

    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_invocation_scope_counts_are_dropped_when_retries_run_out(
      self,
  ) -> None:
    """Counts left at the retry limit are dropped when the run ends."""
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )

    await self._run_flaky_agent(plugin, num_invocations=2, fail_every_call=True)

    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_invocation_scope_counts_are_dropped_when_the_run_fails(
      self,
  ) -> None:
    """Counts are dropped when the plugin re-raises and the run fails."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    with self.assertRaises(Exception):
      await self._run_flaky_agent(
          plugin, num_invocations=1, fail_every_call=True
      )

    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_global_scope_counts_survive_the_end_of_a_run(self) -> None:
    """Global failure counts are shared across runs, so they are kept."""
    plugin = ReflectAndRetryToolPlugin(
        max_retries=5,
        throw_exception_if_retry_exceeded=False,
        tracking_scope=TrackingScope.GLOBAL,
    )

    await self._run_flaky_agent(plugin, num_invocations=1, fail_every_call=True)

    self.assertEqual(
        plugin._scoped_failure_counters,
        {GLOBAL_SCOPE_KEY: {"flaky_tool": 2}},
    )

  async def test_custom_invocation_scope_key_counts_are_dropped_when_runs_end(
      self,
  ) -> None:
    """Counts under an overridden per-invocation scope key are dropped too."""

    class _UserInvocationScopedPlugin(ReflectAndRetryToolPlugin):

      def _get_scope_key(self, tool_context: ToolContext) -> str:
        return f"{tool_context.user_id}:{tool_context.invocation_id}"

    plugin = _UserInvocationScopedPlugin(max_retries=1)
    await self._run_flaky_agent(
        plugin, num_invocations=1, fail_every_call=False
    )
    self.assertEqual(plugin._scoped_failure_counters, {})

    with self.assertRaises(Exception):
      await self._run_flaky_agent(
          plugin, num_invocations=1, fail_every_call=True
      )
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_custom_user_scope_key_counts_survive_the_end_of_runs(
      self,
  ) -> None:
    """Counts under an overridden per-user scope key outlive each run."""

    class _UserScopedPlugin(ReflectAndRetryToolPlugin):

      def _get_scope_key(self, tool_context: ToolContext) -> str:
        return f"user:{tool_context.user_id}"

    plugin = _UserScopedPlugin(
        max_retries=5, throw_exception_if_retry_exceeded=False
    )
    await self._run_flaky_agent(plugin, num_invocations=1, fail_every_call=True)
    self.assertEqual(
        plugin._scoped_failure_counters, {"user:test_user": {"flaky_tool": 2}}
    )

    plugin.max_retries = 2
    plugin.throw_exception_if_retry_exceeded = True
    with self.assertRaises(Exception):
      await self._run_flaky_agent(
          plugin, num_invocations=1, fail_every_call=True
      )
    self.assertEqual(
        plugin._scoped_failure_counters, {"user:test_user": {"flaky_tool": 3}}
    )

  async def test_pending_long_running_tool_keeps_the_retry_count(
      self,
  ) -> None:
    """The answer to a pending long-running call continues the retry budget."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A tool that always fails."""
      raise ValueError("tool failed")

    def pending_lro() -> dict[str, str]:
      """A long-running tool that reports a pending status."""
      return {"status": "pending"}

    call_failing_tool = types.Part.from_function_call(
        name="failing_tool", args={}
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=LlmAgent(
                name="root_agent",
                model=testing_utils.MockModel.create(
                    responses=[
                        call_failing_tool,
                        types.Part.from_function_call(
                            name="pending_lro", args={}
                        ),
                        types.Part.from_text(text="waiting"),
                        call_failing_tool,
                    ]
                ),
                tools=[failing_tool, LongRunningFunctionTool(func=pending_lro)],
            ),
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    events = await runner.run_async("go")
    self.assertEqual(
        plugin._scoped_failure_counters,
        {events[0].invocation_id: {"failing_tool": 1}},
    )
    with self.assertRaisesRegex(RuntimeError, "tool failed"):
      await runner.run_async(_answer(events, "pending_lro"))
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_pause_in_non_resumable_app_keeps_the_retry_count(
      self,
  ) -> None:
    """The answer to an interrupt continues a non-resumable invocation too."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A tool that always fails."""
      raise ValueError("tool failed")

    def wait_for_approval() -> None:
      """A long-running tool that leaves its call pending."""
      return None

    call_failing_tool = types.Part.from_function_call(
        name="failing_tool", args={}
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=LlmAgent(
                name="root_agent",
                model=testing_utils.MockModel.create(
                    responses=[
                        call_failing_tool,
                        types.Part.from_function_call(
                            name="wait_for_approval", args={}
                        ),
                        call_failing_tool,
                    ]
                ),
                tools=[
                    failing_tool,
                    LongRunningFunctionTool(func=wait_for_approval),
                ],
            ),
            plugins=[plugin],
        )
    )

    events = await runner.run_async("go")
    self.assertEqual(
        plugin._scoped_failure_counters,
        {events[0].invocation_id: {"failing_tool": 1}},
    )
    with self.assertRaisesRegex(RuntimeError, "tool failed"):
      await runner.run_async(_answer(events, "wait_for_approval"))
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_after_agent_callback_on_a_pause_keeps_the_retry_count(
      self,
  ) -> None:
    """An after-agent callback event emitted on a pause keeps the counts."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A tool that always fails."""
      raise ValueError("tool failed")

    def wait_for_approval() -> None:
      """A long-running tool that leaves its call pending."""
      return None

    def record_state(callback_context: CallbackContext) -> None:
      callback_context.state["after_agent_ran"] = True

    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=LlmAgent(
                name="root_agent",
                model=testing_utils.MockModel.create(
                    responses=[
                        types.Part.from_function_call(
                            name="failing_tool", args={}
                        ),
                        types.Part.from_function_call(
                            name="wait_for_approval", args={}
                        ),
                    ]
                ),
                tools=[
                    failing_tool,
                    LongRunningFunctionTool(func=wait_for_approval),
                ],
                after_agent_callback=record_state,
            ),
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    events = await runner.run_async("go")
    self.assertEqual(events[-1].actions.state_delta, {"after_agent_ran": True})
    self.assertEqual(
        plugin._scoped_failure_counters,
        {events[0].invocation_id: {"failing_tool": 1}},
    )

  async def test_invocation_scope_counts_carry_over_a_pause(self) -> None:
    """A paused invocation keeps its retry budget into the resumed run."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A tool that always fails."""
      raise ValueError("tool failed")

    def wait_for_approval() -> None:
      """A long-running tool that leaves its call pending."""
      return None

    call_failing_tool = types.Part.from_function_call(
        name="failing_tool", args={}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[
            call_failing_tool,
            types.Part.from_function_call(name="wait_for_approval", args={}),
            call_failing_tool,
            types.Part.from_text(text="done"),
        ]
    )
    agent = LlmAgent(
        name="root_agent",
        model=mock_model,
        tools=[failing_tool, LongRunningFunctionTool(func=wait_for_approval)],
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=agent,
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    paused_events = await runner.run_async("go")
    invocation_id = paused_events[0].invocation_id
    self.assertEqual(
        plugin._scoped_failure_counters, {invocation_id: {"failing_tool": 1}}
    )

    approval_call = next(
        fc
        for event in paused_events
        for fc in event.get_function_calls()
        if fc.name == "wait_for_approval"
    )
    with self.assertRaisesRegex(RuntimeError, "tool failed"):
      await runner.run_async(
          invocation_id=invocation_id,
          new_message=types.UserContent(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id=approval_call.id,
                          name="wait_for_approval",
                          response={"approved": True},
                      )
                  )
              ]
          ),
      )
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_pause_in_one_parallel_branch_keeps_the_retry_count(
      self,
  ) -> None:
    """A branch paused while its sibling finishes later keeps its counts."""
    plugin = ReflectAndRetryToolPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A tool that always fails."""
      raise ValueError("tool failed")

    def wait_for_approval() -> None:
      """A long-running tool that leaves its call pending."""
      return None

    async def slow_tool() -> dict[str, str]:
      """A tool that finishes after the other branch has paused."""
      await asyncio.sleep(0.05)
      return {"status": "ok"}

    paused_branch = LlmAgent(
        name="paused_branch",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(name="failing_tool", args={}),
                types.Part.from_function_call(
                    name="wait_for_approval", args={}
                ),
            ]
        ),
        tools=[failing_tool, LongRunningFunctionTool(func=wait_for_approval)],
    )
    finishing_branch = LlmAgent(
        name="finishing_branch",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(name="slow_tool", args={}),
                types.Part.from_text(text="done"),
            ]
        ),
        tools=[slow_tool],
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=ParallelAgent(
                name="root_agent",
                sub_agents=[paused_branch, finishing_branch],
            ),
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    events = await runner.run_async("go")
    self.assertEqual(events[-1].author, "finishing_branch")
    self.assertEqual(
        plugin._scoped_failure_counters,
        {events[0].invocation_id: {"failing_tool": 1}},
    )

  async def test_confirmation_request_does_not_reset_the_retry_count(
      self,
  ) -> None:
    """A confirmation-gated tool that keeps failing runs out of retries."""

    class _ErrorDictPlugin(ReflectAndRetryToolPlugin):

      async def extract_error_from_result(
          self,
          *,
          tool: BaseTool,
          tool_args: dict[str, Any],
          tool_context: ToolContext,
          result: Any,
      ) -> dict[str, Any] | None:
        if isinstance(result, dict) and "error" in result:
          return result
        return None

    plugin = _ErrorDictPlugin(max_retries=1)

    def failing_tool() -> dict[str, str]:
      """A confirmation-gated tool that always fails."""
      raise ValueError("tool failed")

    call_failing_tool = types.Part.from_function_call(
        name="failing_tool", args={}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[
            call_failing_tool,
            call_failing_tool,
            types.Part.from_text(text="done"),
        ]
    )
    agent = LlmAgent(
        name="root_agent",
        model=mock_model,
        tools=[FunctionTool(func=failing_tool, require_confirmation=True)],
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=agent,
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    def confirm(events: list[Any]) -> types.Content:
      confirmation_call = next(
          fc
          for event in events
          for fc in event.get_function_calls()
          if fc.name == functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
      )
      return types.UserContent(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=confirmation_call.id,
                      name=functions.REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
                      response={"confirmed": True},
                  )
              )
          ]
      )

    events = await runner.run_async("go")
    invocation_id = events[0].invocation_id
    self.assertEqual(plugin._scoped_failure_counters, {})
    events = await runner.run_async(
        invocation_id=invocation_id, new_message=confirm(events)
    )
    self.assertEqual(
        plugin._scoped_failure_counters, {invocation_id: {"failing_tool": 1}}
    )

    with self.assertRaisesRegex(RuntimeError, "tool failed"):
      await runner.run_async(
          invocation_id=invocation_id, new_message=confirm(events)
      )
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_credential_request_does_not_reset_the_retry_count(
      self,
  ) -> None:
    """A credential-gated tool that keeps failing runs out of retries."""

    class _ErrorDictPlugin(ReflectAndRetryToolPlugin):

      async def extract_error_from_result(
          self,
          *,
          tool: BaseTool,
          tool_args: dict[str, Any],
          tool_context: ToolContext,
          result: Any,
      ) -> dict[str, Any] | None:
        if isinstance(result, dict) and "error" in result:
          return result
        return None

    plugin = _ErrorDictPlugin(max_retries=1)
    auth_config = AuthConfig(
        auth_scheme=APIKey(**{"in": APIKeyIn.header, "name": "X-Api-Key"}),
        raw_auth_credential=AuthCredential(
            auth_type=AuthCredentialTypes.API_KEY,
            api_key="placeholder",
        ),
        credential_key="test_api_key",
    )
    needs_credential = True
    credential_requests = 0

    def failing_tool(tool_context: ToolContext) -> dict[str, str]:
      """A credential-gated tool that always fails once credentials arrive."""
      nonlocal needs_credential, credential_requests
      if needs_credential:
        needs_credential = False
        credential_requests += 1
        tool_context.request_credential(auth_config)
        if credential_requests == 1:
          return {"error": "credential required"}
        return {"status": "pending"}
      needs_credential = True
      raise ValueError("tool failed")

    call_failing_tool = types.Part.from_function_call(
        name="failing_tool", args={}
    )
    mock_model = testing_utils.MockModel.create(
        responses=[
            call_failing_tool,
            call_failing_tool,
            types.Part.from_text(text="done"),
        ]
    )
    agent = LlmAgent(
        name="root_agent",
        model=mock_model,
        tools=[FunctionTool(func=failing_tool)],
    )
    runner = testing_utils.InMemoryRunner(
        app=App(
            name="test_app",
            root_agent=agent,
            plugins=[plugin],
            resumability_config=ResumabilityConfig(is_resumable=True),
        )
    )

    def supply_credential(events: list[Any]) -> types.Content:
      auth_call = next(
          fc
          for event in events
          for fc in event.get_function_calls()
          if fc.name == functions.REQUEST_EUC_FUNCTION_CALL_NAME
      )
      supplied = auth_config.model_copy(
          update={
              "exchanged_auth_credential": AuthCredential(
                  auth_type=AuthCredentialTypes.API_KEY,
                  api_key="secret",
              )
          }
      )
      return types.UserContent(
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=auth_call.id,
                      name=functions.REQUEST_EUC_FUNCTION_CALL_NAME,
                      response=supplied.model_dump(by_alias=True),
                  )
              )
          ]
      )

    events = await runner.run_async("go")
    invocation_id = events[0].invocation_id
    self.assertEqual(plugin._scoped_failure_counters, {})
    events = await runner.run_async(
        invocation_id=invocation_id, new_message=supply_credential(events)
    )
    self.assertEqual(
        plugin._scoped_failure_counters, {invocation_id: {"failing_tool": 1}}
    )

    with self.assertRaisesRegex(RuntimeError, "tool failed"):
      await runner.run_async(
          invocation_id=invocation_id, new_message=supply_credential(events)
      )
    self.assertEqual(plugin._scoped_failure_counters, {})

  async def test_long_running_tool_failures_are_dropped_when_run_ends(
      self,
  ) -> None:
    """Failed long-running tool calls do not leak invocation retry counts."""

    def failing_lro() -> dict[str, str]:
      """A long-running tool that fails synchronously."""
      raise ValueError("lro failed")

    call_failing_lro = types.Part.from_function_call(
        name="failing_lro", args={}
    )
    done = types.Part.from_text(text="done")

    plugin_recovered = ReflectAndRetryToolPlugin(max_retries=1)
    runner_recovered = testing_utils.TestInMemoryRunner(
        agent=LlmAgent(
            name="root_agent",
            model=testing_utils.MockModel.create(
                responses=[call_failing_lro, done]
            ),
            tools=[LongRunningFunctionTool(func=failing_lro)],
        ),
        plugins=[plugin_recovered],
    )
    await runner_recovered.run_async_with_new_session("go")
    self.assertEqual(plugin_recovered._scoped_failure_counters, {})

    plugin_exhausted = ReflectAndRetryToolPlugin(max_retries=1)
    runner_exhausted = testing_utils.TestInMemoryRunner(
        agent=LlmAgent(
            name="root_agent",
            model=testing_utils.MockModel.create(
                responses=[call_failing_lro, call_failing_lro, done]
            ),
            tools=[LongRunningFunctionTool(func=failing_lro)],
        ),
        plugins=[plugin_exhausted],
    )
    with self.assertRaisesRegex(RuntimeError, "lro failed"):
      await runner_exhausted.run_async_with_new_session("go")
    self.assertEqual(plugin_exhausted._scoped_failure_counters, {})
