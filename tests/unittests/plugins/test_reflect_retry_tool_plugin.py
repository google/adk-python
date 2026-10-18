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
import dataclasses
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.reflect_retry_tool_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ToolReflectionContext
from google.adk.tools.base_tool import BaseTool
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
    return Mock(spec=ToolContext)

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

  async def test_tool_reflection_context_is_frozen_dataclass(self):
    """ToolReflectionContext is an immutable frozen dataclass."""
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("boom")

    ctx = ToolReflectionContext(
        tool_name="test_tool",
        tool_args=sample_tool_args,
        error=error,
        error_details="ValueError: boom",
        retry_count=1,
        max_retries=3,
        tool_context=mock_tool_context,
    )

    self.assertTrue(dataclasses.is_dataclass(ctx))
    self.assertEqual(ctx.tool_name, "test_tool")
    self.assertEqual(ctx.tool_args, sample_tool_args)
    self.assertIs(ctx.error, error)
    self.assertEqual(ctx.error_details, "ValueError: boom")
    self.assertEqual(ctx.retry_count, 1)
    self.assertEqual(ctx.max_retries, 3)
    self.assertIs(ctx.tool_context, mock_tool_context)

    with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
      ctx.tool_name = "mutated"  # type: ignore[misc]

  async def test_string_reflection_guidance_normal_retry(self):
    """Static string reflection guidance overrides default guidance on retry."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Something failed")
    plugin = ReflectAndRetryToolPlugin(
        max_retries=2,
        reflection_guidance="Static custom retry guidance.",
    )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertIsNotNone(result)
    self.assertEqual(result["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE)
    self.assertEqual(result["error_type"], "ValueError")
    self.assertEqual(result["error_details"], "Something failed")
    self.assertEqual(result["retry_count"], 1)
    self.assertEqual(
        result["reflection_guidance"], "Static custom retry guidance."
    )

  async def test_string_retry_exceeded_guidance_when_limit_exceeded(self):
    """Static retry exceeded guidance is returned when max retries exceeded."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = TimeoutError("Timed out")
    plugin = ReflectAndRetryToolPlugin(
        max_retries=1,
        throw_exception_if_retry_exceeded=False,
        retry_exceeded_guidance="Static custom exceeded guidance.",
    )

    first = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertIn("Reflection Guidance:", first["reflection_guidance"])
    self.assertEqual(first["retry_count"], 1)

    exceeded = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertIsNotNone(exceeded)
    self.assertEqual(exceeded["retry_count"], 1)
    self.assertEqual(
        exceeded["reflection_guidance"], "Static custom exceeded guidance."
    )
    self.assertEqual(exceeded["error_type"], "TimeoutError")
    self.assertEqual(exceeded["error_details"], "Timed out")

  async def test_string_retry_exceeded_guidance_zero_retries(self):
    """Static retry exceeded guidance returned on zero-retry without throwing."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = RuntimeError("Immediate fatal")
    plugin = ReflectAndRetryToolPlugin(
        max_retries=0,
        throw_exception_if_retry_exceeded=False,
        retry_exceeded_guidance="No retries allowed guidance.",
    )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertIsNotNone(result)
    self.assertEqual(result["retry_count"], 0)
    self.assertEqual(
        result["reflection_guidance"], "No retries allowed guidance."
    )
    self.assertEqual(result["error_type"], "RuntimeError")

  async def test_sync_callable_provider_context_data_normal_retry(self):
    """Sync callable guidance provider receives accurate ToolReflectionContext."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Invalid arg")

    captured_ctx: list[ToolReflectionContext] = []

    def provider(ctx: ToolReflectionContext) -> str:
      captured_ctx.append(ctx)
      return (
          f"Retry {ctx.tool_name} attempt {ctx.retry_count}/{ctx.max_retries}"
      )

    plugin = ReflectAndRetryToolPlugin(
        max_retries=3,
        reflection_guidance=provider,
    )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertEqual(len(captured_ctx), 1)
    ctx = captured_ctx[0]
    self.assertEqual(ctx.tool_name, "test_tool_id")
    self.assertEqual(ctx.tool_args, sample_tool_args)
    self.assertIs(ctx.error, error)
    self.assertEqual(ctx.error_details, "ValueError: Invalid arg")
    self.assertEqual(ctx.retry_count, 1)
    self.assertEqual(ctx.max_retries, 3)
    self.assertIs(ctx.tool_context, mock_tool_context)

    self.assertEqual(
        result["reflection_guidance"], "Retry test_tool_id attempt 1/3"
    )

  async def test_sync_callable_provider_context_data_exceeded_retry(self):
    """Sync callable exceeded provider receives accurate context on exceeded."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ConnectionError("Dropped")

    captured_ctx: list[ToolReflectionContext] = []

    def exceeded_provider(ctx: ToolReflectionContext) -> str:
      captured_ctx.append(ctx)
      return f"Exceeded {ctx.tool_name} count={ctx.retry_count}"

    plugin = ReflectAndRetryToolPlugin(
        max_retries=2,
        throw_exception_if_retry_exceeded=False,
        retry_exceeded_guidance=exceeded_provider,
    )

    for _ in range(2):
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertEqual(len(captured_ctx), 1)
    ctx = captured_ctx[0]
    self.assertEqual(ctx.tool_name, "test_tool_id")
    self.assertEqual(ctx.retry_count, 2)
    self.assertEqual(ctx.max_retries, 2)
    self.assertEqual(
        result["reflection_guidance"], "Exceeded test_tool_id count=2"
    )

  async def test_async_callable_provider_normal_and_exceeded(self):
    """Async callable guidance providers are awaited properly."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Async test error")

    async def async_retry(ctx: ToolReflectionContext) -> str:
      await asyncio.sleep(0.001)
      return f"Async retry for {ctx.tool_name}"

    async def async_exceeded(ctx: ToolReflectionContext) -> str:
      await asyncio.sleep(0.001)
      return f"Async exceeded for {ctx.tool_name}"

    plugin = ReflectAndRetryToolPlugin(
        max_retries=1,
        throw_exception_if_retry_exceeded=False,
        reflection_guidance=async_retry,
        retry_exceeded_guidance=async_exceeded,
    )

    first = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(
        first["reflection_guidance"], "Async retry for test_tool_id"
    )

    second = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(
        second["reflection_guidance"], "Async exceeded for test_tool_id"
    )

  async def test_subclass_override_build_reflection_guidance(self):
    """Subclass overriding build_reflection_guidance generates custom guidance."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Hook test")

    class HookPlugin(ReflectAndRetryToolPlugin):

      async def build_reflection_guidance(
          self, ctx: ToolReflectionContext
      ) -> str:
        return f"Hook reflection: {ctx.tool_name} attempt={ctx.retry_count}"

    plugin = HookPlugin(max_retries=2)
    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(
        result["reflection_guidance"], "Hook reflection: test_tool_id attempt=1"
    )

  async def test_subclass_override_build_retry_exceeded_guidance(self):
    """Subclass hook build_retry_exceeded_guidance operates on exceeded path."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Exceeded hook test")

    class HookExceededPlugin(ReflectAndRetryToolPlugin):

      async def build_retry_exceeded_guidance(
          self, ctx: ToolReflectionContext
      ) -> str:
        return f"Hook exceeded: {ctx.tool_name} max={ctx.max_retries}"

    plugin_zero = HookExceededPlugin(
        max_retries=0, throw_exception_if_retry_exceeded=False
    )
    res_zero = await plugin_zero.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(
        res_zero["reflection_guidance"], "Hook exceeded: test_tool_id max=0"
    )

    plugin_one = HookExceededPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )
    await plugin_one.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    res_exceeded = await plugin_one.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(
        res_exceeded["reflection_guidance"], "Hook exceeded: test_tool_id max=1"
    )

  async def test_empty_string_guidance_is_valid(self):
    """Empty strings are valid custom guidance and do not fall back."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Empty test")

    plugin_empty_str = ReflectAndRetryToolPlugin(
        max_retries=1,
        reflection_guidance="",
        retry_exceeded_guidance="",
        throw_exception_if_retry_exceeded=False,
    )
    res1 = await plugin_empty_str.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(res1["reflection_guidance"], "")

    res2 = await plugin_empty_str.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(res2["reflection_guidance"], "")

    plugin_empty_callable = ReflectAndRetryToolPlugin(
        max_retries=1,
        reflection_guidance=lambda ctx: "",
    )
    res3 = await plugin_empty_callable.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(res3["reflection_guidance"], "")

  async def test_reject_non_string_provider_results(self):
    """Non-string results returned by providers or hooks raise TypeError."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Type test")

    plugin_none = ReflectAndRetryToolPlugin(
        max_retries=1,
        reflection_guidance=lambda ctx: None,  # type: ignore[arg-type]
    )
    with self.assertRaises(TypeError):
      await plugin_none.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    plugin_int = ReflectAndRetryToolPlugin(
        max_retries=1,
        reflection_guidance=lambda ctx: 12345,  # type: ignore[arg-type]
    )
    with self.assertRaises(TypeError):
      await plugin_int.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    async def async_bad(ctx: ToolReflectionContext) -> int:
      return 999

    plugin_async_bad = ReflectAndRetryToolPlugin(
        max_retries=1,
        reflection_guidance=async_bad,  # type: ignore[arg-type]
    )
    with self.assertRaises(TypeError):
      await plugin_async_bad.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

    class BadHookPlugin(ReflectAndRetryToolPlugin):

      async def build_reflection_guidance(
          self, ctx: ToolReflectionContext
      ) -> str:
        return 42  # type: ignore[return-value]

    with self.assertRaises(TypeError):
      await BadHookPlugin().on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )

  async def test_reject_non_string_non_callable_provider_instance(self):
    """Non-string non-callable passed as provider raises TypeError."""
    with self.assertRaises(TypeError):
      ReflectAndRetryToolPlugin(
          reflection_guidance=123  # type: ignore[arg-type]
      )

    with self.assertRaises(TypeError):
      ReflectAndRetryToolPlugin(
          retry_exceeded_guidance=["invalid"]  # type: ignore[arg-type]
      )

  async def test_propagate_provider_exceptions(self):
    """Exceptions raised inside providers or subclass hooks propagate uncaught."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Initial error")

    def sync_fail(ctx: ToolReflectionContext) -> str:
      raise ArithmeticError("Sync provider explosion")

    plugin_sync = ReflectAndRetryToolPlugin(
        max_retries=2, reflection_guidance=sync_fail
    )
    with self.assertRaises(ArithmeticError) as cm_sync:
      await plugin_sync.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
    self.assertEqual(str(cm_sync.exception), "Sync provider explosion")

    async def async_fail(ctx: ToolReflectionContext) -> str:
      raise KeyError("Async provider explosion")

    plugin_async = ReflectAndRetryToolPlugin(
        max_retries=2, reflection_guidance=async_fail
    )
    with self.assertRaises(KeyError) as cm_async:
      await plugin_async.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
    self.assertIn("Async provider explosion", str(cm_async.exception))

    class FailHookPlugin(ReflectAndRetryToolPlugin):

      async def build_reflection_guidance(
          self, ctx: ToolReflectionContext
      ) -> str:
        raise OSError("Hook explosion")

    with self.assertRaises(OSError) as cm_hook:
      await FailHookPlugin().on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
    self.assertEqual(str(cm_hook.exception), "Hook explosion")

  async def test_await_provider_outside_tracker_lock(self):
    """Async guidance providers are evaluated outside the failure tracker lock."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Lock check error")

    plugin: ReflectAndRetryToolPlugin | None = None
    lock_was_free = False
    concurrent_lock_acquired = False

    async def lock_checking_provider(ctx: ToolReflectionContext) -> str:
      nonlocal lock_was_free, concurrent_lock_acquired
      assert plugin is not None
      lock_was_free = not plugin._lock.locked()
      async with plugin._lock:
        concurrent_lock_acquired = True
      return "Lock check passed"

    plugin = ReflectAndRetryToolPlugin(
        max_retries=2, reflection_guidance=lock_checking_provider
    )

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    self.assertTrue(lock_was_free)
    self.assertTrue(concurrent_lock_acquired)
    self.assertEqual(result["reflection_guidance"], "Lock check passed")

  async def test_throwing_mode_behavior(self):
    """Throwing mode returns guidance on retry and raises on exceeded."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Mode test")

    exceeded_called = False

    def on_exceeded(ctx: ToolReflectionContext) -> str:
      nonlocal exceeded_called
      exceeded_called = True
      return "Should not be called"

    plugin = ReflectAndRetryToolPlugin(
        max_retries=1,
        throw_exception_if_retry_exceeded=True,
        reflection_guidance="Retry custom",
        retry_exceeded_guidance=on_exceeded,
    )

    res1 = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertEqual(res1["reflection_guidance"], "Retry custom")
    self.assertFalse(exceeded_called)

    with self.assertRaises(ValueError):
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
    self.assertFalse(exceeded_called)

    zero_plugin = ReflectAndRetryToolPlugin(
        max_retries=0,
        throw_exception_if_retry_exceeded=True,
        retry_exceeded_guidance=on_exceeded,
    )
    with self.assertRaises(ValueError):
      await zero_plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args=sample_tool_args,
          tool_context=mock_tool_context,
          error=error,
      )
    self.assertFalse(exceeded_called)

  async def test_exact_defaults_preserved_character_for_character(self):
    """Exact default reflection and exceeded guidance strings match original."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()
    error = ValueError("Default check")

    plugin = ReflectAndRetryToolPlugin(
        max_retries=1, throw_exception_if_retry_exceeded=False
    )

    res_retry = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertIn(
        "The call to tool `test_tool_id` failed.",
        res_retry["reflection_guidance"],
    )
    self.assertIn("retry attempt **1 of 1**", res_retry["reflection_guidance"])
    self.assertIn("ValueError: Default check", res_retry["reflection_guidance"])
    self.assertEqual(res_retry["retry_count"], 1)
    self.assertEqual(
        res_retry["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE
    )
    self.assertEqual(res_retry["error_type"], "ValueError")
    self.assertEqual(res_retry["error_details"], "Default check")

    res_exceeded = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        error=error,
    )
    self.assertIn(
        "has failed consecutively 1 times and the retry limit has been"
        " exceeded.",
        res_exceeded["reflection_guidance"],
    )
    self.assertIn(
        "Do not attempt to use the `test_tool_id` tool again for this task.",
        res_exceeded["reflection_guidance"],
    )
    self.assertEqual(res_exceeded["retry_count"], 1)
    self.assertEqual(
        res_exceeded["response_type"], REFLECT_AND_RETRY_RESPONSE_TYPE
    )
    self.assertEqual(res_exceeded["error_type"], "ValueError")
    self.assertEqual(res_exceeded["error_details"], "Default check")

  async def test_custom_provider_with_extracted_error_from_result(self):
    """Custom provider handles non-exception dict errors from after_tool."""
    mock_tool = self.get_mock_tool()
    mock_tool_context = self.get_mock_tool_context()
    sample_tool_args = self.get_sample_tool_args()

    captured_ctx: list[ToolReflectionContext] = []

    def provider(ctx: ToolReflectionContext) -> str:
      captured_ctx.append(ctx)
      return f"Handled dict error with code: {ctx.error.get('code')}"

    class CustomExtractor(ReflectAndRetryToolPlugin):

      async def extract_error_from_result(
          self, *, tool, tool_args, tool_context, result
      ):
        if isinstance(result, dict) and result.get("status") == "error":
          return result
        return None

    plugin = CustomExtractor(max_retries=2, reflection_guidance=provider)
    dict_error = {"status": "error", "code": "E1001", "msg": "Bad input"}

    result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=sample_tool_args,
        tool_context=mock_tool_context,
        result=dict_error,
    )

    self.assertIsNotNone(result)
    self.assertEqual(
        result["reflection_guidance"], "Handled dict error with code: E1001"
    )
    self.assertEqual(result["error_type"], "ToolError")
    self.assertEqual(result["error_details"], str(dict_error))
    self.assertEqual(len(captured_ctx), 1)
    self.assertEqual(captured_ctx[0].error, dict_error)
    self.assertEqual(captured_ctx[0].error_details, str(dict_error))
