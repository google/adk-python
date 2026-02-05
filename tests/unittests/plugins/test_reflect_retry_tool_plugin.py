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

from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.reflect_retry_tool_plugin import REFLECT_AND_RETRY_RESPONSE_TYPE
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
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


# Inheriting from IsolatedAsyncioTestCase ensures consistent behavior.
# See https://github.com/pytest-dev/pytest-asyncio/issues/1039
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
