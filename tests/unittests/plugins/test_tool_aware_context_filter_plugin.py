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

"""Tests for ToolAwareContextFilterPlugin."""

import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.tool_aware_context_filter_plugin import (
    ToolAwareContextFilterPlugin,
)
from google.genai.types import Content, FunctionCall, FunctionResponse, Part


class TestToolAwareContextFilterPlugin:
  """Tests for ToolAwareContextFilterPlugin."""

  def test_init(self):
    """Test plugin initialization."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=2)
    assert plugin._num_invocations_to_keep == 2
    assert plugin._custom_filter is None

  def test_no_filtering_when_disabled(self):
    """Test that no filtering occurs when num_invocations_to_keep is None."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=None)

    contents = [
        Content(role="user", parts=[Part(text="Hello")]),
        Content(role="model", parts=[Part(text="Hi")]),
        Content(role="user", parts=[Part(text="How are you?")]),
        Content(role="model", parts=[Part(text="I'm good")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # No filtering should occur
    assert len(request.contents) == 4

  def test_simple_invocations_no_tool_calls(self):
    """Test filtering simple Q&A without tool calls."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=2)

    # Create 3 simple invocations
    contents = [
        # Invocation 1
        Content(role="user", parts=[Part(text="Hello")]),
        Content(role="model", parts=[Part(text="Hi")]),
        # Invocation 2
        Content(role="user", parts=[Part(text="How are you?")]),
        Content(role="model", parts=[Part(text="I'm good")]),
        # Invocation 3
        Content(role="user", parts=[Part(text="What's your name?")]),
        Content(role="model", parts=[Part(text="I'm Claude")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should keep last 2 invocations (indices 2-5)
    assert len(request.contents) == 4
    assert request.contents[0].parts[0].text == "How are you?"
    assert request.contents[-1].parts[0].text == "I'm Claude"

  def test_tool_call_sequence_kept_together(self):
    """Test that function_call and function_response stay together."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=1)

    # Create invocations where the last one has a tool call
    contents = [
        # Invocation 1 (should be removed)
        Content(role="user", parts=[Part(text="Hello")]),
        Content(role="model", parts=[Part(text="Hi")]),
        # Invocation 2 (should be kept - has tool call)
        Content(role="user", parts=[Part(text="What's the weather?")]),
        Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="get_weather", args={"location": "SF"}
                    )
                )
            ],
        ),
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="get_weather", response={"temp": 72}
                    )
                )
            ],
        ),
        Content(role="model", parts=[Part(text="It's 72°F")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should keep entire tool call sequence (4 messages)
    assert len(request.contents) == 4
    assert request.contents[0].parts[0].text == "What's the weather?"
    assert hasattr(request.contents[1].parts[0], "function_call")
    assert hasattr(request.contents[2].parts[0], "function_response")
    assert request.contents[3].parts[0].text == "It's 72°F"

  def test_orphaned_function_response_prevented(self):
    """Test that function_response is never orphaned without function_call."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=2)

    contents = [
        # Invocation 1
        Content(role="user", parts=[Part(text="Hello")]),
        Content(role="model", parts=[Part(text="Hi")]),
        # Invocation 2 (with tool call)
        Content(role="user", parts=[Part(text="Query 1")]),
        Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="tool1", args={}
                    )
                )
            ],
        ),
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="tool1", response={}
                    )
                )
            ],
        ),
        Content(role="model", parts=[Part(text="Response 1")]),
        # Invocation 3 (with tool call)
        Content(role="user", parts=[Part(text="Query 2")]),
        Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="tool2", args={}
                    )
                )
            ],
        ),
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="tool2", response={}
                    )
                )
            ],
        ),
        Content(role="model", parts=[Part(text="Response 2")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should keep invocations 2 and 3 (indices 2-9)
    assert len(request.contents) == 8

    # Verify no orphaned function_response
    for i, content in enumerate(request.contents):
      if content.role == "user" and any(
          hasattr(p, "function_response") and p.function_response
          for p in content.parts
      ):
        # There must be a preceding model message with function_call
        assert i > 0
        prev_content = request.contents[i - 1]
        assert prev_content.role == "model"
        assert any(
            hasattr(p, "function_call") and p.function_call
            for p in prev_content.parts
        )

  def test_consecutive_user_messages_grouped(self):
    """Test that consecutive user messages are grouped together."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=1)

    contents = [
        # Invocation 1 (should be removed)
        Content(role="user", parts=[Part(text="Hello")]),
        Content(role="model", parts=[Part(text="Hi")]),
        # Invocation 2 (should be kept)
        Content(role="user", parts=[Part(text="For context:")]),
        Content(role="user", parts=[Part(text="Tell me about X")]),
        Content(role="model", parts=[Part(text="Here's about X")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should keep the last invocation with both user messages
    assert len(request.contents) == 3
    assert request.contents[0].parts[0].text == "For context:"
    assert request.contents[1].parts[0].text == "Tell me about X"
    assert request.contents[2].parts[0].text == "Here's about X"

  def test_custom_filter_applied(self):
    """Test that custom filter is applied after invocation filtering."""
    # Custom filter that removes all model messages
    def custom_filter(contents):
      return [c for c in contents if c.role != "model"]

    plugin = ToolAwareContextFilterPlugin(
        num_invocations_to_keep=2, custom_filter=custom_filter
    )

    contents = [
        Content(role="user", parts=[Part(text="Query 1")]),
        Content(role="model", parts=[Part(text="Response 1")]),
        Content(role="user", parts=[Part(text="Query 2")]),
        Content(role="model", parts=[Part(text="Response 2")]),
    ]

    request = LlmRequest(model="test", contents=contents)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should have only user messages
    assert len(request.contents) == 2
    assert all(c.role == "user" for c in request.contents)

  def test_empty_contents(self):
    """Test handling of empty contents list."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=2)

    request = LlmRequest(model="test", contents=[])
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Run the plugin
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should handle empty contents gracefully
    assert len(request.contents) == 0

  def test_error_handling(self):
    """Test that errors are caught and logged without crashing."""
    plugin = ToolAwareContextFilterPlugin(num_invocations_to_keep=2)

    # Create a malformed request that might cause errors
    request = LlmRequest(model="test", contents=None)
    context = CallbackContext(invocation_id="test", agent_name="test")

    # Should not raise an exception
    result = pytest.mark.asyncio(
        plugin.before_model_callback(
            callback_context=context, llm_request=request
        )
    )

    # Should return None without crashing
    assert result is None