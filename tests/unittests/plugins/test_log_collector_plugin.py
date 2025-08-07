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

"""Unit tests for the LogCollectorPlugin."""

from __future__ import annotations

from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import LogCollectorPlugin
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


@pytest.fixture
def plugin() -> LogCollectorPlugin:
  """Provides a clean LogCollectorPlugin instance for each test."""
  return LogCollectorPlugin()


def create_mock_invocation_context(session_id: str) -> Mock:
    mock_context = Mock(spec=InvocationContext)
    mock_context.session = Mock(spec=Session)
    mock_context.session.id = session_id
    return mock_context


def create_mock_callback_context(session_id: str) -> Mock:
    mock_context = Mock(spec=CallbackContext)
    mock_context._invocation_context = create_mock_invocation_context(session_id)
    return mock_context


def create_mock_tool_context(session_id: str) -> Mock:
    mock_context = Mock(spec=ToolContext)
    mock_context._invocation_context = create_mock_invocation_context(session_id)
    return mock_context


@pytest.mark.asyncio
async def test_on_user_message_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_invocation_context("session1")
    mock_context.invocation_id = "inv1"
    user_message = types.Content(parts=[types.Part(text="Hello")])

    await plugin.on_user_message_callback(
        invocation_context=mock_context, user_message=user_message
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "on_user_message"
    assert log["invocation_id"] == "inv1"
    assert log["user_message"] == "Hello"


@pytest.mark.asyncio
async def test_before_agent_callback(plugin: LogCollectorPlugin):
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_context = create_mock_callback_context("session1")
    mock_context.invocation_id = "inv1"

    await plugin.before_agent_callback(agent=mock_agent, callback_context=mock_context)

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "before_agent"
    assert log["agent_name"] == "test_agent"
    assert log["invocation_id"] == "inv1"


@pytest.mark.asyncio
async def test_after_agent_callback(plugin: LogCollectorPlugin):
    mock_agent = Mock(spec=BaseAgent)
    mock_agent.name = "test_agent"
    mock_context = create_mock_callback_context("session1")
    mock_context.invocation_id = "inv1"

    await plugin.after_agent_callback(agent=mock_agent, callback_context=mock_context)

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "after_agent"
    assert log["agent_name"] == "test_agent"
    assert log["invocation_id"] == "inv1"


@pytest.mark.asyncio
async def test_before_model_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_callback_context("session1")
    mock_context.agent_name = "test_agent"
    mock_request = Mock(spec=LlmRequest)
    mock_request.model_dump.return_value = {"model": "gemini"}

    await plugin.before_model_callback(
        callback_context=mock_context, llm_request=mock_request
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "before_model"
    assert log["agent_name"] == "test_agent"
    assert log["request"] == {"model": "gemini"}


@pytest.mark.asyncio
async def test_after_model_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_callback_context("session1")
    mock_context.agent_name = "test_agent"
    mock_response = Mock(spec=LlmResponse)
    mock_response.model_dump.return_value = {"text": "response"}

    await plugin.after_model_callback(
        callback_context=mock_context, llm_response=mock_response
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "after_model"
    assert log["agent_name"] == "test_agent"
    assert log["response"] == {"text": "response"}


@pytest.mark.asyncio
async def test_on_event_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_invocation_context("session1")
    mock_event = Mock(spec=Event)
    mock_event.id = "event1"
    mock_event.author = "test_author"
    mock_event.content = Mock(spec=types.Content)
    mock_event.content.parts = [types.Part(text="event content")]
    mock_event.is_final_response.return_value = True

    await plugin.on_event_callback(invocation_context=mock_context, event=mock_event)

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "on_event"
    assert log["event_id"] == "event1"
    assert log["author"] == "test_author"
    assert log["content"] == "event content"
    assert log["is_final"] is True


@pytest.mark.asyncio
async def test_before_run_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_invocation_context("session1")
    mock_context.invocation_id = "inv1"
    mock_context.agent = Mock(spec=BaseAgent)
    mock_context.agent.name = "test_agent"

    await plugin.before_run_callback(invocation_context=mock_context)

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "before_run"
    assert log["invocation_id"] == "inv1"
    assert log["agent_name"] == "test_agent"


@pytest.mark.asyncio
async def test_after_run_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_invocation_context("session1")
    mock_context.invocation_id = "inv1"
    mock_context.agent = Mock(spec=BaseAgent)
    mock_context.agent.name = "test_agent"

    await plugin.after_run_callback(invocation_context=mock_context)

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "after_run"
    assert log["invocation_id"] == "inv1"
    assert log["agent_name"] == "test_agent"


@pytest.mark.asyncio
async def test_on_model_error_callback(plugin: LogCollectorPlugin):
    mock_context = create_mock_callback_context("session1")
    mock_context.agent_name = "test_agent"
    mock_request = Mock(spec=LlmRequest)
    mock_request.model_dump.return_value = {"model": "gemini"}
    error = ValueError("test error")

    await plugin.on_model_error_callback(
        callback_context=mock_context, llm_request=mock_request, error=error
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "on_model_error"
    assert log["agent_name"] == "test_agent"
    assert log["request"] == {"model": "gemini"}
    assert log["error"] == "test error"


@pytest.mark.asyncio
async def test_before_tool_callback(plugin: LogCollectorPlugin):
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_context = create_mock_tool_context("session1")
    mock_context.agent_name = "test_agent"
    mock_context.function_call_id = "func1"
    tool_args = {"arg1": "value1"}

    await plugin.before_tool_callback(
        tool=mock_tool, tool_args=tool_args, tool_context=mock_context
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "before_tool"
    assert log["tool_name"] == "test_tool"
    assert log["agent_name"] == "test_agent"
    assert log["function_call_id"] == "func1"
    assert log["args"] == {"arg1": "value1"}


@pytest.mark.asyncio
async def test_after_tool_callback(plugin: LogCollectorPlugin):
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_context = create_mock_tool_context("session1")
    mock_context.agent_name = "test_agent"
    mock_context.function_call_id = "func1"
    tool_args = {"arg1": "value1"}
    result = {"result": "success"}

    await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=mock_context,
        result=result,
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "after_tool"
    assert log["tool_name"] == "test_tool"
    assert log["agent_name"] == "test_agent"
    assert log["function_call_id"] == "func1"
    assert log["args"] == {"arg1": "value1"}
    assert log["result"] == {"result": "success"}


@pytest.mark.asyncio
async def test_on_tool_error_callback(plugin: LogCollectorPlugin):
    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    mock_context = create_mock_tool_context("session1")
    mock_context.agent_name = "test_agent"
    mock_context.function_call_id = "func1"
    tool_args = {"arg1": "value1"}
    error = ValueError("tool error")

    await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=mock_context,
        error=error,
    )

    assert len(plugin.logs["session1"]) == 1
    log = plugin.logs["session1"][0]
    assert log["callback_type"] == "on_tool_error"
    assert log["tool_name"] == "test_tool"
    assert log["agent_name"] == "test_agent"
    assert log["function_call_id"] == "func1"
    assert log["args"] == {"arg1": "value1"}
    assert log["error"] == "tool error"


@pytest.mark.asyncio
async def test_multiple_sessions(plugin: LogCollectorPlugin):
    mock_context1 = create_mock_invocation_context("session1")
    mock_context1.invocation_id = "inv1"
    user_message1 = types.Content(parts=[types.Part(text="Hello from session 1")])

    mock_context2 = create_mock_invocation_context("session2")
    mock_context2.invocation_id = "inv2"
    user_message2 = types.Content(parts=[types.Part(text="Hello from session 2")])

    await plugin.on_user_message_callback(
        invocation_context=mock_context1, user_message=user_message1
    )
    await plugin.on_user_message_callback(
        invocation_context=mock_context2, user_message=user_message2
    )

    assert len(plugin.logs["session1"]) == 1
    assert len(plugin.logs["session2"]) == 1

    log1 = plugin.logs["session1"][0]
    assert log1["callback_type"] == "on_user_message"
    assert log1["invocation_id"] == "inv1"
    assert log1["user_message"] == "Hello from session 1"

    log2 = plugin.logs["session2"][0]
    assert log2["callback_type"] == "on_user_message"
    assert log2["invocation_id"] == "inv2"
    assert log2["user_message"] == "Hello from session 2"
