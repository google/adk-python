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

from pathlib import Path
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest
import yaml


@pytest.fixture
def debug_output_file(tmp_path):
  """Fixture to provide a temporary file path for debug output."""
  return tmp_path / "debug_output.yaml"


@pytest.fixture
def mock_session():
  """Create a mock session."""
  session = Mock(spec=Session)
  session.id = "test-session-id"
  session.app_name = "test-app"
  session.user_id = "test-user"
  session.state = {"key1": "value1", "key2": 123}
  session.events = []
  return session


@pytest.fixture
def mock_invocation_context(mock_session):
  """Create a mock invocation context."""
  ctx = Mock(spec=InvocationContext)
  ctx.invocation_id = "test-invocation-id"
  ctx.session = mock_session
  ctx.user_id = "test-user"
  ctx.app_name = "test-app"
  ctx.branch = None
  ctx.agent = Mock()
  ctx.agent.name = "test-agent"
  return ctx


@pytest.fixture
def mock_callback_context(mock_invocation_context):
  """Create a mock callback context."""
  ctx = Mock(spec=CallbackContext)
  ctx.invocation_id = mock_invocation_context.invocation_id
  ctx.agent_name = "test-agent"
  ctx._invocation_context = mock_invocation_context
  ctx.state = {}
  return ctx


@pytest.fixture
def mock_tool_context(mock_invocation_context):
  """Create a mock tool context."""
  ctx = Mock(spec=ToolContext)
  ctx.invocation_id = mock_invocation_context.invocation_id
  ctx.agent_name = "test-agent"
  ctx.function_call_id = "test-function-call-id"
  return ctx


class TestDebugLoggingPluginInitialization:
  """Tests for DebugLoggingPlugin initialization."""

  def test_default_initialization(self):
    """Test plugin initialization with default values."""
    plugin = DebugLoggingPlugin()
    assert plugin.name == "debug_logging_plugin"
    assert plugin._output_path == Path("adk_debug.yaml")
    assert plugin._include_session_state is True
    assert plugin._include_system_instruction is True

  def test_custom_initialization(self, debug_output_file):
    """Test plugin initialization with custom values."""
    plugin = DebugLoggingPlugin(
        name="custom_debug",
        output_path=str(debug_output_file),
        include_session_state=False,
        include_system_instruction=False,
    )
    assert plugin.name == "custom_debug"
    assert plugin._output_path == debug_output_file
    assert plugin._include_session_state is False
    assert plugin._include_system_instruction is False


class TestDebugLoggingPluginCallbacks:
  """Tests for DebugLoggingPlugin callback methods."""

  async def test_before_run_callback_initializes_state(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that before_run_callback initializes debug state."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    result = await plugin.before_run_callback(
        invocation_context=mock_invocation_context
    )

    assert result is None
    assert mock_invocation_context.invocation_id in plugin._invocation_states
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    assert state.invocation_id == mock_invocation_context.invocation_id
    assert state.session_id == mock_invocation_context.session.id
    assert len(state.entries) == 1
    assert state.entries[0].entry_type == "invocation_start"

  async def test_on_user_message_callback_logs_message(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that on_user_message_callback logs user messages."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text="Hello, world!")]
    )

    result = await plugin.on_user_message_callback(
        invocation_context=mock_invocation_context, user_message=user_message
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    user_message_entries = [
        e for e in state.entries if e.entry_type == "user_message"
    ]
    assert len(user_message_entries) == 1
    assert user_message_entries[0].data["content"]["role"] == "user"
    assert user_message_entries[0].data["content"]["parts"][0]["text"] == (
        "Hello, world!"
    )

  async def test_before_model_callback_logs_request(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """Test that before_model_callback logs LLM requests."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    llm_request = LlmRequest(
        model="gemini-2.0-flash",
        contents=[
            types.Content(
                role="user", parts=[types.Part.from_text(text="Test prompt")]
            )
        ],
    )
    llm_request.config.system_instruction = "You are a helpful assistant."

    result = await plugin.before_model_callback(
        callback_context=mock_callback_context, llm_request=llm_request
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    llm_entries = [e for e in state.entries if e.entry_type == "llm_request"]
    assert len(llm_entries) == 1
    assert llm_entries[0].data["model"] == "gemini-2.0-flash"
    assert llm_entries[0].data["content_count"] == 1
    assert "config" in llm_entries[0].data
    assert (
        llm_entries[0].data["config"]["system_instruction"]
        == "You are a helpful assistant."
    )

  async def test_after_model_callback_logs_response(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """Test that after_model_callback logs LLM responses."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    llm_response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Hello! How can I help?")],
        ),
        turn_complete=True,
    )

    result = await plugin.after_model_callback(
        callback_context=mock_callback_context, llm_response=llm_response
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    llm_entries = [e for e in state.entries if e.entry_type == "llm_response"]
    assert len(llm_entries) == 1
    assert llm_entries[0].data["turn_complete"] is True
    assert llm_entries[0].data["content"]["role"] == "model"

  async def test_before_tool_callback_logs_tool_call(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """Test that before_tool_callback logs tool calls."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    tool_args = {"param1": "value1", "param2": 42}

    result = await plugin.before_tool_callback(
        tool=mock_tool, tool_args=tool_args, tool_context=mock_tool_context
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    tool_entries = [e for e in state.entries if e.entry_type == "tool_call"]
    assert len(tool_entries) == 1
    assert tool_entries[0].data["tool_name"] == "test_tool"
    assert tool_entries[0].data["args"]["param1"] == "value1"
    assert tool_entries[0].data["args"]["param2"] == 42

  async def test_after_tool_callback_logs_tool_response(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """Test that after_tool_callback logs tool responses."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    tool_args = {"param1": "value1"}
    result_data = {"output": "success", "data": [1, 2, 3]}

    result = await plugin.after_tool_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=mock_tool_context,
        result=result_data,
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    tool_entries = [e for e in state.entries if e.entry_type == "tool_response"]
    assert len(tool_entries) == 1
    assert tool_entries[0].data["tool_name"] == "test_tool"
    assert tool_entries[0].data["result"]["output"] == "success"

  async def test_on_event_callback_logs_event(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that on_event_callback logs events."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    event = Event(
        author="test-agent",
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text="Response text")],
        ),
    )

    result = await plugin.on_event_callback(
        invocation_context=mock_invocation_context, event=event
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    event_entries = [e for e in state.entries if e.entry_type == "event"]
    assert len(event_entries) == 1
    assert event_entries[0].data["author"] == "test-agent"
    assert event_entries[0].data["event_id"] == event.id

  async def test_on_model_error_callback_logs_error(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """Test that on_model_error_callback logs LLM errors."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    llm_request = LlmRequest(model="gemini-2.0-flash")
    error = ValueError("Test error message")

    result = await plugin.on_model_error_callback(
        callback_context=mock_callback_context,
        llm_request=llm_request,
        error=error,
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    error_entries = [e for e in state.entries if e.entry_type == "llm_error"]
    assert len(error_entries) == 1
    assert error_entries[0].data["error_type"] == "ValueError"
    assert error_entries[0].data["error_message"] == "Test error message"

  async def test_on_tool_error_callback_logs_error(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """Test that on_tool_error_callback logs tool errors."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state first
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    mock_tool = Mock(spec=BaseTool)
    mock_tool.name = "test_tool"
    tool_args = {"param1": "value1"}
    error = RuntimeError("Tool execution failed")

    result = await plugin.on_tool_error_callback(
        tool=mock_tool,
        tool_args=tool_args,
        tool_context=mock_tool_context,
        error=error,
    )

    assert result is None
    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    error_entries = [e for e in state.entries if e.entry_type == "tool_error"]
    assert len(error_entries) == 1
    assert error_entries[0].data["tool_name"] == "test_tool"
    assert error_entries[0].data["error_type"] == "RuntimeError"


class TestDebugLoggingPluginFileOutput:
  """Tests for DebugLoggingPlugin file output."""

  async def test_after_run_callback_writes_to_file(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that after_run_callback writes debug data to file."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # Initialize state
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    # Add some entries
    user_message = types.Content(
        role="user", parts=[types.Part.from_text(text="Test message")]
    )
    await plugin.on_user_message_callback(
        invocation_context=mock_invocation_context, user_message=user_message
    )

    # Finalize
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    # Verify file was written
    assert debug_output_file.exists()

    # Parse and verify content (YAML format with --- separator)
    with open(debug_output_file, "r") as f:
      documents = list(yaml.safe_load_all(f))

    assert len(documents) == 1
    data = documents[0]
    assert data["invocation_id"] == "test-invocation-id"
    assert data["session_id"] == "test-session-id"
    assert (
        len(data["entries"]) >= 2
    )  # At least invocation_start and user_message

  async def test_after_run_callback_includes_session_state(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that session state is included when enabled."""
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), include_session_state=True
    )

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    with open(debug_output_file, "r") as f:
      documents = list(yaml.safe_load_all(f))

    data = documents[0]
    session_state_entries = [
        e
        for e in data["entries"]
        if e["entry_type"] == "session_state_snapshot"
    ]
    assert len(session_state_entries) == 1
    assert session_state_entries[0]["data"]["state"]["key1"] == "value1"

  async def test_after_run_callback_excludes_session_state_when_disabled(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that session state is excluded when disabled."""
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), include_session_state=False
    )

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    with open(debug_output_file, "r") as f:
      documents = list(yaml.safe_load_all(f))

    data = documents[0]
    session_state_entries = [
        e
        for e in data["entries"]
        if e["entry_type"] == "session_state_snapshot"
    ]
    assert not session_state_entries

  async def test_multiple_invocations_append_to_file(
      self, debug_output_file, mock_session
  ):
    """Test that multiple invocations append to the same file."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    # First invocation
    ctx1 = Mock(spec=InvocationContext)
    ctx1.invocation_id = "invocation-1"
    ctx1.session = mock_session
    ctx1.user_id = "test-user"
    ctx1.branch = None
    ctx1.agent = Mock()
    ctx1.agent.name = "agent-1"

    await plugin.before_run_callback(invocation_context=ctx1)
    await plugin.after_run_callback(invocation_context=ctx1)

    # Second invocation
    ctx2 = Mock(spec=InvocationContext)
    ctx2.invocation_id = "invocation-2"
    ctx2.session = mock_session
    ctx2.user_id = "test-user"
    ctx2.branch = None
    ctx2.agent = Mock()
    ctx2.agent.name = "agent-2"

    await plugin.before_run_callback(invocation_context=ctx2)
    await plugin.after_run_callback(invocation_context=ctx2)

    # Verify both invocations are in the file (as separate YAML documents)
    with open(debug_output_file, "r") as f:
      documents = list(yaml.safe_load_all(f))

    assert len(documents) == 2
    assert documents[0]["invocation_id"] == "invocation-1"
    assert documents[1]["invocation_id"] == "invocation-2"

  async def test_after_run_callback_cleans_up_state(
      self, debug_output_file, mock_invocation_context
  ):
    """Test that invocation state is cleaned up after writing."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    assert mock_invocation_context.invocation_id in plugin._invocation_states

    await plugin.after_run_callback(invocation_context=mock_invocation_context)
    assert (
        mock_invocation_context.invocation_id not in plugin._invocation_states
    )


class TestDebugLoggingPluginSerialization:
  """Tests for content serialization."""

  def test_serialize_content_with_text(self):
    """Test serialization of text content."""
    plugin = DebugLoggingPlugin()
    content = types.Content(
        role="user", parts=[types.Part.from_text(text="Hello")]
    )

    result = plugin._serialize_content(content)

    assert result["role"] == "user"
    assert len(result["parts"]) == 1
    assert result["parts"][0]["text"] == "Hello"

  def test_serialize_content_with_function_call(self):
    """Test serialization of function call content."""
    plugin = DebugLoggingPlugin()
    content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    id="fc-1", name="test_func", args={"arg1": "val1"}
                )
            )
        ],
    )

    result = plugin._serialize_content(content)

    assert result["parts"][0]["function_call"]["name"] == "test_func"
    assert result["parts"][0]["function_call"]["args"]["arg1"] == "val1"

  def test_serialize_content_with_none(self):
    """Test serialization of None content."""
    plugin = DebugLoggingPlugin()
    result = plugin._serialize_content(None)
    assert result is None

  def test_safe_serialize_handles_bytes(self):
    """Test that bytes are safely serialized."""
    plugin = DebugLoggingPlugin()
    result = plugin._safe_serialize(b"binary data")
    assert result == "<bytes: 11 bytes>"

  def test_safe_serialize_handles_nested_structures(self):
    """Test that nested structures are serialized."""
    plugin = DebugLoggingPlugin()
    data = {
        "list": [1, 2, {"nested": "value"}],
        "tuple": (3, 4),
        "string": "text",
    }

    result = plugin._safe_serialize(data)

    assert result["list"] == [1, 2, {"nested": "value"}]
    assert result["tuple"] == [3, 4]  # Tuple becomes list
    assert result["string"] == "text"


class TestDebugLoggingPluginSystemInstructionConfig:
  """Tests for system instruction configuration."""

  async def test_system_instruction_included_when_enabled(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """Test that full system instruction is included when enabled."""
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), include_system_instruction=True
    )

    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    llm_request = LlmRequest(model="gemini-2.0-flash")
    llm_request.config.system_instruction = "Full system instruction text"

    await plugin.before_model_callback(
        callback_context=mock_callback_context, llm_request=llm_request
    )

    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    llm_entries = [e for e in state.entries if e.entry_type == "llm_request"]
    assert (
        llm_entries[0].data["config"]["system_instruction"]
        == "Full system instruction text"
    )

  async def test_system_instruction_length_only_when_disabled(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """Test that only length is included when system instruction is disabled."""
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), include_system_instruction=False
    )

    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    llm_request = LlmRequest(model="gemini-2.0-flash")
    llm_request.config.system_instruction = "Full system instruction text"

    await plugin.before_model_callback(
        callback_context=mock_callback_context, llm_request=llm_request
    )

    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    llm_entries = [e for e in state.entries if e.entry_type == "llm_request"]
    assert "system_instruction" not in llm_entries[0].data.get("config", {})
    assert llm_entries[0].data["config"]["system_instruction_length"] == 28
