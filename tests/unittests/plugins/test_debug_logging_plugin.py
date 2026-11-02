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

from datetime import datetime
import logging
import os
from pathlib import Path
import stat
from typing import Any
from typing import Optional
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ToolResultClassification
from google.adk.plugins.debug_logging_plugin import DebugLoggingPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.sessions.session import Session
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pydantic import BaseModel
import pytest
import yaml

from .. import testing_utils

_SENTINEL_ACCESS_TOKEN = "sentinel-access-token-4f7a21"
_SENTINEL_REFRESH_TOKEN = "sentinel-refresh-token-91cc03"
_SENTINEL_CLIENT_SECRET = "sentinel-client-secret-b58d6e"
_SENTINEL_AUTH_CODE = "sentinel-auth-code-2ad914"
_SENTINEL_CODE_VERIFIER = "sentinel-code-verifier-7be055"
_SENTINEL_PRIVATE_KEY = (
    "-----BEGIN PRIVATE KEY-----\nsentinel-key-body\n-----END PRIVATE KEY-----"
)


def _oauth_credential() -> AuthCredential:
  """An exchanged OAuth2 credential carrying sentinel secret values."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test-client-id",
          client_secret=_SENTINEL_CLIENT_SECRET,
          access_token=_SENTINEL_ACCESS_TOKEN,
          refresh_token=_SENTINEL_REFRESH_TOKEN,
      ),
  )


class _CredentialCarrier(BaseModel):
  """A model that is not itself a credential but holds one under any name."""

  label: str
  payload: AuthCredential


class _TypedCredentialCarrier(BaseModel):
  """A carrier whose other fields are not JSON-native."""

  kind: AuthCredentialTypes
  issued_at: datetime
  payload: AuthCredential


class _SelfReferentialCarrier(BaseModel):
  """A carrier that can be pointed at itself."""

  label: str
  payload: AuthCredential
  parent: Optional[Any] = None


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
        model="gemini-2.5-flash",
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
    assert llm_entries[0].data["model"] == "gemini-2.5-flash"
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

    llm_request = LlmRequest(model="gemini-2.5-flash")
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


class TestDebugLoggingPluginRedaction:
  """Tests that credentials never reach the shareable debug file."""

  async def test_session_state_credential_model_is_redacted(
      self, debug_output_file, mock_invocation_context
  ):
    """Credentials stored in session state must not be written out."""
    mock_invocation_context.session.state = {
        "key1": "value1",
        "temp:oauth2_credential": _oauth_credential(),
    }
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    raw = debug_output_file.read_text()
    assert _SENTINEL_ACCESS_TOKEN not in raw
    assert _SENTINEL_REFRESH_TOKEN not in raw
    assert _SENTINEL_CLIENT_SECRET not in raw

    documents = list(yaml.safe_load_all(raw))
    snapshots = [
        e
        for e in documents[0]["entries"]
        if e["entry_type"] == "session_state_snapshot"
    ]
    assert len(snapshots) == 1
    state = snapshots[0]["data"]["state"]
    assert state["temp:oauth2_credential"] == "[REDACTED]"
    # Non-credential state is still useful for debugging.
    assert state["key1"] == "value1"

  async def test_session_state_credential_dict_is_redacted(
      self, debug_output_file, mock_invocation_context
  ):
    """Credentials rehydrated from a session store are plain dicts."""
    mock_invocation_context.session.state = {
        "temp:oauth2_credential": {
            "oauth2": {"access_token": _SENTINEL_ACCESS_TOKEN}
        },
        "user:profile": {
            "name": "test-user",
            "refresh_token": _SENTINEL_REFRESH_TOKEN,
        },
    }
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    raw = debug_output_file.read_text()
    assert _SENTINEL_ACCESS_TOKEN not in raw
    assert _SENTINEL_REFRESH_TOKEN not in raw

    documents = list(yaml.safe_load_all(raw))
    state = [
        e
        for e in documents[0]["entries"]
        if e["entry_type"] == "session_state_snapshot"
    ][0]["data"]["state"]
    assert state["temp:oauth2_credential"] == "[REDACTED]"
    assert state["user:profile"]["refresh_token"] == "[REDACTED]"
    assert state["user:profile"]["name"] == "test-user"

  async def test_state_delta_credential_is_redacted(
      self, debug_output_file, mock_invocation_context
  ):
    """Credentials also flow through event state deltas."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    event = Event(
        author="test-agent",
        actions=EventActions(
            state_delta={
                "temp:oauth2_credential": _oauth_credential(),
                "counter": 7,
            }
        ),
    )

    await plugin.on_event_callback(
        invocation_context=mock_invocation_context, event=event
    )

    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    event_entries = [e for e in state.entries if e.entry_type == "event"]
    state_delta = event_entries[0].data["actions"]["state_delta"]
    assert state_delta["temp:oauth2_credential"] == "[REDACTED]"
    assert state_delta["counter"] == 7

  def test_credential_nested_in_non_credential_model_is_redacted(self):
    """A credential survives `model_dump` as a plain dict, so walk fields."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize(
        _CredentialCarrier(label="anything", payload=_oauth_credential())
    )

    assert result == {"label": "anything", "payload": "[REDACTED]"}

  def test_credential_in_container_under_arbitrary_key_is_redacted(self):
    """Neither the key name nor the nesting depth may matter."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "some_users_own_key": [
            {"inner": (_oauth_credential(), "keep-me")},
            _CredentialCarrier(label="deep", payload=_oauth_credential()),
        ],
    })

    nested = result["some_users_own_key"]
    assert nested[0]["inner"] == ["[REDACTED]", "keep-me"]
    assert nested[1] == {"label": "deep", "payload": "[REDACTED]"}
    assert _SENTINEL_ACCESS_TOKEN not in str(result)
    assert _SENTINEL_CLIENT_SECRET not in str(result)

  def test_carrier_fields_are_normalized_to_yaml_safe_values(self):
    """A walked carrier skips `model_dump`, so it normalizes its own fields."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize(
        _TypedCredentialCarrier(
            kind=AuthCredentialTypes.OAUTH2,
            issued_at=datetime(2026, 1, 2, 3, 4, 5),
            payload=_oauth_credential(),
        )
    )

    assert result == {
        "kind": "oauth2",
        "issued_at": "2026-01-02T03:04:05",
        "payload": "[REDACTED]",
    }
    assert yaml.safe_load(yaml.dump(result)) == result

  def test_auth_config_serializes_to_loadable_yaml(self):
    """`AuthConfig` is the carrier ADK itself puts in session state."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize(
        AuthConfig(
            auth_scheme=OpenIdConnectWithConfig(
                openIdConnectUrl="https://example.com/openid-configuration",
                authorization_endpoint="https://example.com/auth",
                token_endpoint="https://example.com/token",
                scopes=["openid"],
            ),
            raw_auth_credential=_oauth_credential(),
        )
    )

    assert result["raw_auth_credential"] == "[REDACTED]"
    assert result["auth_scheme"]["type_"] == "openIdConnect"
    assert yaml.safe_load(yaml.dump(result)) == result
    assert _SENTINEL_ACCESS_TOKEN not in str(result)

  def test_self_referential_value_is_bounded(self):
    """A cycle must not recurse until the interpreter gives up."""
    plugin = DebugLoggingPlugin()
    carrier = _SelfReferentialCarrier(label="loop", payload=_oauth_credential())
    carrier.parent = carrier
    cyclic_dict = {"credential": _oauth_credential()}
    cyclic_dict["itself"] = cyclic_dict

    from_model = plugin._safe_serialize(carrier)
    from_dict = plugin._safe_serialize(cyclic_dict)

    assert from_model["payload"] == "[REDACTED]"
    assert from_dict["credential"] == "[REDACTED]"
    for result in (from_model, from_dict):
      assert yaml.safe_load(yaml.dump(result)) == result
      assert _SENTINEL_ACCESS_TOKEN not in str(result)

  def test_hyphenated_sensitive_keys_are_redacted(self):
    """Header spellings reach the plugin as tool arguments."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "headers": {
            "X-Api-Key": _SENTINEL_CLIENT_SECRET,
            "Proxy-Authorization": _SENTINEL_ACCESS_TOKEN,
            "Content-Type": "application/json",
        }
    })

    assert result["headers"]["X-Api-Key"] == "[REDACTED]"
    assert result["headers"]["Proxy-Authorization"] == "[REDACTED]"
    assert result["headers"]["Content-Type"] == "application/json"

  def test_oauth_authorization_code_keys_are_redacted(self):
    """A dumped credential kept under a non-`temp:` key leaves only keys."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "apikey_scheme_existing_exchanged_credential": {
            "oauth2": {
                "auth_code": _SENTINEL_AUTH_CODE,
                "auth_response_uri": f"https://x/cb?code={_SENTINEL_AUTH_CODE}",
                "code_verifier": _SENTINEL_CODE_VERIFIER,
                "client_id": "test-client-id",
            }
        }
    })

    oauth2 = result["apikey_scheme_existing_exchanged_credential"]["oauth2"]
    assert oauth2["auth_code"] == "[REDACTED]"
    assert oauth2["auth_response_uri"] == "[REDACTED]"
    assert oauth2["code_verifier"] == "[REDACTED]"
    assert oauth2["client_id"] == "test-client-id"

  def test_scoped_state_keys_are_redacted(self):
    """A state scope prefix says nothing about whether the value is a secret."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "api_key": _SENTINEL_CLIENT_SECRET,
        "user:api_key": _SENTINEL_CLIENT_SECRET,
        "app:client_secret": _SENTINEL_CLIENT_SECRET,
        "user:profile": {"name": "test-user"},
    })

    assert result["api_key"] == "[REDACTED]"
    assert result["user:api_key"] == "[REDACTED]"
    assert result["app:client_secret"] == "[REDACTED]"
    assert result["user:profile"] == {"name": "test-user"}

  def test_key_spelling_variants_are_redacted(self):
    """Camel case and compound names name the same secrets."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "apiKey": _SENTINEL_CLIENT_SECRET,
        "secret_key": _SENTINEL_CLIENT_SECRET,
        "bearer_token": _SENTINEL_ACCESS_TOKEN,
        "credentials": _SENTINEL_CLIENT_SECRET,
        "serviceAccountCredentials": _SENTINEL_CLIENT_SECRET,
    })

    assert set(result.values()) == {"[REDACTED]"}

  def test_usage_counters_survive_key_matching(self):
    """Counters end in the word `token` and are the point of the log."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "usage_metadata": {
            "prompt_token_count": 12,
            "candidates_token_count": 34,
            "total_token_count": 46,
        },
        "max_output_tokens": 1024,
        "cache_key": "abc",
    })

    assert result["usage_metadata"]["prompt_token_count"] == 12
    assert result["usage_metadata"]["total_token_count"] == 46
    assert result["max_output_tokens"] == 1024
    assert result["cache_key"] == "abc"

  def test_private_key_in_a_string_value_is_redacted(self):
    """A service account file pasted into state has no telling key name."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "user:uploaded_file": (
            '{"type": "service_account", "client_email": "a@b.example.com",'
            f' "private_key": "{_SENTINEL_PRIVATE_KEY}"}}'
        ),
        "notes": ["harmless", _SENTINEL_PRIVATE_KEY],
    })

    assert result["notes"] == ["harmless", "[REDACTED]"]
    assert "sentinel-key-body" not in str(result)

  def test_only_the_private_key_block_is_cut_from_the_string(self):
    """The surrounding prompt is what the log exists to show."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize(
        f"here is my key {_SENTINEL_PRIVATE_KEY} please rotate it"
    )

    assert result == "here is my key [REDACTED] please rotate it"

  def test_armor_header_variants_are_redacted(self):
    """The header is matched as a unit, not as loose fragments."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "pgp": (
            "-----BEGIN PGP PRIVATE KEY BLOCK-----\nsentinel-key-body\n"
            "-----END PGP PRIVATE KEY BLOCK-----"
        ),
        "rsa": (
            "-----BEGIN RSA PRIVATE KEY-----\nsentinel-key-body\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        "unterminated": "-----BEGIN PRIVATE KEY-----\nsentinel-key-body\n",
    })

    assert set(result.values()) == {"[REDACTED]"}

  def test_prose_quoting_armor_fragments_is_kept(self):
    """Two fragments in any order are not a key block."""
    plugin = DebugLoggingPlugin()
    prose = "notes about a PRIVATE KEY----- and -----BEGIN elsewhere"

    assert plugin._safe_serialize(prose) == prose

  def test_none_and_scalars_pass_through_unchanged(self):
    """Redaction runs over whatever the callbacks hand it."""
    plugin = DebugLoggingPlugin()

    assert plugin._safe_serialize(None) is None
    assert plugin._safe_serialize("plain") == "plain"
    assert plugin._safe_serialize(7) == 7

  def test_a_secret_nested_in_a_list_is_redacted(self):
    """A callback payload is commonly a list of dicts."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize([None, {"token": _SENTINEL_ACCESS_TOKEN}])

    assert result == [None, {"token": "[REDACTED]"}]

  def test_the_walk_depth_bound_truncates_instead_of_recursing(self):
    """A self-referential object would otherwise never terminate."""
    plugin = DebugLoggingPlugin()
    deep: Any = {"api_key": _SENTINEL_CLIENT_SECRET}
    for _ in range(60):
      deep = {"level": deep}

    result = plugin._safe_serialize(deep)

    assert "<dict ...>" in str(result)
    assert _SENTINEL_CLIENT_SECRET not in str(result)

  def test_non_credential_values_are_not_redacted(self):
    """Redaction must not swallow ordinary debug data."""
    plugin = DebugLoggingPlugin()

    result = plugin._safe_serialize({
        "nested": {"list": [1, "two", {"deep": "value"}]},
        "model": types.FunctionCall(id="fc-1", name="do_it", args={"a": 1}),
    })

    assert result["nested"]["list"] == [1, "two", {"deep": "value"}]
    assert result["model"]["name"] == "do_it"
    assert result["model"]["args"] == {"a": 1}

  @pytest.mark.skipif(
      os.name == "nt", reason="POSIX file permissions differ on Windows"
  )
  async def test_output_file_is_not_world_readable(
      self, debug_output_file, mock_invocation_context
  ):
    """The debug file holds whole conversations; keep it owner-only."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    assert debug_output_file.exists()
    mode = stat.S_IMODE(debug_output_file.stat().st_mode)
    assert mode & 0o077 == 0

  @pytest.mark.skipif(
      os.name == "nt", reason="POSIX file permissions differ on Windows"
  )
  async def test_pre_existing_world_readable_file_is_flagged(
      self, debug_output_file, mock_invocation_context, caplog
  ):
    """A file from an earlier run keeps its mode, so warn instead."""
    debug_output_file.write_text("")
    debug_output_file.chmod(0o644)
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))

    with caplog.at_level(logging.WARNING, logger="google_adk"):
      await plugin.before_run_callback(
          invocation_context=mock_invocation_context
      )
      await plugin.after_run_callback(
          invocation_context=mock_invocation_context
      )

    assert any(
        "readable beyond its owner" in record.message
        for record in caplog.records
    )


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

    llm_request = LlmRequest(model="gemini-2.5-flash")
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

    llm_request = LlmRequest(model="gemini-2.5-flash")
    llm_request.config.system_instruction = "Full system instruction text"

    await plugin.before_model_callback(
        callback_context=mock_callback_context, llm_request=llm_request
    )

    state = plugin._invocation_states[mock_invocation_context.invocation_id]
    llm_entries = [e for e in state.entries if e.entry_type == "llm_request"]
    assert "system_instruction" not in llm_entries[0].data.get("config", {})
    assert llm_entries[0].data["config"]["system_instruction_length"] == 28


def _tool(name: str) -> BaseTool:
  tool = Mock(spec=BaseTool)
  tool.name = name
  return tool


def _node_event(
    *, message_as_output: Optional[bool] = None, **fields: Any
) -> Event:
  """Returns an event of the workflow node at path `wf@1/step@2`."""
  return Event(
      author="step",
      node_info=NodeInfo(
          path="wf@1/step@2", message_as_output=message_as_output
      ),
      **fields,
  )


def _entries(plugin: DebugLoggingPlugin, entry_type: str) -> list[Any]:
  """Returns the data of the recorded entries of one type, in order."""
  state = plugin._invocation_states["test-invocation-id"]
  return [e.data for e in state.entries if e.entry_type == entry_type]


async def _retry_answer(
    tool: BaseTool, tool_context: ToolContext, error: Exception
) -> dict[str, Any]:
  """Returns ReflectAndRetryToolPlugin's real answer to a raised error."""
  return await ReflectAndRetryToolPlugin(max_retries=3).on_tool_error_callback(
      tool=tool, tool_args={}, tool_context=tool_context, error=error
  )


# The dump of an MCP CallToolResult that reports a failure.
_MCP_ERROR_RESULT = {
    "content": [{"type": "text", "text": "permission denied"}],
    "isError": True,
}


def _classify_status_field(*, tool, tool_args, tool_context, result):
  """Classifies a result whose `status` is `ERROR` as a tool error."""
  del tool, tool_args, tool_context  # Unused.
  if isinstance(result, dict) and result.get("status") == "ERROR":
    return ToolResultClassification(
        status="ERROR", error_message=result.get("reason")
    )
  return None


class _RecordingOkClassifier:
  """Classifies every result as a completed call, recording each one."""

  def __init__(self):
    self.results = []

  def __call__(self, *, tool, tool_args, tool_context, result):
    del tool, tool_args, tool_context  # Unused.
    self.results.append(result)
    return ToolResultClassification(status="OK")


def _failing_classifier(*, tool, tool_args, tool_context, result):
  del tool, tool_args, tool_context  # Unused.
  raise RuntimeError(f"cannot classify {result!r}")


async def _async_classifier(*, tool, tool_args, tool_context, result):
  del tool, tool_args, tool_context, result  # Unused.
  return None


def _classifier_without_result(*, tool, tool_args, tool_context):
  del tool, tool_args, tool_context  # Unused.
  return None


class TestDebugLoggingPluginModelTermination:
  """Tests for the termination fields of LLM response entries."""

  async def test_final_response_records_finish_reason(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    await plugin.after_model_callback(
        callback_context=mock_callback_context,
        llm_response=LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text="hi")]),
            finish_reason=types.FinishReason.STOP,
        ),
    )

    [response] = _entries(plugin, "llm_response")
    assert response["finish_reason"] == str(types.FinishReason.STOP)

  async def test_streamed_chunk_leaves_termination_to_final_response(
      self, debug_output_file, mock_invocation_context, mock_callback_context
  ):
    """A stream's error and finish reason are recorded once, at its end.

    The final response of a stream carries the same values as the chunk, so
    recording them on the chunk too would record one termination twice.
    """
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    chunk = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="par")]),
        partial=True,
        error_code="MAX_TOKENS",
        error_message="cut off",
        finish_reason=types.FinishReason.MAX_TOKENS,
    )
    final = chunk.model_copy(update={"partial": False})

    for llm_response in (chunk, final):
      await plugin.after_model_callback(
          callback_context=mock_callback_context, llm_response=llm_response
      )

    chunk_data, final_data = _entries(plugin, "llm_response")
    assert chunk_data["partial"] is True
    assert not {"error_code", "error_message", "finish_reason"} & set(
        chunk_data
    )
    assert final_data["error_code"] == "MAX_TOKENS"
    assert final_data["error_message"] == "cut off"
    assert final_data["finish_reason"] == str(types.FinishReason.MAX_TOKENS)


class TestDebugLoggingPluginNodeFields:
  """Tests for the workflow node fields of event entries."""

  async def _record_event(
      self, debug_output_file, mock_invocation_context, event: Event
  ) -> dict[str, Any]:
    """Records one event and returns the data of its entry."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.on_event_callback(
        invocation_context=mock_invocation_context, event=event
    )
    [event_data] = _entries(plugin, "event")
    return event_data

  async def test_records_node_path_and_output(
      self, debug_output_file, mock_invocation_context
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        _node_event(output={"answer": 42}),
    )

    assert data["node_path"] == "wf@1/step@2"
    assert data["node_output"] == {"answer": 42}
    assert "node_error" not in data

  async def test_skips_output_its_message_carries(
      self, debug_output_file, mock_invocation_context
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        _node_event(
            content=types.Content(parts=[types.Part(text="the answer")]),
            output="the answer",
            message_as_output=True,
        ),
    )

    assert data["node_path"] == "wf@1/step@2"
    assert "node_output" not in data

  async def test_records_node_error_beside_its_output(
      self, debug_output_file, mock_invocation_context
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        _node_event(
            error_code="ValueError",
            error_message="invalid input",
            output={"processed": 3},
        ),
    )

    assert data["node_error"] == {
        "error_code": "ValueError",
        "error_message": "invalid input",
    }
    assert data["node_output"] == {"processed": 3}

  async def test_partial_event_records_only_node_path(
      self, debug_output_file, mock_invocation_context
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        _node_event(
            partial=True,
            output={"processed": 3},
            error_code="ValueError",
            error_message="invalid input",
        ),
    )

    assert data["node_path"] == "wf@1/step@2"
    assert "node_output" not in data
    assert "node_error" not in data

  @pytest.mark.parametrize(
      "error_code",
      ["MAX_TOKENS", "SAFETY", "MODEL_ARMOR", types.FinishReason.MAX_TOKENS],
  )
  async def test_model_termination_is_not_a_node_error(
      self, debug_output_file, mock_invocation_context, error_code
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        _node_event(error_code=error_code, error_message="model stopped"),
    )

    assert data["node_path"] == "wf@1/step@2"
    assert "node_error" not in data

  async def test_event_outside_a_workflow_has_no_node_fields(
      self, debug_output_file, mock_invocation_context
  ):
    data = await self._record_event(
        debug_output_file,
        mock_invocation_context,
        Event(
            author="agent",
            output={"processed": 3},
            error_code="ValueError",
            error_message="invalid input",
        ),
    )

    assert not {"node_path", "node_output", "node_error"} & set(data)

  async def test_node_output_credential_is_redacted(
      self, debug_output_file, mock_invocation_context
  ):
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    await plugin.on_event_callback(
        invocation_context=mock_invocation_context,
        event=_node_event(
            output=_CredentialCarrier(label="db", payload=_oauth_credential())
        ),
    )
    await plugin.after_run_callback(invocation_context=mock_invocation_context)

    text = debug_output_file.read_text()
    assert _SENTINEL_ACCESS_TOKEN not in text
    [document] = yaml.safe_load_all(text)
    [event_entry] = [
        e for e in document["entries"] if e["entry_type"] == "event"
    ]
    assert event_entry["data"]["node_output"]["label"] == "db"
    assert event_entry["data"]["node_output"]["payload"] == "[REDACTED]"


class TestDebugLoggingPluginToolResultErrors:
  """Tests for tool results that report a failure without raising."""

  async def test_result_with_error_keys_is_a_tool_response(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """A result that merely has error-like keys is the tool's data."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    await plugin.after_tool_callback(
        tool=_tool("lookup"),
        tool_args={},
        tool_context=mock_tool_context,
        result={"error": "none", "error_details": "none"},
    )

    assert len(_entries(plugin, "tool_response")) == 1
    assert not _entries(plugin, "tool_error")

  async def test_retry_answer_is_recorded_as_tool_error(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    tool = _tool("lookup")
    answer = await _retry_answer(
        tool, mock_tool_context, RuntimeError("quota exceeded")
    )

    await plugin.after_tool_callback(
        tool=tool,
        tool_args={"q": "x"},
        tool_context=mock_tool_context,
        result=answer,
    )

    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["tool_name"] == "lookup"
    assert tool_error["function_call_id"] == "test-function-call-id"
    assert tool_error["args"] == {"q": "x"}
    assert tool_error["error_message"] == "quota exceeded"
    assert tool_error["result"]["response_type"] == (
        "ERROR_HANDLED_BY_REFLECT_AND_RETRY_PLUGIN"
    )
    assert not _entries(plugin, "tool_response")

  async def test_mcp_error_result_is_recorded_as_tool_error(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    await plugin.after_tool_callback(
        tool=_tool("mcp_tool"),
        tool_args={},
        tool_context=mock_tool_context,
        result=_MCP_ERROR_RESULT,
    )

    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["error_message"] == (
        "Tool returned an MCP result with isError=true."
    )
    assert tool_error["result"] == _MCP_ERROR_RESULT
    assert not _entries(plugin, "tool_response")

  async def test_classified_error_is_recorded_as_tool_error(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file),
        tool_result_classifier=_classify_status_field,
    )
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    await plugin.after_tool_callback(
        tool=_tool("app_tool"),
        tool_args={},
        tool_context=mock_tool_context,
        result={"status": "ERROR", "reason": "row not found"},
    )

    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["error_message"] == "row not found"
    assert not _entries(plugin, "tool_response")

  async def test_classifier_ok_verdict_overrides_mcp_rule(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    classifier = _RecordingOkClassifier()
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), tool_result_classifier=classifier
    )
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    await plugin.after_tool_callback(
        tool=_tool("mcp_tool"),
        tool_args={},
        tool_context=mock_tool_context,
        result=_MCP_ERROR_RESULT,
    )

    assert classifier.results == [_MCP_ERROR_RESULT]
    assert len(_entries(plugin, "tool_response")) == 1
    assert not _entries(plugin, "tool_error")

  async def test_classifier_cannot_hide_retry_answer(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """A retry answer stands for a raised error; no classifier sees it."""
    classifier = _RecordingOkClassifier()
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file), tool_result_classifier=classifier
    )
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    tool = _tool("lookup")
    answer = await _retry_answer(
        tool, mock_tool_context, RuntimeError("quota exceeded")
    )

    await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=mock_tool_context, result=answer
    )

    assert classifier.results == []
    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["error_message"] == "quota exceeded"
    assert not _entries(plugin, "tool_response")

  async def test_failing_classifier_falls_back_to_built_in_rule(
      self,
      debug_output_file,
      mock_invocation_context,
      mock_tool_context,
      caplog,
  ):
    """The warning names only the tool, so the result stays out of the log."""
    plugin = DebugLoggingPlugin(
        output_path=str(debug_output_file),
        tool_result_classifier=_failing_classifier,
    )
    await plugin.before_run_callback(invocation_context=mock_invocation_context)

    with caplog.at_level(logging.WARNING, logger="google_adk"):
      await plugin.after_tool_callback(
          tool=_tool("app_tool"),
          tool_args={},
          tool_context=mock_tool_context,
          result={**_MCP_ERROR_RESULT, "secret": "sentinel-result-7c1d"},
      )

    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["error_message"] == (
        "Tool returned an MCP result with isError=true."
    )
    assert (
        "tool_result_classifier failed or returned an unusable verdict for"
        " tool app_tool"
        in caplog.text
    )
    assert "sentinel-result-7c1d" not in caplog.text

  @pytest.mark.parametrize(
      "classifier",
      ["not callable", _async_classifier, _classifier_without_result],
      ids=["not_callable", "async", "missing_result_keyword"],
  )
  def test_init_rejects_unusable_classifier(self, classifier):
    with pytest.raises(ValueError, match="tool_result_classifier"):
      DebugLoggingPlugin(tool_result_classifier=classifier)

  async def test_answer_to_a_recorded_tool_error_is_not_recorded_again(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    """A later plugin's answer to the error reaches after_tool_callback."""
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    tool = _tool("lookup")
    error = RuntimeError("quota exceeded")

    await plugin.on_tool_error_callback(
        tool=tool, tool_args={}, tool_context=mock_tool_context, error=error
    )
    answer = await _retry_answer(tool, mock_tool_context, error)
    await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=mock_tool_context, result=answer
    )

    [tool_error] = _entries(plugin, "tool_error")
    assert tool_error["error_type"] == "RuntimeError"
    assert tool_error["error_message"] == "quota exceeded"
    assert not _entries(plugin, "tool_response")

  async def test_recorded_tool_error_does_not_skip_another_call(
      self, debug_output_file, mock_invocation_context, mock_tool_context
  ):
    plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    await plugin.before_run_callback(invocation_context=mock_invocation_context)
    tool = _tool("lookup")
    other_call = Mock(spec=ToolContext)
    other_call.invocation_id = mock_invocation_context.invocation_id
    other_call.agent_name = "test-agent"
    other_call.function_call_id = "other-function-call-id"

    await plugin.on_tool_error_callback(
        tool=tool,
        tool_args={},
        tool_context=mock_tool_context,
        error=RuntimeError("boom"),
    )
    await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=other_call, result={"rows": 1}
    )

    [tool_response] = _entries(plugin, "tool_response")
    assert tool_response["function_call_id"] == "other-function-call-id"

  async def test_plugin_registered_after_the_answering_plugin_records_it(
      self, tmp_path, mock_invocation_context, mock_tool_context
  ):
    """Each plugin instance skips only the error it recorded itself.

    Registered as first, ReflectAndRetryToolPlugin, second: the retry plugin
    answers the error before second's on_tool_error_callback runs, so the
    answer is the only record of the error that second gets.
    """
    first = DebugLoggingPlugin(output_path=str(tmp_path / "first.yaml"))
    second = DebugLoggingPlugin(output_path=str(tmp_path / "second.yaml"))
    for debug_plugin in (first, second):
      await debug_plugin.before_run_callback(
          invocation_context=mock_invocation_context
      )
    tool = _tool("lookup")
    error = RuntimeError("quota exceeded")

    await first.on_tool_error_callback(
        tool=tool, tool_args={}, tool_context=mock_tool_context, error=error
    )
    answer = await _retry_answer(tool, mock_tool_context, error)
    for debug_plugin in (first, second):
      await debug_plugin.after_tool_callback(
          tool=tool,
          tool_args={},
          tool_context=mock_tool_context,
          result=answer,
      )

    for debug_plugin in (first, second):
      [tool_error] = _entries(debug_plugin, "tool_error")
      assert tool_error["error_message"] == "quota exceeded"
      assert not _entries(debug_plugin, "tool_response")

  @pytest.mark.parametrize(
      "debug_first", [False, True], ids=["retry_first", "debug_first"]
  )
  async def test_raised_error_answered_by_retry_plugin_is_recorded_once(
      self, debug_output_file, debug_first
  ):
    """Either plugin order writes one tool_error and no tool_response."""

    def raising_tool() -> str:
      raise RuntimeError("quota exceeded")

    debug_plugin = DebugLoggingPlugin(output_path=str(debug_output_file))
    retry_plugin = ReflectAndRetryToolPlugin(max_retries=3)
    agent = LlmAgent(
        name="agent",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(name="raising_tool", args={}),
                "done",
            ]
        ),
        tools=[FunctionTool(raising_tool)],
    )
    runner = testing_utils.InMemoryRunner(
        app=testing_utils.App(
            name="test_app",
            root_agent=agent,
            plugins=(
                [debug_plugin, retry_plugin]
                if debug_first
                else [retry_plugin, debug_plugin]
            ),
        )
    )

    await runner.run_async(testing_utils.UserContent("go"))

    with open(debug_output_file, "r") as f:
      [document] = yaml.safe_load_all(f)
    entry_types = [e["entry_type"] for e in document["entries"]]
    tool_errors = [
        e["data"]
        for e in document["entries"]
        if e["entry_type"] == "tool_error"
    ]
    assert [e["error_message"] for e in tool_errors] == ["quota exceeded"]
    assert "tool_response" not in entry_types
