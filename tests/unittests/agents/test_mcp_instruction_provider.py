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

"""Unit tests for McpInstructionProvider."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.agents.readonly_context import ReadonlyContext
import pytest


class TestMcpInstructionProvider:
  """Unit tests for McpInstructionProvider."""

  def setup_method(self):
    """Sets up the test environment."""
    self.connection_params = {"host": "localhost", "port": 8000}
    self.prompt_name = "test_prompt"
    self.mock_mcp_session_manager_cls = patch(
        "google.adk.agents.mcp_instruction_provider.MCPSessionManager"
    ).start()
    self.mock_mcp_session_manager = (
        self.mock_mcp_session_manager_cls.return_value
    )
    self.mock_session = MagicMock()
    self.mock_session.list_prompts = AsyncMock()
    self.mock_session.get_prompt = AsyncMock()
    self.mock_mcp_session_manager.create_session = AsyncMock(
        return_value=self.mock_session
    )
    self.provider = McpInstructionProvider(
        self.connection_params, self.prompt_name
    )

  @pytest.mark.asyncio
  async def test_call_success_no_args(self):
    """Tests __call__ with a prompt that has no arguments."""
    mock_msg1 = MagicMock()
    mock_msg1.content.type = "text"
    mock_msg1.content.text = "instruction part 1. "
    mock_msg2 = MagicMock()
    mock_msg2.content.type = "text"
    mock_msg2.content.text = "instruction part 2"
    self.mock_session.get_prompt.return_value = MagicMock(
        messages=[mock_msg1, mock_msg2]
    )

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {}
    context = ReadonlyContext(mock_invocation_context)

    # Call
    instruction = await self.provider(context)

    # Assert
    assert instruction == "instruction part 1. instruction part 2"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments=None
    )

  @pytest.mark.asyncio
  async def test_call_does_not_send_session_state(self):
    """Tests __call__ sends no session state, whatever the server declares."""
    # The server declares argument names that match session state keys.
    mock_arg1 = MagicMock()
    mock_arg1.name = "user:email"
    mock_arg2 = MagicMock()
    mock_arg2.name = "adk_oauth2_credential"
    mock_prompt = MagicMock()
    mock_prompt.name = self.prompt_name
    mock_prompt.arguments = [mock_arg1, mock_arg2]
    self.mock_session.list_prompts.return_value = MagicMock(
        prompts=[mock_prompt]
    )

    mock_msg = MagicMock()
    mock_msg.content.type = "text"
    mock_msg.content.text = "instruction"
    self.mock_session.get_prompt.return_value = MagicMock(messages=[mock_msg])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {
        "user:email": "user@example.com",
        "adk_oauth2_credential": "credential-value",
    }
    context = ReadonlyContext(mock_invocation_context)

    instruction = await self.provider(context)

    assert instruction == "instruction"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments=None
    )

  @pytest.mark.asyncio
  async def test_call_sends_static_arguments(self):
    """Tests __call__ sends the mapping the developer supplied."""
    provider = McpInstructionProvider(
        self.connection_params,
        self.prompt_name,
        arguments={"locale": "en-US"},
    )
    mock_msg = MagicMock()
    mock_msg.content.type = "text"
    mock_msg.content.text = "instruction"
    self.mock_session.get_prompt.return_value = MagicMock(messages=[mock_msg])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {"user:email": "user@example.com"}
    context = ReadonlyContext(mock_invocation_context)

    assert await provider(context) == "instruction"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments={"locale": "en-US"}
    )

  @pytest.mark.asyncio
  async def test_call_sends_arguments_from_callable(self):
    """Tests __call__ sends only the state keys the callable selects."""
    provider = McpInstructionProvider(
        self.connection_params,
        self.prompt_name,
        arguments=lambda ctx: {"locale": ctx.state["locale"]},
    )
    mock_msg = MagicMock()
    mock_msg.content.type = "text"
    mock_msg.content.text = "instruction"
    self.mock_session.get_prompt.return_value = MagicMock(messages=[mock_msg])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {
        "locale": "en-US",
        "adk_oauth2_credential": "credential-value",
    }
    context = ReadonlyContext(mock_invocation_context)

    assert await provider(context) == "instruction"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments={"locale": "en-US"}
    )

  @pytest.mark.asyncio
  async def test_call_get_prompt_returns_no_messages(self):
    """Tests __call__ when get_prompt returns no messages."""
    # Setup mocks
    self.mock_session.get_prompt.return_value = MagicMock(messages=[])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {}
    context = ReadonlyContext(mock_invocation_context)

    # Call and assert
    with pytest.raises(
        ValueError, match="Failed to load MCP prompt 'test_prompt'."
    ):
      await self.provider(context)

    # Assert
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments=None
    )

  @pytest.mark.asyncio
  async def test_call_ignore_non_text_messages(self):
    """Tests __call__ ignores non-text messages."""
    # Setup mocks
    mock_msg1 = MagicMock()
    mock_msg1.content.type = "text"
    mock_msg1.content.text = "instruction part 1. "

    mock_msg2 = MagicMock()
    mock_msg2.content.type = "image"
    mock_msg2.content.text = "ignored"

    mock_msg3 = MagicMock()
    mock_msg3.content.type = "text"
    mock_msg3.content.text = "instruction part 2"

    self.mock_session.get_prompt.return_value = MagicMock(
        messages=[mock_msg1, mock_msg2, mock_msg3]
    )

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {}
    context = ReadonlyContext(mock_invocation_context)

    # Call
    instruction = await self.provider(context)

    # Assert
    assert instruction == "instruction part 1. instruction part 2"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments=None
    )
