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

"""Unit tests for McpInstructionProvider."""
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.readonly_context import ReadonlyContext
import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="MCP instruction provider requires Python 3.10+",
)

# Import dependencies with version checking
try:
  from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyClass:
      pass

    McpInstructionProvider = DummyClass
  else:
    raise e


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
    mock_prompt = MagicMock()
    mock_prompt.name = self.prompt_name
    mock_prompt.arguments = None
    self.mock_session.list_prompts.return_value = MagicMock(
        prompts=[mock_prompt]
    )

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
        self.prompt_name, arguments={}
    )

  @pytest.mark.asyncio
  async def test_call_success_with_args(self):
    """Tests __call__ with a prompt that has arguments."""
    mock_arg1 = MagicMock()
    mock_arg1.name = "arg1"
    mock_prompt = MagicMock()
    mock_prompt.name = self.prompt_name
    mock_prompt.arguments = [mock_arg1]
    self.mock_session.list_prompts.return_value = MagicMock(
        prompts=[mock_prompt]
    )

    mock_msg = MagicMock()
    mock_msg.content.type = "text"
    mock_msg.content.text = "instruction with arg1"
    self.mock_session.get_prompt.return_value = MagicMock(messages=[mock_msg])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {"arg1": "value1", "arg2": "value2"}
    context = ReadonlyContext(mock_invocation_context)

    instruction = await self.provider(context)

    assert instruction == "instruction with arg1"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments={"arg1": "value1"}
    )

  @pytest.mark.asyncio
  async def test_call_prompt_not_found_in_list_prompts(self):
    """Tests __call__ when list_prompts doesn't return the prompt."""
    self.mock_session.list_prompts.return_value = MagicMock(prompts=[])

    mock_msg = MagicMock()
    mock_msg.content.type = "text"
    mock_msg.content.text = "instruction"
    self.mock_session.get_prompt.return_value = MagicMock(messages=[mock_msg])

    mock_invocation_context = MagicMock()
    mock_invocation_context.session.state = {"arg1": "value1"}
    context = ReadonlyContext(mock_invocation_context)

    instruction = await self.provider(context)

    assert instruction == "instruction"
    self.mock_session.get_prompt.assert_called_once_with(
        self.prompt_name, arguments={}
    )

  @pytest.mark.asyncio
  async def test_call_get_prompt_returns_no_messages(self):
    """Tests __call__ when get_prompt returns no messages."""
    # Setup mocks
    self.mock_session.list_prompts.return_value = MagicMock(prompts=[])
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
        self.prompt_name, arguments={}
    )

  @pytest.mark.asyncio
  async def test_call_ignore_non_text_messages(self):
    """Tests __call__ ignores non-text messages."""
    # Setup mocks
    mock_prompt = MagicMock()
    mock_prompt.name = self.prompt_name
    mock_prompt.arguments = None
    self.mock_session.list_prompts.return_value = MagicMock(
        prompts=[mock_prompt]
    )

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
        self.prompt_name, arguments={}
    )
