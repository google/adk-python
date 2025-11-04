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

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.sessions.session import Session
from google.genai import types
import pytest


@pytest.mark.asyncio
async def test_global_instruction_plugin_with_string():
  """Test GlobalInstructionPlugin with a string global instruction."""
  plugin = GlobalInstructionPlugin(
      global_instruction=(
          "You are a helpful assistant with a friendly personality."
      )
  )

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should now contain the global instruction
  assert (
      "You are a helpful assistant with a friendly personality."
      in llm_request.config.system_instruction
  )


@pytest.mark.asyncio
async def test_global_instruction_plugin_with_instruction_provider():
  """Test GlobalInstructionPlugin with an InstructionProvider function."""

  async def build_global_instruction(readonly_context: ReadonlyContext) -> str:
    return f"You are assistant for user {readonly_context.session.user_id}."

  plugin = GlobalInstructionPlugin(global_instruction=build_global_instruction)

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="alice", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context
  mock_callback_context.session = mock_session

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(system_instruction=""),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should contain the dynamically generated instruction
  assert (
      "You are assistant for user alice."
      in llm_request.config.system_instruction
  )


@pytest.mark.asyncio
async def test_global_instruction_plugin_empty_instruction():
  """Test GlobalInstructionPlugin with empty global instruction."""
  plugin = GlobalInstructionPlugin(global_instruction="")

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(
          system_instruction="Original instruction"
      ),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should remain unchanged
  assert llm_request.config.system_instruction == "Original instruction"


@pytest.mark.asyncio
async def test_global_instruction_plugin_leads_existing():
  """Test that GlobalInstructionPlugin prepends global instructions."""
  plugin = GlobalInstructionPlugin(
      global_instruction="You are a helpful assistant."
  )

  # Create mock objects
  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(
          system_instruction="Existing instructions."
      ),
  )

  # Execute the plugin's before_model_callback
  result = await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  # Plugin should return None to allow normal processing
  assert result is None

  # System instruction should contain global instruction before existing ones
  expected = "You are a helpful assistant.\n\nExisting instructions."
  assert llm_request.config.system_instruction == expected


@pytest.mark.asyncio
async def test_global_instruction_plugin_prepends_to_list():
  """Test GlobalInstructionPlugin prepends to a list of instructions."""
  plugin = GlobalInstructionPlugin(global_instruction="Global instruction.")

  mock_session = Session(
      app_name="test_app", user_id="test_user", id="test_session", state={}
  )

  mock_invocation_context = Mock(spec=InvocationContext)
  mock_invocation_context.session = mock_session

  mock_callback_context = Mock(spec=CallbackContext)
  mock_callback_context._invocation_context = mock_invocation_context

  llm_request = LlmRequest(
      model="gemini-1.5-flash",
      config=types.GenerateContentConfig(
          system_instruction=["Existing instruction."]
      ),
  )

  await plugin.before_model_callback(
      callback_context=mock_callback_context, llm_request=llm_request
  )

  expected = ["Global instruction.", "Existing instruction."]
  assert llm_request.config.system_instruction == expected
