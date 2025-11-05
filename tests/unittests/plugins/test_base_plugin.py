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

from unittest.mock import Mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


class TestablePlugin(BasePlugin):
  __test__ = False
  """A concrete implementation of BasePlugin for testing purposes."""
  pass


class FullOverridePlugin(BasePlugin):
  __test__ = False

  """A plugin that overrides every single callback method for testing."""

  def __init__(self, name: str = "full_override"):
    super().__init__(name)

  async def on_user_message_callback(self, **kwargs) -> str:
    return "overridden_on_user_message"

  async def before_run_callback(self, **kwargs) -> str:
    return "overridden_before_run"

  async def after_run_callback(self, **kwargs) -> str:
    return "overridden_after_run"

  async def on_event_callback(self, **kwargs) -> str:
    return "overridden_on_event"

  async def before_agent_callback(self, **kwargs) -> str:
    return "overridden_before_agent"

  async def after_agent_callback(self, **kwargs) -> str:
    return "overridden_after_agent"

  async def before_tool_callback(self, **kwargs) -> str:
    return "overridden_before_tool"

  async def after_tool_callback(self, **kwargs) -> str:
    return "overridden_after_tool"

  async def on_tool_error_callback(self, **kwargs) -> str:
    return "overridden_on_tool_error"

  async def before_model_callback(self, **kwargs) -> str:
    return "overridden_before_model"

  async def after_model_callback(self, **kwargs) -> str:
    return "overridden_after_model"

  async def on_model_error_callback(self, **kwargs) -> str:
    return "overridden_on_model_error"


def test_base_plugin_initialization():
  """Tests that a plugin is initialized with the correct name."""
  plugin_name = "my_test_plugin"
  plugin = TestablePlugin(name=plugin_name)
  assert plugin.name == plugin_name


@pytest.mark.asyncio
async def test_base_plugin_default_callbacks_return_none():
  """Tests that the default (non-overridden) callbacks in BasePlugin exist

  and return None as expected.
  """
  plugin = TestablePlugin(name="default_plugin")

  # Mocking all necessary context objects
  mock_context = Mock()
  mock_user_message = Mock()

  # The default implementations should do nothing and return None.
  assert (
      await plugin.on_user_message_callback(
          user_message=mock_user_message,
          invocation_context=mock_context,
      )
      is None
  )
  assert (
      await plugin.before_run_callback(invocation_context=mock_context) is None
  )
  assert (
      await plugin.after_run_callback(invocation_context=mock_context) is None
  )
  assert (
      await plugin.on_event_callback(
          invocation_context=mock_context, event=mock_context
      )
      is None
  )
  assert (
      await plugin.before_agent_callback(
          agent=mock_context, callback_context=mock_context
      )
      is None
  )
  assert (
      await plugin.after_agent_callback(
          agent=mock_context, callback_context=mock_context
      )
      is None
  )
  assert (
      await plugin.before_tool_callback(
          tool=mock_context, tool_args={}, tool_context=mock_context
      )
      is None
  )
  assert (
      await plugin.after_tool_callback(
          tool=mock_context, tool_args={}, tool_context=mock_context, result={}
      )
      is None
  )
  assert (
      await plugin.on_tool_error_callback(
          tool=mock_context,
          tool_args={},
          tool_context=mock_context,
          error=Exception(),
      )
      is None
  )
  assert (
      await plugin.before_model_callback(
          callback_context=mock_context, llm_request=mock_context
      )
      is None
  )
  assert (
      await plugin.after_model_callback(
          callback_context=mock_context, llm_response=mock_context
      )
      is None
  )
  assert (
      await plugin.on_model_error_callback(
          callback_context=mock_context,
          llm_request=mock_context,
          error=Exception(),
      )
      is None
  )


@pytest.mark.asyncio
async def test_base_plugin_all_callbacks_can_be_overridden():
  """Verifies that a user can create a subclass of BasePlugin and that all

  overridden methods are correctly called.
  """
  plugin = FullOverridePlugin()

  # Create mock objects for all required arguments. We don't need real
  # objects, just placeholders to satisfy the method signatures.
  mock_user_message = Mock(spec=types.Content)
  mock_invocation_context = Mock(spec=InvocationContext)
  mock_callback_context = Mock(spec=CallbackContext)
  mock_agent = Mock(spec=BaseAgent)
  mock_tool = Mock(spec=BaseTool)
  mock_tool_context = Mock(spec=ToolContext)
  mock_llm_request = Mock(spec=LlmRequest)
  mock_llm_response = Mock(spec=LlmResponse)
  mock_event = Mock(spec=Event)
  mock_error = Mock(spec=Exception)

  # Call each method and assert it returns the unique string from the override.
  # This proves that the subclass's method was executed.
  assert (
      await plugin.on_user_message_callback(
          user_message=mock_user_message,
          invocation_context=mock_invocation_context,
      )
      == "overridden_on_user_message"
  )
  assert (
      await plugin.before_run_callback(
          invocation_context=mock_invocation_context
      )
      == "overridden_before_run"
  )
  assert (
      await plugin.after_run_callback(
          invocation_context=mock_invocation_context
      )
      == "overridden_after_run"
  )
  assert (
      await plugin.on_event_callback(
          invocation_context=mock_invocation_context, event=mock_event
      )
      == "overridden_on_event"
  )
  assert (
      await plugin.before_agent_callback(
          agent=mock_agent, callback_context=mock_callback_context
      )
      == "overridden_before_agent"
  )
  assert (
      await plugin.after_agent_callback(
          agent=mock_agent, callback_context=mock_callback_context
      )
      == "overridden_after_agent"
  )
  assert (
      await plugin.before_model_callback(
          callback_context=mock_callback_context, llm_request=mock_llm_request
      )
      == "overridden_before_model"
  )
  assert (
      await plugin.after_model_callback(
          callback_context=mock_callback_context, llm_response=mock_llm_response
      )
      == "overridden_after_model"
  )
  assert (
      await plugin.before_tool_callback(
          tool=mock_tool, tool_args={}, tool_context=mock_tool_context
      )
      == "overridden_before_tool"
  )
  assert (
      await plugin.after_tool_callback(
          tool=mock_tool,
          tool_args={},
          tool_context=mock_tool_context,
          result={},
      )
      == "overridden_after_tool"
  )
  assert (
      await plugin.on_tool_error_callback(
          tool=mock_tool,
          tool_args={},
          tool_context=mock_tool_context,
          error=mock_error,
      )
      == "overridden_on_tool_error"
  )
  assert (
      await plugin.on_model_error_callback(
          callback_context=mock_callback_context,
          llm_request=mock_llm_request,
          error=mock_error,
      )
      == "overridden_on_model_error"
  )
