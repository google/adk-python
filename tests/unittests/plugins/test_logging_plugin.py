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

"""Unit tests for LoggingPlugin's console rendering of a run."""

from __future__ import annotations

from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest


@pytest.fixture
def plugin():
  return LoggingPlugin()


@pytest.fixture
def callback_context():
  ctx = Mock(spec=CallbackContext)
  ctx.agent_name = 'test-agent'
  ctx.invocation_id = 'test-invocation'
  return ctx


@pytest.fixture
def tool_context():
  ctx = Mock(spec=ToolContext)
  ctx.agent_name = 'test-agent'
  ctx.invocation_id = 'test-invocation'
  ctx.function_call_id = 'call-1'
  return ctx


def _tool(name: str) -> BaseTool:
  tool = Mock(spec=BaseTool)
  tool.name = name
  return tool


async def test_before_model_callback_truncates_long_system_instruction(
    plugin, callback_context, capsys
):
  llm_request = LlmRequest(
      model='test-model',
      config=types.GenerateContentConfig(
          system_instruction='a' * 200 + 'Z' * 50
      ),
  )

  result = await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  out = capsys.readouterr().out
  assert result is None
  assert f"System Instruction: '{'a' * 200}...'" in out
  # Everything past the 200-char budget is dropped, not merely elided.
  assert 'Z' not in out


async def test_before_model_callback_keeps_system_instruction_at_budget(
    plugin, callback_context, capsys
):
  llm_request = LlmRequest(
      model='test-model',
      config=types.GenerateContentConfig(system_instruction='a' * 200),
  )

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  out = capsys.readouterr().out
  assert f"System Instruction: '{'a' * 200}'" in out


async def test_before_model_callback_lists_available_tool_names(
    plugin, callback_context, capsys
):
  llm_request = LlmRequest(model='test-model')
  llm_request.tools_dict = {'alpha': _tool('alpha'), 'beta': _tool('beta')}

  await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  out = capsys.readouterr().out
  assert "Available Tools: ['alpha', 'beta']" in out
  assert 'Model: test-model' in out


async def test_after_model_callback_logs_error_instead_of_content(
    plugin, callback_context, capsys
):
  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part(text='unreachable-text')]),
      error_code='429',
      error_message='rate limited',
  )

  result = await plugin.after_model_callback(
      callback_context=callback_context, llm_response=llm_response
  )

  out = capsys.readouterr().out
  assert result is None
  assert 'ERROR - Code: 429' in out
  assert 'Error Message: rate limited' in out
  # An errored response carries no usable content; logging it would bury the
  # error under an empty "Content:" line.
  assert 'unreachable-text' not in out
  assert 'Content:' not in out


async def test_after_model_callback_logs_content_and_token_usage(
    plugin, callback_context, capsys
):
  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part(text='hello')]),
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=11, candidates_token_count=7
      ),
  )

  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=llm_response
  )

  out = capsys.readouterr().out
  assert "Content: text: 'hello'" in out
  assert 'Token Usage - Input: 11, Output: 7' in out


async def test_on_event_callback_summarizes_function_parts(plugin, capsys):
  event = Event(
      author='test-agent',
      content=types.Content(
          parts=[
              types.Part.from_function_call(name='do_thing', args={'x': 1}),
              types.Part.from_function_response(
                  name='do_thing', response={'ok': True}
              ),
          ]
      ),
  )

  result = await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert result is None
  assert 'Content: function_call: do_thing | function_response: do_thing' in out
  assert "Function Calls: ['do_thing']" in out
  assert "Function Responses: ['do_thing']" in out


async def test_on_event_callback_renders_absent_content_as_none(plugin, capsys):
  event = Event(author='test-agent', content=None)

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert 'Content: None' in out


async def test_on_event_callback_truncates_long_text_part(plugin, capsys):
  event = Event(
      author='test-agent',
      content=types.Content(parts=[types.Part(text='a' * 200 + 'Z' * 50)]),
  )

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert f"text: '{'a' * 200}...'" in out
  assert 'Z' not in out


async def test_before_tool_callback_truncates_long_arguments(
    plugin, tool_context, capsys
):
  tool_args = {'payload': 'a' * 400}

  result = await plugin.before_tool_callback(
      tool=_tool('my_tool'), tool_args=tool_args, tool_context=tool_context
  )

  out = capsys.readouterr().out
  assert result is None
  assert f'Arguments: {str(tool_args)[:300]}...}}' in out
  # The full payload must not reach the console.
  assert str(tool_args) not in out
