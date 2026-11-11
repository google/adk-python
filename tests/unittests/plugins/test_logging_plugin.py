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

import logging
from typing import Any
from typing import Optional
from unittest.mock import Mock

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import ToolResultClassification
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils


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


def _node_event(
    *, message_as_output: Optional[bool] = None, **fields: Any
) -> Event:
  """Returns an event of the workflow node at path ``wf@1/step@2``."""
  return Event(
      author='step',
      node_info=NodeInfo(
          path='wf@1/step@2', message_as_output=message_as_output
      ),
      **fields,
  )


async def _retry_answer(
    tool: BaseTool, tool_context: ToolContext, error: Exception
) -> dict[str, Any]:
  """Returns ReflectAndRetryToolPlugin's real answer to a raised error."""
  return await ReflectAndRetryToolPlugin(max_retries=3).on_tool_error_callback(
      tool=tool, tool_args={}, tool_context=tool_context, error=error
  )


# The dump of an MCP CallToolResult that reports a failure.
_MCP_ERROR_RESULT = {
    'content': [{'type': 'text', 'text': 'permission denied'}],
    'isError': True,
}


def _classify_status_field(*, tool, tool_args, tool_context, result):
  """Classifies a result whose ``status`` is ``ERROR`` as a tool error."""
  del tool, tool_args, tool_context  # Unused.
  if isinstance(result, dict) and result.get('status') == 'ERROR':
    return ToolResultClassification(
        status='ERROR', error_message=result.get('reason')
    )
  return None


class _RecordingOkClassifier:
  """Classifies every result as a completed call, recording each one."""

  def __init__(self):
    self.results = []

  def __call__(self, *, tool, tool_args, tool_context, result):
    del tool, tool_args, tool_context  # Unused.
    self.results.append(result)
    return ToolResultClassification(status='OK')


def _failing_classifier(*, tool, tool_args, tool_context, result):
  del tool, tool_args, tool_context  # Unused.
  raise RuntimeError(f'cannot classify {result!r}')


async def _async_classifier(*, tool, tool_args, tool_context, result):
  del tool, tool_args, tool_context, result  # Unused.
  return None


def _classifier_without_result(*, tool, tool_args, tool_context):
  del tool, tool_args, tool_context  # Unused.
  return None


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


async def test_before_model_callback_formats_content_system_instruction(
    plugin, callback_context, capsys
):
  """ContentUnion instructions are logged without assuming string behavior."""
  llm_request = LlmRequest(
      model='test-model',
      config=types.GenerateContentConfig(
          system_instruction=types.Content(
              role='system',
              parts=[types.Part.from_text(text='Stay concise.')],
          )
      ),
  )

  result = await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert result is None
  assert 'Stay concise.' in capsys.readouterr().out


async def test_before_model_callback_formats_list_system_instruction(
    plugin, callback_context, capsys
):
  """A list instruction renders every entry rather than its repr."""
  llm_request = LlmRequest(
      model='test-model',
      config=types.GenerateContentConfig(
          system_instruction=[
              types.Part.from_text(text='Stay concise.'),
              types.Part.from_text(text='Cite sources.'),
          ]
      ),
  )

  result = await plugin.before_model_callback(
      callback_context=callback_context, llm_request=llm_request
  )

  assert result is None
  out = capsys.readouterr().out
  assert 'Stay concise.' in out
  assert 'Cite sources.' in out


async def test_after_model_callback_logs_finish_reason_of_final_response(
    plugin, callback_context, capsys
):
  llm_response = LlmResponse(
      content=types.Content(parts=[types.Part(text='hello')]),
      finish_reason=types.FinishReason.STOP,
  )

  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=llm_response
  )

  out = capsys.readouterr().out
  assert 'Finish Reason: STOP' in out
  # The reason's value, not the enum's repr.
  assert 'FinishReason.' not in out


async def test_after_model_callback_logs_termination_of_a_stream_once(
    plugin, callback_context, capsys
):
  """A streamed chunk's error and finish reason wait for the final response.

  The final response of a stream carries the same values, so logging them on
  the chunk too would log one termination twice.
  """
  chunk = LlmResponse(
      content=types.Content(parts=[types.Part(text='partial-text')]),
      partial=True,
      error_code='MAX_TOKENS',
      error_message='cut off',
      finish_reason=types.FinishReason.MAX_TOKENS,
  )
  final = chunk.model_copy(update={'partial': False})

  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=chunk
  )
  chunk_out = capsys.readouterr().out
  await plugin.after_model_callback(
      callback_context=callback_context, llm_response=final
  )
  final_out = capsys.readouterr().out

  assert "Content: text: 'partial-text'" in chunk_out
  assert 'Partial: True' in chunk_out
  assert 'ERROR - Code' not in chunk_out
  assert 'Finish Reason' not in chunk_out
  assert 'ERROR - Code: MAX_TOKENS' in final_out
  assert 'Error Message: cut off' in final_out
  assert 'Finish Reason: MAX_TOKENS' in final_out


async def test_on_event_callback_logs_node_path_and_output(plugin, capsys):
  await plugin.on_event_callback(
      invocation_context=None, event=_node_event(output={'answer': 42})
  )

  out = capsys.readouterr().out
  assert 'Node Path: wf@1/step@2' in out
  assert "Node Output: {'answer': 42}" in out
  assert 'Node Error' not in out


async def test_on_event_callback_quotes_string_node_output(plugin, capsys):
  """A string output is quoted, so an empty one still shows."""
  await plugin.on_event_callback(
      invocation_context=None, event=_node_event(output='')
  )

  assert "Node Output: ''" in capsys.readouterr().out


async def test_on_event_callback_skips_output_its_message_carries(
    plugin, capsys
):
  event = _node_event(
      content=types.Content(parts=[types.Part(text='the answer')]),
      output='the answer',
      message_as_output=True,
  )

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert 'Node Path: wf@1/step@2' in out
  assert 'Node Output' not in out


async def test_on_event_callback_logs_node_error_beside_its_output(
    plugin, capsys
):
  event = _node_event(
      error_code='ValueError',
      error_message='invalid input',
      output={'processed': 3},
  )

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert 'Node Error: [ValueError] invalid input' in out
  assert "Node Output: {'processed': 3}" in out


async def test_on_event_callback_logs_node_error_without_message(
    plugin, capsys
):
  await plugin.on_event_callback(
      invocation_context=None, event=_node_event(error_code='ValueError')
  )

  # Neither a trailing space nor 'None' stands in for the missing message.
  assert 'Node Error: [ValueError]\033[0m' in capsys.readouterr().out


async def test_on_event_callback_skips_node_output_and_error_of_partial_event(
    plugin, capsys
):
  event = _node_event(
      partial=True,
      output={'processed': 3},
      error_code='ValueError',
      error_message='invalid input',
  )

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert 'Node Path: wf@1/step@2' in out
  assert 'Node Output' not in out
  assert 'Node Error' not in out


@pytest.mark.parametrize(
    'error_code',
    ['MAX_TOKENS', 'SAFETY', 'MODEL_ARMOR', types.FinishReason.MAX_TOKENS],
)
async def test_on_event_callback_does_not_log_model_termination_as_node_error(
    plugin, capsys, error_code
):
  event = _node_event(error_code=error_code, error_message='model stopped')

  await plugin.on_event_callback(invocation_context=None, event=event)

  out = capsys.readouterr().out
  assert 'Node Path: wf@1/step@2' in out
  assert 'Node Error' not in out


async def test_on_event_callback_logs_no_node_lines_outside_a_workflow(
    plugin, capsys
):
  event = Event(
      author='agent',
      output={'processed': 3},
      error_code='ValueError',
      error_message='invalid input',
  )

  await plugin.on_event_callback(invocation_context=None, event=event)

  assert 'Node ' not in capsys.readouterr().out


async def test_on_event_callback_truncates_long_node_output(plugin, capsys):
  await plugin.on_event_callback(
      invocation_context=None, event=_node_event(output='a' * 400)
  )

  assert f"Node Output: '{'a' * 299}...\033[0m" in capsys.readouterr().out


async def test_after_tool_callback_logs_result_with_error_keys_as_completed(
    plugin, tool_context, capsys
):
  """A result that merely has error-like keys is the tool's data."""
  result = await plugin.after_tool_callback(
      tool=_tool('lookup'),
      tool_args={},
      tool_context=tool_context,
      result={'error': 'none', 'error_details': 'none'},
  )

  out = capsys.readouterr().out
  assert result is None
  assert '🔧 TOOL COMPLETED' in out
  assert 'TOOL ERROR' not in out


async def test_after_tool_callback_logs_retry_answer_as_tool_error(
    plugin, tool_context, capsys
):
  tool = _tool('lookup')
  answer = await _retry_answer(
      tool, tool_context, RuntimeError('quota exceeded')
  )

  result = await plugin.after_tool_callback(
      tool=tool, tool_args={'q': 'x'}, tool_context=tool_context, result=answer
  )

  out = capsys.readouterr().out
  assert result is None
  assert '🔧 TOOL ERROR' in out
  assert 'Tool Name: lookup' in out
  assert 'Function Call ID: call-1' in out
  assert "Arguments: {'q': 'x'}" in out
  assert 'Error: quota exceeded' in out
  assert 'TOOL COMPLETED' not in out


async def test_after_tool_callback_logs_mcp_error_result_as_tool_error(
    plugin, tool_context, capsys
):
  await plugin.after_tool_callback(
      tool=_tool('mcp_tool'),
      tool_args={},
      tool_context=tool_context,
      result=_MCP_ERROR_RESULT,
  )

  out = capsys.readouterr().out
  assert '🔧 TOOL ERROR' in out
  assert 'Error: Tool returned an MCP result with isError=true.' in out
  assert 'TOOL COMPLETED' not in out


async def test_after_tool_callback_logs_classified_error(tool_context, capsys):
  plugin = LoggingPlugin(tool_result_classifier=_classify_status_field)

  await plugin.after_tool_callback(
      tool=_tool('app_tool'),
      tool_args={},
      tool_context=tool_context,
      result={'status': 'ERROR', 'reason': 'row not found'},
  )

  out = capsys.readouterr().out
  assert '🔧 TOOL ERROR' in out
  assert 'Error: row not found' in out
  assert 'TOOL COMPLETED' not in out


async def test_classifier_ok_verdict_overrides_mcp_rule(tool_context, capsys):
  classifier = _RecordingOkClassifier()
  plugin = LoggingPlugin(tool_result_classifier=classifier)

  await plugin.after_tool_callback(
      tool=_tool('mcp_tool'),
      tool_args={},
      tool_context=tool_context,
      result=_MCP_ERROR_RESULT,
  )

  out = capsys.readouterr().out
  assert classifier.results == [_MCP_ERROR_RESULT]
  assert '🔧 TOOL COMPLETED' in out
  assert 'TOOL ERROR' not in out


async def test_classifier_cannot_hide_retry_answer(tool_context, capsys):
  """A retry answer stands for a raised error; no classifier sees it."""
  classifier = _RecordingOkClassifier()
  plugin = LoggingPlugin(tool_result_classifier=classifier)
  tool = _tool('lookup')
  answer = await _retry_answer(
      tool, tool_context, RuntimeError('quota exceeded')
  )

  await plugin.after_tool_callback(
      tool=tool, tool_args={}, tool_context=tool_context, result=answer
  )

  out = capsys.readouterr().out
  assert classifier.results == []
  assert 'Error: quota exceeded' in out
  assert 'TOOL COMPLETED' not in out


async def test_failing_classifier_falls_back_to_built_in_rule(
    tool_context, capsys, caplog
):
  """The warning names only the tool, so the result stays out of the log."""
  plugin = LoggingPlugin(tool_result_classifier=_failing_classifier)
  result = {**_MCP_ERROR_RESULT, 'secret': 'sentinel-result-7c1d'}

  with caplog.at_level(logging.WARNING, logger='google_adk'):
    await plugin.after_tool_callback(
        tool=_tool('app_tool'),
        tool_args={},
        tool_context=tool_context,
        result=result,
    )

  out = capsys.readouterr().out
  assert 'Error: Tool returned an MCP result with isError=true.' in out
  assert (
      'tool_result_classifier failed or returned an unusable verdict for tool'
      ' app_tool'
      in caplog.text
  )
  assert 'sentinel-result-7c1d' not in caplog.text


@pytest.mark.parametrize(
    'classifier',
    ['not callable', _async_classifier, _classifier_without_result],
    ids=['not_callable', 'async', 'missing_result_keyword'],
)
def test_init_rejects_unusable_classifier(classifier):
  with pytest.raises(ValueError, match='tool_result_classifier'):
    LoggingPlugin(tool_result_classifier=classifier)


async def test_answer_to_a_logged_tool_error_is_not_logged_again(
    plugin, tool_context, capsys
):
  """A later plugin's answer to the error reaches after_tool_callback."""
  tool = _tool('lookup')
  error = RuntimeError('quota exceeded')

  await plugin.on_tool_error_callback(
      tool=tool, tool_args={}, tool_context=tool_context, error=error
  )
  answer = await _retry_answer(tool, tool_context, error)
  await plugin.after_tool_callback(
      tool=tool, tool_args={}, tool_context=tool_context, result=answer
  )

  out = capsys.readouterr().out
  assert out.count('TOOL ERROR') == 1
  assert 'Error: quota exceeded' in out
  assert 'TOOL COMPLETED' not in out


async def test_logged_tool_error_does_not_skip_another_call(
    plugin, tool_context, capsys
):
  tool = _tool('lookup')
  other_call = Mock(spec=ToolContext)
  other_call.agent_name = 'test-agent'
  other_call.invocation_id = 'test-invocation'
  other_call.function_call_id = 'call-2'

  await plugin.on_tool_error_callback(
      tool=tool,
      tool_args={},
      tool_context=tool_context,
      error=RuntimeError('boom'),
  )
  await plugin.after_tool_callback(
      tool=tool, tool_args={}, tool_context=other_call, result={'rows': 1}
  )

  out = capsys.readouterr().out
  assert '🔧 TOOL COMPLETED' in out
  assert 'Function Call ID: call-2' in out


async def test_plugin_registered_after_the_answering_plugin_logs_the_answer(
    tool_context, capsys
):
  """Each plugin instance skips only the error it logged itself.

  Registered as first, ReflectAndRetryToolPlugin, second: the retry plugin
  answers the error before second's on_tool_error_callback runs, so the
  answer is the only record of the error that second gets.
  """
  first = LoggingPlugin(name='first')
  second = LoggingPlugin(name='second')
  tool = _tool('lookup')
  error = RuntimeError('quota exceeded')

  await first.on_tool_error_callback(
      tool=tool, tool_args={}, tool_context=tool_context, error=error
  )
  answer = await _retry_answer(tool, tool_context, error)
  for logging_plugin in (first, second):
    await logging_plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=tool_context, result=answer
    )

  out = capsys.readouterr().out
  assert out.count('[first] 🔧 TOOL ERROR') == 1
  assert out.count('[second] 🔧 TOOL ERROR') == 1
  assert 'TOOL COMPLETED' not in out


@pytest.mark.parametrize(
    'logging_first', [False, True], ids=['retry_first', 'logging_first']
)
async def test_raised_error_answered_by_retry_plugin_is_logged_once(
    capsys, logging_first
):
  """Either plugin order logs one TOOL ERROR and no TOOL COMPLETED."""

  def raising_tool() -> str:
    raise RuntimeError('quota exceeded')

  logging_plugin = LoggingPlugin()
  retry_plugin = ReflectAndRetryToolPlugin(max_retries=3)
  agent = LlmAgent(
      name='agent',
      model=testing_utils.MockModel.create(
          responses=[
              types.Part.from_function_call(name='raising_tool', args={}),
              'done',
          ]
      ),
      tools=[FunctionTool(raising_tool)],
  )
  runner = testing_utils.InMemoryRunner(
      app=testing_utils.App(
          name='test_app',
          root_agent=agent,
          plugins=(
              [logging_plugin, retry_plugin]
              if logging_first
              else [retry_plugin, logging_plugin]
          ),
      )
  )

  await runner.run_async(testing_utils.UserContent('go'))

  out = capsys.readouterr().out
  assert out.count('🔧 TOOL ERROR') == 1
  assert 'Error: quota exceeded' in out
  assert 'TOOL COMPLETED' not in out
