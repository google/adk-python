# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native stdio evaluation must complete inference and close its MCP session."""

import logging
import sys
import textwrap

from google.adk import runners
from google.adk.agents.llm_agent import LlmAgent
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceStatus
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_set import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.tools.mcp_tool import mcp_toolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types
from mcp import StdioServerParameters
from mcp.client import stdio
import pytest
import pytest_asyncio

from tests.unittests.testing_utils import MockModel


@pytest_asyncio.fixture
async def native_stdio_toolset(tmp_path):
  calls = tmp_path / 'calls.txt'
  ready = tmp_path / 'ready.txt'
  server = tmp_path / 'server.py'
  # A finite wire peer avoids a test-only dependency on one SDK's server API.
  server.write_text(
      textwrap.dedent("""\
      import json
      from pathlib import Path
      import sys

      tool_error = False
      Path(@@READY@@).write_text('ready\\n')
      for line in sys.stdin:
          request = json.loads(line)
          assert request['jsonrpc'] == '2.0'
          method = request['method']
          if method == 'notifications/initialized':
              assert 'id' not in request
              continue
          request_id = request['id']
          params = request.get('params', {})
          if method == 'initialize':
              result = {
                  'protocolVersion': params['protocolVersion'],
                  'capabilities': {'tools': {'listChanged': False}},
                  'serverInfo': {'name': 'finite-evaluation-test', 'version': '1'},
              }
          elif method == 'ping':
              result = {}
          elif method == 'tools/list':
              result = {'tools': [{
                  'name': 'probe',
                  'description': 'Return one recorded finite probe.',
                  'inputSchema': {
                      'type': 'object',
                      'properties': {'value': {'type': 'string'}},
                      'required': ['value'],
                  },
              }]}
          elif method == 'tools/call':
              assert params['name'] == 'probe'
              value = params['arguments']['value']
              assert isinstance(value, str)
              with Path(@@CALLS@@).open('a') as output:
                  output.write(value + '\\n')
              result = {
                  'content': [{'type': 'text', 'text': 'native-result:' + value}],
                  'isError': tool_error,
              }
          else:
              raise RuntimeError('unexpected MCP method: ' + method)
          print(json.dumps({
              'jsonrpc': '2.0', 'id': request_id, 'result': result,
          }), flush=True)
      """)
      .replace('@@READY@@', repr(str(ready)))
      .replace('@@CALLS@@', repr(str(calls)))
  )
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(
              command=sys.executable, args=[str(server)]
          ),
          timeout=5,
      ),
      tool_filter=['probe'],
  )
  real_close = toolset.close
  try:
    yield toolset, calls, ready, server
  finally:
    # Test failures must not leave the fixture's real subprocess running.
    await real_close()


async def _run_native_eval(mocker, model, toolset):
  agent = LlmAgent(name='native_mcp_agent', model=model, tools=[toolset])
  case = EvalCase(
      eval_id='native_mcp_case',
      conversation=[
          Invocation(
              user_content=types.Content(
                  parts=[types.Part(text='Run the probe.')]
              )
          )
      ],
  )
  manager = mocker.create_autospec(EvalSetsManager)
  manager.get_eval_case.return_value = case
  manager.get_eval_set.return_value = EvalSet(
      eval_set_id='native_mcp_set', eval_cases=[case]
  )
  service = LocalEvalService(root_agent=agent, eval_sets_manager=manager)
  request = InferenceRequest(
      app_name='native_mcp_app',
      eval_set_id='native_mcp_set',
      inference_config=InferenceConfig(parallelism=1),
  )
  return [result async for result in service.perform_inference(request)]


def _assert_successful_native_inference(
    results, model, calls, ready, toolset, processes, caplog
):
  assert len(results) == 1
  result = results[0]
  assert (
      result.status == InferenceStatus.SUCCESS
  ), f'MCP inference failed: {result.error_message}'
  assert result.error_message is None
  assert result.inferences is not None and len(result.inferences) == 1
  assert ready.is_file(), 'MCP server did not start'
  assert ready.read_text() == 'ready\n'
  assert calls.is_file(), 'MCP tool was not called'
  assert calls.read_text().splitlines() == ['checked']
  invocation = result.inferences[0]
  assert invocation.final_response.parts[0].text == 'Probe complete.'
  responses = [
      part.function_response
      for content in model.requests[-1].contents
      for part in content.parts
      if part.function_response is not None
  ]
  assert len(responses) == 1 and responses[0].name == 'probe'
  assert responses[0].response['isError'] is False, 'MCP tool reported an error'
  assert responses[0].response['content'][0]['text'] == 'native-result:checked'
  assert (
      len(processes.spy_return_list) == 1
  ), 'MCP server spawn was not observed'
  process = processes.spy_return_list[0]
  assert process.returncode is not None, 'MCP server did not exit'
  assert not toolset._mcp_session_manager._sessions, 'MCP session was retained'
  assert not toolset._mcp_session_manager._session_contexts
  assert not [
      record.getMessage()
      for record in caplog.records
      if (
          record.name.startswith(mcp_toolset.logger.name.rsplit('.', 1)[0])
          or record.name == runners.logger.name
          or record.name.startswith('mcp.client.stdio')
      )
      and record.levelno >= logging.WARNING
  ], 'MCP cleanup reported a warning or error'
  return invocation


@pytest.mark.asyncio
async def test_native_stdio_eval_calls_tool_and_closes(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, _ = native_stdio_toolset
  processes = mocker.spy(stdio, '_create_platform_compatible_process')
  close = mocker.spy(toolset, 'close')
  model = MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='probe', args={'value': 'checked'}
          ),
          types.Part(text='Probe complete.'),
      ]
  )

  _assert_successful_native_inference(
      await _run_native_eval(mocker, model, toolset),
      model,
      calls,
      ready,
      toolset,
      processes,
      caplog,
  )
  close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_unavailable_toolset_cannot_satisfy_native_control(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, server = native_stdio_toolset
  processes = mocker.spy(stdio, '_create_platform_compatible_process')
  server.write_text("raise RuntimeError('unexpected setup failure')\n")
  close = mocker.spy(toolset, 'close')
  model = MockModel.create(responses=['A response without the MCP tool.'])

  results = await _run_native_eval(mocker, model, toolset)

  assert len(results) == 1
  # Toolset loading deliberately logs lost capability and continues inference.
  # A successful model response therefore cannot prove that MCP was exercised.
  assert results[0].status == InferenceStatus.SUCCESS
  assert results[0].error_message is None
  assert model.requests
  assert not ready.exists() and not calls.exists()
  assert any(
      'will run without the tools from toolset McpToolset'
      in record.getMessage()
      and record.levelno == logging.ERROR
      for record in caplog.records
  )
  close.assert_awaited_once_with()
  with pytest.raises(AssertionError, match='MCP server did not start'):
    _assert_successful_native_inference(
        results, model, calls, ready, toolset, processes, caplog
    )


@pytest.mark.asyncio
async def test_native_stdio_tool_error_is_not_success(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, server = native_stdio_toolset
  processes = mocker.spy(stdio, '_create_platform_compatible_process')
  server.write_text(
      server.read_text().replace('tool_error = False', 'tool_error = True')
  )
  close = mocker.spy(toolset, 'close')
  model = MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='probe', args={'value': 'checked'}
          ),
          types.Part(text='Probe complete.'),
      ]
  )

  results = await _run_native_eval(mocker, model, toolset)

  # A tool error can reach the model as ordinary content and still end inference.
  # Matching text alone must not turn that error result into MCP success.
  assert results[0].status == InferenceStatus.SUCCESS
  assert ready.read_text() == 'ready\n'
  assert calls.read_text().splitlines() == ['checked']
  responses = [
      part.function_response
      for content in model.requests[-1].contents
      for part in content.parts
      if part.function_response is not None
  ]
  assert len(responses) == 1 and responses[0].name == 'probe'
  response = responses[0]
  assert response.response['isError'] is True
  assert response.response['content'][0]['text'] == 'native-result:checked'
  close.assert_awaited_once_with()
  with pytest.raises(AssertionError, match='MCP tool reported an error'):
    _assert_successful_native_inference(
        results, model, calls, ready, toolset, processes, caplog
    )


@pytest.mark.asyncio
async def test_native_stdio_model_failure_is_not_success(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, _ = native_stdio_toolset
  processes = mocker.spy(stdio, '_create_platform_compatible_process')
  close = mocker.spy(toolset, 'close')
  model = MockModel.create(
      responses=[], error=RuntimeError('unexpected model failure')
  )

  results = await _run_native_eval(mocker, model, toolset)

  assert ready.read_text() == 'ready\n'
  assert len(results) == 1
  assert results[0].status == InferenceStatus.FAILURE
  assert results[0].inferences is None
  assert results[0].error_message == 'unexpected model failure'
  close.assert_awaited_once_with()
  with pytest.raises(AssertionError, match='unexpected model failure'):
    _assert_successful_native_inference(
        results, model, calls, ready, toolset, processes, caplog
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('release_session', [False, True])
async def test_native_stdio_cleanup_failure_is_not_success(
    mocker, native_stdio_toolset, caplog, release_session
):
  toolset, calls, ready, _ = native_stdio_toolset
  processes = mocker.spy(stdio, '_create_platform_compatible_process')
  real_close = toolset.close
  model = MockModel.create(
      responses=[
          types.Part.from_function_call(
              name='probe', args={'value': 'checked'}
          ),
          types.Part(text='Probe complete.'),
      ]
  )

  async def fail_close():
    if release_session:
      await real_close()
    raise RuntimeError('unexpected cleanup failure')

  close = mocker.patch.object(toolset, 'close', side_effect=fail_close)
  try:
    results = await _run_native_eval(mocker, model, toolset)
    assert results[0].status == InferenceStatus.SUCCESS
    assert calls.read_text().splitlines() == ['checked']
    close.assert_awaited_once_with()
    assert any(
        record.name == runners.logger.name
        and record.levelno == logging.ERROR
        and 'unexpected cleanup failure' in record.getMessage()
        for record in caplog.records
    )
    expected = (
        'MCP cleanup reported a warning or error'
        if release_session
        else 'MCP server did not exit'
    )
    with pytest.raises(AssertionError, match=expected):
      _assert_successful_native_inference(
          results, model, calls, ready, toolset, processes, caplog
      )
  finally:
    await real_close()
  assert processes.spy_return_list[0].returncode is not None
  assert not toolset._mcp_session_manager._sessions
