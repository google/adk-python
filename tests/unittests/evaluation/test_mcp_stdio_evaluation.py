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
import pytest

from tests.unittests.testing_utils import MockModel


@pytest.fixture
def native_stdio_toolset(tmp_path):
  calls = tmp_path / 'calls.txt'
  ready = tmp_path / 'ready.txt'
  server = tmp_path / 'server.py'
  server.write_text(textwrap.dedent(f"""\
      from pathlib import Path
      from mcp.server import MCPServer
      from mcp.types import CallToolResult, TextContent

      app = MCPServer('finite-evaluation-test')

      @app.tool()
      def probe(value: str):
          with Path({str(calls)!r}).open('a') as output:
              output.write(value + '\\n')
          return CallToolResult(
              content=[TextContent(type='text', text='native-result:' + value)],
              is_error=False,
          )

      Path({str(ready)!r}).write_text('ready\\n')
      app.run(transport='stdio')
      """))
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(
              command=sys.executable, args=[str(server)]
          ),
          timeout=5,
      ),
      tool_filter=['probe'],
  )
  return toolset, calls, ready, server


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


def _assert_successful_native_inference(results, model, calls, ready):
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
  return invocation


@pytest.mark.asyncio
async def test_native_stdio_eval_calls_tool_and_closes(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, _ = native_stdio_toolset
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
      await _run_native_eval(mocker, model, toolset), model, calls, ready
  )
  close.assert_awaited_once_with()
  assert not [
      record.getMessage()
      for record in caplog.records
      if record.name.startswith(mcp_toolset.logger.name.rsplit('.', 1)[0])
      and record.levelno >= logging.WARNING
  ]


@pytest.mark.asyncio
async def test_unavailable_toolset_cannot_satisfy_native_control(
    mocker, native_stdio_toolset, caplog
):
  toolset, calls, ready, server = native_stdio_toolset
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
    _assert_successful_native_inference(results, model, calls, ready)


@pytest.mark.asyncio
async def test_native_stdio_tool_error_is_not_success(
    mocker, native_stdio_toolset
):
  toolset, calls, ready, server = native_stdio_toolset
  server.write_text(
      server.read_text().replace('is_error=False', 'is_error=True')
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
    _assert_successful_native_inference(results, model, calls, ready)


@pytest.mark.asyncio
async def test_native_stdio_model_failure_is_not_success(
    mocker, native_stdio_toolset
):
  toolset, calls, ready, _ = native_stdio_toolset
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
    _assert_successful_native_inference(results, model, calls, ready)
