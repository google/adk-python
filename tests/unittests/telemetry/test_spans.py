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

import json
from typing import Any
from typing import Dict
from typing import Optional
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry.tracing import trace_agent_invocation
from google.adk.telemetry.tracing import trace_call_llm
from google.adk.telemetry.tracing import trace_merged_tool_calls
from google.adk.telemetry.tracing import trace_tool_call
from google.adk.tools.base_tool import BaseTool
from google.genai import types
import pytest


class Event:

  def __init__(self, event_id: str, event_content: Any):
    self.id = event_id
    self.content = event_content

  def model_dumps_json(self, exclude_none: bool = False) -> str:
    # This is just a stub for the spec. The mock will provide behavior.
    return ''


@pytest.fixture
def mock_span_fixture():
  return mock.MagicMock()


@pytest.fixture
def mock_tool_fixture():
  tool = mock.Mock(spec=BaseTool)
  tool.name = 'sample_tool'
  tool.description = 'A sample tool for testing.'
  return tool


@pytest.fixture
def mock_event_fixture():
  event_mock = mock.create_autospec(Event, instance=True)
  event_mock.model_dumps_json.return_value = (
      '{"default_event_key": "default_event_value"}'
  )
  return event_mock


async def _create_invocation_context(
    agent: LlmAgent, state: Optional[dict[str, Any]] = None
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user', state=state
  )
  invocation_context = InvocationContext(
      invocation_id='test_id',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  return invocation_context


@pytest.mark.asyncio
async def test_trace_agent_invocation(mock_span_fixture):
  """Test trace_agent_invocation sets span attributes correctly."""
  agent = LlmAgent(name='test_llm_agent', model='gemini-pro')
  agent.description = 'Test agent description'
  invocation_context = await _create_invocation_context(agent)

  trace_agent_invocation(mock_span_fixture, agent, invocation_context)

  expected_calls = [
      mock.call('gen_ai.operation.name', 'invoke_agent'),
      mock.call('gen_ai.agent.description', agent.description),
      mock.call('gen_ai.agent.name', agent.name),
      mock.call(
          'gen_ai.conversation.id',
          invocation_context.session.id,
      ),
  ]
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )
  assert mock_span_fixture.set_attribute.call_count == len(expected_calls)


@pytest.mark.asyncio
async def test_trace_call_llm(monkeypatch, mock_span_fixture):
  """Test trace_call_llm sets all telemetry attributes correctly with normal content."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
      model='gemini-pro',
      contents=[
          types.Content(
              role='user',
              parts=[types.Part(text='Hello, how are you?')],
          ),
      ],
      config=types.GenerateContentConfig(
          top_p=0.95,
          max_output_tokens=1024,
      ),
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=100,
          prompt_token_count=50,
          candidates_token_count=50,
      ),
  )
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  expected_calls = [
      mock.call('gen_ai.system', 'gcp.vertex.agent'),
      mock.call('gen_ai.request.top_p', 0.95),
      mock.call('gen_ai.request.max_tokens', 1024),
      mock.call('gcp.vertex.agent.llm_response', mock.ANY),
      mock.call('gen_ai.usage.input_tokens', 50),
      mock.call('gen_ai.usage.output_tokens', 50),
      mock.call('gen_ai.response.finish_reasons', ['stop']),
  ]
  assert mock_span_fixture.set_attribute.call_count == 12
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )


@pytest.mark.asyncio
async def test_trace_call_llm_with_binary_content(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm handles binary content serialization correctly."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
      model='gemini-pro',
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part.from_function_response(
                      name='test_function_1',
                      response={
                          'result': b'test_data',
                      },
                  ),
              ],
          ),
          types.Content(
              role='user',
              parts=[
                  types.Part.from_function_response(
                      name='test_function_2',
                      response={
                          'result': types.Part.from_bytes(
                              data=b'test_data',
                              mime_type='application/octet-stream',
                          ),
                      },
                  ),
              ],
          ),
      ],
      config=types.GenerateContentConfig(),
  )
  llm_response = LlmResponse(turn_complete=True)
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  # Verify basic telemetry attributes are set
  expected_calls = [
      mock.call('gen_ai.system', 'gcp.vertex.agent'),
  ]
  assert mock_span_fixture.set_attribute.call_count == 7
  mock_span_fixture.set_attribute.assert_has_calls(expected_calls)

  # Verify binary content is replaced with '<not serializable>' in JSON
  llm_request_json_str = None
  for call_obj in mock_span_fixture.set_attribute.call_args_list:
    if call_obj.args[0] == 'gcp.vertex.agent.llm_request':
      llm_request_json_str = call_obj.args[1]
      break

  assert (
      llm_request_json_str is not None
  ), "Attribute 'gcp.vertex.agent.llm_request' was not set on the span."

  assert llm_request_json_str.count('<not serializable>') == 2


def test_trace_tool_call_with_scalar_response(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, Any] = {'param_a': 'value_a', 'param_b': 100}
  test_tool_call_id: str = 'tool_call_id_001'
  test_event_id: str = 'event_id_001'
  scalar_function_response: Any = 'Scalar result'

  expected_processed_response = {'result': scalar_function_response}

  mock_event_fixture.id = test_event_id
  mock_event_fixture.content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=test_tool_call_id,
                  name='test_function_1',
                  response={'result': scalar_function_response},
              )
          ),
      ],
  )

  # Act
  trace_tool_call(
      tool=mock_tool_fixture,
      args=test_args,
      function_response_event=mock_event_fixture,
  )

  # Assert
  expected_calls = [
      mock.call('gen_ai.operation.name', 'execute_tool'),
      mock.call('gen_ai.tool.name', mock_tool_fixture.name),
      mock.call('gen_ai.tool.description', mock_tool_fixture.description),
      mock.call('gen_ai.tool.type', 'BaseTool'),
      mock.call('gen_ai.tool.call.id', test_tool_call_id),
      mock.call('gcp.vertex.agent.tool_call_args', json.dumps(test_args)),
      mock.call('gcp.vertex.agent.event_id', test_event_id),
      mock.call(
          'gcp.vertex.agent.tool_response',
          json.dumps(expected_processed_response),
      ),
      mock.call('gcp.vertex.agent.llm_request', '{}'),
      mock.call('gcp.vertex.agent.llm_response', '{}'),
  ]

  assert mock_span_fixture.set_attribute.call_count == len(expected_calls)
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )


def test_trace_tool_call_with_dict_response(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  # Arrange
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, Any] = {'query': 'details', 'id_list': [1, 2, 3]}
  test_tool_call_id: str = 'tool_call_id_002'
  test_event_id: str = 'event_id_dict_002'
  dict_function_response: Dict[str, Any] = {
      'data': 'structured_data',
      'count': 5,
  }

  mock_event_fixture.id = test_event_id
  mock_event_fixture.content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id=test_tool_call_id,
                  name='test_function_1',
                  response=dict_function_response,
              )
          ),
      ],
  )

  # Act
  trace_tool_call(
      tool=mock_tool_fixture,
      args=test_args,
      function_response_event=mock_event_fixture,
  )

  # Assert
  expected_calls = [
      mock.call('gen_ai.operation.name', 'execute_tool'),
      mock.call('gen_ai.tool.name', mock_tool_fixture.name),
      mock.call('gen_ai.tool.description', mock_tool_fixture.description),
      mock.call('gen_ai.tool.type', 'BaseTool'),
      mock.call('gen_ai.tool.call.id', test_tool_call_id),
      mock.call('gcp.vertex.agent.tool_call_args', json.dumps(test_args)),
      mock.call('gcp.vertex.agent.event_id', test_event_id),
      mock.call(
          'gcp.vertex.agent.tool_response', json.dumps(dict_function_response)
      ),
      mock.call('gcp.vertex.agent.llm_request', '{}'),
      mock.call('gcp.vertex.agent.llm_response', '{}'),
  ]

  assert mock_span_fixture.set_attribute.call_count == len(expected_calls)
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )


def test_trace_merged_tool_calls_sets_correct_attributes(
    monkeypatch, mock_span_fixture, mock_event_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_response_event_id = 'merged_evt_id_001'
  custom_event_json_output = (
      '{"custom_event_payload": true, "details": "merged_details"}'
  )
  mock_event_fixture.model_dumps_json.return_value = custom_event_json_output

  trace_merged_tool_calls(
      response_event_id=test_response_event_id,
      function_response_event=mock_event_fixture,
  )

  expected_calls = [
      mock.call('gen_ai.operation.name', 'execute_tool'),
      mock.call('gen_ai.tool.name', '(merged tools)'),
      mock.call('gen_ai.tool.description', '(merged tools)'),
      mock.call('gen_ai.tool.call.id', test_response_event_id),
      mock.call('gcp.vertex.agent.tool_call_args', 'N/A'),
      mock.call('gcp.vertex.agent.event_id', test_response_event_id),
      mock.call('gcp.vertex.agent.tool_response', custom_event_json_output),
      mock.call('gcp.vertex.agent.llm_request', '{}'),
      mock.call('gcp.vertex.agent.llm_response', '{}'),
  ]

  assert mock_span_fixture.set_attribute.call_count == len(expected_calls)
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )
  mock_event_fixture.model_dumps_json.assert_called_once_with(exclude_none=True)
