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
from google.adk.telemetry.tracing import ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS
from google.adk.telemetry.tracing import trace_agent_invocation
from google.adk.telemetry.tracing import trace_call_llm
from google.adk.telemetry.tracing import trace_generate_content_result
from google.adk.telemetry.tracing import trace_merged_tool_calls
from google.adk.telemetry.tracing import trace_send_data
from google.adk.telemetry.tracing import trace_tool_call
from google.adk.telemetry.tracing import use_generate_content_span
from google.adk.tools.base_tool import BaseTool
from google.genai import types
from opentelemetry._logs import LogRecord
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_REQUEST_MODEL
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
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
async def test_trace_call_llm_with_no_usage_metadata(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm handles usage metadata with None token counts."""
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
      usage_metadata=types.GenerateContentResponseUsageMetadata(),
  )
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  expected_calls = [
      mock.call('gen_ai.system', 'gcp.vertex.agent'),
      mock.call('gen_ai.request.top_p', 0.95),
      mock.call('gen_ai.request.max_tokens', 1024),
      mock.call('gcp.vertex.agent.llm_response', mock.ANY),
      mock.call('gen_ai.response.finish_reasons', ['stop']),
  ]
  assert mock_span_fixture.set_attribute.call_count == 10
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

  # Verify binary values are properly serialized as base64
  llm_request_json_str = None
  for call_obj in mock_span_fixture.set_attribute.call_args_list:
    arg_name, arg_value = call_obj.args
    if arg_name == 'gcp.vertex.agent.llm_request':
      llm_request_json_str = arg_value
      break

  assert llm_request_json_str is not None

  # Verify bytes are base64 encoded (b'test_data' -> 'dGVzdF9kYXRh')
  assert 'dGVzdF9kYXRh' in llm_request_json_str

  # Verify no serialization failures
  assert '<not serializable>' not in llm_request_json_str


@pytest.mark.asyncio
async def test_trace_call_llm_with_thought_signature(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm handles thought_signature bytes correctly.

  This test verifies that thought_signature bytes from Gemini 3.0 models
  are properly serialized as base64 in telemetry traces.
  """
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)

  # multi-turn conversation where the model's response contains
  # thought_signature bytes
  thought_signature_bytes = b'thought_signature'
  llm_request = LlmRequest(
      model='gemini-3-pro-preview',
      contents=[
          types.Content(
              role='user',
              parts=[types.Part(text='Hello')],
          ),
          types.Content(
              role='model',
              parts=[
                  types.Part(
                      thought=True,
                      thought_signature=thought_signature_bytes,
                  )
              ],
          ),
          types.Content(
              role='user',
              parts=[types.Part(text='Follow up question')],
          ),
      ],
      config=types.GenerateContentConfig(),
  )
  llm_response = LlmResponse(turn_complete=True)

  # should not raise TypeError for bytes serialization
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  llm_request_json_str = None
  for call_obj in mock_span_fixture.set_attribute.call_args_list:
    arg_name, arg_value = call_obj.args
    if arg_name == 'gcp.vertex.agent.llm_request':
      llm_request_json_str = arg_value
      break

  assert (
      llm_request_json_str is not None
  ), "Attribute 'gcp.vertex.agent.llm_request' was not set on the span."

  # no serialization failures
  assert '<not serializable>' not in llm_request_json_str
  # llm request is valid JSON
  parsed = json.loads(llm_request_json_str)
  assert parsed['model'] == 'gemini-3-pro-preview'
  assert len(parsed['contents']) == 3


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


@pytest.mark.asyncio
async def test_call_llm_disabling_request_response_content(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm sets placeholders when capture is disabled."""
  # Arrange
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'false')
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
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
  )

  # Act
  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  # Assert
  assert (
      'gcp.vertex.agent.llm_request',
      '{}',
  ) in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )
  assert (
      'gcp.vertex.agent.llm_response',
      '{}',
  ) in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )


def test_trace_tool_call_disabling_request_response_content(
    monkeypatch,
    mock_span_fixture,
    mock_tool_fixture,
    mock_event_fixture,
):
  """Test trace_tool_call sets placeholders when capture is disabled."""
  # Arrange
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'false')
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
  assert (
      'gcp.vertex.agent.tool_call_args',
      '{}',
  ) in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )
  assert (
      'gcp.vertex.agent.tool_response',
      '{}',
  ) in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )


def test_trace_merged_tool_disabling_request_response_content(
    monkeypatch,
    mock_span_fixture,
    mock_event_fixture,
):
  """Test trace_merged_tool_calls sets placeholders when capture is disabled."""
  # Arrange
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'false')
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_response_event_id = 'merged_evt_id_001'
  custom_event_json_output = (
      '{"custom_event_payload": true, "details": "merged_details"}'
  )
  mock_event_fixture.model_dumps_json.return_value = custom_event_json_output

  # Act
  trace_merged_tool_calls(
      response_event_id=test_response_event_id,
      function_response_event=mock_event_fixture,
  )

  # Assert
  assert (
      'gcp.vertex.agent.tool_response',
      '{}',
  ) in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )


@pytest.mark.asyncio
async def test_trace_send_data_disabling_request_response_content(
    monkeypatch, mock_span_fixture
):
  """Test trace_send_data sets placeholders when capture is disabled."""
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'false')
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)

  trace_send_data(
      invocation_context=invocation_context,
      event_id='test_event_id',
      data=[
          types.Content(
              role='user',
              parts=[types.Part(text='hi')],
          )
      ],
  )

  assert ('gcp.vertex.agent.data', '{}') in (
      call_obj.args
      for call_obj in mock_span_fixture.set_attribute.call_args_list
  )


@pytest.mark.asyncio
@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch('google.adk.telemetry.tracing.tracer')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='test_system',
)
@pytest.mark.parametrize('capture_content', [True, False])
async def test_generate_content_span(
    mock_guess_system_name,
    mock_tracer,
    mock_otel_logger,
    monkeypatch,
    capture_content,
):
  """Test native generate_content span creation with attributes and logs."""
  # Arrange
  monkeypatch.setenv(
      'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT',
      str(capture_content).lower(),
  )
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: False,
  )

  agent = LlmAgent(name='test_agent', model='not-a-gemini-model')
  invocation_context = await _create_invocation_context(agent)

  system_instruction = types.Content(
      parts=[types.Part.from_text(text='You are a helpful assistant.')],
  )
  user_content1 = types.Content(role='user', parts=[types.Part(text='Hello')])
  user_content2 = types.Content(role='user', parts=[types.Part(text='World')])

  model_content = types.Content(
      role='model', parts=[types.Part(text='Response')]
  )

  llm_request = LlmRequest(
      model='some-model',
      contents=[user_content1, user_content2],
      config=types.GenerateContentConfig(system_instruction=system_instruction),
  )
  llm_response = LlmResponse(
      content=model_content,
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=20,
      ),
  )

  model_response_event = mock.MagicMock()
  model_response_event.id = 'event-123'

  mock_span = (
      mock_tracer.start_as_current_span.return_value.__enter__.return_value
  )

  # Act
  with use_generate_content_span(
      llm_request, invocation_context, model_response_event
  ) as span:
    assert span is mock_span

    trace_generate_content_result(span, llm_response)

  # Assert Span
  mock_tracer.start_as_current_span.assert_called_once_with(
      'generate_content some-model'
  )

  mock_span.set_attribute.assert_any_call(GEN_AI_SYSTEM, 'test_system')
  mock_span.set_attribute.assert_any_call(
      GEN_AI_OPERATION_NAME, 'generate_content'
  )
  mock_span.set_attribute.assert_any_call(GEN_AI_REQUEST_MODEL, 'some-model')
  mock_span.set_attribute.assert_any_call(
      GEN_AI_RESPONSE_FINISH_REASONS, ['stop']
  )
  mock_span.set_attribute.assert_any_call(GEN_AI_USAGE_INPUT_TOKENS, 10)
  mock_span.set_attribute.assert_any_call(GEN_AI_USAGE_OUTPUT_TOKENS, 20)

  mock_span.set_attributes.assert_called_once_with({
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      USER_ID: invocation_context.session.user_id,
      'gcp.vertex.agent.event_id': 'event-123',
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  })

  # Assert Logs
  assert mock_otel_logger.emit.call_count == 4

  expected_system_body = {
      'content': (
          system_instruction.model_dump() if capture_content else '<elided>'
      )
  }
  expected_user1_body = {
      'content': user_content1.model_dump() if capture_content else '<elided>'
  }
  expected_user2_body = {
      'content': user_content2.model_dump() if capture_content else '<elided>'
  }
  expected_choice_body = {
      'content': model_content.model_dump() if capture_content else '<elided>',
      'index': 0,
      'finish_reason': 'STOP',
  }

  log_records: list[LogRecord] = [
      call.args[0] for call in mock_otel_logger.emit.call_args_list
  ]

  system_log = next(
      (lr for lr in log_records if lr.event_name == 'gen_ai.system.message'),
      None,
  )
  assert system_log is not None
  assert system_log.body == expected_system_body
  assert system_log.attributes == {GEN_AI_SYSTEM: 'test_system'}

  user_logs = [
      lr for lr in log_records if lr.event_name == 'gen_ai.user.message'
  ]
  assert len(user_logs) == 2
  assert expected_user1_body == user_logs[0].body
  assert expected_user2_body == user_logs[1].body
  for log in user_logs:
    assert log.attributes == {GEN_AI_SYSTEM: 'test_system'}

  choice_log = next(
      (lr for lr in log_records if lr.event_name == 'gen_ai.choice'),
      None,
  )
  assert choice_log is not None
  assert choice_log.body == expected_choice_body
  assert choice_log.attributes == {GEN_AI_SYSTEM: 'test_system'}
