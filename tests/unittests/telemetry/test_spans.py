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
from google.adk.agents.run_config import RunConfig
from google.adk.errors.tool_execution_error import ToolErrorType
from google.adk.errors.tool_execution_error import ToolExecutionError
from google.adk.events.event import Event
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry import TelemetryConfig
from google.adk.telemetry import tracing
from google.adk.telemetry._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT
from google.adk.telemetry._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT
from google.adk.telemetry._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT
from google.adk.telemetry._adk_attributes import ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED
from google.adk.telemetry._experimental_semconv import _safe_json_serialize_no_whitespaces
from google.adk.telemetry._stable_semconv import USER_CONTENT_ELIDED
from google.adk.telemetry.context import ADK_EXPERIMENTAL_TELEMETRY
from google.adk.telemetry.tracing import _ADK_CAPTURE_MCP_HTTP_BODIES
from google.adk.telemetry.tracing import _HTTP_REQUEST_BODY_CONTENT
from google.adk.telemetry.tracing import _HTTP_RESPONSE_BODY_CONTENT
from google.adk.telemetry.tracing import _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST
from google.adk.telemetry.tracing import _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE
from google.adk.telemetry.tracing import _should_report_mcp_http_exchanges
from google.adk.telemetry.tracing import _trace_mcp_http_exchange
from google.adk.telemetry.tracing import _use_extra_generate_content_attributes
from google.adk.telemetry.tracing import ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS
from google.adk.telemetry.tracing import GCP_MCP_SERVER_DESTINATION_ID
from google.adk.telemetry.tracing import GenerateContentSpan
from google.adk.telemetry.tracing import resolve_error_type
from google.adk.telemetry.tracing import safe_json_serialize
from google.adk.telemetry.tracing import trace_agent_invocation
from google.adk.telemetry.tracing import trace_call_llm
from google.adk.telemetry.tracing import trace_generate_content_result
from google.adk.telemetry.tracing import trace_inference_result
from google.adk.telemetry.tracing import trace_merged_tool_calls
from google.adk.telemetry.tracing import trace_send_data
from google.adk.telemetry.tracing import trace_tool_call
from google.adk.telemetry.tracing import use_generate_content_span
from google.adk.telemetry.tracing import use_inference_span
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import errors as genai_errors
from google.genai import types
from mcp import ClientSession as McpClientSession
from mcp import ListToolsResult as McpListToolsResult
from mcp import Tool as McpTool
from opentelemetry._logs import LogRecord
from opentelemetry._logs import SeverityNumber
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_AGENT_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_INPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OUTPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_REQUEST_MODEL
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM_INSTRUCTIONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_PROTOCOL_VERSION
from opentelemetry.semconv._incubating.attributes.mcp_attributes import MCP_SESSION_ID
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.semconv.attributes.http_attributes import HTTP_REQUEST_METHOD
from opentelemetry.semconv.attributes.http_attributes import HTTP_RESPONSE_STATUS_CODE
from opentelemetry.semconv.attributes.server_attributes import SERVER_ADDRESS
from opentelemetry.semconv.attributes.server_attributes import SERVER_PORT
from opentelemetry.semconv.attributes.url_attributes import URL_FULL
from opentelemetry.trace import StatusCode
from pydantic import BaseModel
import pytest

try:
  from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_DEFINITIONS
except ImportError:
  GEN_AI_TOOL_DEFINITIONS = 'gen_ai.tool.definitions'


# Create a minimal concrete BaseTool for testing
class SimpleTestTool(BaseTool):

  async def run_async(
      self, *, args: dict[str, object], tool_context: ToolContext
  ) -> object:
    return 'SimpleTestTool result'


@pytest.fixture
def mock_span_fixture():
  return mock.MagicMock()


@pytest.fixture
def mock_tool_fixture():
  return SimpleTestTool(
      name='sample_tool',
      description='A sample tool for testing.',
  )


@pytest.fixture
def mock_event_fixture():
  return Event(id='test_event_id', author='test_agent')


async def _create_invocation_context(
    agent: LlmAgent,
    state: Optional[dict[str, object]] = None,
    run_config: Optional[RunConfig] = None,
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
      run_config=run_config or RunConfig(),
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
          thinking_config=types.ThinkingConfig(thinking_budget=10),
      ),
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          total_token_count=100,
          prompt_token_count=50,
          candidates_token_count=50,
          thoughts_token_count=10,
      ),
  )
  # We dynamically assign system_instruction_tokens rather than passing it
  # to the GenerateContentResponseUsageMetadata constructor to ensure backward
  # compatibility with older versions of the google-genai SDK that do not have
  # this property defined in their Pydantic models.
  try:
    llm_response.usage_metadata.system_instruction_tokens = 5
  except Exception:
    pass

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  expected_calls = [
      mock.call('gen_ai.system', 'gcp.vertex.agent'),
      mock.call('gen_ai.request.top_p', 0.95),
      mock.call('gen_ai.request.max_tokens', 1024),
      mock.call('gcp.vertex.agent.llm_response', mock.ANY),
      mock.call('gen_ai.usage.experimental.reasoning_tokens_limit', 10),
      mock.call('gen_ai.response.finish_reasons', ['stop']),
  ]

  expected_usage_attrs = {
      'gen_ai.usage.input_tokens': 50,
      'gen_ai.usage.output_tokens': 60,
      'gen_ai.usage.reasoning.output_tokens': 10,
  }
  if hasattr(llm_response.usage_metadata, 'system_instruction_tokens'):
    expected_usage_attrs[
        'gen_ai.usage.experimental.system_instruction_tokens'
    ] = 5

  assert mock_span_fixture.set_attribute.call_count == len(expected_calls) + 5
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )
  mock_span_fixture.set_attributes.assert_called_once_with(expected_usage_attrs)


@pytest.mark.asyncio
async def test_trace_call_llm_skips_non_recording_span(monkeypatch):
  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(model='gemini-pro')
  llm_response = LlmResponse(turn_complete=True)
  span = mock.MagicMock()
  span.is_recording.return_value = False
  get_telemetry_config = mock.Mock()
  serialize_request = mock.Mock(return_value='{}')
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._telemetry_config_from_invocation_context',
      get_telemetry_config,
  )
  monkeypatch.setattr(
      'google.adk.telemetry.tracing.safe_json_serialize', serialize_request
  )

  trace_call_llm(
      invocation_context,
      'test_event_id',
      llm_request,
      llm_response,
      span=span,
  )

  get_telemetry_config.assert_not_called()
  serialize_request.assert_not_called()
  span.set_attribute.assert_not_called()
  span.set_attributes.assert_not_called()


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


_EXPERIMENTAL_TELEMETRY_ON = RunConfig(
    telemetry=TelemetryConfig(adk_experimental_telemetry_opt_in=True)
)


@pytest.mark.asyncio
async def test_trace_call_llm_with_active_context_cache(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm records the state of an active context cache."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(
      agent, run_config=_EXPERIMENTAL_TELEMETRY_ON
  )
  llm_request = LlmRequest(
      model='gemini-pro',
      contents=[
          types.Content(role='user', parts=[types.Part(text='Hello')]),
      ],
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      cache_metadata=CacheMetadata(
          cache_name='projects/p/locations/l/cachedContents/c',
          expire_time=1893456000.0,
          fingerprint='fp-123',
          invocations_used=3,
          contents_count=5,
      ),
  )

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  mock_span_fixture.set_attributes.assert_any_call({
      ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT: True,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT: 'fp-123',
      ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT: 5,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED: 3,
  })


@pytest.mark.asyncio
async def test_trace_call_llm_with_fingerprint_only_context_cache(
    monkeypatch, mock_span_fixture
):
  """Test trace_call_llm records a miss when no cache is active."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(
      agent, run_config=_EXPERIMENTAL_TELEMETRY_ON
  )
  llm_request = LlmRequest(
      model='gemini-pro',
      contents=[
          types.Content(role='user', parts=[types.Part(text='Hello')]),
      ],
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      cache_metadata=CacheMetadata(fingerprint='fp-123', contents_count=8),
  )

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  # The exact dict also pins that invocations_used is left out when unset.
  mock_span_fixture.set_attributes.assert_any_call({
      ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT: False,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT: 'fp-123',
      ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT: 8,
  })


@pytest.mark.asyncio
async def test_trace_call_llm_omits_context_cache_without_opt_in(
    monkeypatch, mock_span_fixture
):
  """Test context cache attributes stay off unless experimental is opted in."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
      model='gemini-pro',
      contents=[
          types.Content(role='user', parts=[types.Part(text='Hello')]),
      ],
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      cache_metadata=CacheMetadata(
          cache_name='projects/p/locations/l/cachedContents/c',
          expire_time=1893456000.0,
          fingerprint='fp-123',
          invocations_used=3,
          contents_count=5,
      ),
  )

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  # The span is still traced; only the experimental attributes are withheld.
  mock_span_fixture.set_attribute.assert_any_call(
      'gen_ai.system', 'gcp.vertex.agent'
  )
  set_keys = [
      call.args[0] for call in mock_span_fixture.set_attribute.call_args_list
  ]
  for call in mock_span_fixture.set_attributes.call_args_list:
    set_keys.extend(call.args[0])
  assert not [key for key in set_keys if key.startswith('adk.experimental.')]


@pytest.mark.asyncio
async def test_trace_inference_result_with_context_cache(mock_span_fixture):
  """Test the generate_content span also carries context cache state."""
  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(
      agent, run_config=_EXPERIMENTAL_TELEMETRY_ON
  )
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      cache_metadata=CacheMetadata(
          cache_name='projects/p/locations/l/cachedContents/c',
          expire_time=1893456000.0,
          fingerprint='fp-123',
          invocations_used=3,
          contents_count=5,
      ),
  )

  trace_inference_result(invocation_context, mock_span_fixture, llm_response)

  mock_span_fixture.set_attributes.assert_any_call({
      ADK_EXPERIMENTAL_CONTEXT_CACHE_HIT: True,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_FINGERPRINT: 'fp-123',
      ADK_EXPERIMENTAL_CONTEXT_CACHE_CONTENTS_COUNT: 5,
      ADK_EXPERIMENTAL_CONTEXT_CACHE_INVOCATIONS_USED: 3,
  })


@pytest.mark.asyncio
async def test_trace_inference_result_omits_context_cache_without_opt_in(
    mock_span_fixture,
):
  """Test the generate_content span withholds cache state without opt-in."""
  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_response = LlmResponse(
      turn_complete=True,
      finish_reason=types.FinishReason.STOP,
      cache_metadata=CacheMetadata(fingerprint='fp-123', contents_count=5),
  )

  trace_inference_result(invocation_context, mock_span_fixture, llm_response)

  set_keys = []
  for call in mock_span_fixture.set_attributes.call_args_list:
    set_keys.extend(call.args[0])
  assert not [key for key in set_keys if key.startswith('adk.experimental.')]


@pytest.mark.asyncio
async def test_trace_inference_result_allows_a_response_without_cache_metadata(
    mock_span_fixture,
):
  """Test a caller's own response object without cache_metadata is accepted."""

  class ResponseWithoutCacheMetadata:
    """Stands in for a caller's response type outside adk."""

    partial = False
    finish_reason = types.FinishReason.STOP
    usage_metadata = None
    content = None
    model_version = 'gemini-pro'

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)

  trace_inference_result(
      invocation_context, mock_span_fixture, ResponseWithoutCacheMetadata()
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


def test_trace_tool_call_with_destination_id(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  """Test trace_tool_call sets destination ID span attribute when present."""
  # Arrange
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_dest_id = 'urn:mcp:googleapis.com:project:1234:location:global:bigquery'
  tool = mock_tool_fixture
  tool.custom_metadata = {
      GCP_MCP_SERVER_DESTINATION_ID: test_dest_id,
      'other_meta': 'value',
  }

  # Act
  trace_tool_call(
      tool=tool,
      args={},
      function_response_event=mock_event_fixture,
  )

  # Assert
  mock_span_fixture.set_attribute.assert_any_call(
      GCP_MCP_SERVER_DESTINATION_ID, test_dest_id
  )


def test_trace_tool_call_without_destination_id(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  """Test trace_tool_call does not set destination ID span attribute when not present."""
  # Arrange
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  tool = mock_tool_fixture
  tool.custom_metadata = {
      'other_meta': 'value',
  }

  # Act
  trace_tool_call(
      tool=tool,
      args={},
      function_response_event=mock_event_fixture,
  )

  # Assert
  called_with_dest_id = any(
      call_args[0][0] == GCP_MCP_SERVER_DESTINATION_ID
      for call_args in mock_span_fixture.set_attribute.call_args_list
  )
  assert not called_with_dest_id


def test_trace_tool_call_with_empty_custom_metadata(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  """Test trace_tool_call handles empty custom_metadata gracefully."""
  # Arrange
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  tool = mock_tool_fixture
  tool.custom_metadata = {}

  # Act
  trace_tool_call(
      tool=tool,
      args={},
      function_response_event=mock_event_fixture,
  )

  # Assert
  called_with_dest_id = any(
      call_args[0][0] == GCP_MCP_SERVER_DESTINATION_ID
      for call_args in mock_span_fixture.set_attribute.call_args_list
  )
  assert not called_with_dest_id


def test_trace_tool_call_with_scalar_response(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, object] = {'param_a': 'value_a', 'param_b': 100}
  test_tool_call_id: str = 'tool_call_id_001'
  test_event_id: str = 'event_id_001'
  scalar_function_response: object = 'Scalar result'

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
      mock.call('gen_ai.tool.type', 'SimpleTestTool'),
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

  test_args: Dict[str, object] = {'query': 'details', 'id_list': [1, 2, 3]}
  test_tool_call_id: str = 'tool_call_id_002'
  test_event_id: str = 'event_id_dict_002'
  dict_function_response: Dict[str, object] = {
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
      mock.call('gen_ai.tool.type', 'SimpleTestTool'),
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


def _trace_mcp_exchange(**overrides):
  """Reports a plausible MCP exchange, with `overrides` applied."""
  _trace_mcp_http_exchange(**{
      'method': 'POST',
      'url': 'https://mcp.example.com/messages',
      'server_address': 'mcp.example.com',
      'server_port': None,
      'status_code': 200,
      'mcp_session_id': None,
      'mcp_protocol_version': None,
      'request_headers': {'x-req': 'val'},
      'request_body': '{"method": "tools/call"}',
      'response_headers': {'content-type': 'application/json'},
      'response_body': '{"result": {}}',
      **overrides,
  })


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_emits_debug_log_record(
    mock_otel_logger, monkeypatch
):
  """Test that an exchange lands as one record with semconv attributes."""
  monkeypatch.setenv(_ADK_CAPTURE_MCP_HTTP_BODIES, 'true')

  _trace_mcp_exchange(
      url='https://mcp.example.com:8443/messages?sessionId=REDACTED',
      server_port=8443,
      mcp_session_id='sess-1',
      mcp_protocol_version='2025-06-18',
  )

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert (
      log_record.event_name == 'adk.experimental.mcp.http.client.response.end'
  )
  assert log_record.severity_number == SeverityNumber.DEBUG
  assert log_record.attributes == {
      HTTP_REQUEST_METHOD: 'POST',
      URL_FULL: 'https://mcp.example.com:8443/messages?sessionId=REDACTED',
      SERVER_ADDRESS: 'mcp.example.com',
      SERVER_PORT: 8443,
      MCP_SESSION_ID: 'sess-1',
      MCP_PROTOCOL_VERSION: '2025-06-18',
      HTTP_RESPONSE_STATUS_CODE: 200,
  }
  assert log_record.body == {
      _HTTP_REQUEST_BODY_CONTENT: '{"method": "tools/call"}',
      _HTTP_RESPONSE_BODY_CONTENT: '{"result": {}}',
  }


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_elides_bodies_by_default(
    mock_otel_logger, monkeypatch
):
  """Bodies carry user content, so recording them has to be asked for."""
  monkeypatch.delenv(_ADK_CAPTURE_MCP_HTTP_BODIES, raising=False)

  _trace_mcp_exchange()

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  # The attributes still describe the exchange; only the payload is elided.
  assert log_record.attributes[HTTP_RESPONSE_STATUS_CODE] == 200
  assert log_record.body == {
      _HTTP_REQUEST_BODY_CONTENT: USER_CONTENT_ELIDED,
      _HTTP_RESPONSE_BODY_CONTENT: USER_CONTENT_ELIDED,
  }
  assert SERVER_PORT not in log_record.attributes
  assert MCP_SESSION_ID not in log_record.attributes
  assert MCP_PROTOCOL_VERSION not in log_record.attributes


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_records_no_header_unasked(
    mock_otel_logger, monkeypatch
):
  """No header is recorded that the OTel env vars do not name."""
  monkeypatch.delenv(
      _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST, raising=False
  )
  monkeypatch.delenv(
      _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE, raising=False
  )

  _trace_mcp_exchange(
      request_headers={'x-req': 'val', 'content-type': 'application/json'}
  )

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert not [key for key in log_record.attributes if '.header.' in key]


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_captures_allowlisted_headers(
    mock_otel_logger, monkeypatch
):
  """The OTel httpx env vars are the whole allowlist, regexes included."""
  monkeypatch.setenv(
      _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST,
      'x-req,x-trace-.*',
  )
  monkeypatch.setenv(
      _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_RESPONSE, 'x-resp'
  )

  _trace_mcp_exchange(
      request_headers={
          'X-Req': 'val',
          'x-trace-id': 'abc',
          'x-other': 'dropped',
          # Allowlisting a credential header still yields only the marker,
          # because the caller redacts before we ever see it.
          'authorization': '<redacted>',
      },
      response_headers={'x-resp': 'val'},
  )

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert log_record.attributes['http.request.header.x-req'] == ['val']
  assert log_record.attributes['http.request.header.x-trace-id'] == ['abc']
  assert log_record.attributes['http.response.header.x-resp'] == ['val']
  assert 'http.request.header.x-other' not in log_record.attributes
  assert 'http.request.header.authorization' not in log_record.attributes


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_ignores_malformed_header_pattern(
    mock_otel_logger, monkeypatch
):
  """A bad regex drops its own entry rather than the whole record."""
  monkeypatch.setenv(
      _OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST, 'x-re[,x-req'
  )

  _trace_mcp_exchange(request_headers={'x-req': 'val'})

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert log_record.attributes['http.request.header.x-req'] == ['val']


@pytest.mark.parametrize(
    'env_value,reported',
    [(None, False), ('false', False), ('true', True), ('1', True)],
)
def test_should_report_mcp_http_exchanges_follows_the_experimental_opt_in(
    env_value, reported, monkeypatch
):
  """The record is experimental, so nothing is reported without the opt-in."""
  if env_value is None:
    monkeypatch.delenv(ADK_EXPERIMENTAL_TELEMETRY, raising=False)
  else:
    monkeypatch.setenv(ADK_EXPERIMENTAL_TELEMETRY, env_value)

  assert _should_report_mcp_http_exchanges() is reported


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_mcp_http_exchange_sets_error_type_on_failure(mock_otel_logger):
  """Test that a 4xx or 5xx status is also recorded as `error.type`."""
  _trace_mcp_exchange(
      server_address=None, status_code=403, response_body='Forbidden'
  )

  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert log_record.attributes[ERROR_TYPE] == '403'
  # An unknown host is omitted rather than recorded as None.
  assert SERVER_ADDRESS not in log_record.attributes


def test_trace_merged_tool_calls_sets_correct_attributes(
    monkeypatch, mock_span_fixture, mock_event_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_response_event_id = 'merged_evt_id_001'
  mock_event_fixture.content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id='tool_call_id_003',
                  name='test_function_1',
                  response={'data': 'merged_details'},
              )
          ),
      ],
  )

  trace_merged_tool_calls(
      response_event_id=test_response_event_id,
      function_response_event=mock_event_fixture,
  )

  expected_event_json = mock_event_fixture.model_dump_json(exclude_none=True)
  expected_calls = [
      mock.call('gen_ai.operation.name', 'execute_tool'),
      mock.call('gen_ai.tool.name', '(merged tools)'),
      mock.call('gen_ai.tool.description', '(merged tools)'),
      mock.call('gen_ai.tool.call.id', test_response_event_id),
      mock.call('gcp.vertex.agent.tool_call_args', 'N/A'),
      mock.call('gcp.vertex.agent.event_id', test_response_event_id),
      mock.call('gcp.vertex.agent.tool_response', expected_event_json),
      mock.call('gcp.vertex.agent.llm_request', '{}'),
      mock.call('gcp.vertex.agent.llm_response', '{}'),
  ]

  assert mock_span_fixture.set_attribute.call_count == len(expected_calls)
  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )
  # The merged response must be the real serialized event, not the
  # "<not serializable>" fallback.
  recorded_response = next(
      call_obj.args[1]
      for call_obj in mock_span_fixture.set_attribute.call_args_list
      if call_obj.args[0] == 'gcp.vertex.agent.tool_response'
  )
  parsed = json.loads(recorded_response)
  assert parsed['id'] == 'test_event_id'
  assert 'merged_details' in recorded_response


def test_trace_tool_call_skips_non_recording_span(
    monkeypatch, mock_tool_fixture, mock_event_fixture
):
  span = mock.MagicMock()
  span.is_recording.return_value = False
  get_telemetry_config = mock.Mock()
  serialize = mock.Mock(return_value='{}')
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._telemetry_config_from_invocation_context',
      get_telemetry_config,
  )
  monkeypatch.setattr(
      'google.adk.telemetry.tracing.safe_json_serialize', serialize
  )
  mock_event_fixture.content = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id='tool_call_id_004',
                  name='test_function_1',
                  response={'data': 'structured_data'},
              )
          ),
      ],
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'query': 'details'},
      function_response_event=mock_event_fixture,
      span=span,
  )

  get_telemetry_config.assert_not_called()
  serialize.assert_not_called()
  span.set_attribute.assert_not_called()


def test_trace_merged_tool_calls_skips_non_recording_span(
    monkeypatch, mock_event_fixture
):
  span = mock.MagicMock()
  span.is_recording.return_value = False
  monkeypatch.setattr('opentelemetry.trace.get_current_span', lambda: span)
  get_telemetry_config = mock.Mock()
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._telemetry_config_from_invocation_context',
      get_telemetry_config,
  )

  with mock.patch.object(
      Event, 'model_dump_json', autospec=True
  ) as serialize_event:
    trace_merged_tool_calls(
        response_event_id='merged_evt_id_002',
        function_response_event=mock_event_fixture,
    )

  get_telemetry_config.assert_not_called()
  serialize_event.assert_not_called()
  span.set_attribute.assert_not_called()


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

  test_args: Dict[str, object] = {'query': 'details', 'id_list': [1, 2, 3]}
  test_tool_call_id: str = 'tool_call_id_002'
  test_event_id: str = 'event_id_dict_002'
  dict_function_response: Dict[str, object] = {
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
async def test_trace_call_llm_summarizes_response_inline_data(
    monkeypatch, mock_span_fixture
):
  """Inline binary data in the response is described, not copied to the span."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
      model='gemini-pro', config=types.GenerateContentConfig()
  )
  llm_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(text='hi'),
              types.Part.from_bytes(data=b'test_data', mime_type='audio/pcm'),
          ],
      )
  )

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  llm_response_json = next(
      call_obj.args[1]
      for call_obj in mock_span_fixture.set_attribute.call_args_list
      if call_obj.args[0] == 'gcp.vertex.agent.llm_response'
  )

  # b'test_data' base64-encodes to 'dGVzdF9kYXRh'.
  assert 'dGVzdF9kYXRh' not in llm_response_json
  assert 'hi' in llm_response_json
  assert '<inline_data: audio/pcm, 9 bytes>' in llm_response_json


@pytest.mark.asyncio
async def test_trace_send_data_summarizes_inline_data(
    monkeypatch, mock_span_fixture
):
  """Inline binary data is described on the span, never copied onto it."""
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'true')
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
              parts=[
                  types.Part(text='hi'),
                  types.Part.from_bytes(
                      data=b'test_data', mime_type='audio/pcm'
                  ),
              ],
          )
      ],
  )

  data_json = next(
      call_obj.args[1]
      for call_obj in mock_span_fixture.set_attribute.call_args_list
      if call_obj.args[0] == 'gcp.vertex.agent.data'
  )

  # b'test_data' base64-encodes to 'dGVzdF9kYXRh'.
  assert 'dGVzdF9kYXRh' not in data_json
  assert 'hi' in data_json
  assert '<inline_data: audio/pcm, 9 bytes>' in data_json


@pytest.mark.asyncio
async def test_trace_send_data_summarizes_blob_without_mime_type(
    monkeypatch, mock_span_fixture
):
  """A blob is described even when its mime type and bytes are unset.

  The parts-less content in the same call pins that summarizing tolerates
  ``Content.parts`` being unset.
  """
  monkeypatch.setenv(ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS, 'true')
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  agent = LlmAgent(name='test_agent')
  invocation_context = await _create_invocation_context(agent)

  trace_send_data(
      invocation_context=invocation_context,
      event_id='test_event_id',
      data=[
          types.Content(role='user'),
          types.Content(
              role='user', parts=[types.Part(inline_data=types.Blob())]
          ),
      ],
  )

  data_json = next(
      call_obj.args[1]
      for call_obj in mock_span_fixture.set_attribute.call_args_list
      if call_obj.args[0] == 'gcp.vertex.agent.data'
  )

  assert '<inline_data: unknown, 0 bytes>' in data_json
  assert 'inlineData' not in data_json


@pytest.mark.asyncio
@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch('google.adk.telemetry.tracing.tracer')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='test_system',
)
# (env_value, captured) pairs: pin both the documented OTel four-state
# values that enable LogRecord content ('EVENT_ONLY' and 'SPAN_AND_EVENT')
# and the cases that disable it (empty string and 'SPAN_ONLY' -- the latter
# puts content on the span only).
@pytest.mark.parametrize(
    'env_capture_value,capture_content',
    [
        ('EVENT_ONLY', True),
        ('SPAN_AND_EVENT', True),
        ('', False),
        ('SPAN_ONLY', False),
    ],
)
@pytest.mark.parametrize('user_id', ['some-user-id', None])
async def test_generate_content_span(
    mock_guess_system_name,
    mock_tracer,
    mock_otel_logger,
    monkeypatch,
    env_capture_value,
    capture_content,
    user_id,
):
  """Test native generate_content span creation with attributes and logs."""
  # Arrange
  monkeypatch.setenv(
      'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT',
      env_capture_value,
  )
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: False,
  )

  agent = LlmAgent(name='test_agent', model='not-a-gemini-model')
  invocation_context = await _create_invocation_context(agent)
  invocation_context.session.user_id = user_id
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
  async with use_inference_span(
      llm_request, invocation_context, model_response_event
  ) as gc_span:
    assert gc_span.span is mock_span

    trace_inference_result(invocation_context, gc_span, llm_response)

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

  mock_span.set_attributes.assert_any_call({
      GEN_AI_USAGE_INPUT_TOKENS: 10,
      GEN_AI_USAGE_OUTPUT_TOKENS: 20,
  })
  mock_span.set_attributes.assert_any_call({
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      'gcp.vertex.agent.event_id': 'event-123',
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  })

  all_set_attribute_keys = [
      call.args[0] for call in mock_span.set_attribute.call_args_list
  ]
  assert USER_ID not in all_set_attribute_keys

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
  expected_user_log_attributes = {GEN_AI_SYSTEM: 'test_system'}
  if capture_content and user_id is not None:
    expected_user_log_attributes[USER_ID] = user_id
  for log in user_logs:
    assert log.attributes == expected_user_log_attributes

  choice_log = next(
      (lr for lr in log_records if lr.event_name == 'gen_ai.choice'),
      None,
  )
  assert choice_log is not None
  assert choice_log.body == expected_choice_body
  assert choice_log.attributes == {GEN_AI_SYSTEM: 'test_system'}


@pytest.mark.asyncio
@mock.patch(
    'google.adk.telemetry.tracing._use_extra_generate_content_attributes'
)
async def test_generate_content_span_with_genai_instrumentation(
    mock_use_extra,
    monkeypatch,
):
  """Test that genai-instrumentation delegation branch does not forward USER_ID in attributes."""
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: True,
  )
  # _is_gemini_agent returns true for gemini models.
  agent = LlmAgent(name='test_agent', model='gemini-1.5-pro')
  invocation_context = await _create_invocation_context(agent)

  llm_request = LlmRequest(
      model='gemini-1.5-pro',
      contents=[types.Content(role='user', parts=[types.Part(text='Hello')])],
  )

  model_response_event = mock.MagicMock()
  model_response_event.id = 'event-123'

  mock_cm = mock.MagicMock()
  mock_use_extra.return_value = mock_cm

  async with use_inference_span(
      llm_request, invocation_context, model_response_event
  ):
    pass

  mock_use_extra.assert_called_once()
  args, _ = mock_use_extra.call_args
  common_attributes = args[0]

  assert GEN_AI_AGENT_NAME in common_attributes
  assert GEN_AI_CONVERSATION_ID in common_attributes
  assert 'gcp.vertex.agent.event_id' in common_attributes
  assert 'gcp.vertex.agent.invocation_id' in common_attributes

  # USER_ID should NOT be in common_attributes passed to the genai instrumentor
  assert USER_ID not in common_attributes


def _mock_callable_tool():
  """Description of some tool."""
  return 'result'


def _mock_mcp_tool():
  return McpTool(
      name='mcp_tool',
      description='A standalone mcp tool',
      inputSchema={
          'type': 'object',
          'properties': {'id': {'type': 'integer'}},
      },
  )


def _mock_tool_dict() -> types.ToolDict:
  return types.ToolDict(
      function_declarations=[
          types.FunctionDeclarationDict(
              name='mock_tool', description='Description of mock tool.'
          ),
      ],
      google_maps=types.GoogleMaps(),
  )


@pytest.mark.asyncio
@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch('google.adk.telemetry.tracing.tracer')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='test_system',
)
@pytest.mark.parametrize(
    'capture_content',
    ['SPAN_AND_EVENT', 'EVENT_ONLY', 'SPAN_ONLY', 'NO_CONTENT'],
)
@pytest.mark.parametrize('user_id', ['some-user-id', None])
async def test_generate_content_span_with_experimental_semconv(
    mock_guess_system_name,
    mock_tracer,
    mock_otel_logger,
    monkeypatch,
    capture_content,
    user_id,
):
  """Test native generate_content span creation with attributes and logs with experimental semconv enabled."""
  # Arrange
  monkeypatch.setenv(
      'OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT',
      str(capture_content).lower(),
  )
  monkeypatch.setenv(
      'OTEL_SEMCONV_STABILITY_OPT_IN',
      'gen_ai_latest_experimental',
  )
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: False,
  )

  agent = LlmAgent(name='test_agent', model='not-a-gemini-model')
  invocation_context = await _create_invocation_context(agent)
  invocation_context.session.user_id = user_id

  system_instruction = types.Content(
      parts=[types.Part.from_text(text='You are a helpful assistant.')],
  )

  user_content1 = types.Content(role='user', parts=[types.Part(text='Hello')])
  user_content2 = types.Content(role='user', parts=[types.Part(text='World')])

  model_content = types.Content(
      role='model', parts=[types.Part(text='Response')]
  )

  tools = [
      _mock_callable_tool,
      _mock_tool_dict(),
      _mock_mcp_tool(),
  ]

  llm_request = LlmRequest(
      model='some-model',
      contents=[user_content1, user_content2],
      config=types.GenerateContentConfig(
          system_instruction=system_instruction, tools=tools
      ),
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
  async with use_inference_span(
      llm_request,
      invocation_context,
      model_response_event,
  ) as gc_span:
    assert gc_span.span is mock_span

    trace_inference_result(invocation_context, gc_span, llm_response)

  # Expected attributes
  expected_system_instructions = [
      {
          'content': 'You are a helpful assistant.',
          'type': 'text',
      },
  ]
  expected_input_messages = [
      {
          'role': 'user',
          'parts': [
              {'content': 'Hello', 'type': 'text'},
          ],
      },
      {
          'role': 'user',
          'parts': [
              {'content': 'World', 'type': 'text'},
          ],
      },
  ]
  expected_output_messages = [{
      'role': 'assistant',
      'parts': [
          {'content': 'Response', 'type': 'text'},
      ],
      'finish_reason': 'stop',
  }]
  expected_tool_definitions = [
      {
          'name': '_mock_callable_tool',
          'description': 'Description of some tool.',
          'parameters': None,
          'type': 'function',
      },
      {
          'name': 'mock_tool',
          'description': 'Description of mock tool.',
          'parameters': None,
          'type': 'function',
      },
      {
          'name': 'google_maps',
          'type': 'google_maps',
      },
      {
          'name': 'mcp_tool',
          'description': 'A standalone mcp tool',
          'parameters': {
              'type': 'object',
              'properties': {'id': {'type': 'integer'}},
          },
          'type': 'function',
      },
  ]
  expected_tool_definitions_no_content = [
      {
          'name': '_mock_callable_tool',
          'description': 'Description of some tool.',
          'parameters': None,
          'type': 'function',
      },
      {
          'name': 'mock_tool',
          'description': 'Description of mock tool.',
          'parameters': None,
          'type': 'function',
      },
      {
          'name': 'google_maps',
          'type': 'google_maps',
      },
      {
          'name': 'mcp_tool',
          'description': 'A standalone mcp tool',
          'parameters': None,
          'type': 'function',
      },
  ]
  expected_tool_definitions_json = (
      '[{"name":"_mock_callable_tool","description":"Description of some'
      ' tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description'
      ' of mock'
      ' tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"A'
      ' standalone mcp'
      ' tool","parameters":{"type":"object","properties":{"id":{"type":"integer"}}},"type":"function"}]'
  )

  expected_tool_definitions_no_content_json = (
      '[{"name":"_mock_callable_tool","description":"Description of some'
      ' tool.","parameters":null,"type":"function"},{"name":"mock_tool","description":"Description'
      ' of mock'
      ' tool.","parameters":null,"type":"function"},{"name":"google_maps","type":"google_maps"},{"name":"mcp_tool","description":"A'
      ' standalone mcp tool","parameters":null,"type":"function"}]'
  )
  # Assert Span
  mock_tracer.start_as_current_span.assert_called_once_with(
      'generate_content some-model'
  )

  mock_span.set_attribute.assert_any_call(
      GEN_AI_OPERATION_NAME, 'generate_content'
  )
  mock_span.set_attribute.assert_any_call(GEN_AI_REQUEST_MODEL, 'some-model')
  mock_span.set_attribute.assert_any_call(
      GEN_AI_RESPONSE_FINISH_REASONS, ['stop']
  )

  mock_span.set_attributes.assert_any_call({
      GEN_AI_USAGE_INPUT_TOKENS: 10,
      GEN_AI_USAGE_OUTPUT_TOKENS: 20,
  })
  mock_span.set_attributes.assert_any_call({
      GEN_AI_AGENT_NAME: invocation_context.agent.name,
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      'gcp.vertex.agent.event_id': 'event-123',
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  })

  all_set_attribute_keys = [
      call.args[0] for call in mock_span.set_attribute.call_args_list
  ]
  assert USER_ID not in all_set_attribute_keys

  if capture_content in ['SPAN_AND_EVENT', 'SPAN_ONLY']:
    mock_span.set_attribute.assert_any_call(
        GEN_AI_SYSTEM_INSTRUCTIONS,
        '[{"content":"You are a helpful assistant.","type":"text"}]',
    )
    mock_span.set_attribute.assert_any_call(
        GEN_AI_INPUT_MESSAGES,
        '[{"role":"user","parts":[{"content":"Hello","type":"text"}]},{"role":"user","parts":[{"content":"World","type":"text"}]}]',
    )
    mock_span.set_attribute.assert_any_call(
        GEN_AI_OUTPUT_MESSAGES,
        '[{"role":"assistant","parts":[{"content":"Response","type":"text"}],"finish_reason":"stop"}]',
    )
    mock_span.set_attribute.assert_any_call(
        GEN_AI_TOOL_DEFINITIONS, expected_tool_definitions_json
    )
  else:
    all_attribute_calls = mock_span.set_attribute.call_args_list
    assert GEN_AI_SYSTEM_INSTRUCTIONS not in all_attribute_calls
    assert GEN_AI_INPUT_MESSAGES not in all_attribute_calls
    assert GEN_AI_OUTPUT_MESSAGES not in all_attribute_calls
    mock_span.set_attribute.assert_any_call(
        GEN_AI_TOOL_DEFINITIONS, expected_tool_definitions_no_content_json
    )

  # Assert Logs
  assert mock_otel_logger.emit.call_count == 1

  log_records: list[LogRecord] = [
      call.args[0] for call in mock_otel_logger.emit.call_args_list
  ]

  operation_details_log = next(
      (
          lr
          for lr in log_records
          if lr.event_name == 'gen_ai.client.inference.operation.details'
      ),
      None,
  )

  assert operation_details_log is not None
  assert operation_details_log.attributes is not None

  attributes = operation_details_log.attributes

  if (
      capture_content in ['EVENT_ONLY', 'SPAN_AND_EVENT']
      and user_id is not None
  ):
    assert USER_ID in attributes
    assert attributes[USER_ID] == user_id
  else:
    assert USER_ID not in attributes

  if capture_content in ['SPAN_AND_EVENT', 'EVENT_ONLY']:
    assert GEN_AI_SYSTEM_INSTRUCTIONS in attributes
    assert (
        attributes[GEN_AI_SYSTEM_INSTRUCTIONS] == expected_system_instructions
    )
    assert GEN_AI_INPUT_MESSAGES in attributes
    assert attributes[GEN_AI_INPUT_MESSAGES] == expected_input_messages
    assert GEN_AI_OUTPUT_MESSAGES in attributes
    assert attributes[GEN_AI_OUTPUT_MESSAGES] == expected_output_messages
    assert GEN_AI_TOOL_DEFINITIONS in attributes
    assert attributes[GEN_AI_TOOL_DEFINITIONS] == expected_tool_definitions
  else:
    assert GEN_AI_SYSTEM_INSTRUCTIONS not in attributes
    assert GEN_AI_INPUT_MESSAGES not in attributes
    assert GEN_AI_OUTPUT_MESSAGES not in attributes
    assert GEN_AI_TOOL_DEFINITIONS in attributes
    assert (
        attributes[GEN_AI_TOOL_DEFINITIONS]
        == expected_tool_definitions_no_content
    )

  assert GEN_AI_USAGE_INPUT_TOKENS in attributes
  assert attributes[GEN_AI_USAGE_INPUT_TOKENS] == 10
  assert GEN_AI_USAGE_OUTPUT_TOKENS in attributes
  assert attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 20
  assert 'gcp.vertex.agent.event_id' in attributes
  assert attributes['gcp.vertex.agent.event_id'] == 'event-123'
  assert 'gcp.vertex.agent.invocation_id' in attributes
  assert (
      attributes['gcp.vertex.agent.invocation_id']
      == invocation_context.invocation_id
  )
  assert GEN_AI_AGENT_NAME in attributes
  assert attributes[GEN_AI_AGENT_NAME] == invocation_context.agent.name
  assert GEN_AI_CONVERSATION_ID in attributes
  assert attributes[GEN_AI_CONVERSATION_ID] == invocation_context.session.id


def test_trace_tool_call_with_tool_execution_error(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, object] = {'param_a': 'value_a'}
  test_error = ToolExecutionError(
      message='Internal server error',
      error_type=ToolErrorType.INTERNAL_SERVER_ERROR,
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args=test_args,
      function_response_event=None,
      error=test_error,
  )

  expected_calls = [
      mock.call('gen_ai.operation.name', 'execute_tool'),
      mock.call('gen_ai.tool.name', mock_tool_fixture.name),
      mock.call('gen_ai.tool.description', mock_tool_fixture.description),
      mock.call('gen_ai.tool.type', 'SimpleTestTool'),
      mock.call('error.type', 'INTERNAL_SERVER_ERROR'),
      mock.call('gcp.vertex.agent.tool_call_args', json.dumps(test_args)),
      mock.call(
          'gcp.vertex.agent.tool_response', '{"result": "<not specified>"}'
      ),
      mock.call('gcp.vertex.agent.llm_request', '{}'),
      mock.call('gcp.vertex.agent.llm_response', '{}'),
      mock.call('gen_ai.tool.call.id', '<not specified>'),
  ]

  mock_span_fixture.set_attribute.assert_has_calls(
      expected_calls, any_order=True
  )


def test_trace_tool_call_with_timeout_error(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, object] = {'param_a': 'value_a'}
  test_error = ToolExecutionError(
      message='Request timed out',
      error_type=ToolErrorType.REQUEST_TIMEOUT,
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args=test_args,
      function_response_event=None,
      error=test_error,
  )

  assert (
      mock.call('error.type', 'REQUEST_TIMEOUT')
      in mock_span_fixture.set_attribute.call_args_list
  )


def test_trace_tool_call_with_standard_error(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_args: Dict[str, object] = {'param': 1}
  test_error = ValueError('Invalid arguments')

  trace_tool_call(
      tool=mock_tool_fixture,
      args=test_args,
      function_response_event=None,
      error=test_error,
  )

  assert (
      mock.call('error.type', 'ValueError')
      in mock_span_fixture.set_attribute.call_args_list
  )


def test_trace_tool_call_with_genai_api_error_uses_status_code(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  """A genai APIError surfaces its HTTP status code (not ``ClientError``)."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  test_error = genai_errors.ClientError(
      429, {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'param': 1},
      function_response_event=None,
      error=test_error,
  )

  assert (
      mock.call('error.type', '429')
      in mock_span_fixture.set_attribute.call_args_list
  )


def test_trace_tool_call_with_dict_error_marks_span_as_failed(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  """A tool reporting failure in its response dict must not render as green."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'param': 1},
      function_response_event=None,
      error_type='MCP_TOOL_ERROR',
  )

  mock_span_fixture.set_status.assert_called_once()
  status = mock_span_fixture.set_status.call_args.args[0]
  assert status.status_code is StatusCode.ERROR
  assert status.description == 'MCP_TOOL_ERROR'
  mock_span_fixture.record_exception.assert_not_called()


def test_trace_tool_call_with_error_marks_span_as_failed_and_records_it(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )
  test_error = ToolExecutionError(
      message='Internal server error',
      error_type=ToolErrorType.INTERNAL_SERVER_ERROR,
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'param': 1},
      function_response_event=None,
      error=test_error,
  )

  mock_span_fixture.record_exception.assert_called_once_with(test_error)
  mock_span_fixture.set_status.assert_called_once()
  status = mock_span_fixture.set_status.call_args.args[0]
  assert status.status_code is StatusCode.ERROR
  # The type, not the message, so tool content stays out of an attribute the
  # content toggle cannot elide.
  assert status.description == 'INTERNAL_SERVER_ERROR'


def test_trace_tool_call_without_error_leaves_span_status_unset(
    monkeypatch, mock_span_fixture, mock_tool_fixture, mock_event_fixture
):
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'param': 1},
      function_response_event=mock_event_fixture,
  )

  mock_span_fixture.set_status.assert_not_called()
  mock_span_fixture.record_exception.assert_not_called()


@pytest.mark.asyncio
@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch('google.adk.telemetry.tracing.tracer')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='deployment_default',
)
@pytest.mark.parametrize(
    'model,expected_system',
    [
        ('claude-sonnet-4-5', 'anthropic'),
        ('claude-3-5-haiku-latest', 'anthropic'),
        ('anthropic/claude-sonnet-4-5', 'anthropic'),
        (
            'projects/p/locations/l/publishers/anthropic/models/claude-sonnet-4-5',
            'anthropic',
        ),
        ('openai/gpt-4o', 'openai'),
        (
            'projects/p/locations/l/publishers/meta/models/llama-3',
            'deployment_default',
        ),
        ('tunedModels/my-tuned-model', 'deployment_default'),
        ('gemini-2.0-flash', 'deployment_default'),
        ('gemini/gemini-2.0-flash', 'deployment_default'),
        ('some-model', 'deployment_default'),
    ],
)
async def test_generate_content_span_system_name_follows_model(
    mock_guess_system_name,
    mock_tracer,
    mock_otel_logger,
    monkeypatch,
    model,
    expected_system,
):
  """The system name follows the served model, not just the deployment env."""
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: False,
  )
  agent = LlmAgent(name='test_agent', model=model)
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(model=model, contents=[])
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')]),
      finish_reason=types.FinishReason.STOP,
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = 'event-123'
  mock_span = (
      mock_tracer.start_as_current_span.return_value.__enter__.return_value
  )

  async with use_inference_span(
      llm_request, invocation_context, model_response_event
  ) as gc_span:
    trace_inference_result(invocation_context, gc_span, llm_response)

  mock_span.set_attribute.assert_any_call(GEN_AI_SYSTEM, expected_system)
  log_records: list[LogRecord] = [
      call.args[0] for call in mock_otel_logger.emit.call_args_list
  ]
  assert log_records
  for log_record in log_records:
    assert log_record.attributes[GEN_AI_SYSTEM] == expected_system


# Model ids that are resource paths: the leading segment is a resource
# collection, so reading it as a provider prefix mislabels the model.
_RESOURCE_PATH_MODELS = [
    'projects/p/locations/l/publishers/meta/models/llama-3',
    'projects/p/locations/l/endpoints/123456',
    'tunedModels/my-tuned-model',
]


@pytest.mark.parametrize(
    'model,expected_system',
    [
        # A Model Garden path names its publisher mid-path, never up front.
        ('projects/p/locations/l/publishers/meta/models/llama-3', 'gemini'),
        (
            'projects/p/locations/l/publishers/anthropic/models/claude-4-5',
            'anthropic',
        ),
        # Tuned models arrive as a path too, on either backend.
        ('projects/p/locations/l/endpoints/123456', 'gemini'),
        ('tunedModels/my-tuned-model', 'gemini'),
        ('gemini-2.0-flash', 'gemini'),
        ('claude-sonnet-4-5', 'anthropic'),
        # What the provider-prefix rule is actually for.
        ('openai/gpt-4o', 'openai'),
    ],
)
def test_resolve_gen_ai_system_name(monkeypatch, model, expected_system):
  """Only a real provider prefix names the provider; a path segment must not."""
  monkeypatch.setattr(tracing, '_guess_gemini_system_name', lambda: 'gemini')

  assert tracing._resolve_gen_ai_system_name(model) == expected_system


@pytest.mark.parametrize('model', _RESOURCE_PATH_MODELS)
def test_resolve_gen_ai_system_name_never_names_a_path_segment(
    monkeypatch, model
):
  """A resource path must not be read as a `<provider>/<model>` id."""
  monkeypatch.setattr(tracing, '_guess_gemini_system_name', lambda: 'gemini')

  system_name = tracing._resolve_gen_ai_system_name(model)

  assert system_name == 'gemini'
  assert system_name not in {segment.lower() for segment in model.split('/')}


def test_safe_json_serialize_circular_dict_returns_not_serializable():
  obj = {}
  obj['self'] = obj
  assert safe_json_serialize(obj) == '<not serializable>'


def test_safe_json_serialize_no_whitespaces_circular_dict_returns_not_serializable():
  obj = {}
  obj['self'] = obj
  assert _safe_json_serialize_no_whitespaces(obj) == '<not serializable>'


def test_safe_json_serialize_recursion_error_returns_not_serializable():
  with mock.patch.object(
      json, 'dumps', side_effect=RecursionError('maximum recursion depth')
  ):
    assert safe_json_serialize({'a': 1}) == '<not serializable>'


def test_safe_json_serialize_no_whitespaces_recursion_error_returns_not_serializable():
  with mock.patch.object(
      json, 'dumps', side_effect=RecursionError('maximum recursion depth')
  ):
    assert _safe_json_serialize_no_whitespaces({'a': 1}) == '<not serializable>'


def test_use_extra_generate_content_attributes_upgraded_version(monkeypatch):
  # Arrange: Mock the presence of the new event-only context key in the contrib module
  from opentelemetry.instrumentation import google_genai

  mock_event_only_key = 'MOCKED_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY'
  monkeypatch.setattr(
      google_genai,
      'GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY',
      mock_event_only_key,
      raising=False,
  )

  # Act: Run the helper with mock.patch on the otel context
  with mock.patch('opentelemetry.context.set_value') as mock_set_value:
    with _use_extra_generate_content_attributes(
        extra_attributes={'span.attr': 'value'},
        log_only_extra_attributes={USER_ID: 'user_123'},
    ):
      pass

    # Assert: Verify set_value was called with the mocked event-only key
    mock_set_value.assert_any_call(
        mock_event_only_key,
        {USER_ID: 'user_123'},
        context=mock.ANY,
    )


def test_use_extra_generate_content_attributes_older_version(monkeypatch):
  # Arrange: Simulate an older version by deleting the key if present
  from opentelemetry.instrumentation import google_genai

  if hasattr(
      google_genai, 'GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY'
  ):
    monkeypatch.delattr(
        google_genai, 'GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY'
    )

  # Act & Assert: Ensure execution does not throw any ImportError/AttributeError
  try:
    with _use_extra_generate_content_attributes(
        extra_attributes={'span.attr': 'value'},
        log_only_extra_attributes={USER_ID: 'user_123'},
    ):
      pass
  except Exception as e:  # pylint: disable=broad-exception-caught
    pytest.fail(f'Graceful degradation failed: {e}')


# ---------------------------------------------------------------------------
# Tests for _detect_error_in_response
# ---------------------------------------------------------------------------


class _ErrorDetectingTool(BaseTool):
  """A test tool whose _detect_error_in_response raises."""

  async def run_async(self, *, args, tool_context):
    return 'result'

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    raise RuntimeError('detection exploded')


def test_base_tool_does_not_define_detect_error_in_response():
  """BaseTool intentionally does not expose _detect_error_in_response as a public hook."""
  tool = SimpleTestTool(name='t', description='d')
  # The hook is opt-in per subclass; BaseTool itself must not declare it so
  # that telemetry callers can use getattr(...) to skip detection.
  assert not hasattr(tool, '_detect_error_in_response')


def test_detect_error_function_tool_error():
  from google.adk.tools.function_tool import FunctionTool

  tool = FunctionTool(func=lambda: None)
  assert (
      tool._detect_error_in_response({'error': 'missing arg'}) == 'TOOL_ERROR'
  )


def test_detect_error_function_tool_no_error():
  from google.adk.tools.function_tool import FunctionTool

  tool = FunctionTool(func=lambda: None)
  assert tool._detect_error_in_response({'result': 'ok'}) is None
  assert tool._detect_error_in_response('plain string') is None
  assert tool._detect_error_in_response(None) is None


def test_detect_error_rest_api_tool():
  from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool

  tool = RestApiTool.__new__(RestApiTool)
  assert (
      tool._detect_error_in_response({'error': 'Status Code: 404'})
      == 'HTTP_ERROR'
  )
  assert tool._detect_error_in_response({'result': 'ok'}) is None
  assert tool._detect_error_in_response({'text': 'html response'}) is None


def test_detect_error_mcp_tool():
  from google.adk.tools.mcp_tool.mcp_tool import McpTool as AdkMcpTool

  tool = AdkMcpTool.__new__(AdkMcpTool)
  assert (
      tool._detect_error_in_response({'isError': True, 'content': []})
      == 'MCP_TOOL_ERROR'
  )
  assert (
      tool._detect_error_in_response({'isError': False, 'content': []}) is None
  )
  assert tool._detect_error_in_response({'content': [{'text': 'ok'}]}) is None


def test_detect_error_google_tool():
  from google.adk.tools.google_tool import GoogleTool

  tool = GoogleTool.__new__(GoogleTool)
  assert (
      tool._detect_error_in_response(
          {'status': 'ERROR', 'error_details': 'fail'}
      )
      == 'TOOL_ERROR'
  )
  assert tool._detect_error_in_response({'status': 'OK', 'data': []}) is None
  assert (
      tool._detect_error_in_response({'error': 'something'}) is None
  )  # GoogleTool checks status, not error key


def test_detect_error_bash_tool():
  from google.adk.tools.bash_tool import ExecuteBashTool

  tool = ExecuteBashTool.__new__(ExecuteBashTool)
  assert (
      tool._detect_error_in_response({'error': 'Execution failed'})
      == 'TOOL_ERROR'
  )
  assert (
      tool._detect_error_in_response(
          {'error': 'timeout', 'stdout': '', 'stderr': ''}
      )
      == 'TOOL_ERROR'
  )
  assert (
      tool._detect_error_in_response({'stdout': 'ok', 'returncode': 0}) is None
  )


def _environment_tool_classes():
  from google.adk.tools.environment._edit_file_tool import EditFileTool
  from google.adk.tools.environment._execute_tool import ExecuteTool
  from google.adk.tools.environment._read_file_tool import ReadFileTool
  from google.adk.tools.environment._write_file_tool import WriteFileTool

  return [ExecuteTool, ReadFileTool, WriteFileTool, EditFileTool]


@pytest.mark.parametrize(
    'cls',
    _environment_tool_classes(),
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize(
    'response,expected',
    [
        ({'status': 'error', 'error': 'fail'}, 'TOOL_ERROR'),
        ({'status': 'ok', 'message': 'done'}, None),
        # Environment tools check status, not the error key.
        ({'error': 'something'}, None),
    ],
    ids=['status_error', 'status_ok', 'error_key_only'],
)
def test_detect_error_environment_tools(cls, response, expected):
  tool = cls.__new__(cls)
  assert tool._detect_error_in_response(response) == expected


@pytest.mark.parametrize(
    'cls_name',
    ['LoadSkillTool', 'LoadSkillResourceTool', 'RunSkillScriptTool'],
)
@pytest.mark.parametrize(
    'response,expected',
    [
        (
            {'error': 'missing', 'error_code': 'INVALID_ARGUMENTS'},
            'INVALID_ARGUMENTS',
        ),
        ({'error': 'generic'}, 'TOOL_ERROR'),
        ({'skill_name': 'x', 'instructions': 'y'}, None),
    ],
    ids=['with_error_code', 'error_no_code', 'no_error'],
)
def test_detect_error_skill_tools(cls_name, response, expected):
  skill_toolset = pytest.importorskip('google.adk.tools.skill_toolset')
  cls = getattr(skill_toolset, cls_name)
  tool = cls.__new__(cls)
  assert tool._detect_error_in_response(response) == expected


def test_detect_error_discovery_engine_search_tool():
  mod = pytest.importorskip('google.adk.tools.discovery_engine_search_tool')
  DiscoveryEngineSearchTool = mod.DiscoveryEngineSearchTool

  tool = DiscoveryEngineSearchTool.__new__(DiscoveryEngineSearchTool)
  assert (
      tool._detect_error_in_response(
          {'status': 'error', 'error_message': 'fail'}
      )
      == 'TOOL_ERROR'
  )
  assert tool._detect_error_in_response({'status': 'ok', 'results': []}) is None


# ---------------------------------------------------------------------------
# Tests for trace_tool_call with error_type parameter
# ---------------------------------------------------------------------------


def test_trace_tool_call_with_error_type(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  """error_type sets the span error.type attribute when no exception."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'x': 1},
      function_response_event=None,
      error=None,
      error_type='HTTP_ERROR',
  )

  mock_span_fixture.set_attribute.assert_any_call('error.type', 'HTTP_ERROR')


def test_trace_tool_call_error_takes_precedence_over_error_type(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  """When both error and error_type are provided, error takes precedence."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'x': 1},
      function_response_event=None,
      error=ValueError('boom'),
      error_type='HTTP_ERROR',
  )

  # ValueError should be set, not HTTP_ERROR.
  mock_span_fixture.set_attribute.assert_any_call('error.type', 'ValueError')
  error_type_calls = [
      c
      for c in mock_span_fixture.set_attribute.call_args_list
      if c == mock.call('error.type', mock.ANY)
  ]
  assert len(error_type_calls) == 1


def test_trace_tool_call_no_error_no_error_type(
    monkeypatch, mock_span_fixture, mock_tool_fixture
):
  """When neither error nor error_type is set, no error.type attribute."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )

  trace_tool_call(
      tool=mock_tool_fixture,
      args={'x': 1},
      function_response_event=None,
      error=None,
      error_type=None,
  )

  error_type_calls = [
      c
      for c in mock_span_fixture.set_attribute.call_args_list
      if c == mock.call('error.type', mock.ANY)
  ]
  assert len(error_type_calls) == 0


def test_build_llm_request_for_trace_excludes_thought_signatures():
  """Opaque signature bytes must not be base64-encoded onto a span attribute.

  A thought signature stays in history and is replayed on every later request,
  so leaving it in would grow the serialized request on each call of a session.
  """
  from google.adk.telemetry.tracing import _build_llm_request_for_trace

  llm_request = LlmRequest(
      model='gemini-2.0-flash',
      contents=[
          types.Content(
              role='model',
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          id='call-1', name='search', args={}
                      ),
                      thought_signature=b'opaque-signature-bytes' * 40,
                  )
              ],
          )
      ],
  )

  serialized = json.dumps(_build_llm_request_for_trace(llm_request))

  assert 'thoughtSignature' not in serialized
  assert 'thought_signature' not in serialized
  # The part itself is still described on the span.
  assert 'search' in serialized


@pytest.mark.asyncio
async def test_trace_call_llm_excludes_response_thought_signature(
    monkeypatch, mock_span_fixture
):
  """A signature on the model's own response must not reach the span either."""
  monkeypatch.setattr(
      'opentelemetry.trace.get_current_span', lambda: mock_span_fixture
  )
  invocation_context = await _create_invocation_context(
      LlmAgent(name='test_agent')
  )
  llm_request = LlmRequest(
      model='gemini-pro', config=types.GenerateContentConfig()
  )
  llm_response = LlmResponse(
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  text='done',
                  thought_signature=b'opaque-signature-bytes' * 40,
              )
          ],
      )
  )

  trace_call_llm(invocation_context, 'test_event_id', llm_request, llm_response)

  llm_response_json = next(
      call_obj.args[1]
      for call_obj in mock_span_fixture.set_attribute.call_args_list
      if call_obj.args[0] == 'gcp.vertex.agent.llm_response'
  )
  assert 'thoughtSignature' not in llm_response_json
  assert 'thought_signature' not in llm_response_json
  assert 'done' in llm_response_json


def test_build_llm_request_for_trace_excludes_live_http_clients():
  """Tracing must not crash when config.http_options holds live SDK clients.

  HttpOptions.{httpx_client, httpx_async_client, aiohttp_client} are live
  transport objects that pydantic cannot serialize; they must be excluded so
  the trace serialization does not raise PydanticSerializationError.
  """
  from google.adk.telemetry.tracing import _build_llm_request_for_trace
  import httpx

  llm_request = LlmRequest(
      model='gemini-2.0-flash',
      config=types.GenerateContentConfig(
          temperature=0.1,
          http_options=types.HttpOptions(
              httpx_async_client=httpx.AsyncClient()
          ),
      ),
  )

  result = _build_llm_request_for_trace(llm_request)

  # Must be JSON-serializable (raised PydanticSerializationError before the fix).
  json.dumps(result)
  assert 'httpx_async_client' not in result['config'].get('http_options', {})
  assert result['config']['temperature'] == 0.1


def test_build_llm_request_for_trace_excludes_http_option_credentials():
  """Credential-bearing http_options fields must never reach a span attribute.

  `RunConfig.http_options` is a documented place for callers to put custom
  headers (including `Authorization`), and it is copied onto
  `llm_request.config.http_options`. Serializing it verbatim would export the
  caller's credentials to the tracing backend on every model call.
  """
  from google.adk.telemetry.tracing import _build_llm_request_for_trace

  llm_request = LlmRequest(
      model='gemini-2.0-flash',
      config=types.GenerateContentConfig(
          temperature=0.1,
          http_options=types.HttpOptions(
              base_url='https://example.test',
              headers={'Authorization': 'Bearer sentinel-secret-token'},
              extra_body={'api_key': 'sentinel-secret-token'},
              client_args={'auth': 'sentinel-secret-token'},
              async_client_args={'auth': 'sentinel-secret-token'},
          ),
      ),
  )

  result = _build_llm_request_for_trace(llm_request)

  assert 'sentinel-secret-token' not in json.dumps(result)
  http_options = result['config'].get('http_options', {})
  for field in ('headers', 'extra_body', 'client_args', 'async_client_args'):
    assert field not in http_options
  # Non-sensitive http_options fields are still traced.
  assert http_options['base_url'] == 'https://example.test'


# ---------------------------------------------------------------------------
# safe_json_serialize tests
# ---------------------------------------------------------------------------


class _SampleToolResult(BaseModel):
  query: str
  total: int
  items: list[str] = []


class _NestedModel(BaseModel):
  inner: _SampleToolResult


def test_safe_json_serialize_plain_dict():
  """Plain dicts serialize normally."""
  result = safe_json_serialize({'key': 'value', 'num': 42})
  assert json.loads(result) == {'key': 'value', 'num': 42}


def test_safe_json_serialize_pydantic_model_in_dict():
  """Pydantic models nested in a dict are serialized via model_dump."""
  model = _SampleToolResult(query='test', total=2, items=['a', 'b'])
  result = safe_json_serialize({'result': model})
  parsed = json.loads(result)
  assert parsed == {
      'result': {'query': 'test', 'total': 2, 'items': ['a', 'b']}
  }


def test_safe_json_serialize_nested_pydantic_model():
  """Nested Pydantic models are fully serialized."""
  inner = _SampleToolResult(query='q', total=0, items=[])
  outer = _NestedModel(inner=inner)
  result = safe_json_serialize({'result': outer})
  parsed = json.loads(result)
  assert parsed['result']['inner'] == {'query': 'q', 'total': 0, 'items': []}


def test_safe_json_serialize_top_level_pydantic_model():
  """A top-level Pydantic model (not wrapped in a dict) is serialized."""
  model = _SampleToolResult(query='direct', total=1, items=['x'])
  result = safe_json_serialize(model)
  parsed = json.loads(result)
  assert parsed == {'query': 'direct', 'total': 1, 'items': ['x']}


def test_safe_json_serialize_non_serializable_fallback():
  """Objects that are neither JSON-native nor Pydantic fall back gracefully."""
  result = safe_json_serialize({'value': object()})
  assert '<not serializable>' in result


# ---------------------------------------------------------------------------
# resolve_error_type precedence.
#
# The three individual branches are exercised through ``trace_tool_call``
# above; what is pinned here is which one wins when more than one applies.
# ---------------------------------------------------------------------------


def test_resolve_error_type_prefers_a_pre_classified_type_over_the_status():
  """An ADK-classified type outranks the HTTP status: it is the higher

  resolution label, and the status is only a fallback for SDK errors that
  collapse every 4xx into one class.
  """
  error = genai_errors.ClientError(429, {'error': {'code': 429}})
  error.error_type = 'QUOTA_EXHAUSTED'

  assert resolve_error_type(error) == 'QUOTA_EXHAUSTED'


def test_resolve_error_type_stringifies_a_non_string_classification():
  """``error.type`` is a string span attribute, so a numeric classification

  has to be coerced rather than handed to OTel as an int.
  """
  error = ToolExecutionError(message='boom')
  error.error_type = 500

  assert resolve_error_type(error) == '500'


# ---------------------------------------------------------------------------
# GenerateContentSpan.
# ---------------------------------------------------------------------------


def test_generate_content_span_attribute_stores_are_per_instance(
    mock_span_fixture,
):
  """Each inference call accumulates its own experimental-semconv attributes;

  sharing the dicts across instances would leak one call's prompt/response
  attributes onto the next.
  """
  first = GenerateContentSpan(mock_span_fixture)
  second = GenerateContentSpan(mock_span_fixture)

  first.operation_details_attributes['some_key'] = 'some_value'
  first.operation_details_common_attributes['other_key'] = 'other_value'

  assert first.span is mock_span_fixture
  assert second.operation_details_attributes == {}
  assert second.operation_details_common_attributes == {}


# ---------------------------------------------------------------------------
# The deprecated use_generate_content_span / trace_generate_content_result
# pair, kept until callers move to use_inference_span /
# trace_inference_result.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch('google.adk.telemetry.tracing.tracer')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='test_system',
)
async def test_use_generate_content_span_yields_the_bare_span(
    mock_guess_system_name,
    mock_tracer,
    mock_otel_logger,
    monkeypatch,
):
  """The deprecated manager yields the raw OTel span rather than the

  ``GenerateContentSpan`` its replacement yields, because its result helper
  takes a plain span.
  """
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: False,
  )
  agent = LlmAgent(name='test_agent', model='not-a-gemini-model')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(
      model='some-model',
      contents=[types.Content(role='user', parts=[types.Part(text='Hello')])],
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = 'event-123'

  mock_span = (
      mock_tracer.start_as_current_span.return_value.__enter__.return_value
  )

  with use_generate_content_span(
      llm_request, invocation_context, model_response_event
  ) as span:
    assert span is mock_span

  mock_tracer.start_as_current_span.assert_called_once_with(
      'generate_content some-model'
  )
  mock_span.set_attribute.assert_any_call(GEN_AI_SYSTEM, 'test_system')
  mock_span.set_attribute.assert_any_call(
      GEN_AI_OPERATION_NAME, 'generate_content'
  )
  mock_span.set_attribute.assert_any_call(GEN_AI_REQUEST_MODEL, 'some-model')
  mock_span.set_attributes.assert_any_call({
      GEN_AI_AGENT_NAME: 'test_agent',
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      'gcp.vertex.agent.event_id': 'event-123',
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  })


@pytest.mark.asyncio
@mock.patch(
    'google.adk.telemetry.tracing._use_extra_generate_content_attributes'
)
async def test_use_generate_content_span_delegates_to_the_genai_instrumentor(
    mock_use_extra,
    monkeypatch,
):
  """With the genai instrumentation library wrapping a Gemini call, the span

  belongs to that library: nothing is yielded, and the ADK attributes are
  only stashed on the context for the library to pick up.
  """
  monkeypatch.setattr(
      'google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: True,
  )
  agent = LlmAgent(name='test_agent', model='gemini-1.5-pro')
  invocation_context = await _create_invocation_context(agent)
  llm_request = LlmRequest(model='gemini-1.5-pro')
  model_response_event = mock.MagicMock()
  model_response_event.id = 'event-123'

  with use_generate_content_span(
      llm_request, invocation_context, model_response_event
  ) as span:
    assert span is None

  mock_use_extra.assert_called_once()
  (common_attributes,) = mock_use_extra.call_args.args
  assert common_attributes == {
      GEN_AI_AGENT_NAME: 'test_agent',
      GEN_AI_CONVERSATION_ID: invocation_context.session.id,
      'gcp.vertex.agent.event_id': 'event-123',
      'gcp.vertex.agent.invocation_id': invocation_context.invocation_id,
  }


@mock.patch('google.adk.telemetry.tracing.otel_logger')
@mock.patch(
    'google.adk.telemetry.tracing._guess_gemini_system_name',
    return_value='test_system',
)
def test_trace_generate_content_result_records_outcome_and_choice_log(
    mock_guess_system_name,
    mock_otel_logger,
    mock_span_fixture,
):
  """The finish reason is lower-cased into a list (semconv allows several)

  and the token usage lands on the span, alongside a choice log record.
  """
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='hi')]),
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=20,
      ),
  )

  trace_generate_content_result(mock_span_fixture, llm_response)

  mock_span_fixture.set_attribute.assert_called_once_with(
      GEN_AI_RESPONSE_FINISH_REASONS, ['stop']
  )
  mock_span_fixture.set_attributes.assert_called_once_with({
      GEN_AI_USAGE_INPUT_TOKENS: 10,
      GEN_AI_USAGE_OUTPUT_TOKENS: 20,
  })
  log_record: LogRecord = mock_otel_logger.emit.call_args.args[0]
  assert log_record.event_name == 'gen_ai.choice'
  assert log_record.attributes == {GEN_AI_SYSTEM: 'test_system'}


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_generate_content_result_skips_a_partial_response(
    mock_otel_logger,
    mock_span_fixture,
):
  """A partial streaming chunk is not the operation's result.

  Recording it would emit a choice log per chunk and report a finish reason for
  a call that has not finished.
  """
  llm_response = LlmResponse(
      partial=True,
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=20,
      ),
  )

  trace_generate_content_result(mock_span_fixture, llm_response)

  mock_span_fixture.set_attribute.assert_not_called()
  mock_span_fixture.set_attributes.assert_not_called()
  mock_otel_logger.emit.assert_not_called()


@mock.patch('google.adk.telemetry.tracing.otel_logger')
def test_trace_generate_content_result_without_a_span_emits_nothing(
    mock_otel_logger,
):
  """No span means the inference was not traced at all, so the choice log

  would be an orphan; it must be suppressed too.
  """
  trace_generate_content_result(
      None, LlmResponse(finish_reason=types.FinishReason.STOP)
  )

  mock_otel_logger.emit.assert_not_called()
