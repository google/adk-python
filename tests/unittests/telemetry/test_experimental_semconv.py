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

"""Tests for the experimental OTel GenAI semconv attribute setters.

The attribute keys and the per-part value shapes written by these setters are
the wire contract consumed downstream, so the assertions compare whole
attribute mappings instead of probing individual keys.
"""

from __future__ import annotations

from typing import Optional

from google.adk.dependencies._mcp import Tool as McpTool
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry._experimental_semconv import _model_dump_to_tool_definition
from google.adk.telemetry._experimental_semconv import set_operation_details_attributes_from_request
from google.adk.telemetry._experimental_semconv import set_operation_details_attributes_from_response
from google.adk.telemetry._stable_semconv import choice_body
from google.adk.telemetry.context import ContentCapturingMode
from google.adk.telemetry.context import TelemetryConfig
from google.genai import types
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_INPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OUTPUT_MESSAGES
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_RESPONSE_FINISH_REASONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_SYSTEM_INSTRUCTIONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_TOOL_DEFINITIONS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_INPUT_TOKENS
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_USAGE_OUTPUT_TOKENS
import pytest

_CACHE_READ_INPUT_TOKENS = 'gen_ai.usage.cache_read.input_tokens'


def _request_attributes(llm_request: LlmRequest) -> dict:
  attributes: dict = {}
  set_operation_details_attributes_from_request(attributes, llm_request)
  return attributes


def _response_attributes(llm_response: LlmResponse) -> tuple[dict, dict]:
  """Returns the (details, common) mappings written for `llm_response`."""
  details: dict = {}
  common: dict = {}
  set_operation_details_attributes_from_response(llm_response, details, common)
  return details, common


# ---------------------------------------------------------------------------
# set_operation_details_attributes_from_request
# ---------------------------------------------------------------------------


def test_request_attributes_always_write_the_three_wire_keys():
  """An empty request still emits every key, with empty lists as values.

  Key names are asserted as literals because consumers read them off the
  wire, not through the semconv constants.
  """
  attributes = {'pre.existing': 'kept'}

  set_operation_details_attributes_from_request(
      attributes, LlmRequest(model='some-model')
  )

  assert attributes == {
      'pre.existing': 'kept',
      'gen_ai.input.messages': [],
      'gen_ai.system_instructions': [],
      'gen_ai.tool.definitions': [],
  }


def test_request_attributes_render_every_supported_part_shape():
  """Each genai part maps to its own tagged dict; unknown parts are dropped."""
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='hi'),
          types.Part(
              inline_data=types.Blob(mime_type='image/png', data=b'\x89PNG')
          ),
          types.Part(
              file_data=types.FileData(
                  mime_type='audio/wav', file_uri='https://example/a.wav'
              )
          ),
          types.Part(
              function_call=types.FunctionCall(
                  id='call-1', name='get_weather', args={'city': 'Zurich'}
              )
          ),
          types.Part(
              function_response=types.FunctionResponse(
                  id='call-1', name='get_weather', response={'temp_c': 21}
              )
          ),
          types.Part(),
      ],
  )

  attributes = _request_attributes(
      LlmRequest(model='some-model', contents=[content])
  )

  assert attributes[GEN_AI_INPUT_MESSAGES] == [{
      'role': 'user',
      'parts': [
          {'content': 'hi', 'type': 'text'},
          {'mime_type': 'image/png', 'data': b'\x89PNG', 'type': 'blob'},
          {
              'mime_type': 'audio/wav',
              'uri': 'https://example/a.wav',
              'type': 'file_data',
          },
          {
              'id': 'call-1',
              'name': 'get_weather',
              'arguments': {'city': 'Zurich'},
              'type': 'tool_call',
          },
          {
              'id': 'call-1',
              'response': {'temp_c': 21},
              'type': 'tool_call_response',
          },
      ],
  }]


def test_request_attributes_synthesize_missing_tool_call_ids():
  """A missing call id becomes `<name>_<part index>`, or the index alone."""
  content = types.Content(
      role='user',
      parts=[
          types.Part(text='hi'),
          types.Part(function_call=types.FunctionCall(name='lookup')),
          types.Part(function_response=types.FunctionResponse(response={})),
      ],
  )

  attributes = _request_attributes(
      LlmRequest(model='some-model', contents=[content])
  )

  parts = attributes[GEN_AI_INPUT_MESSAGES][0]['parts']
  assert parts[1]['id'] == 'lookup_1'
  assert parts[2]['id'] == '2'


@pytest.mark.parametrize(
    'role,expected',
    [
        ('user', 'user'),
        ('model', 'assistant'),
        ('tool', ''),
        (None, ''),
    ],
)
def test_request_attributes_map_genai_roles_to_otel_roles(
    role: Optional[str], expected: str
):
  content = types.Content(role=role, parts=[types.Part(text='hi')])

  attributes = _request_attributes(
      LlmRequest(model='some-model', contents=[content])
  )

  assert attributes[GEN_AI_INPUT_MESSAGES] == [
      {'role': expected, 'parts': [{'content': 'hi', 'type': 'text'}]}
  ]


def test_request_attributes_flatten_system_instruction_to_parts():
  """System instructions are emitted as bare parts, with no role wrapper."""
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(system_instruction='Be terse.'),
  )

  attributes = _request_attributes(llm_request)

  assert attributes[GEN_AI_SYSTEM_INSTRUCTIONS] == [
      {'content': 'Be terse.', 'type': 'text'}
  ]


def test_request_attributes_describe_function_tools_with_parameters():
  """A declared function tool becomes a `function` definition with a schema."""
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name='get_weather',
                          description='Gets the weather.',
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  'city': types.Schema(type=types.Type.STRING)
                              },
                              required=['city'],
                          ),
                      )
                  ]
              )
          ]
      ),
  )

  attributes = _request_attributes(llm_request)

  assert attributes[GEN_AI_TOOL_DEFINITIONS] == [{
      'name': 'get_weather',
      'description': 'Gets the weather.',
      'parameters': {
          'type': 'OBJECT',
          'properties': {'city': {'type': 'STRING'}},
          'required': ['city'],
      },
      'type': 'function',
  }]


def test_request_attributes_describe_a_raw_mcp_tool():
  """A caller bypassing ADK's tool pipeline can hand genai a raw MCP tool.

  Recognizing it means asking `sys.modules` whether the MCP SDK is loaded, so
  the SDK's name has to be the one the dependencies seam resolves. Get that
  wrong and every MCP tool is reported as an unserializable one instead.
  """
  llm_request = LlmRequest(
      model='some-model',
      config=types.GenerateContentConfig(
          tools=[
              McpTool(
                  name='get_weather',
                  description='Gets the weather.',
                  inputSchema={
                      'type': 'object',
                      'properties': {'city': {'type': 'string'}},
                      'required': ['city'],
                  },
              )
          ]
      ),
  )

  attributes = _request_attributes(llm_request)

  assert attributes[GEN_AI_TOOL_DEFINITIONS] == [{
      'name': 'get_weather',
      'description': 'Gets the weather.',
      'parameters': {
          'type': 'object',
          'properties': {'city': {'type': 'string'}},
          'required': ['city'],
      },
      'type': 'function',
  }]


# ---------------------------------------------------------------------------
# set_operation_details_attributes_from_response
# ---------------------------------------------------------------------------


def test_response_attributes_split_between_details_and_common():
  """Messages go to the details mapping; finish reason and usage to common."""
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')]),
      finish_reason=types.FinishReason.STOP,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=20,
          cached_content_token_count=4,
      ),
  )

  details, common = _response_attributes(llm_response)

  assert details == {
      'gen_ai.output.messages': [{
          'role': 'assistant',
          'parts': [{'content': 'Response', 'type': 'text'}],
          'finish_reason': 'stop',
      }]
  }
  assert common == {
      'gen_ai.response.finish_reasons': ['stop'],
      'gen_ai.usage.input_tokens': 10,
      'gen_ai.usage.output_tokens': 20,
      'gen_ai.usage.cache_read.input_tokens': 4,
  }


def test_response_attributes_accumulate_output_messages_across_a_stream():
  """Keeping only the last chunk truncated the log to it."""
  details: dict = {}
  common: dict = {}

  for text in ('text ', 'response'):
    set_operation_details_attributes_from_response(
        LlmResponse(
            content=types.Content(role='model', parts=[types.Part(text=text)]),
            finish_reason=types.FinishReason.STOP,
        ),
        details,
        common,
    )

  assert details == {
      'gen_ai.output.messages': [
          {
              'role': 'assistant',
              'parts': [{'content': 'text ', 'type': 'text'}],
              'finish_reason': 'stop',
          },
          {
              'role': 'assistant',
              'parts': [{'content': 'response', 'type': 'text'}],
              'finish_reason': 'stop',
          },
      ]
  }


def test_streamed_chunks_are_reported_one_message_each():
  """Each chunk is its own message, as the OTel instrumentor reports a stream."""
  details: dict = {}
  common: dict = {}

  # Only the chunk that ends the turn reports why generation stopped.
  for text, finish_reason in (
      ('text ', types.FinishReason.FINISH_REASON_UNSPECIFIED),
      ('response', types.FinishReason.STOP),
  ):
    set_operation_details_attributes_from_response(
        LlmResponse(
            content=types.Content(role='model', parts=[types.Part(text=text)]),
            finish_reason=finish_reason,
        ),
        details,
        common,
    )

  assert [
      message['parts'][0]['content']
      for message in details['gen_ai.output.messages']
  ] == ['text ', 'response']
  assert common == {'gen_ai.response.finish_reasons': ['stop']}


def test_response_attributes_omit_output_messages_without_content():
  """An error-only response writes no output-message key at all."""
  llm_response = LlmResponse(
      error_code='UNAVAILABLE',
      finish_reason=types.FinishReason.OTHER,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=7
      ),
  )

  details, common = _response_attributes(llm_response)

  assert details == {}
  assert common == {
      GEN_AI_RESPONSE_FINISH_REASONS: ['error'],
      GEN_AI_USAGE_INPUT_TOKENS: 7,
  }


def test_response_attributes_omit_finish_reasons_but_keep_empty_message_field():
  """No finish reason drops the common key; the message field becomes ''."""
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')])
  )

  details, common = _response_attributes(llm_response)

  assert common == {}
  assert details[GEN_AI_OUTPUT_MESSAGES][0]['finish_reason'] == ''


@pytest.mark.parametrize(
    'finish_reason,expected',
    [
        (types.FinishReason.STOP, 'stop'),
        (types.FinishReason.MAX_TOKENS, 'length'),
        (types.FinishReason.OTHER, 'error'),
        (types.FinishReason.SAFETY, 'safety'),
    ],
)
def test_response_attributes_normalize_finish_reason(
    finish_reason: types.FinishReason, expected: str
):
  """genai finish reasons are mapped onto the OTel-allowed vocabulary."""
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')]),
      finish_reason=finish_reason,
  )

  details, common = _response_attributes(llm_response)

  assert common[GEN_AI_RESPONSE_FINISH_REASONS] == [expected]
  assert details[GEN_AI_OUTPUT_MESSAGES][0]['finish_reason'] == expected


def test_response_attributes_treat_unspecified_finish_reason_as_unreported():
  """The proto3 zero value means unreported, not failed."""
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')]),
      finish_reason=types.FinishReason.FINISH_REASON_UNSPECIFIED,
  )

  details, common = _response_attributes(llm_response)

  assert GEN_AI_RESPONSE_FINISH_REASONS not in common
  assert details[GEN_AI_OUTPUT_MESSAGES][0]['finish_reason'] == ''


def test_response_attributes_omit_token_usage_without_metadata():
  llm_response = LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text='Response')]),
      finish_reason=types.FinishReason.STOP,
  )

  _, common = _response_attributes(llm_response)

  assert common == {GEN_AI_RESPONSE_FINISH_REASONS: ['stop']}
  assert GEN_AI_USAGE_INPUT_TOKENS not in common
  assert GEN_AI_USAGE_OUTPUT_TOKENS not in common


# ---------------------------------------------------------------------------
# stable vs experimental divergence
# ---------------------------------------------------------------------------


def test_stable_and_experimental_encode_the_same_choice_differently():
  """The two variants disagree on finish-reason casing and on `index`.

  Stable `gen_ai.choice` reports the raw genai enum value and an explicit
  candidate index; the experimental output message reports the normalized
  OTel token and no index.
  """
  content = types.Content(role='model', parts=[types.Part(text='Response')])
  llm_response = LlmResponse(
      content=content, finish_reason=types.FinishReason.MAX_TOKENS
  )

  stable = choice_body(
      llm_response,
      TelemetryConfig(capture_message_content=ContentCapturingMode.EVENT_ONLY),
  )
  details, _ = _response_attributes(llm_response)
  experimental = details[GEN_AI_OUTPUT_MESSAGES][0]

  assert stable == {
      'content': content.model_dump(),
      'index': 0,
      'finish_reason': 'MAX_TOKENS',
  }
  assert experimental == {
      'role': 'assistant',
      'parts': [{'content': 'Response', 'type': 'text'}],
      'finish_reason': 'length',
  }


class _DumpingTool:
  """A pydantic-shaped tool whose dump uses the given schema key."""

  def __init__(self, schema_key: str):
    self._schema_key = schema_key

  def model_dump(self, *, exclude_none: bool = False) -> dict[str, object]:
    del exclude_none
    return {
        'name': 'mcp_tool',
        'description': 'A standalone mcp tool',
        self._schema_key: {
            'type': 'object',
            'properties': {'id': {'type': 'integer'}},
        },
    }


class TestModelDumpToolDefinitionSchemaKey:
  """The schema key an MCP tool dumps under changes with the SDK version.

  A dumped tool reaches telemetry through `_model_dump_to_tool_definition`.
  Reading the wrong key is not an error there: the tool is still reported, and
  reported with no parameters, so the loss shows up only as an emptier span.
  """

  @pytest.mark.parametrize(
      'schema_key', ['parameters', 'inputSchema', 'input_schema']
  )
  def test_parameters_are_read_under_every_spelling(self, schema_key):
    definition = _model_dump_to_tool_definition(_DumpingTool(schema_key))

    assert definition['parameters'] == {
        'type': 'object',
        'properties': {'id': {'type': 'integer'}},
    }

  def test_an_unknown_spelling_is_still_reported_without_parameters(self):
    """The fallback must stay lossy-but-alive, not raise.

    Telemetry is not worth failing a tool call over.
    """
    definition = _model_dump_to_tool_definition(_DumpingTool('schemaOfInput'))

    assert definition['name'] == 'mcp_tool'
    assert definition.get('parameters') is None
