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

from __future__ import annotations

import logging
from unittest import mock

from google.genai import types
from pydantic import BaseModel

# The repository may be tested locally with a google-genai version older than
# the source tree expects (run_config.py imports types.AvatarConfig at import
# time, so the shim must be installed before importing the OpenAI model below).
# The empty shim is installed process-wide and intentionally left in place: it
# is harmless, and other test modules (e.g. test_live_streaming_configs) that
# run later read ``types.AvatarConfig`` at runtime, so removing it on teardown
# would break them.
if not hasattr(types, 'AvatarConfig'):
  types.AvatarConfig = type('AvatarConfig', (BaseModel,), {})

from google.adk.integrations.openai._openai_common import OpenAIGenerateContentConfig
from google.adk.integrations.openai._openai_responses_llm import _content_to_response_input_items
from google.adk.integrations.openai._openai_responses_llm import _function_declaration_to_response_tool
from google.adk.integrations.openai._openai_responses_llm import _loads_json_object
from google.adk.integrations.openai._openai_responses_llm import _response_to_llm_response
from google.adk.integrations.openai._openai_responses_llm import _serialize_system_instruction
from google.adk.integrations.openai._openai_responses_llm import _tool_choice
from google.adk.integrations.openai._openai_responses_llm import AzureOpenAIResponsesLlm
from google.adk.integrations.openai._openai_responses_llm import OpenAIResponsesLlm
from google.adk.integrations.openai._openai_schema import enforce_strict_openai_schema
from google.adk.models.llm_request import LlmRequest
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from openai.types.responses import FunctionToolParam
from openai.types.responses import Response
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses import ResponseFunctionToolCallParam
from openai.types.responses import ResponseInputFileParam
from openai.types.responses import ResponseInputImageParam
from openai.types.responses import ResponseInputItemParam
from openai.types.responses import ResponseInputTextParam
from openai.types.responses import ResponseOutputMessage
from openai.types.responses import ResponseOutputText
from openai.types.responses import ResponseReasoningItem
from openai.types.responses import ResponseReasoningItemParam
from openai.types.responses import ResponseStreamEvent
from openai.types.responses import ResponseUsage
from openai.types.responses import ToolParam
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import InputTokensDetails
from openai.types.responses.response_usage import OutputTokensDetails
import pytest


class _FakeAsyncStream:

  def __init__(self, events: list[dict]):
    self._events = events

  def __aiter__(self):
    return self._iter()

  async def _iter(self):
    for event in self._events:
      yield event


class _CaptureResponses:

  def __init__(self, response):
    self.response = response
    self.kwargs = None

  async def create(self, **kwargs):
    self.kwargs = kwargs
    return self.response


class _CaptureClient:

  def __init__(self, response):
    self.responses = _CaptureResponses(response)


def test_openai_responses_package_exports_required_types():
  """The supported OpenAI SDK range exposes the Responses API types we use."""
  assert EasyInputMessageParam
  assert FunctionToolParam
  assert Response
  assert ResponseFunctionToolCall
  assert ResponseFunctionToolCallParam
  assert ResponseInputFileParam
  assert ResponseInputImageParam
  assert ResponseInputItemParam
  assert ResponseInputTextParam
  assert ResponseOutputMessage
  assert ResponseOutputText
  assert ResponseReasoningItem
  assert ResponseReasoningItemParam
  assert ResponseStreamEvent
  assert ResponseUsage
  assert ToolParam
  assert Summary
  assert InputTokensDetails
  assert OutputTokensDetails


def test_request_kwargs_use_responses_api_shape():
  """A standard (non-reasoning) chat model maps to the Responses shape."""
  llm = OpenAIResponsesLlm(model='gpt-4o')
  llm_request = LlmRequest(
      model='gpt-4o',
      previous_interaction_id='resp_previous',
      contents=[
          types.Content(
              role='user',
              parts=[types.Part.from_text(text='What is the weather?')],
          ),
          types.Content(
              role='tool',
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id='call_weather',
                          name='get_weather',
                          response={'temperature': '70 F'},
                      )
                  )
              ],
          ),
      ],
      config=types.GenerateContentConfig(
          system_instruction='You are concise.',
          temperature=0.2,
          top_p=0.9,
          max_output_tokens=128,
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name='get_weather',
                          description='Get weather',
                          parameters=types.Schema(
                              type=types.Type.OBJECT,
                              properties={
                                  'location': types.Schema(
                                      type=types.Type.STRING
                                  )
                              },
                              required=['location'],
                          ),
                      )
                  ]
              )
          ],
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['model'] == 'gpt-4o'
  assert kwargs['instructions'] == 'You are concise.'
  assert kwargs['previous_response_id'] == 'resp_previous'
  assert kwargs['stream'] is False
  # A non-reasoning model keeps temperature / top_p.
  assert kwargs['temperature'] == 0.2
  assert kwargs['top_p'] == 0.9
  assert kwargs['max_output_tokens'] == 128
  assert kwargs['input'] == [
      {
          'type': 'message',
          'role': 'user',
          'content': [{'type': 'input_text', 'text': 'What is the weather?'}],
      },
      {
          'type': 'function_call_output',
          'call_id': 'call_weather',
          'output': '{"temperature": "70 F"}',
      },
  ]
  assert kwargs['tools'] == [{
      'type': 'function',
      'name': 'get_weather',
      'description': 'Get weather',
      'parameters': {
          'type': 'object',
          'properties': {'location': {'type': 'string'}},
          'required': ['location'],
      },
      'strict': False,
  }]


def test_reasoning_request_kwargs_pass_reasoning_options_and_drop_sampling():
  """A reasoning model forwards reasoning/store/include and drops temp/top_p."""
  llm = OpenAIResponsesLlm(
      model='gpt-5',
      store=False,
      include=['reasoning.encrypted_content'],
      reasoning={'effort': 'medium'},
  )
  llm_request = LlmRequest(
      model='gpt-5',
      contents=[
          types.Content(
              role='user',
              parts=[types.Part.from_text(text='What is the weather?')],
          )
      ],
      config=types.GenerateContentConfig(
          temperature=0.2, top_p=0.9, max_output_tokens=128
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['model'] == 'gpt-5'
  assert kwargs['store'] is False
  assert kwargs['include'] == ['reasoning.encrypted_content']
  assert kwargs['reasoning'] == {'effort': 'medium'}
  assert kwargs['max_output_tokens'] == 128
  # Reasoning models reject non-default temperature/top_p, so they are dropped.
  assert 'temperature' not in kwargs
  assert 'top_p' not in kwargs


@pytest.mark.parametrize('model', ['gpt-5.6-sol', 'gpt-6-astra', 'o3-mini'])
def test_reasoning_model_drops_temperature_and_top_p(model, caplog):
  """Reasoning models reject non-default temperature/top_p; drop them."""
  llm = OpenAIResponsesLlm(model=model)
  llm_request = LlmRequest(
      model=model,
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
      config=types.GenerateContentConfig(temperature=0.2, top_p=0.9),
  )

  with caplog.at_level(logging.WARNING):
    kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert 'temperature' not in kwargs
  assert 'top_p' not in kwargs
  assert 'Ignoring temperature' in caplog.text
  assert 'Ignoring top_p' in caplog.text


@pytest.mark.parametrize('model', ['gpt-5.6-sol', 'gpt-6-astra', 'o3-mini'])
def test_reasoning_model_keeps_default_temperature(model):
  """Reasoning models accept the default temperature/top_p (1); keep them."""
  llm = OpenAIResponsesLlm(model=model)
  llm_request = LlmRequest(
      model=model,
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
      config=types.GenerateContentConfig(temperature=1, top_p=1),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['temperature'] == 1
  assert kwargs['top_p'] == 1


def test_llm_request_model_overrides_self_model_for_reasoning_detection():
  """llm_request.model overrides self.model; reasoning detection reads it.

  self.model is a reasoning model, but the request overrides it with a
  non-reasoning model. temperature/top_p must survive because detection reads
  the effective (overridden) model in kwargs, not self.model.
  """
  llm = OpenAIResponsesLlm(model='gpt-5', api_key='k')
  llm_request = LlmRequest(
      model='gpt-4o',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='hi')])
      ],
      config=types.GenerateContentConfig(temperature=0.2, top_p=0.9),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['model'] == 'gpt-4o'
  assert kwargs['temperature'] == 0.2
  assert kwargs['top_p'] == 0.9


def test_content_mapping_preserves_model_tool_calls_and_reasoning():
  """Model tool calls/text replay while synthetic reasoning is skipped."""
  function_call_part = types.Part.from_function_call(
      name='get_weather', args={'location': 'Paris'}
  )
  function_call_part.function_call.id = 'call_123'
  thought_part = types.Part(text='Need weather first.', thought=True)
  content = types.Content(
      role='model',
      parts=[thought_part, function_call_part, types.Part.from_text(text='Hi')],
  )

  items = _content_to_response_input_items(content)

  assert items == [
      {
          'type': 'function_call',
          'call_id': 'call_123',
          'name': 'get_weather',
          'arguments': '{"location": "Paris"}',
      },
      {
          'type': 'message',
          'role': 'assistant',
          'content': 'Hi',
      },
  ]


def test_content_mapping_preserves_reasoning_signature():
  """Replayed thoughts are skipped because synthetic IDs are invalid."""
  thought_part = types.Part(text='Need weather first.', thought=True)
  thought_part.thought_signature = b'encrypted_reasoning'
  redacted_part = types.Part(
      thought=True, thought_signature=b'redacted_reasoning'
  )
  content = types.Content(role='model', parts=[thought_part, redacted_part])

  items = _content_to_response_input_items(content)

  assert items == []


def test_content_mapping_sanitizes_function_call_ids_per_request():
  """Invalid IDs get stable fallbacks and missing IDs do not collide."""
  invalid_call = types.Part.from_function_call(name='tool', args={})
  invalid_call.function_call.id = 'invalid id!'
  invalid_response = types.Part(
      function_response=types.FunctionResponse(
          id='invalid id!', name='tool', response={'result': 'ok'}
      )
  )
  missing_call_1 = types.Part.from_function_call(name='tool', args={})
  missing_call_2 = types.Part.from_function_call(name='tool', args={})
  content = types.Content(
      role='model',
      parts=[invalid_call, invalid_response, missing_call_1, missing_call_2],
  )

  items = OpenAIResponsesLlm()._get_response_input(
      LlmRequest(contents=[content])
  )

  assert items[0]['call_id'] == 'call_adk_fallback_0'
  assert items[1]['call_id'] == 'call_adk_fallback_0'
  assert items[2]['call_id'] == 'call_adk_fallback_1'
  assert items[3]['call_id'] == 'call_adk_fallback_2'


def test_function_response_serializes_mcp_content_as_text():
  """MCP-style text content is flattened for function_call_output."""
  content = types.Content(
      role='tool',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  id='call_123',
                  name='tool',
                  response={
                      'content': [
                          {'type': 'text', 'text': 'first'},
                          {'type': 'text', 'text': 'second'},
                      ]
                  },
              )
          )
      ],
  )

  items = _content_to_response_input_items(content)

  assert items == [{
      'type': 'function_call_output',
      'call_id': 'call_123',
      'output': 'first\nsecond',
  }]


def test_image_and_file_parts_use_responses_content_types():
  """Image and file parts become Responses input_image/input_file content."""
  content = types.Content(
      role='user',
      parts=[
          types.Part(
              inline_data=types.Blob(data=b'image', mime_type='image/png')
          ),
          types.Part(
              inline_data=types.Blob(
                  data=b'hello', mime_type='text/plain', display_name='a.txt'
              )
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri='file-abc', mime_type='application/pdf'
              )
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri='https://example.com/doc.pdf',
                  mime_type='application/pdf',
              )
          ),
          types.Part(
              file_data=types.FileData(
                  file_uri='https://example.com/image.png',
                  mime_type='image/png',
              )
          ),
      ],
  )

  items = _content_to_response_input_items(content)

  assert items[0]['content'][0]['type'] == 'input_image'
  assert items[0]['content'][0]['image_url'].startswith(
      'data:image/png;base64,'
  )
  assert items[0]['content'][1] == {
      'type': 'input_file',
      'filename': 'a.txt',
      'file_data': 'data:text/plain;base64,aGVsbG8=',
  }
  assert items[0]['content'][2] == {
      'type': 'input_file',
      'file_id': 'file-abc',
  }
  assert items[0]['content'][3] == {
      'type': 'input_file',
      'file_url': 'https://example.com/doc.pdf',
  }
  assert items[0]['content'][4] == {
      'type': 'input_image',
      'detail': 'auto',
      'image_url': 'https://example.com/image.png',
  }


def test_assistant_media_is_filtered(caplog):
  """Assistant media is skipped instead of creating invalid input blocks."""
  content = types.Content(
      role='model',
      parts=[
          types.Part.from_text(text='before'),
          types.Part(
              inline_data=types.Blob(data=b'image', mime_type='image/png')
          ),
          types.Part.from_text(text='after'),
      ],
  )

  items = _content_to_response_input_items(content)

  assert items == [
      {'type': 'message', 'role': 'assistant', 'content': 'before'},
      {'type': 'message', 'role': 'assistant', 'content': 'after'},
  ]
  assert (
      'Media data is not supported in Responses assistant turns.' in caplog.text
  )


def test_code_parts_are_preserved_as_text():
  """Code parts use the same lossy text fallback as other adapters."""
  content = types.Content(
      role='user',
      parts=[
          types.Part(
              executable_code=types.ExecutableCode(
                  language='PYTHON', code='print(1)'
              )
          ),
          types.Part(
              code_execution_result=types.CodeExecutionResult(
                  output='1', outcome=types.Outcome.OUTCOME_OK
              )
          ),
      ],
  )

  items = _content_to_response_input_items(content)

  assert items[0]['content'] == [
      {'type': 'input_text', 'text': 'Code:```python\nprint(1)\n```'},
      {
          'type': 'input_text',
          'text': 'Execution Result:```code_output\n1\n```',
      },
  ]


def test_function_declaration_uses_responses_tool_shape():
  """Function declarations use top-level Responses function tool fields."""
  declaration = types.FunctionDeclaration(
      name='search',
      description='Search docs',
      parameters_json_schema={
          'type': 'OBJECT',
          'properties': {'query': {'type': 'STRING'}},
      },
  )

  tool = _function_declaration_to_response_tool(declaration)

  assert tool == {
      'type': 'function',
      'name': 'search',
      'description': 'Search docs',
      'parameters': {
          'type': 'object',
          'properties': {'query': {'type': 'string'}},
      },
      'strict': False,
  }


def test_function_declaration_to_response_tool_parameters_json_schema_ignores_parameters_required():
  declaration = types.FunctionDeclaration(
      name='custom_tool',
      description='Tool with both schemas',
      parameters=types.Schema(
          type=types.Type.OBJECT,
          required=['legacy_param'],
      ),
      parameters_json_schema={
          'type': 'object',
          'properties': {
              'query': {'type': 'string'},
          },
      },
  )
  tool = _function_declaration_to_response_tool(declaration)
  params = tool['parameters']
  assert 'required' not in params


def test_function_declaration_to_response_tool_prefers_parameters_json_schema_over_parameters():
  declaration = types.FunctionDeclaration(
      name='custom_tool',
      description='Tool with both schemas',
      parameters=types.Schema(
          type=types.Type.OBJECT,
          properties={
              'legacy_param': types.Schema(type=types.Type.STRING),
          },
          required=['legacy_param'],
      ),
      parameters_json_schema={
          'type': 'object',
          'properties': {
              'query': {'type': 'string'},
          },
          'required': ['query'],
      },
  )
  tool = _function_declaration_to_response_tool(declaration)
  params = tool['parameters']
  assert 'query' in params['properties']
  assert 'legacy_param' not in params['properties']
  assert params['required'] == ['query']


def test_structured_output_uses_responses_text_format():
  """ADK response schemas become Responses text.format json_schema."""

  class Answer(BaseModel):
    answer: str

  llm = OpenAIResponsesLlm(model='gpt-5')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(response_schema=Answer),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['text']['format']['type'] == 'json_schema'
  assert kwargs['text']['format']['name'] == 'Answer'
  assert kwargs['text']['format']['strict'] is True
  assert kwargs['text']['format']['schema']['additionalProperties'] is False
  assert kwargs['text']['format']['schema']['required'] == ['answer']


@pytest.mark.parametrize(
    ('model', 'effort'),
    [
        ('gpt-6-astra', 'max'),
        ('gpt-5.6-sol', 'xhigh'),
        ('gpt-5', 'minimal'),
        ('o3', 'high'),
    ],
)
def test_effort_maps_to_responses_reasoning(model, effort):
  """OpenAIGenerateContentConfig.effort maps to the Responses reasoning object."""
  llm = OpenAIResponsesLlm(model=model)
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=OpenAIGenerateContentConfig(effort=effort),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['reasoning'] == {'effort': effort, 'summary': 'concise'}


def test_effort_overrides_only_effort_key_of_static_reasoning():
  """A configured effort overrides only the effort of the model-level default."""
  llm = OpenAIResponsesLlm(model='gpt-5', reasoning={'effort': 'medium'})
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=OpenAIGenerateContentConfig(effort='high'),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  # effort is overridden; nothing else is injected (no forced 'concise').
  assert kwargs['reasoning'] == {'effort': 'high'}


def test_effort_preserves_static_reasoning_summary():
  """A configured effort keeps a model-level summary instead of clobbering it."""
  llm = OpenAIResponsesLlm(
      model='gpt-5', reasoning={'effort': 'medium', 'summary': 'detailed'}
  )
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=OpenAIGenerateContentConfig(effort='high'),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['reasoning'] == {'effort': 'high', 'summary': 'detailed'}


def test_effort_unsupported_tier_raises(monkeypatch):
  """An effort tier the model does not accept raises before the request."""
  # Tier validation only runs against the default OpenAI host, so clear any
  # ambient base-url overrides that would otherwise skip it and mask the raise.
  monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
  monkeypatch.delenv('AZURE_OPENAI_ENDPOINT', raising=False)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=OpenAIGenerateContentConfig(effort='xhigh'),
  )

  with pytest.raises(ValueError, match="not supported by model 'gpt-5'"):
    llm._get_response_create_kwargs(llm_request, stream=False)


def _unsupported_effort_request():
  return LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=OpenAIGenerateContentConfig(effort='xhigh'),
  )


def test_effort_unsupported_tier_passes_through_with_base_url():
  """A custom base_url skips tier validation; the effort passes through."""
  llm = OpenAIResponsesLlm(model='gpt-5', base_url='https://host.example/v1')

  kwargs = llm._get_response_create_kwargs(
      _unsupported_effort_request(), stream=False
  )

  assert kwargs['reasoning']['effort'] == 'xhigh'


def test_effort_unsupported_tier_passes_through_with_injected_client():
  """An injected client skips tier validation; the effort passes through."""
  llm = OpenAIResponsesLlm(model='gpt-5', client=AsyncOpenAI(api_key='x'))

  kwargs = llm._get_response_create_kwargs(
      _unsupported_effort_request(), stream=False
  )

  assert kwargs['reasoning']['effort'] == 'xhigh'


def test_effort_unsupported_tier_passes_through_with_openai_base_url_env(
    monkeypatch,
):
  """OPENAI_BASE_URL skips tier validation; the effort passes through."""
  monkeypatch.setenv('OPENAI_BASE_URL', 'https://host.example/v1')
  llm = OpenAIResponsesLlm(model='gpt-5')

  kwargs = llm._get_response_create_kwargs(
      _unsupported_effort_request(), stream=False
  )

  assert kwargs['reasoning']['effort'] == 'xhigh'


def test_effort_unsupported_tier_passes_through_with_azure_endpoint():
  """An Azure endpoint skips tier validation; the effort passes through."""
  llm = AzureOpenAIResponsesLlm(
      model='gpt-5', azure_endpoint='https://example.openai.azure.com/'
  )

  kwargs = llm._get_response_create_kwargs(
      _unsupported_effort_request(), stream=False
  )

  assert kwargs['reasoning']['effort'] == 'xhigh'


def test_no_effort_falls_back_to_static_reasoning():
  """Without a configured effort, the model-level reasoning default applies."""
  llm = OpenAIResponsesLlm(model='gpt-5', reasoning={'effort': 'medium'})
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['reasoning'] == {'effort': 'medium'}


def test_standard_thinking_config_warns_and_is_ignored(caplog):
  """Standard thinking_config is ignored (with a warning) for OpenAI models."""
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(
          thinking_config=types.ThinkingConfig(
              thinking_level=types.ThinkingLevel.HIGH
          )
      ),
  )

  with caplog.at_level(logging.WARNING):
    kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert 'not supported for OpenAI models' in caplog.text
  # Ignored -> no reasoning override (the None default is filtered out).
  assert 'reasoning' not in kwargs


def test_response_parsing_maps_text_reasoning_tool_calls_and_usage():
  """Responses output items become ADK text, thought, and function parts."""
  response = {
      'id': 'resp_123',
      'model': 'gpt-5',
      'status': 'completed',
      'usage': {
          'input_tokens': 11,
          'output_tokens': 7,
          'total_tokens': 18,
          'input_tokens_details': {'cached_tokens': 3},
          'output_tokens_details': {'reasoning_tokens': 4},
      },
      'output': [
          {
              'type': 'reasoning',
              'id': 'rs_1',
              'summary': [{'type': 'summary_text', 'text': 'Think.'}],
              'encrypted_content': 'encrypted',
          },
          {
              'type': 'message',
              'role': 'assistant',
              'content': [{'type': 'output_text', 'text': 'Calling a tool.'}],
          },
          {
              'type': 'function_call',
              'call_id': 'call_123',
              'name': 'get_weather',
              'arguments': '{"location": "Paris"}',
          },
      ],
  }

  llm_response = _response_to_llm_response(response)

  assert llm_response.interaction_id == 'resp_123'
  assert llm_response.model_version == 'gpt-5'
  assert llm_response.finish_reason == types.FinishReason.STOP
  assert llm_response.usage_metadata.prompt_token_count == 11
  assert llm_response.usage_metadata.candidates_token_count == 7
  assert llm_response.usage_metadata.total_token_count == 18
  assert llm_response.usage_metadata.cached_content_token_count == 3
  assert llm_response.usage_metadata.thoughts_token_count == 4
  assert llm_response.content.parts[0].thought is True
  assert llm_response.content.parts[0].text == 'Think.'
  assert llm_response.content.parts[0].thought_signature == b'encrypted'
  assert llm_response.content.parts[1].text == 'Calling a tool.'
  function_call = llm_response.content.parts[2].function_call
  assert function_call.id == 'call_123'
  assert function_call.name == 'get_weather'
  assert function_call.args == {'location': 'Paris'}
  assert llm_response.custom_metadata['openai_response']['reasoning'] == [
      {'encrypted_content': 'encrypted', 'id': 'rs_1'}
  ]


def test_response_parsing_accepts_openai_sdk_response_types():
  """OpenAI SDK Response objects are parsed through typed paths."""
  response = Response(
      id='resp_typed',
      created_at=1.0,
      model='gpt-5',
      object='response',
      output=[
          ResponseReasoningItem(
              id='rs_typed',
              type='reasoning',
              summary=[Summary(type='summary_text', text='Typed thought.')],
              encrypted_content='encrypted_typed',
          ),
          ResponseOutputMessage(
              id='msg_typed',
              type='message',
              role='assistant',
              status='completed',
              content=[
                  ResponseOutputText(
                      type='output_text', text='Typed hello.', annotations=[]
                  )
              ],
          ),
          ResponseFunctionToolCall(
              type='function_call',
              call_id='call_typed',
              name='get_weather',
              arguments='{"city": "Tokyo"}',
          ),
      ],
      parallel_tool_calls=True,
      tool_choice='auto',
      tools=[],
      status='completed',
      usage=ResponseUsage(
          input_tokens=3,
          input_tokens_details=InputTokensDetails(
              cached_tokens=1, cache_write_tokens=0
          ),
          output_tokens=5,
          output_tokens_details=OutputTokensDetails(reasoning_tokens=2),
          total_tokens=8,
      ),
  )

  llm_response = _response_to_llm_response(response)

  assert llm_response.interaction_id == 'resp_typed'
  assert llm_response.content.parts[0].thought is True
  assert llm_response.content.parts[0].text == 'Typed thought.'
  assert llm_response.content.parts[0].thought_signature == b'encrypted_typed'
  assert llm_response.content.parts[1].text == 'Typed hello.'
  assert llm_response.content.parts[2].function_call.id == 'call_typed'
  assert llm_response.content.parts[2].function_call.args == {'city': 'Tokyo'}
  assert llm_response.usage_metadata.total_token_count == 8
  assert llm_response.custom_metadata['openai_response']['reasoning'] == [
      {'encrypted_content': 'encrypted_typed', 'id': 'rs_typed'}
  ]


def test_response_parsing_preserves_redacted_reasoning():
  """Encrypted-only reasoning becomes a signature-only thought part."""
  response = {
      'id': 'resp_123',
      'model': 'gpt-5',
      'status': 'completed',
      'output': [
          {
              'type': 'reasoning',
              'id': 'rs_1',
              'encrypted_content': 'encrypted_only',
          },
      ],
  }

  llm_response = _response_to_llm_response(response)

  part = llm_response.content.parts[0]
  assert part.thought is True
  assert part.text is None
  assert part.thought_signature == b'encrypted_only'


@pytest.mark.asyncio
async def test_generate_content_async_calls_responses_create():
  """Non-streaming generation calls responses.create and parses the result."""
  response = {
      'id': 'resp_123',
      'model': 'gpt-5',
      'status': 'completed',
      'output': [{
          'type': 'message',
          'role': 'assistant',
          'content': [{'type': 'output_text', 'text': 'Hello'}],
      }],
  }
  client = _CaptureClient(response)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [item async for item in llm.generate_content_async(llm_request)]

  assert client.responses.kwargs['model'] == 'gpt-5'
  assert client.responses.kwargs['stream'] is False
  assert responses[0].content.parts[0].text == 'Hello'
  assert responses[0].interaction_id == 'resp_123'


@pytest.mark.asyncio
async def test_callable_api_key_wrapped_as_async_provider():
  """A callable api_key becomes the async provider AsyncOpenAI refreshes.

  A Vertex OAuth bearer token expires ~1h, so ``AsyncOpenAI`` awaits its api_key
  provider on every request rather than freezing the key at construction. A sync
  callable is adapted into that async provider; the callable is not consumed at
  construction time and is re-invoked on each await.
  """
  calls = {'n': 0}

  def key_provider() -> str:
    calls['n'] += 1
    return f'token-{calls["n"]}'

  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = OpenAIResponsesLlm(model='gpt-5', api_key=key_provider)._openai_client

  client_cls.assert_called_once()
  provider = client_cls.call_args.kwargs['api_key']
  # Not resolved eagerly at construction...
  assert calls['n'] == 0
  # ...and re-invoked (awaited) on each request, yielding a fresh token.
  assert await provider() == 'token-1'
  assert await provider() == 'token-2'
  assert calls['n'] == 2


@pytest.mark.asyncio
async def test_base_url_is_passed_to_client():
  """base_url is forwarded to the default AsyncOpenAI client."""
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = OpenAIResponsesLlm(
        model='gpt-5', api_key='secret', base_url='https://host.example/v1'
    )._openai_client
  client_cls.assert_called_once_with(
      api_key='secret', base_url='https://host.example/v1'
  )


@pytest.mark.asyncio
async def test_azure_falls_back_to_base_url_without_azure_endpoint():
  """AzureOpenAIResponsesLlm uses base_url when azure_endpoint is unset."""
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = AzureOpenAIResponsesLlm(
        model='gpt-5', api_key='secret', base_url='https://host.example/v1'
    )._openai_client
  client_cls.assert_called_once_with(
      api_key='secret', base_url='https://host.example/v1'
  )


@pytest.mark.asyncio
async def test_generate_content_async_can_skip_response_metadata():
  """Response metadata can be omitted from LlmResponse.custom_metadata."""
  response = {
      'id': 'resp_123',
      'model': 'gpt-5',
      'status': 'completed',
      'usage': {
          'input_tokens': 1,
          'output_tokens': 2,
          'total_tokens': 3,
      },
      'output': [{
          'type': 'message',
          'role': 'assistant',
          'content': [{'type': 'output_text', 'text': 'Hello'}],
      }],
  }
  client = _CaptureClient(response)
  llm = OpenAIResponsesLlm(model='gpt-5', include_response_metadata=False)
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [item async for item in llm.generate_content_async(llm_request)]

  assert responses[0].custom_metadata is None
  assert responses[0].usage_metadata.total_token_count == 3


@pytest.mark.asyncio
async def test_streaming_generation_yields_partials_and_final_response():
  """Streaming generation yields text/thought deltas and a final response."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.created',
          'response': {'id': 'resp_stream', 'model': 'gpt-5'},
      },
      {'type': 'response.reasoning_summary_text.delta', 'delta': 'Think'},
      {'type': 'response.output_text.delta', 'delta': 'Hel'},
      {'type': 'response.output_text.delta', 'delta': 'lo'},
      {
          'type': 'response.completed',
          'response': {
              'id': 'resp_stream',
              'model': 'gpt-5',
              'status': 'completed',
              'output': [
                  {
                      'type': 'reasoning',
                      'summary': [{'type': 'summary_text', 'text': 'Think'}],
                  },
                  {
                      'type': 'message',
                      'role': 'assistant',
                      'content': [{'type': 'output_text', 'text': 'Hello'}],
                  },
              ],
          },
      },
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert client.responses.kwargs['stream'] is True
  assert responses[0].partial is True
  assert responses[0].content.parts[0].thought is True
  assert responses[0].content.parts[0].text == 'Think'
  assert responses[1].partial is True
  assert responses[1].content is None
  assert responses[1].custom_metadata == {
      'openai_response': {
          'stream_event': {
              'type': 'response.output_text.delta',
              'reasoning_done': True,
          }
      }
  }
  assert responses[2].content.parts[0].text == 'Hel'
  assert responses[3].content.parts[0].text == 'lo'
  assert responses[4].partial is None
  assert responses[4].content.parts[0].thought is True
  assert responses[4].content.parts[1].text == 'Hello'


@pytest.mark.asyncio
async def test_streaming_generation_can_skip_response_metadata():
  """Metadata-only stream boundary events are omitted when metadata is off."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.created',
          'response': {'id': 'resp_stream', 'model': 'gpt-5'},
      },
      {'type': 'response.reasoning_summary_text.delta', 'delta': 'Think'},
      {'type': 'response.output_text.delta', 'delta': 'Hello'},
      {
          'type': 'response.completed',
          'response': {
              'id': 'resp_stream',
              'model': 'gpt-5',
              'status': 'completed',
              'output': [
                  {
                      'type': 'reasoning',
                      'summary': [{'type': 'summary_text', 'text': 'Think'}],
                  },
                  {
                      'type': 'message',
                      'role': 'assistant',
                      'content': [{'type': 'output_text', 'text': 'Hello'}],
                  },
              ],
          },
      },
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5', include_response_metadata=False)
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert [response.custom_metadata for response in responses] == [
      None,
      None,
      None,
  ]
  assert responses[0].content.parts[0].thought is True
  assert responses[1].content.parts[0].text == 'Hello'
  assert responses[2].partial is None


@pytest.mark.asyncio
async def test_streaming_generation_fallback_preserves_output_item_order():
  """Fallback final response preserves separate reasoning/text items."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.created',
          'response': {'id': 'resp_stream', 'model': 'gpt-5'},
      },
      {
          'type': 'response.output_item.added',
          'output_index': 0,
          'item': {'id': 'rs_1', 'type': 'reasoning', 'summary': []},
      },
      {
          'type': 'response.reasoning_summary_text.delta',
          'output_index': 0,
          'summary_index': 0,
          'delta': 'Think',
      },
      {
          'type': 'response.reasoning_summary_text.done',
          'output_index': 0,
          'summary_index': 0,
          'text': 'Think',
      },
      {
          'type': 'response.output_item.added',
          'output_index': 1,
          'item': {'id': 'msg_1', 'type': 'message', 'content': []},
      },
      {
          'type': 'response.output_text.delta',
          'output_index': 1,
          'content_index': 0,
          'delta': 'Hel',
      },
      {
          'type': 'response.output_text.delta',
          'output_index': 1,
          'content_index': 0,
          'delta': 'lo',
      },
      {
          'type': 'response.output_item.added',
          'output_index': 2,
          'item': {'id': 'rs_2', 'type': 'reasoning', 'summary': []},
      },
      {
          'type': 'response.reasoning_summary_text.delta',
          'output_index': 2,
          'summary_index': 0,
          'delta': 'Again',
      },
      {
          'type': 'response.output_item.added',
          'output_index': 3,
          'item': {'id': 'msg_2', 'type': 'message', 'content': []},
      },
      {
          'type': 'response.output_text.delta',
          'output_index': 3,
          'content_index': 0,
          'delta': 'Bye',
      },
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  final_response = responses[-1]
  assert final_response.partial is False
  parts = final_response.content.parts
  assert [(part.text, part.thought) for part in parts] == [
      ('Think', True),
      ('Hello', None),
      ('Again', True),
      ('Bye', None),
  ]
  boundaries = [
      response
      for response in responses
      if response.custom_metadata
      and response.custom_metadata['openai_response']['stream_event'][
          'reasoning_done'
      ]
  ]
  assert [
      boundary.custom_metadata['openai_response']['stream_event']['type']
      for boundary in boundaries
  ] == [
      'response.reasoning_summary_text.done',
      'response.output_item.added',
  ]


@pytest.mark.asyncio
async def test_streaming_generation_aggregates_function_call_without_completed_event():
  """Streaming function-call events become a final ADK function call."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.output_item.added',
          'output_index': 0,
          'item': {
              'type': 'function_call',
              'call_id': 'call_123',
              'name': 'get_weather',
              'arguments': '',
          },
      },
      {
          'type': 'response.function_call_arguments.delta',
          'output_index': 0,
          'delta': '{"location"',
      },
      {
          'type': 'response.function_call_arguments.delta',
          'output_index': 0,
          'delta': ': "Paris"}',
      },
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert len(responses) == 4
  partial_responses = responses[:-1]
  assert all(response.partial is True for response in partial_responses)
  assert [
      response.content.parts[0].function_call.partial_args
      for response in partial_responses
  ] == [
      None,
      None,
      [
          types.PartialArg(
              json_path='$.location', string_value='Paris', will_continue=False
          )
      ],
  ]
  assert [
      response.content.parts[0].function_call.id
      for response in partial_responses
  ] == ['call_123', 'call_123', 'call_123']
  assert all(
      response.content.parts[0].function_call.will_continue
      for response in partial_responses
  )

  final_response = responses[-1]
  assert final_response.partial is False
  assert final_response.finish_reason == types.FinishReason.STOP
  function_call = final_response.content.parts[0].function_call
  assert function_call.id == 'call_123'
  assert function_call.name == 'get_weather'
  assert function_call.args == {'location': 'Paris'}


@pytest.mark.asyncio
async def test_streaming_generation_uses_function_arguments_done_event():
  """Final function-call arguments can arrive in a done event."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.output_item.added',
          'output_index': 0,
          'item': {
              'type': 'function_call',
              'call_id': 'call_123',
              'name': 'get_weather',
              'arguments': '',
          },
      },
      {
          'type': 'response.function_call_arguments.done',
          'output_index': 0,
          'arguments': '{"location": "Paris"}',
      },
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert len(responses) == 3
  assert responses[0].partial is True
  assert responses[0].content.parts[0].function_call.partial_args is None
  assert responses[1].partial is True
  assert (
      responses[1].content.parts[0].function_call.partial_args[0].json_path
      == '$.location'
  )
  assert (
      responses[1].content.parts[0].function_call.partial_args[0].string_value
      == 'Paris'
  )

  final_response = responses[-1]
  assert final_response.partial is False
  function_call = final_response.content.parts[0].function_call
  assert function_call.id == 'call_123'
  assert function_call.args == {'location': 'Paris'}


@pytest.mark.asyncio
async def test_streaming_generation_failed_event_is_terminal():
  """A failed stream does not also emit a successful fallback final."""
  stream = _FakeAsyncStream([
      {'type': 'response.output_text.delta', 'delta': 'partial'},
      {'type': 'response.failed', 'response': {'id': 'resp_123'}},
  ])
  client = _CaptureClient(stream)
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert len(responses) == 2
  assert responses[0].partial is True
  assert responses[1].finish_reason == types.FinishReason.OTHER
  assert responses[1].error_code == types.FinishReason.OTHER


def test_azure_client_uses_openai_v1_base_url():
  """Azure model uses the Azure OpenAI /openai/v1 base URL."""
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    llm = AzureOpenAIResponsesLlm(
        model='deployment',
        azure_endpoint='https://example.openai.azure.com/',
        api_key='key',
    )

    _ = llm._openai_client

  client_cls.assert_called_once_with(
      api_key='key',
      base_url='https://example.openai.azure.com/openai/v1/',
  )


@pytest.mark.asyncio
async def test_azure_responses_inherits_partial_function_call_streaming():
  """Azure Responses uses the shared partial function-call stream handling."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.output_item.added',
          'output_index': 0,
          'item': {
              'type': 'function_call',
              'call_id': 'call_azure',
              'name': 'get_weather',
              'arguments': '',
          },
      },
      {
          'type': 'response.function_call_arguments.delta',
          'output_index': 0,
          'delta': '{"city": "Seattle"}',
      },
  ])
  client = _CaptureClient(stream)
  llm = AzureOpenAIResponsesLlm(
      model='deployment', azure_endpoint='https://example.openai.azure.com/'
  )
  llm.__dict__['_openai_client'] = client
  llm_request = LlmRequest(
      contents=[
          types.Content(
              role='user', parts=[types.Part.from_text(text='Weather?')]
          )
      ]
  )

  responses = [
      item
      async for item in llm.generate_content_async(llm_request, stream=True)
  ]

  assert len(responses) == 3
  assert responses[0].partial is True
  assert responses[1].partial is True
  assert (
      responses[1].content.parts[0].function_call.partial_args[0].json_path
      == '$.city'
  )
  assert (
      responses[1].content.parts[0].function_call.partial_args[0].string_value
      == 'Seattle'
  )
  assert responses[-1].partial is False
  assert responses[-1].content.parts[0].function_call.args == {
      'city': 'Seattle'
  }


def _user_request(**config_kwargs) -> LlmRequest:
  return LlmRequest(
      model='gpt-5',
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(**config_kwargs),
  )


def test_provided_client_is_used():
  """A pre-configured client is used verbatim instead of constructing one."""
  client = AsyncOpenAI(api_key='x')
  llm = OpenAIResponsesLlm(model='gpt-5', client=client)
  assert llm._openai_client is client


def test_default_client_built_with_resolved_api_key():
  """Without a client, AsyncOpenAI is constructed with the resolved key."""
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    llm = OpenAIResponsesLlm(model='gpt-5', api_key='secret')
    _ = llm._openai_client

  client_cls.assert_called_once_with(api_key='secret')


@pytest.mark.asyncio
async def test_api_key_callable_wrapped():
  """A sync api_key callable is wrapped in an async provider, not resolved."""
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = OpenAIResponsesLlm(
        model='gpt-5', api_key=lambda: 'dynamic'
    )._openai_client

  provider = client_cls.call_args.kwargs['api_key']
  assert await provider() == 'dynamic'


@pytest.mark.asyncio
async def test_async_api_key_callable_supported():
  """An async api_key provider is passed through for AsyncOpenAI to await."""

  async def _key() -> str:
    return 'k'

  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = OpenAIResponsesLlm(model='gpt-5', api_key=_key)._openai_client

  provider = client_cls.call_args.kwargs['api_key']
  assert await provider() == 'k'


def test_azure_api_key_env_fallback(monkeypatch):
  """Azure falls back to AZURE_OPENAI_API_KEY when no key is provided."""
  monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'env-key')
  with mock.patch(
      'google.adk.integrations.openai._openai_responses_llm.AsyncOpenAI'
  ) as client_cls:
    _ = AzureOpenAIResponsesLlm(
        model='deployment',
        azure_endpoint='https://example.openai.azure.com/',
    )._openai_client

  # A string env key passes through build_api_key unchanged.
  assert client_cls.call_args.kwargs['api_key'] == 'env-key'


def test_extra_request_args_override_and_merge_extra_body():
  """extra_request_args overrides kwargs but merges (not clobbers) extra_body."""
  llm = OpenAIResponsesLlm(
      model='gpt-5',
      extra_request_args={'temperature': 0.9, 'extra_body': {'foo': 'bar'}},
  )
  kwargs = llm._get_response_create_kwargs(
      _user_request(temperature=0.1, stop_sequences=['STOP']), stream=False
  )

  assert kwargs['temperature'] == 0.9
  assert kwargs['extra_body'] == {'stop': ['STOP'], 'foo': 'bar'}


def test_structured_output_schema_name_is_sanitized():
  """Schema names are sanitized to OpenAI's ^[a-zA-Z0-9_-]+$ requirement."""
  llm = OpenAIResponsesLlm(model='gpt-5')
  kwargs = llm._get_response_create_kwargs(
      _user_request(
          response_json_schema={
              'title': 'My Schema!',
              'type': 'object',
              'properties': {'x': {'type': 'integer'}},
          }
      ),
      stream=False,
  )

  assert kwargs['text']['format']['name'] == 'My_Schema_'


def test_structured_output_preserves_any_of_for_genai_schema():
  llm = OpenAIResponsesLlm(model='gpt-5')
  schema = types.Schema(
      type=types.Type.OBJECT,
      properties={
          'choice': types.Schema(
              any_of=[
                  types.Schema(type=types.Type.STRING),
                  types.Schema(type=types.Type.INTEGER),
              ]
          )
      },
  )
  kwargs = llm._get_response_create_kwargs(
      _user_request(response_schema=schema),
      stream=False,
  )
  schema_dict = kwargs['text']['format']['schema']
  assert 'any_of' not in schema_dict['properties']['choice']
  assert schema_dict['properties']['choice']['anyOf'] == [
      {'type': 'string'},
      {'type': 'integer'},
  ]


def test_enforce_strict_openai_schema_handles_nested_refs():
  """Strict transform recurses into $defs, properties, anyOf, and items."""
  schema = {
      'type': 'object',
      'properties': {
          'items': {'type': 'array', 'items': {'$ref': '#/$defs/Item'}},
          'choice': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
      },
      '$defs': {
          'Item': {'type': 'object', 'properties': {'n': {'type': 'integer'}}}
      },
  }

  enforce_strict_openai_schema(schema)

  assert schema['additionalProperties'] is False
  assert schema['required'] == ['choice', 'items']
  assert schema['$defs']['Item']['additionalProperties'] is False
  assert schema['$defs']['Item']['required'] == ['n']


@pytest.mark.parametrize(
    ('mode', 'expected'),
    [
        (types.FunctionCallingConfigMode.ANY, 'required'),
        (types.FunctionCallingConfigMode.NONE, 'none'),
        (types.FunctionCallingConfigMode.AUTO, 'auto'),
    ],
)
def test_tool_choice_maps_function_calling_mode(mode, expected):
  """function_calling_config.mode maps to the Responses tool_choice value."""
  config = types.GenerateContentConfig(
      tool_config=types.ToolConfig(
          function_calling_config=types.FunctionCallingConfig(mode=mode)
      )
  )
  assert _tool_choice(config) == expected


def test_tool_choice_is_omitted_without_tools():
  """tool_choice is not sent when there are no tools to choose from."""
  llm = OpenAIResponsesLlm(model='gpt-5', api_key='k')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(
          tool_config=types.ToolConfig(
              function_calling_config=types.FunctionCallingConfig(
                  mode=types.FunctionCallingConfigMode.ANY
              )
          )
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert 'tool_choice' not in kwargs
  assert 'tools' not in kwargs


def test_tool_choice_applied_when_tools_from_extra_request_args():
  """tool_choice is resolved even when tools arrive via extra_request_args."""
  llm = OpenAIResponsesLlm(
      model='gpt-5',
      api_key='k',
      extra_request_args={
          'tools': [{'type': 'function', 'name': 'a', 'parameters': {}}]
      },
  )
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(
          tool_config=types.ToolConfig(
              function_calling_config=types.FunctionCallingConfig(
                  mode=types.FunctionCallingConfigMode.ANY
              )
          )
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['tools']
  assert kwargs['tool_choice'] == 'required'


def test_tool_choice_applied_when_tools_from_config():
  """tool_choice is resolved when tools come from config.tools + tool_config."""
  llm = OpenAIResponsesLlm(model='gpt-5', api_key='k')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(name='a', description='A')
                  ]
              )
          ],
          tool_config=types.ToolConfig(
              function_calling_config=types.FunctionCallingConfig(
                  mode=types.FunctionCallingConfigMode.ANY
              )
          ),
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert kwargs['tools']
  assert kwargs['tool_choice'] == 'required'


def test_tool_without_function_declarations_is_skipped_with_warning(caplog):
  """A tool with no function declarations is skipped and logged, not sent."""
  llm = OpenAIResponsesLlm(model='gpt-5', api_key='k')
  llm_request = LlmRequest(
      contents=[
          types.Content(role='user', parts=[types.Part.from_text(text='Hi')])
      ],
      config=types.GenerateContentConfig(
          tools=[
              types.Tool(function_declarations=None),
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(name='a', description='A')
                  ]
              ),
          ]
      ),
  )

  kwargs = llm._get_response_create_kwargs(llm_request, stream=False)

  assert len(kwargs['tools']) == 1
  assert 'no function declarations' in caplog.text


def test_response_parsing_incomplete_max_tokens_sets_error():
  """An incomplete max-tokens response maps to MAX_TOKENS with an error."""
  response = {
      'id': 'resp_1',
      'model': 'gpt-5',
      'status': 'incomplete',
      'incomplete_details': {'reason': 'max_output_tokens'},
      'output': [],
  }

  llm_response = _response_to_llm_response(response)

  assert llm_response.finish_reason == types.FinishReason.MAX_TOKENS
  assert llm_response.error_code == types.FinishReason.MAX_TOKENS
  assert 'max_output_tokens' in llm_response.error_message


def test_response_parsing_failed_status_sets_error():
  """A failed response maps to OTHER and surfaces the error payload."""
  response = {
      'id': 'resp_1',
      'model': 'gpt-5',
      'status': 'failed',
      'error': {'message': 'boom'},
      'output': [],
  }

  llm_response = _response_to_llm_response(response)

  assert llm_response.finish_reason == types.FinishReason.OTHER
  assert llm_response.error_code == types.FinishReason.OTHER
  assert 'boom' in llm_response.error_message


def test_response_parsing_maps_refusal_to_prefixed_text():
  """Refusal content becomes prefixed text rather than being dropped."""
  response = {
      'id': 'resp_1',
      'model': 'gpt-5',
      'status': 'completed',
      'output': [{
          'type': 'message',
          'role': 'assistant',
          'content': [{'type': 'refusal', 'refusal': 'I cannot help.'}],
      }],
  }

  llm_response = _response_to_llm_response(response)

  assert llm_response.content.parts[0].text == 'OpenAI refusal: I cannot help.'


def test_loads_json_object_handles_malformed_arguments():
  """Malformed or non-object function arguments degrade to an empty dict."""
  assert _loads_json_object('not json') == {}
  assert _loads_json_object('[1, 2]') == {}
  assert _loads_json_object('') == {}
  assert _loads_json_object('{"a": 1}') == {'a': 1}


def test_code_parts_handle_missing_inner_fields():
  """Code parts with unset code/output do not crash the conversion."""
  content = types.Content(
      role='user',
      parts=[
          types.Part(executable_code=types.ExecutableCode(language='PYTHON')),
      ],
  )

  items = _content_to_response_input_items(content)

  assert items[0]['content'][0]['text'] == 'Code:```python\n\n```'


@pytest.mark.asyncio
async def test_streaming_incomplete_event_sets_max_tokens():
  """A streamed incomplete response yields a MAX_TOKENS final response."""
  stream = _FakeAsyncStream([
      {'type': 'response.output_text.delta', 'delta': 'Hi'},
      {
          'type': 'response.incomplete',
          'response': {
              'id': 'resp_stream',
              'model': 'gpt-5',
              'status': 'incomplete',
              'incomplete_details': {'reason': 'max_output_tokens'},
              'output': [{
                  'type': 'message',
                  'role': 'assistant',
                  'content': [{'type': 'output_text', 'text': 'Hi'}],
              }],
          },
      },
  ])
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = _CaptureClient(stream)

  responses = [
      item
      async for item in llm.generate_content_async(_user_request(), stream=True)
  ]

  assert responses[-1].finish_reason == types.FinishReason.MAX_TOKENS


@pytest.mark.asyncio
async def test_streaming_output_item_done_uses_done_item_text():
  """A done output item without a completed response feeds the fallback final."""
  stream = _FakeAsyncStream([
      {
          'type': 'response.output_item.added',
          'output_index': 0,
          'item': {'type': 'message', 'content': []},
      },
      {
          'type': 'response.output_item.done',
          'output_index': 0,
          'item': {
              'type': 'message',
              'role': 'assistant',
              'content': [{'type': 'output_text', 'text': 'Done text'}],
          },
      },
  ])
  llm = OpenAIResponsesLlm(model='gpt-5')
  llm.__dict__['_openai_client'] = _CaptureClient(stream)

  responses = [
      item
      async for item in llm.generate_content_async(_user_request(), stream=True)
  ]

  assert responses[-1].content.parts[0].text == 'Done text'


def test_serialize_system_instruction_part_shaped_mapping():
  """A Part-shaped mapping serializes to its text."""
  assert _serialize_system_instruction({'text': 'Be concise.'}) == 'Be concise.'


def test_serialize_system_instruction_content_shaped_mapping():
  """A Content-shaped mapping is serialized instead of raising ValidationError.

  Previously the Mapping branch did types.Part(**mapping), which raised an
  uncaught pydantic ValidationError on a {'role': ..., 'parts': [...]} dict.
  """
  mapping = {
      'role': 'system',
      'parts': [{'text': 'Be '}, {'text': 'concise.'}],
  }
  assert _serialize_system_instruction(mapping) == 'Be concise.'


def test_serialize_system_instruction_unparseable_mapping_returns_none():
  """A mapping that fits neither Part nor Content is dropped, not raised."""
  assert _serialize_system_instruction({'not_a_field': 123}) is None
