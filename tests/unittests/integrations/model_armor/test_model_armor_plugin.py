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

"""Tests for ModelArmorPlugin."""

from __future__ import annotations

import json
from typing import Any
from typing import Optional
from unittest import mock

from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.integrations.model_armor import ModelArmorPlugin
from google.adk.integrations.model_armor._config import _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE
from google.adk.integrations.model_armor._plugin import _MAX_TOOL_OUTPUT_CHARS
from google.adk.integrations.model_armor._plugin import _regional_endpoint
from google.adk.integrations.model_armor._plugin import _shared_template_location
from google.adk.integrations.model_armor._plugin import _TOOL_OUTPUT_CHUNK_OVERLAP_CHARS
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.auth.credentials import AnonymousCredentials
from google.cloud import modelarmor_v1
from google.genai import types
import pytest

_PROMPT_TEMPLATE_PATH = (
    'projects/test-project/locations/us-central1/templates/test-prompt'
)
_RESPONSE_TEMPLATE_PATH = (
    'projects/test-project/locations/us-central1/templates/test-response'
)


def _sanitization_result(
    *,
    match: bool = False,
    invocation_result=modelarmor_v1.InvocationResult.SUCCESS,
):
  return modelarmor_v1.SanitizationResult(
      filter_match_state=(
          modelarmor_v1.FilterMatchState.MATCH_FOUND
          if match
          else modelarmor_v1.FilterMatchState.NO_MATCH_FOUND
      ),
      invocation_result=invocation_result,
  )


def _sdk_client(*, result=None, raises: bool = False) -> mock.Mock:
  """A Model Armor SDK client answering both directions with ``result``."""
  result = _sanitization_result() if result is None else result
  client = mock.Mock()
  client.sanitize_user_prompt = mock.AsyncMock(
      return_value=modelarmor_v1.SanitizeUserPromptResponse(
          sanitization_result=result
      )
  )
  client.sanitize_model_response = mock.AsyncMock(
      return_value=modelarmor_v1.SanitizeModelResponseResponse(
          sanitization_result=result
      )
  )
  if raises:
    unreachable = RuntimeError('model armor is unreachable')
    client.sanitize_user_prompt.side_effect = unreachable
    client.sanitize_model_response.side_effect = unreachable
  return client


def _screened(client: mock.Mock) -> list[tuple[str, str]]:
  """The ``(direction, text)`` of every call that reached the SDK client."""
  screened = []
  for name, _, kwargs in client.mock_calls:
    if name == 'sanitize_user_prompt':
      screened.append(('input', kwargs['request'].user_prompt_data.text))
    elif name == 'sanitize_model_response':
      screened.append(('output', kwargs['request'].model_response_data.text))
  return screened


def _config(**overrides) -> ModelArmorConfig:
  defaults = dict(
      prompt_template_name=_PROMPT_TEMPLATE_PATH,
      response_template_name=_RESPONSE_TEMPLATE_PATH,
      input_blocked_message='input blocked',
      output_blocked_message='output blocked',
  )
  defaults.update(overrides)
  return ModelArmorConfig(**defaults)


def _plugin(*, result=None, raises: bool = False, **config_overrides):
  """Returns a ``(plugin, client)`` pair sharing one fake SDK client."""
  client = _sdk_client(result=result, raises=raises)
  plugin = ModelArmorPlugin(config=_config(**config_overrides), client=client)
  return plugin, client


def _user_request(text: str) -> LlmRequest:
  return LlmRequest(
      contents=[types.Content(role='user', parts=[types.Part(text=text)])]
  )


def _text_response(text: str) -> LlmResponse:
  """Model output as content parts, the way a unary turn carries it."""
  return LlmResponse(
      content=types.Content(role='model', parts=[types.Part(text=text)])
  )


def _transcription_response(text: str) -> LlmResponse:
  """Model output as a transcription, the way a live turn carries it."""
  return LlmResponse(
      output_transcription=types.Transcription(text=text, finished=False)
  )


async def _screen_input(plugin, llm_request) -> Optional[LlmResponse]:
  return await plugin.before_model_callback(
      callback_context=mock.Mock(), llm_request=llm_request
  )


async def _screen_output(plugin, llm_response) -> Optional[LlmResponse]:
  return await plugin.after_model_callback(
      callback_context=mock.Mock(), llm_response=llm_response
  )


# --- Input screening --------------------------------------------------------


@pytest.mark.asyncio
async def test_matched_input_is_replaced_with_the_blocked_message():
  """User input that matches the prompt template is blocked."""
  plugin, client = _plugin(result=_sanitization_result(match=True))

  result = await _screen_input(plugin, _user_request('bad input'))

  assert result.content.parts[0].text == _config().input_blocked_message
  assert result.custom_metadata['model_armor_blocked'] is True
  assert _screened(client) == [('input', 'bad input')]


@pytest.mark.asyncio
async def test_clean_input_passes_through():
  """User input that doesn't match the prompt template is allowed."""
  plugin, client = _plugin()

  result = await _screen_input(plugin, _user_request('hello'))

  assert result is None
  assert _screened(client) == [('input', 'hello')]


@pytest.mark.asyncio
async def test_thought_parts_are_left_out_of_the_screened_text():
  """Thoughts are model reasoning, which ADK hides from context by default."""
  plugin, client = _plugin()
  request = LlmRequest(
      contents=[
          types.Content(
              role='user',
              parts=[
                  types.Part(text='reasoning about the answer', thought=True),
                  types.Part(text='the visible question'),
              ],
          )
      ]
  )

  await _screen_input(plugin, request)

  assert _screened(client) == [('input', 'the visible question')]


# --- Output screening -------------------------------------------------------


@pytest.mark.asyncio
async def test_matched_output_content_is_replaced_with_the_blocked_message():
  """Unary model output arrives as content parts."""
  plugin, client = _plugin(result=_sanitization_result(match=True))

  result = await _screen_output(plugin, _text_response('harmful output'))

  assert result.content.parts[0].text == _config().output_blocked_message
  assert _screened(client) == [('output', 'harmful output')]


@pytest.mark.asyncio
async def test_matched_output_transcription_is_replaced():
  """Live model output carries no text parts, only a transcription."""
  plugin, client = _plugin(result=_sanitization_result(match=True))

  result = await _screen_output(plugin, _transcription_response('a secret'))

  assert result.content.parts[0].text == _config().output_blocked_message
  assert _screened(client) == [('output', 'a secret')]


@pytest.mark.asyncio
async def test_clean_output_passes_through():
  """Model output that doesn't match the response template is allowed."""
  plugin, client = _plugin()

  result = await _screen_output(plugin, _transcription_response('all clear'))

  assert result is None
  assert _screened(client) == [('output', 'all clear')]


# --- Templates opt each direction in ----------------------------------------


@pytest.mark.asyncio
async def test_input_is_not_screened_without_a_prompt_template():
  """If no prompt template is configured, input screening is skipped."""
  plugin, client = _plugin(
      result=_sanitization_result(match=True), prompt_template_name=None
  )

  result = await _screen_input(plugin, _user_request('bad input'))

  assert result is None
  assert _screened(client) == []


@pytest.mark.asyncio
async def test_output_is_not_screened_without_a_response_template():
  """If no response template is configured, output screening is skipped."""
  plugin, client = _plugin(
      result=_sanitization_result(match=True), response_template_name=None
  )

  result = await _screen_output(plugin, _text_response('harmful output'))

  assert result is None
  assert _screened(client) == []


# --- Nothing to screen passes through ---------------------------------------


@pytest.mark.asyncio
async def test_request_without_text_is_not_screened():
  """If there's no text, the request is not screened."""
  plugin, client = _plugin(result=_sanitization_result(match=True))

  result = await _screen_input(plugin, LlmRequest())

  assert result is None
  assert _screened(client) == []


@pytest.mark.asyncio
async def test_empty_response_is_not_screened():
  """If there's no text or output transcription, the response is not screened."""
  plugin, client = _plugin(result=_sanitization_result(match=True))

  result = await _screen_output(plugin, LlmResponse())

  assert result is None
  assert _screened(client) == []


# --- Model Armor call failures ----------------------------------------


_SCREENING_FAILURE = [
    pytest.param({'raises': True}, id='call_failed'),
    pytest.param(
        {
            'result': _sanitization_result(
                invocation_result=modelarmor_v1.InvocationResult.FAILURE
            )
        },
        id='failure_invocation',
    ),
    pytest.param(
        {
            'result': _sanitization_result(
                invocation_result=modelarmor_v1.InvocationResult.PARTIAL
            )
        },
        id='partial_invocation',
    ),
    pytest.param(
        {
            'result': _sanitization_result(
                invocation_result=(
                    modelarmor_v1.InvocationResult.INVOCATION_RESULT_UNSPECIFIED
                )
            )
        },
        id='unspecified_invocation',
    ),
]


@pytest.mark.parametrize('screening', _SCREENING_FAILURE)
@pytest.mark.asyncio
async def test_screening_failure_blocks_by_default(screening):
  """By default, a screening failure blocks, with that direction's message."""
  plugin, client = _plugin(**screening)

  blocked_in = await _screen_input(plugin, _user_request('hello'))
  blocked_out = await _screen_output(plugin, _text_response('hi there'))

  assert blocked_in.content.parts[0].text == _config().input_blocked_message
  assert blocked_out.content.parts[0].text == _config().output_blocked_message
  assert _screened(client) == [('input', 'hello'), ('output', 'hi there')]


@pytest.mark.parametrize('screening', _SCREENING_FAILURE)
@pytest.mark.asyncio
async def test_screening_failure_passes_through_when_configured(screening):
  """If configured, a Model Armor screening failure passes through."""
  plugin, _ = _plugin(block_on_screening_failure=False, **screening)

  result = await _screen_input(plugin, _user_request('hello'))

  assert result is None


@pytest.mark.asyncio
async def test_fully_screened_clean_content_passes_through():
  """Only SUCCESS means every configured filter actually ran."""
  plugin, _ = _plugin(
      result=_sanitization_result(
          invocation_result=modelarmor_v1.InvocationResult.SUCCESS
      ),
      block_on_screening_failure=True,
  )

  result = await _screen_output(plugin, _transcription_response('all clear'))

  assert result is None


# --- Regional endpoint ------------------------------------------------------


def test_regional_endpoint_format():
  """The host pattern Model Armor serves regional traffic on."""
  assert (
      _regional_endpoint('us-central1')
      == 'modelarmor.us-central1.rep.googleapis.com'
  )


@pytest.mark.asyncio
async def test_regional_endpoint_comes_from_the_template_location():
  """Model Armor is regional, so the template's own location picks the host.

  Async only for the event loop: the ``grpc.aio`` channel binds to it at
  construction, so a sync version fails on a worker with no current loop.
  """
  location = 'europe-west1'
  config = _config(
      prompt_template_name=(
          f'projects/test-project/locations/{location}/templates/eu-prompt'
      ),
      response_template_name=(
          f'projects/test-project/locations/{location}/templates/eu-response'
      ),
  )
  plugin = ModelArmorPlugin(config=config, credentials=AnonymousCredentials())

  try:
    assert plugin.client.api_endpoint == _regional_endpoint(location)
  finally:
    await plugin.close()


def test_templates_in_different_regions_are_rejected():
  """One client serves one endpoint, so both templates must share a region."""
  with pytest.raises(ValueError, match='same location'):
    _shared_template_location(
        'projects/test-project/locations/us-central1/templates/us-prompt',
        'projects/test-project/locations/europe-west1/templates/eu-response',
    )


def test_short_template_name_is_rejected():
  """Only full resource names carry the location the endpoint is built from."""
  with pytest.raises(ValueError, match='full resource names'):
    _shared_template_location('test-prompt-template')


# --- The SDK interface ------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_user_prompt_sends_the_prompt_template_and_text():
  """The plugin passes the prompt text and template name to the SDK."""
  plugin, client = _plugin()

  await plugin._sanitize_user_prompt('hello', _PROMPT_TEMPLATE_PATH)

  request = client.sanitize_user_prompt.call_args.kwargs['request']
  assert isinstance(request, modelarmor_v1.SanitizeUserPromptRequest)
  assert request.name == _PROMPT_TEMPLATE_PATH
  assert request.user_prompt_data.text == 'hello'


@pytest.mark.asyncio
async def test_sanitize_model_response_sends_the_response_template_and_text():
  """The plugin passes the response text and template name to the SDK."""
  plugin, client = _plugin()

  await plugin._sanitize_model_response('hi there', _RESPONSE_TEMPLATE_PATH)

  request = client.sanitize_model_response.call_args.kwargs['request']
  assert isinstance(request, modelarmor_v1.SanitizeModelResponseRequest)
  assert request.name == _RESPONSE_TEMPLATE_PATH
  assert request.model_response_data.text == 'hi there'


@pytest.mark.asyncio
async def test_sanitize_user_prompt_returns_the_sanitization_result():
  """The plugin unwraps the response and returns the result unchanged."""
  plugin, _ = _plugin(result=_sanitization_result(match=True))

  result = await plugin._sanitize_user_prompt('hello', _PROMPT_TEMPLATE_PATH)

  assert isinstance(result, modelarmor_v1.SanitizationResult)
  assert result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND


@pytest.mark.asyncio
async def test_sanitize_model_response_returns_the_sanitization_result():
  """The plugin unwraps the response and returns the result unchanged."""
  plugin, _ = _plugin(result=_sanitization_result(match=True))

  result = await plugin._sanitize_model_response(
      'hi there', _RESPONSE_TEMPLATE_PATH
  )

  assert isinstance(result, modelarmor_v1.SanitizationResult)
  assert result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND


# --- Shutdown ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_only_closes_own_client():
  """A supplied SDK client belongs to its caller, so close() must leave it."""
  supplied = mock.Mock()
  supplied.transport.close = mock.AsyncMock()

  await ModelArmorPlugin(config=_config(), client=supplied).close()

  supplied.transport.close.assert_not_awaited()

  plugin = ModelArmorPlugin(
      config=_config(), credentials=AnonymousCredentials()
  )
  built_client = plugin.client  # Opens the channel the plugin owns.
  built_client.transport.close = mock.AsyncMock()

  await plugin.close()

  built_client.transport.close.assert_awaited_once()


# --- Tool output screening --------------------------------------------------

_TOOL_OUTPUT_TEMPLATE_PATH = (
    'projects/test-project/locations/us-central1/templates/test-tool-output'
)


def _tool_plugin(*, result=None, raises: bool = False, **config_overrides):
  """Returns a (plugin, client) pair configured for tool output screening."""
  client = _sdk_client(result=result, raises=raises)
  config = ModelArmorConfig(
      tool_output_template_name=_TOOL_OUTPUT_TEMPLATE_PATH,
      **config_overrides,
  )
  plugin = ModelArmorPlugin(config=config, client=client)
  return plugin, client


async def _screen_tool_output(plugin, result: Any):
  return await plugin.after_tool_callback(
      tool=mock.Mock(),
      tool_args={},
      tool_context=mock.Mock(),
      result=result,
  )


@pytest.mark.asyncio
async def test_matched_tool_output_is_blocked():
  """Tool output matching the template is replaced with the blocked message."""
  plugin, client = _tool_plugin(result=_sanitization_result(match=True))

  result = await _screen_tool_output(plugin, {'result': 'malicious content'})

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}
  assert _screened(client) == [('input', '{"result": "malicious content"}')]


@pytest.mark.asyncio
async def test_clean_tool_output_passes_through():
  """Tool output that does not match the template returns None."""
  plugin, client = _tool_plugin()

  result = await _screen_tool_output(plugin, {'result': 'safe content'})

  assert result is None
  assert _screened(client) == [('input', '{"result": "safe content"}')]


@pytest.mark.asyncio
async def test_tool_output_not_screened_without_template():
  """If tool_output_template_name is unset, after_tool_callback is a no-op."""
  client = _sdk_client(result=_sanitization_result(match=True))
  plugin = ModelArmorPlugin(
      config=_config(),  # no tool_output_template_name
      client=client,
  )

  result = await _screen_tool_output(plugin, {'result': 'anything'})

  assert result is None
  assert _screened(client) == []


@pytest.mark.asyncio
async def test_empty_tool_result_is_not_screened():
  """An empty tool result is skipped without calling Model Armor."""
  plugin, client = _tool_plugin()

  assert await _screen_tool_output(plugin, {}) is None
  assert await _screen_tool_output(plugin, []) is None
  assert await _screen_tool_output(plugin, '') is None
  assert await _screen_tool_output(plugin, None) is None
  assert _screened(client) == []


@pytest.mark.asyncio
async def test_tool_output_json_serialization_screens_whole_dict():
  """All fields in a result dict are serialized and screened, preventing bypass."""
  plugin, client = _tool_plugin()

  await _screen_tool_output(plugin, {'foo': 'bar', 'baz': 123})

  screened = _screened(client)
  assert len(screened) == 1
  assert screened[0][0] == 'input'

  parsed = json.loads(screened[0][1])
  assert parsed == {'foo': 'bar', 'baz': 123}


@pytest.mark.asyncio
async def test_tool_output_screening_failure_blocks_by_default():
  """A screening failure blocks tool output when block_on_screening_failure=True."""
  plugin, _ = _tool_plugin(raises=True, block_on_screening_failure=True)

  result = await _screen_tool_output(plugin, {'result': 'content'})

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}


@pytest.mark.asyncio
async def test_tool_output_screening_failure_passes_when_configured():
  """A screening failure passes through when block_on_screening_failure=False."""
  plugin, _ = _tool_plugin(raises=True, block_on_screening_failure=False)

  result = await _screen_tool_output(plugin, {'result': 'content'})

  assert result is None


@pytest.mark.parametrize('screening', _SCREENING_FAILURE)
@pytest.mark.asyncio
async def test_tool_output_non_success_invocation_blocks_by_default(screening):
  """Non-SUCCESS invocation results block tool output by default."""
  plugin, _ = _tool_plugin(**screening)

  result = await _screen_tool_output(plugin, {'result': 'content'})

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}


def test_tool_output_template_location_validated():
  """tool_output_template_name must share location with other templates."""
  config = ModelArmorConfig(
      prompt_template_name=(
          'projects/test-project/locations/us-central1/templates/prompt'
      ),
      tool_output_template_name=(
          'projects/test-project/locations/europe-west1/templates/tool'
      ),
  )

  with pytest.raises(ValueError, match='same location'):
    ModelArmorPlugin(config=config)


@pytest.mark.asyncio
async def test_tool_output_string_result_is_screened():
  """Tool output returned as a plain string is screened without crashing."""
  plugin, client = _tool_plugin(result=_sanitization_result(match=True))

  result = await _screen_tool_output(plugin, 'malicious content')

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}
  assert _screened(client) == [('input', 'malicious content')]


@pytest.mark.asyncio
async def test_tool_output_list_result_is_screened():
  """Tool output returned as a list is serialized and screened."""
  plugin, client = _tool_plugin(result=_sanitization_result(match=True))

  result = await _screen_tool_output(plugin, ['item1', 'malicious content'])

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}
  assert _screened(client) == [('input', '["item1", "malicious content"]')]


@pytest.mark.asyncio
async def test_tool_output_multi_field_dict_screens_all_fields():
  """A tool result dict containing 'result' alongside other fields must screen all fields."""
  plugin, client = _tool_plugin()

  await _screen_tool_output(
      plugin,
      {'result': 'benign output', 'injected_payload': 'malicious instruction'},
  )

  screened = _screened(client)
  assert len(screened) == 1
  assert 'injected_payload' in screened[0][1]
  assert 'malicious instruction' in screened[0][1]


@pytest.mark.asyncio
async def test_non_ascii_tool_output_is_not_escaped():
  """Non-ASCII characters in tool output reach Model Armor unescaped."""
  plugin, client = _tool_plugin()

  await _screen_tool_output(plugin, {'result': 'instructions précédentes'})

  assert _screened(client) == [
      ('input', '{"result": "instructions précédentes"}')
  ]


@pytest.mark.asyncio
async def test_oversized_tool_output_is_screened_in_chunks():
  """Oversized tool output is screened in overlapping 65,536-char chunks so boundary-spanning payloads are caught."""
  assert _MAX_TOOL_OUTPUT_CHARS == 65_536
  plugin, client = _tool_plugin()
  client.sanitize_user_prompt = mock.AsyncMock(
      side_effect=[
          modelarmor_v1.SanitizeUserPromptResponse(
              sanitization_result=_sanitization_result(match=False)
          ),
          modelarmor_v1.SanitizeUserPromptResponse(
              sanitization_result=_sanitization_result(match=True)
          ),
      ]
  )
  oversized = 'a' * (_MAX_TOOL_OUTPUT_CHARS - 5) + 'malicious tail'

  result = await _screen_tool_output(plugin, oversized)

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}
  assert _screened(client) == [
      ('input', 'a' * (_MAX_TOOL_OUTPUT_CHARS - 5) + 'malic'),
      (
          'input',
          'a' * (_TOOL_OUTPUT_CHUNK_OVERLAP_CHARS - 5) + 'malicious tail',
      ),
  ]

  clean_plugin, clean_client = _tool_plugin()
  assert (
      await _screen_tool_output(clean_plugin, 'a' * _MAX_TOOL_OUTPUT_CHARS)
      is None
  )
  assert _screened(clean_client) == [('input', 'a' * _MAX_TOOL_OUTPUT_CHARS)]


@pytest.mark.parametrize(
    'first_chunk_effect',
    [
        pytest.param(
            RuntimeError('model armor is unreachable'),
            id='call_failed',
        ),
        pytest.param(
            modelarmor_v1.SanitizeUserPromptResponse(
                sanitization_result=_sanitization_result(
                    match=False,
                    invocation_result=modelarmor_v1.InvocationResult.PARTIAL,
                )
            ),
            id='partial_invocation',
        ),
    ],
)
@pytest.mark.asyncio
async def test_oversized_tool_output_fail_open_chunk_continues_to_match(
    first_chunk_effect,
):
  """With block_on_screening_failure=False, a failed first chunk continues to screen the next chunk."""
  plugin, client = _tool_plugin(block_on_screening_failure=False)
  client.sanitize_user_prompt = mock.AsyncMock(
      side_effect=[
          first_chunk_effect,
          modelarmor_v1.SanitizeUserPromptResponse(
              sanitization_result=_sanitization_result(match=True)
          ),
      ]
  )
  oversized = 'a' * (_MAX_TOOL_OUTPUT_CHARS - 5) + 'malicious tail'

  result = await _screen_tool_output(plugin, oversized)

  assert result == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}
  assert _screened(client) == [
      ('input', 'a' * (_MAX_TOOL_OUTPUT_CHARS - 5) + 'malic'),
      (
          'input',
          'a' * (_TOOL_OUTPUT_CHUNK_OVERLAP_CHARS - 5) + 'malicious tail',
      ),
  ]


@pytest.mark.asyncio
async def test_partial_invocation_with_match_blocks_when_fail_open():
  """A PARTIAL result that still reports MATCH_FOUND blocks across input, output, and tool output even when block_on_screening_failure=False."""
  plugin, _ = _plugin(
      result=_sanitization_result(
          match=True,
          invocation_result=modelarmor_v1.InvocationResult.PARTIAL,
      ),
      tool_output_template_name=_TOOL_OUTPUT_TEMPLATE_PATH,
      block_on_screening_failure=False,
  )

  blocked_in = await _screen_input(plugin, _user_request('bad input'))
  blocked_out = await _screen_output(plugin, _text_response('harmful output'))
  blocked_tool = await _screen_tool_output(
      plugin, {'result': 'malicious content'}
  )

  assert blocked_in.content.parts[0].text == _config().input_blocked_message
  assert blocked_out.content.parts[0].text == _config().output_blocked_message
  assert blocked_tool == {'error': _DEFAULT_TOOL_OUTPUT_BLOCKED_MESSAGE}


@pytest.mark.asyncio
async def test_tool_output_bytes_and_media_parts_are_not_screened():
  """Raw bytes and non-text media Parts in tool output are skipped without calling Model Armor."""
  plugin, client = _tool_plugin(result=_sanitization_result(match=True))
  media_part = types.Part.from_bytes(
      data=b'\x89PNG\r\n\x1a\n', mime_type='image/png'
  )
  file_part = types.Part(
      file_data=types.FileData(
          file_uri='gs://bucket/chart.png', mime_type='image/png'
      )
  )

  assert await _screen_tool_output(plugin, b'\x89PNG\r\n\x1a\n') is None
  assert (
      await _screen_tool_output(plugin, {'image': b'\x89PNG\r\n\x1a\n'}) is None
  )
  assert await _screen_tool_output(plugin, media_part) is None
  assert await _screen_tool_output(plugin, file_part) is None
  assert await _screen_tool_output(plugin, {'chart': media_part}) is None
  assert await _screen_tool_output(plugin, [media_part, file_part]) is None
  assert _screened(client) == []


@pytest.mark.asyncio
async def test_tool_output_media_alongside_text_screens_only_text():
  """When a tool result mixes media/bytes with text, only the text is screened."""
  plugin, client = _tool_plugin()
  media_part = types.Part.from_bytes(
      data=b'\x89PNG\r\n\x1a\n', mime_type='image/png'
  )

  await _screen_tool_output(
      plugin,
      {
          'chart': media_part,
          'raw_bytes': b'\x89PNG\r\n\x1a\n',
          'summary': 'up 3%',
      },
  )
  await _screen_tool_output(plugin, types.Part(text='part text'))

  assert _screened(client) == [
      ('input', '{"summary": "up 3%"}'),
      ('input', 'part text'),
  ]
