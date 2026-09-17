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

from typing import Optional
from unittest import mock

from google.adk.integrations.model_armor import ModelArmorConfig
from google.adk.integrations.model_armor import ModelArmorPlugin
from google.adk.integrations.model_armor._plugin import _regional_endpoint
from google.adk.integrations.model_armor._plugin import _shared_template_location
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
