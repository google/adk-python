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

"""Private OpenAI Realtime connection used by :class:`OpenAILlm`."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator
from collections.abc import Mapping
import json
import logging
import re
from typing import Any
from typing import cast
from typing import Union

from google.genai import types
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.chat import ChatCompletionToolParam
from openai.types.realtime import session_update_event_param

from ...models.base_llm_connection import BaseLlmConnection
from ...models.llm_request import LlmRequest
from ...models.llm_response import LlmResponse
from ._openai_llm import _function_declaration_to_openai_tool

logger = logging.getLogger('google_adk.' + __name__)

_AUDIO_SAMPLE_RATE_HZ = 24_000
_AUDIO_MIME_TYPE = 'audio/pcm;rate=24000'
_REALTIME_TRANSCRIPTION_MODEL = 'gpt-live-transcribe'

_RealtimeInput = Union[
    types.Blob,
    types.ActivityStart,
    types.ActivityEnd,
    types.LiveClientRealtimeInput,
]


def _get_value(obj: object, key: str, default: Any = None) -> Any:
  if obj is None:
    return default
  if isinstance(obj, Mapping):
    return obj.get(key, default)
  return getattr(obj, key, default)


def _text_of(content: types.Content) -> str:
  return ''.join(part.text or '' for part in content.parts or [])


def _instruction_text(value: types.ContentUnion | None) -> str:
  if isinstance(value, str):
    return value
  if isinstance(value, types.Content):
    return _text_of(value)
  if isinstance(value, types.Part):
    return value.text or ''
  if isinstance(value, Mapping):
    return _instruction_text(types.Part(**value))
  if isinstance(value, list):
    return ''.join(
        _instruction_text(item)
        for item in value
        if isinstance(item, (str, types.Part, Mapping))
    )
  return ''


def _function_output(value: object) -> str:
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  return json.dumps(value, default=str, ensure_ascii=False)


def _function_arguments(value: str | None) -> dict[str, Any]:
  if not value:
    return {}
  try:
    result = json.loads(value)
  except json.JSONDecodeError:
    logger.warning('Failed to parse Realtime function arguments as JSON.')
    return {}
  return result if isinstance(result, dict) else {}


def _usage_metadata(
    usage: object,
) -> types.GenerateContentResponseUsageMetadata:
  input_details = _get_value(usage, 'input_token_details')
  return types.GenerateContentResponseUsageMetadata(
      prompt_token_count=_get_value(usage, 'input_tokens'),
      candidates_token_count=_get_value(usage, 'output_tokens'),
      total_token_count=_get_value(usage, 'total_tokens'),
      cached_content_token_count=_get_value(input_details, 'cached_tokens'),
  )


def _voice_name(config: types.LiveConnectConfig) -> str | None:
  speech = config.speech_config
  voice_config = speech.voice_config if speech else None
  prebuilt = voice_config.prebuilt_voice_config if voice_config else None
  return prebuilt.voice_name if prebuilt else None


def _transcription_languages(
    config: types.AudioTranscriptionConfig,
) -> list[str]:
  hints = config.language_hints
  languages = hints.language_codes if hints and hints.language_codes else None
  languages = languages or config.language_codes or []
  normalized = [
      language.strip().lower().split('-', 1)[0]
      for language in languages
      if language.strip()
  ]
  return list(dict.fromkeys(normalized))


def _input_transcription(
    config: types.LiveConnectConfig,
) -> dict[str, Any] | None:
  transcription_config = config.input_audio_transcription
  if transcription_config is None:
    return None
  result: dict[str, Any] = {'model': _REALTIME_TRANSCRIPTION_MODEL}
  languages = _transcription_languages(transcription_config)
  if languages:
    result['languages'] = languages
  if transcription_config.adaptation_phrases:
    result['keywords'] = transcription_config.adaptation_phrases
  return result


def _automatic_activity_detection(
    config: types.LiveConnectConfig,
) -> tuple[bool, dict[str, Any] | None]:
  realtime = config.realtime_input_config
  detection = realtime.automatic_activity_detection if realtime else None
  if detection and detection.disabled:
    return False, None

  result: dict[str, Any] = {
      'type': 'server_vad',
      'create_response': True,
      'interrupt_response': not (
          realtime
          and realtime.activity_handling
          == types.ActivityHandling.NO_INTERRUPTION
      ),
  }
  if detection and detection.prefix_padding_ms is not None:
    result['prefix_padding_ms'] = detection.prefix_padding_ms
  if detection and detection.silence_duration_ms is not None:
    result['silence_duration_ms'] = detection.silence_duration_ms
  return True, result


def _interruptions_allowed(config: types.LiveConnectConfig) -> bool:
  realtime = config.realtime_input_config
  return not (
      realtime
      and realtime.activity_handling == types.ActivityHandling.NO_INTERRUPTION
  )


def _realtime_tools(llm_request: LlmRequest) -> list[dict[str, Any]]:
  tools: list[dict[str, Any]] = []
  for tool in llm_request.config.tools or []:
    if not isinstance(tool, types.Tool):
      continue
    for declaration in tool.function_declarations or []:
      chat_tool: ChatCompletionToolParam = _function_declaration_to_openai_tool(
          declaration
      )
      function = chat_tool['function']
      realtime_tool: dict[str, Any] = {
          'type': 'function',
          'name': function['name'],
          'description': function.get('description', ''),
          'parameters': function.get(
              'parameters', {'type': 'object', 'properties': {}}
          ),
      }
      tools.append(realtime_tool)
  return tools


def _session_payload(
    llm_request: LlmRequest,
) -> tuple[dict[str, Any], bool, bool]:
  config = llm_request.live_connect_config
  instructions = _instruction_text(config.system_instruction)
  if not instructions:
    instructions = _instruction_text(llm_request.config.system_instruction)

  modalities = {
      str(getattr(modality, 'value', modality)).lower()
      for modality in config.response_modalities or []
  }
  if not modalities or modalities == {'audio'}:
    output_modalities = ['audio']
  elif modalities == {'text'}:
    output_modalities = ['text']
  else:
    raise ValueError(
        'OpenAI Realtime supports exactly one output modality: audio or text.'
    )

  automatic_detection, turn_detection = _automatic_activity_detection(config)
  audio_input: dict[str, Any] = {
      'format': {'type': 'audio/pcm', 'rate': _AUDIO_SAMPLE_RATE_HZ},
      'turn_detection': turn_detection,
  }
  transcription = _input_transcription(config)
  if transcription:
    audio_input['transcription'] = transcription

  audio_output: dict[str, Any] = {
      'format': {'type': 'audio/pcm', 'rate': _AUDIO_SAMPLE_RATE_HZ},
  }
  voice = _voice_name(config)
  if voice:
    audio_output['voice'] = voice

  session: dict[str, Any] = {
      'type': 'realtime',
      'output_modalities': output_modalities,
      'audio': {'input': audio_input, 'output': audio_output},
  }
  if instructions:
    session['instructions'] = instructions
  if config.max_output_tokens is not None:
    session['max_output_tokens'] = config.max_output_tokens
  tools = _realtime_tools(llm_request)
  if tools:
    session['tools'] = tools
    session['tool_choice'] = 'auto'
  return session, automatic_detection, _interruptions_allowed(config)


def _conversation_items(content: types.Content) -> list[dict[str, Any]]:
  items: list[dict[str, Any]] = []
  text = _text_of(content)
  role = 'assistant' if content.role in ('model', 'assistant') else 'user'
  if text:
    content_type = 'output_text' if role == 'assistant' else 'input_text'
    items.append({
        'type': 'message',
        'role': role,
        'content': [{'type': content_type, 'text': text}],
    })

  for part in content.parts or []:
    if part.function_call:
      call = part.function_call
      items.append({
          'type': 'function_call',
          'call_id': call.id or '',
          'name': call.name or '',
          'arguments': json.dumps(
              call.args or {}, default=str, ensure_ascii=False
          ),
      })
    elif part.function_response:
      response = part.function_response
      items.append({
          'type': 'function_call_output',
          'call_id': response.id or '',
          'output': _function_output(response.response),
      })
  return items


def _pcm16_24khz(blob: types.Blob) -> bytes:
  mime_type = (blob.mime_type or '').lower().replace(' ', '')
  if not mime_type.startswith('audio/pcm'):
    raise ValueError('OpenAI Realtime only supports PCM16 audio input.')
  rate = re.search(r'(?:^|;)rate=(\d+)(?:;|$)', mime_type)
  if not rate or int(rate.group(1)) != _AUDIO_SAMPLE_RATE_HZ:
    raise ValueError(
        'OpenAI Realtime audio input must declare a 24000 Hz sample rate; '
        f'use {_AUDIO_MIME_TYPE!r}.'
    )
  data = blob.data
  if not isinstance(data, bytes):
    raise ValueError('OpenAI Realtime audio input must contain bytes.')
  if len(data) % 2:
    raise ValueError('PCM16 audio input must contain complete 16-bit samples.')
  return data


def _plain_text_only(content: types.Content) -> bool:
  """Whether content can be safely merged with a partial text turn."""
  return all(
      set(part.model_dump(exclude_none=True)) <= {'text'}
      for part in content.parts or []
  )


def _output_text(item: object) -> str:
  text_parts: list[str] = []
  for content_part in _get_value(item, 'content', []) or []:
    if _get_value(content_part, 'type') == 'output_text':
      text = _get_value(content_part, 'text')
      if text:
        text_parts.append(str(text))
  return ''.join(text_parts)


def _output_index(event: object) -> int | None:
  value = _get_value(event, 'output_index')
  return (
      value if isinstance(value, int) and not isinstance(value, bool) else None
  )


class _OpenAIRealtimeLlmConnection(BaseLlmConnection):
  """Maps ADK's live connection contract onto the OpenAI Realtime SDK."""

  def __init__(
      self,
      session: AsyncRealtimeConnection,
      *,
      model_version: str,
  ) -> None:
    self._session = session
    self._model_version = model_version
    self._live_session_id: str | None = None
    self._automatic_detection = True
    self._interruptions_allowed = True
    self._text_output = False
    self._response_active = False
    self._interruption_announced = False
    self._announced_calls: set[str] = set()
    self._pending_calls: dict[str, object] = {}
    self._pending_text = ''
    self._pending_text_output_index: int | None = None
    self._partial_input_text = ''
    self._closed = False

  async def configure(self, llm_request: LlmRequest) -> None:
    """Applies ADK's live configuration to the connected session."""
    (
        session,
        self._automatic_detection,
        self._interruptions_allowed,
    ) = _session_payload(llm_request)
    self._text_output = session['output_modalities'] == ['text']
    # gpt-live-transcribe's ``languages`` and ``keywords`` fields can precede
    # the SDK's generated TypedDicts. The SDK still validates and serializes
    # this ordinary payload at the WebSocket boundary.
    await self._session.session.update(
        session=cast(session_update_event_param.Session, session)
    )

  async def send_history(self, history: list[types.Content]) -> None:
    last_content_sent = False
    for index, content in enumerate(history):
      items = _conversation_items(content)
      for item in items:
        await self._session.conversation.item.create(item=cast(Any, item))
      if index == len(history) - 1:
        last_content_sent = bool(items)
    if history and history[-1].role == 'user' and last_content_sent:
      await self._session.response.create()

  async def send_content(self, content: types.Content) -> None:
    await self._send_content(content)

  async def _send_content(
      self, content: types.Content, *, partial: bool = False
  ) -> None:
    if partial:
      if not _plain_text_only(content):
        raise ValueError(
            'Partial OpenAI Realtime content may only contain text parts.'
        )
      self._partial_input_text += _text_of(content)
      return

    items: list[dict[str, Any]] = []
    if self._partial_input_text:
      if _plain_text_only(content):
        content = types.Content(
            role=content.role or 'user',
            parts=[
                types.Part.from_text(
                    text=self._partial_input_text + _text_of(content)
                )
            ],
        )
      else:
        items.extend(
            _conversation_items(
                types.Content(
                    role='user',
                    parts=[types.Part.from_text(text=self._partial_input_text)],
                )
            )
        )
      self._partial_input_text = ''
    items.extend(_conversation_items(content))
    for item in items:
      await self._session.conversation.item.create(item=cast(Any, item))
    if items:
      await self._session.response.create()

  async def send_realtime(self, input: _RealtimeInput) -> None:
    if isinstance(input, types.Blob):
      audio = _pcm16_24khz(input)
      await self._session.input_audio_buffer.append(
          audio=base64.b64encode(audio).decode('ascii')
      )
      return
    if isinstance(input, types.ActivityStart):
      if self._response_active and self._interruptions_allowed:
        await self._session.response.cancel()
      if not self._automatic_detection:
        await self._session.input_audio_buffer.clear()
      return
    if isinstance(input, types.ActivityEnd):
      if not self._automatic_detection:
        await self._commit_audio_and_respond()
      return
    if isinstance(input, types.LiveClientRealtimeInput):
      if input.audio_stream_end:
        await self._commit_audio_and_respond()
        return
      logger.warning('Unary LiveClientRealtimeInput not fully supported yet.')
      return
    raise ValueError(f'Unsupported input type: {type(input)}')

  async def _commit_audio_and_respond(self) -> None:
    await self._session.input_audio_buffer.commit()
    await self._session.response.create()

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    async for event in self._session:
      event_type = str(_get_value(event, 'type', ''))

      if event_type in ('session.created', 'session.updated'):
        session = _get_value(event, 'session')
        self._live_session_id = _get_value(session, 'id') or (
            self._live_session_id
        )
        continue

      if event_type == 'response.created':
        self._response_active = True
        self._interruption_announced = False
        continue

      if event_type == 'response.output_audio.delta':
        delta = _get_value(event, 'delta')
        try:
          audio = base64.b64decode(delta, validate=True)
        except (TypeError, ValueError):
          logger.warning('Discarding an invalid Realtime audio delta.')
          continue
        yield self._response(
            content=types.Content(
                role='model',
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=audio, mime_type=_AUDIO_MIME_TYPE
                        )
                    )
                ],
            ),
        )
        continue

      if event_type == 'response.output_text.delta':
        delta = str(_get_value(event, 'delta') or '')
        if delta:
          self._pending_text += delta
          output_index = _output_index(event)
          if output_index is not None:
            self._pending_text_output_index = output_index
          yield self._response(
              content=types.Content(
                  role='model', parts=[types.Part.from_text(text=delta)]
              ),
              partial=True,
          )
        continue

      if event_type == 'response.output_audio_transcript.delta':
        text = str(_get_value(event, 'delta') or '')
        if text:
          yield self._response(
              output_transcription=types.Transcription(
                  text=text, finished=False
              ),
              partial=True,
          )
        continue

      if event_type == 'response.output_audio_transcript.done':
        text = str(_get_value(event, 'transcript') or '')
        yield self._response(
            output_transcription=types.Transcription(text=text, finished=True),
            partial=False,
        )
        continue

      if event_type == 'conversation.item.input_audio_transcription.delta':
        text = str(_get_value(event, 'delta') or '')
        if text:
          yield self._response(
              input_transcription=types.Transcription(
                  text=text, finished=False
              ),
              partial=True,
          )
        continue

      if event_type == 'conversation.item.input_audio_transcription.completed':
        text = str(_get_value(event, 'transcript') or '')
        yield self._response(
            input_transcription=types.Transcription(text=text, finished=True),
            partial=False,
        )
        continue

      if event_type == 'response.function_call_arguments.done':
        self._buffer_function_call(event)
        continue

      if event_type == 'input_audio_buffer.speech_started':
        interrupted = (
            self._response_active
            and self._interruptions_allowed
            and not self._interruption_announced
        )
        yield self._response(
            voice_activity=types.VoiceActivity(
                voice_activity_type=types.VoiceActivityType.ACTIVITY_START,
                audio_offset=f"{_get_value(event, 'audio_start_ms', 0)}ms",
            ),
        )
        if interrupted:
          # Text deltas are consolidated into an interrupted final Content at
          # response.done. For audio-only output, surface interruption now.
          if not self._text_output:
            self._interruption_announced = True
            yield self._response(interrupted=True)
        continue

      if event_type == 'input_audio_buffer.speech_stopped':
        yield self._response(
            voice_activity=types.VoiceActivity(
                voice_activity_type=types.VoiceActivityType.ACTIVITY_END,
                audio_offset=f"{_get_value(event, 'audio_end_ms', 0)}ms",
            )
        )
        continue

      if event_type == 'error':
        error = _get_value(event, 'error')
        code = str(
            _get_value(error, 'code')
            or _get_value(error, 'type')
            or 'realtime_error'
        )
        yield self._response(
            error_code=code,
            error_message=f'OpenAI Realtime error ({code})',
            finish_reason=types.FinishReason.OTHER,
        )
        continue

      if event_type == 'response.done':
        async for response in self._response_done(event):
          yield response
        return

    if not self._closed:
      yield self._response(
          go_away=types.LiveServerGoAway(time_left='0s'),
      )

  async def _response_done(
      self, event: object
  ) -> AsyncGenerator[LlmResponse, None]:
    self._response_active = False
    response = _get_value(event, 'response')
    status = str(_get_value(response, 'status') or 'completed')
    parts = self._final_response_parts(
        response, include_function_calls=status == 'completed'
    )
    self._pending_calls.clear()
    self._pending_text = ''
    self._pending_text_output_index = None

    interruption_emitted_with_content = False
    if parts:
      interruption_emitted_with_content = status == 'cancelled'
      yield self._response(
          content=types.Content(role='model', parts=parts),
          partial=False,
          interrupted=interruption_emitted_with_content or None,
      )

    interrupted = (
        status == 'cancelled'
        and not self._interruption_announced
        and not interruption_emitted_with_content
    )
    usage = _get_value(response, 'usage')
    status_details = _get_value(response, 'status_details')
    incomplete_reason = _get_value(status_details, 'reason')
    finish_reason = types.FinishReason.STOP
    if status == 'incomplete':
      finish_reason = (
          types.FinishReason.SAFETY
          if incomplete_reason == 'content_filter'
          else types.FinishReason.MAX_TOKENS
      )
    kwargs: dict[str, Any] = {
        'turn_complete': True,
        'interrupted': interrupted,
        'finish_reason': finish_reason,
    }
    if usage:
      kwargs['usage_metadata'] = _usage_metadata(usage)
    if status == 'failed':
      error = _get_value(status_details, 'error')
      kwargs.update(
          error_code=str(_get_value(error, 'code') or 'response_failed'),
          error_message='OpenAI Realtime response failed',
          finish_reason=types.FinishReason.OTHER,
      )
    yield self._response(**kwargs)
    self._interruption_announced = False

  def _final_response_parts(
      self, response: object, *, include_function_calls: bool
  ) -> list[types.Part]:
    """Builds one ordered final content from a Realtime response."""
    ordered_parts: list[tuple[int, int, types.Part]] = []
    seen_call_ids: set[str] = set()
    text_added = False
    output = list(_get_value(response, 'output', []) or [])

    for sequence, item in enumerate(output):
      item_type = _get_value(item, 'type')
      if item_type == 'message':
        text = _output_text(item)
        if text:
          ordered_parts.append(
              (sequence, sequence, types.Part.from_text(text=text))
          )
          text_added = True
      elif item_type == 'function_call' and include_function_calls:
        function_part = self._function_call_part(item)
        if function_part is not None:
          call_id, part = function_part
          if call_id in seen_call_ids:
            continue
          seen_call_ids.add(call_id)
          ordered_parts.append((sequence, sequence, part))

    fallback_position = len(output)
    if self._pending_text and not text_added:
      text_position = self._pending_text_output_index
      ordered_parts.append((
          text_position if text_position is not None else fallback_position,
          -1,
          types.Part.from_text(text=self._pending_text),
      ))

    if include_function_calls:
      for sequence, pending_call in enumerate(self._pending_calls.values()):
        function_part = self._function_call_part(pending_call)
        if function_part is None:
          continue
        call_id, part = function_part
        if call_id in seen_call_ids:
          continue
        call_position = _output_index(pending_call)
        ordered_parts.append((
            call_position
            if call_position is not None
            else fallback_position + sequence,
            sequence,
            part,
        ))
        seen_call_ids.add(call_id)

    self._announced_calls.update(seen_call_ids)
    ordered_parts.sort(key=lambda value: (value[0], value[1]))
    return [part for _, _, part in ordered_parts]

  def _buffer_function_call(self, event: object) -> None:
    call_id = str(_get_value(event, 'call_id') or _get_value(event, 'id') or '')
    if call_id and call_id not in self._announced_calls:
      self._pending_calls[call_id] = event

  def _function_call_part(self, event: object) -> tuple[str, types.Part] | None:
    call_id = str(_get_value(event, 'call_id') or _get_value(event, 'id') or '')
    if not call_id or call_id in self._announced_calls:
      return None
    part = types.Part(
        function_call=types.FunctionCall(
            id=call_id,
            name=str(_get_value(event, 'name') or ''),
            args=_function_arguments(_get_value(event, 'arguments')),
        )
    )
    return call_id, part

  def _response(self, **kwargs: Any) -> LlmResponse:
    return LlmResponse(
        model_version=self._model_version,
        live_session_id=self._live_session_id,
        **kwargs,
    )

  async def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    await self._session.close()
