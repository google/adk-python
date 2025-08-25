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

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import AsyncGenerator
from typing import Optional

from google.genai import types
import websockets

from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)


class OpenAILlmConnection(BaseLlmConnection):
  """OpenAI Realtime WebSocket connection implementing BaseLlmConnection.

  This provides the minimal bridging needed for ADK live flows:
  - send_history: forwards user text history as conversation items
  - send_content: forwards new user text or function responses and starts a turn
  - send_realtime: appends audio chunks and commits on activity end
  - receive: yields LlmResponse with audio bytes, text aggregation, tool calls,
    transcription deltas, and turn_complete when done
  """

  def __init__(
      self,
      *,
      websocket: websockets.asyncio.client.ClientConnection,
      model_name: str,
      tool_param_names: dict[str, set[str]] | None = None,
  ):
    self._ws = websocket
    self._model_name = model_name
    self._text_buffer = ''
    self._output_transcript_buffer = ''
    self._input_transcript_buffer = ''
    self._closed = False
    self._tool_param_names = tool_param_names or {}
    # Track pending function calls where arguments stream via deltas
    self._pending_func_calls: dict[str, dict] = {}
    # Note: trimming helpers removed in revert

  async def _send(self, event: dict):
    try:
      await self._ws.send(json.dumps(event))
    except Exception as e:
      logger.error('Failed to send event to OpenAI Realtime: %s', e)
      raise

  # ----------------------
  # BaseLlmConnection API
  # ----------------------

  async def send_history(self, history: list[types.Content]):
    """Send conversation history (text and function outputs) to the session.

    Strategy: only forward user turns and function outputs to prime context;
    then trigger response if last item was a user turn.
    """
    if not history:
      return

    last_role: Optional[str] = None
    for content in history:
      last_role = content.role
      if not content.parts:
        continue
      # Function responses are represented as user role content in ADK
      if content.parts[0].function_response:
        for part in content.parts:
          if not part.function_response:
            continue
          call_id = part.function_response.id
          output = json.dumps(part.function_response.response)
          await self._send({
              'type': 'conversation.item.create',
              'item': {
                  'type': 'function_call_output',
                  'call_id': call_id,
                  'output': output,
              },
          })
      else:
        # Aggregate text parts into one user message
        texts = []
        for p in content.parts:
          if p.text:
            texts.append({'type': 'input_text', 'text': p.text})
        if texts:
          await self._send({
              'type': 'conversation.item.create',
              'item': {
                  'type': 'message',
                  'role': 'user',
                  'content': texts,
              },
          })

    if last_role == 'user':
      await self._send({'type': 'response.create'})

  async def send_content(self, content: types.Content):
    """Send user text or function responses and start a response."""
    assert content.parts
    # Function responses
    if content.parts[0].function_response:
      for part in content.parts:
        if not part.function_response:
          continue
        call_id = part.function_response.id
        output = json.dumps(part.function_response.response)
        await self._send({
            'type': 'conversation.item.create',
            'item': {
                'type': 'function_call_output',
                'call_id': call_id,
                'output': output,
            },
        })
      await self._send({'type': 'response.create'})
      return

    # Plain text
    texts = []
    for p in content.parts:
      if p.text:
        texts.append({'type': 'input_text', 'text': p.text})
    if texts:
      await self._send({
          'type': 'conversation.item.create',
          'item': {
              'type': 'message',
              'role': 'user',
              'content': texts,
          },
      })
      await self._send({'type': 'response.create'})

  async def send_realtime(self, input):
    """Send realtime input: Blob audio or activity markers."""
    # Activity markers map to commit signals in OpenAI Realtime
    if not isinstance(input, types.Blob):
      raise ValueError('Unsupported realtime input type: %s' % type(input))
    audio_b64 = base64.b64encode(input.data).decode('utf-8')
    await self._send({'type': 'input_audio_buffer.append', 'audio': audio_b64})

  def _build_full_text_response(self, text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role='model', parts=[types.Part.from_text(text=text)]
        )
    )

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receive server events and yield mapped LlmResponse objects."""
    try:
      async for message in self._ws:
        try:
          event = json.loads(message)
        except Exception as e:
          logger.error('Invalid JSON from OpenAI Realtime: %s', e)
          continue

        etype = event.get('type')
        for resp in self._dispatch_event(etype, event):
          yield resp
    except websockets.exceptions.ConnectionClosedOK:
      return
    except websockets.exceptions.ConnectionClosed as e:
      logger.error('OpenAI Realtime connection closed unexpectedly: %s', e)
      raise

  async def close(self):
    if self._closed:
      return
    try:
      await self._ws.close()
    finally:
      self._closed = True

  def _normalize_args(self, tool_name: str, args: dict) -> dict:
    """Return arguments as-is; rely on tool schema to guide the model.

    Any vendor-specific argument mapping should be provided via RunConfig-driven
    tool schemas, not hardcoded here.
    """
    return args

  # ----------------------
  # Event dispatch helpers
  # ----------------------

  def _dispatch_event(
      self, etype: Optional[str], event: dict
  ) -> list[LlmResponse]:
    if not etype:
      return []
    if etype == 'response.audio.delta':
      return self._handle_audio_delta(event)
    if etype in ('response.text.delta', 'response.output_text.delta'):
      return self._handle_text_delta(event)
    if etype == 'response.audio_transcript.delta':
      return self._handle_output_transcript_delta(event)
    # item.created tracking removed in revert
    if etype == 'conversation.item.input_audio_transcription.delta':
      return self._handle_input_transcript_delta(event)
    if etype == 'conversation.item.input_audio_transcription.completed':
      return self._handle_input_transcript_completed(event)
    if etype == 'input_audio_buffer.speech_started':
      return self._handle_speech_started()
    if etype in (
        'conversation.item.truncated',
        'input_audio_buffer.timeout_triggered',
    ):
      return self._handle_truncated_or_timeout()
    if etype == 'response.output_item.added':
      return self._handle_output_item_added(event)
    if etype == 'response.function_call_arguments.delta':
      return self._handle_function_args_delta(event)
    if etype == 'response.function_call_arguments.done':
      return self._handle_function_args_done(event)
    if etype == 'response.output_item.done':
      return self._handle_output_item_done(event)
    if etype == 'response.done':
      return self._handle_response_done(event)
    if etype == 'error':
      return self._handle_error(event)
    return []

  def _handle_audio_delta(self, event: dict) -> list[LlmResponse]:
    delta_b64 = event.get('delta', '')
    if not delta_b64:
      return []
    data = base64.b64decode(delta_b64)
    return [
        LlmResponse(
            content=types.Content(
                role='model',
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=data, mime_type='audio/pcm')
                    )
                ],
            ),
            partial=True,
        )
    ]

  def _handle_text_delta(self, event: dict) -> list[LlmResponse]:
    delta = event.get('delta') or ''
    self._text_buffer += delta
    return []

  def _handle_output_transcript_delta(self, event: dict) -> list[LlmResponse]:
    delta = event.get('delta') or ''
    # Stream as partial deltas and buffer for final flush
    self._output_transcript_buffer += delta
    return [
        LlmResponse(
            output_transcription=types.Transcription(text=delta),
            partial=True,
        )
    ]

  def _handle_input_transcript_delta(self, event: dict) -> list[LlmResponse]:
    delta = event.get('delta') or ''
    self._input_transcript_buffer += delta
    return [
        LlmResponse(
            input_transcription=types.Transcription(text=delta),
            partial=True,
        )
    ]

  # item.created handler removed in revert

  def _handle_input_transcript_completed(
      self, event: dict
  ) -> list[LlmResponse]:
    transcript = event.get('transcript') or ''
    # Ensure buffer aligns with final transcript when provider sends completed
    if (
        transcript
        and self._input_transcript_buffer
        and transcript != self._input_transcript_buffer
    ):
      # Reset to final authoritative transcript
      self._input_transcript_buffer = transcript
    else:
      # If no explicit transcript present, use buffered value
      transcript = transcript or self._input_transcript_buffer
    self._input_transcript_buffer = ''
    # Emit as standard user text content so it is stored in session history
    return [
        LlmResponse(
            input_transcription=types.Transcription(text=transcript),
            content=types.Content(
                role='user', parts=[types.Part.from_text(text=transcript)]
            ),
        )
    ]

  def _handle_speech_started(self) -> list[LlmResponse]:
    responses: list[LlmResponse] = []
    if self._text_buffer:
      responses.append(self._build_full_text_response(self._text_buffer))
      self._text_buffer = ''
    # Mark meta-only event as partial to avoid persisting NIL events
    responses.append(LlmResponse(interrupted=True, partial=True))
    return responses

  def _handle_truncated_or_timeout(self) -> list[LlmResponse]:
    responses: list[LlmResponse] = []
    if self._text_buffer:
      responses.append(self._build_full_text_response(self._text_buffer))
      self._text_buffer = ''
    # Mark meta-only event as partial to avoid persisting NIL events
    responses.append(LlmResponse(interrupted=True, partial=True))
    return responses

  def _handle_output_item_added(self, event: dict) -> list[LlmResponse]:
    item = event.get('item', {})
    if item.get('type') != 'function_call':
      return []
    name = item.get('name') or ''
    args_str = item.get('arguments') or '{}'
    try:
      args = json.loads(args_str)
    except Exception:
      args = {}
    item_id = item.get('id') or item.get('call_id') or ''
    self._pending_func_calls[item_id] = {
        'name': name,
        'args_buffer': (
            args_str if isinstance(args_str, str) else json.dumps(args)
        ),
    }
    # Defer emission until done
    return []

  def _handle_function_args_delta(self, event: dict) -> list[LlmResponse]:
    delta = event.get('delta', '')
    item_id = event.get('item_id') or event.get('id') or ''
    if item_id in self._pending_func_calls:
      self._pending_func_calls[item_id]['args_buffer'] += delta
    return []

  def _handle_function_args_done(self, event: dict) -> list[LlmResponse]:
    item_id = event.get('item_id') or event.get('id') or ''
    args_str = event.get('arguments', '')
    if item_id in self._pending_func_calls and isinstance(args_str, str):
      self._pending_func_calls[item_id]['args_buffer'] = args_str
    return []

  def _handle_output_item_done(self, event: dict) -> list[LlmResponse]:
    item = event.get('item', {})
    if item.get('type') != 'function_call':
      return []
    item_id = item.get('id') or item.get('call_id') or ''
    pending = self._pending_func_calls.pop(item_id, None)
    if not pending:
      return []
    try:
      args = json.loads(pending['args_buffer'] or '{}')
    except Exception:
      args = {}
    func_call = types.FunctionCall(
        name=pending['name'], args=self._normalize_args(pending['name'], args)
    )
    func_call.id = item.get('call_id') or item.get('id')
    return [
        LlmResponse(
            content=types.Content(
                role='model', parts=[types.Part(function_call=func_call)]
            )
        )
    ]

  def _handle_response_done(
      self, event: dict | None = None
  ) -> list[LlmResponse]:
    responses: list[LlmResponse] = []
    usage_meta = None
    try:
      if event and isinstance(event, dict):
        usage_raw = event.get('response', {}).get('usage')
        if isinstance(usage_raw, dict):
          # Try common shapes
          prompt = (
              usage_raw.get('input_tokens')
              or usage_raw.get('prompt_tokens')
              or usage_raw.get('prompt_token_count')
          )
          completion = (
              usage_raw.get('output_tokens')
              or usage_raw.get('completion_tokens')
              or usage_raw.get('candidates_token_count')
          )
          total = usage_raw.get('total_tokens') or usage_raw.get(
              'total_token_count'
          )

          # Fall back to details structure used by some Realtime payloads
          if prompt is None:
            details_in = usage_raw.get('input_token_details') or {}
            prompt = (details_in.get('audio_tokens') or 0) + (
                details_in.get('text_tokens') or 0
            )
          if completion is None:
            details_out = usage_raw.get('output_token_details') or {}
            completion = (details_out.get('audio_tokens') or 0) + (
                details_out.get('text_tokens') or 0
            )
          if total is None and (prompt is not None or completion is not None):
            try:
              total = (prompt or 0) + (completion or 0)
            except Exception:
              total = None

          if (
              (prompt is not None)
              or (completion is not None)
              or (total is not None)
          ):
            usage_meta = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=int(prompt or 0),
                candidates_token_count=int(completion or 0),
                total_token_count=int(
                    total or ((prompt or 0) + (completion or 0))
                ),
            )
    except Exception:
      usage_meta = None
    transcript_response: LlmResponse | None = None

    # Flush any accumulated output transcription; do not mark done yet,
    # in case a text response follows.
    if self._output_transcript_buffer:
      transcript_response = LlmResponse(
          output_transcription=types.Transcription(
              text=self._output_transcript_buffer
          ),
          # Emit as model text content so it is stored in session history
          content=types.Content(
              role='model',
              parts=[types.Part.from_text(text=self._output_transcript_buffer)],
          ),
      )
      self._output_transcript_buffer = ''

    # Flush any buffered text; this will carry the turn_complete flag
    if self._text_buffer:
      text_resp = self._build_full_text_response(self._text_buffer)
      self._text_buffer = ''
      # Prepend transcript response if present
      if transcript_response:
        responses.append(transcript_response)
      text_resp.turn_complete = True
      if usage_meta:
        text_resp.usage_metadata = usage_meta
      responses.append(text_resp)
      return responses

    # No text buffer; if we had only transcript, mark that as done
    if transcript_response:
      transcript_response.turn_complete = True
      if usage_meta:
        transcript_response.usage_metadata = usage_meta
      responses.append(transcript_response)
      return responses

    # No content to flush: emit meta-only done as partial to avoid NIL persistence
    lr = LlmResponse(turn_complete=True, partial=True)
    if usage_meta:
      lr.usage_metadata = usage_meta
    responses.append(lr)
    return responses

  def _handle_error(self, event: dict) -> list[LlmResponse]:
    err = event.get('error', {})
    return [
        LlmResponse(
            error_code=str(err.get('code') or 'OPENAI_ERROR'),
            error_message=err.get('message'),
        )
    ]

  # Trimming helpers removed in revert
