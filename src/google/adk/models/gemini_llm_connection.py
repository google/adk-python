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
from typing import AsyncGenerator
from typing import Union

from google.genai import types

from ..utils import model_name_utils
from ..utils.content_utils import filter_audio_parts
from ..utils.context_utils import Aclosing
from ..utils.variant_utils import GoogleLLMVariant
from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

RealtimeInput = Union[types.Blob, types.ActivityStart, types.ActivityEnd]
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.genai import live


class GeminiLlmConnection(BaseLlmConnection):
  """The Gemini model connection."""

  def __init__(
      self,
      gemini_session: live.AsyncSession,
      api_backend: GoogleLLMVariant = GoogleLLMVariant.VERTEX_AI,
      model_version: str | None = None,
  ):
    self._gemini_session = gemini_session
    self._input_transcription_text: str = ''
    self._output_transcription_text: str = ''
    self._api_backend = api_backend
    self._model_version = model_version
    self._is_gemini_3_x_live = model_name_utils._is_gemini_3_x_live(
        model_version
    )
    self._is_gemini_3_5_live_translate = (
        model_name_utils.is_gemini_3_5_live_translate(model_version)
    )

  async def send_history(self, history: list[types.Content]):
    """Sends the conversation history to the gemini model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user; otherwise, it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """

    # TODO: Remove this filter and translate unary contents to streaming
    # contents properly.

    # Filter out audio parts from history because:
    # 1. audio has already been transcribed.
    # 2. sending audio via connection.send or connection.send_live_content is
    # not supported by LIVE API (session will be corrupted).
    # This method is called when:
    # 1. Agent transfer to a new agent
    # 2. Establishing a new live connection with previous ADK session history

    contents = [
        filtered
        for content in history
        if (filtered := filter_audio_parts(content)) is not None
    ]

    if contents:
      logger.debug('Sending history to live connection: %s', contents)
      await self._gemini_session.send_client_content(
          turns=contents,
          turn_complete=contents[-1].role == 'user',
      )
    else:
      logger.info('no content is sent')

  async def send_content(self, content: types.Content):
    """Sends a user content to the gemini model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """
    await self._send_content(content)

  async def _send_content(
      self, content: types.Content, *, partial: bool = False
  ) -> None:
    """Sends content, optionally as a partial (non-turn-completing) update.

    Args:
      content: The content to send to the model.
      partial: Whether this content is a partial turn update that does not
        complete the model turn.
    """
    assert content.parts
    if content.parts[0].function_response:
      # All parts have to be function responses.
      function_responses = [part.function_response for part in content.parts]
      logger.debug('Sending LLM function response: %s', function_responses)
      await self._gemini_session.send_tool_response(
          function_responses=function_responses
      )
    else:
      logger.debug('Sending LLM new content %s', content)
      if (
          not partial
          and self._is_gemini_3_x_live
          and len(content.parts) == 1
          and content.parts[0].text
      ):
        logger.debug('Using send_realtime_input for Gemini 3.x Live text input')
        await self._gemini_session.send_realtime_input(
            text=content.parts[0].text
        )
      else:
        await self._gemini_session.send(
            input=types.LiveClientContent(
                turns=[content],
                turn_complete=not partial,
            )
        )

  async def send_realtime(self, input: RealtimeInput):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    Args:
      input: The input to send to the model.
    """
    if isinstance(input, types.Blob):
      # The blob is binary and is very large. So let's not log it.
      logger.debug('Sending LLM Blob.')
      if self._is_gemini_3_x_live or self._is_gemini_3_5_live_translate:
        if input.mime_type and input.mime_type.startswith('audio/'):
          await self._gemini_session.send_realtime_input(audio=input)
        elif input.mime_type and input.mime_type.startswith('image/'):
          await self._gemini_session.send_realtime_input(video=input)
        else:
          logger.warning(
              'Blob not sent. Unknown or empty mime type for'
              ' send_realtime_input: %s',
              input.mime_type,
          )
      else:
        await self._gemini_session.send_realtime_input(media=input)

    elif isinstance(input, types.ActivityStart):
      logger.debug('Sending LLM activity start signal.')
      await self._gemini_session.send_realtime_input(activity_start=input)
    elif isinstance(input, types.ActivityEnd):
      logger.debug('Sending LLM activity end signal.')
      await self._gemini_session.send_realtime_input(activity_end=input)
    else:
      raise ValueError('Unsupported input type: %s' % type(input))

  @staticmethod
  def _merge_grounding_metadata(
      existing: types.GroundingMetadata | None,
      new: types.GroundingMetadata | None,
  ) -> types.GroundingMetadata | None:
    """Merges two GroundingMetadata instances, accumulating list fields safely."""
    if existing is None:
      return new
    if new is None:
      return existing
    existing_data = existing.model_dump(exclude_none=True)
    new_data = new.model_dump(exclude_none=True)

    # Get offset from existing grounding chunks for shifting support indices
    chunk_offset = len(existing_data.get('grounding_chunks', []))

    for key, val in new_data.items():
      if isinstance(val, list) and all(isinstance(x, str) for x in val):
        existing_list = existing_data.get(key, [])
        for item in val:
          if item not in existing_list:
            existing_list.append(item)
        existing_data[key] = existing_list
      elif key == 'grounding_chunks':
        existing_chunks = existing_data.get('grounding_chunks', [])
        existing_chunks.extend(val)
        existing_data['grounding_chunks'] = existing_chunks
      elif key == 'grounding_supports':
        existing_supports = existing_data.get('grounding_supports', [])
        for support in val:
          if (
              'grounding_chunk_indices' in support
              and support['grounding_chunk_indices']
          ):
            support['grounding_chunk_indices'] = [
                idx + chunk_offset for idx in support['grounding_chunk_indices']
            ]
          existing_supports.append(support)
        existing_data['grounding_supports'] = existing_supports
      else:
        existing_data[key] = val
    return types.GroundingMetadata(**existing_data)

  def __build_full_text_response(
      self,
      text: str,
      is_thought: bool = False,
      grounding_metadata: types.GroundingMetadata | None = None,
      interrupted: bool = False,
  ):
    """Builds a full text response.

    The text should not be partial and the returned LlmResponse is not
    partial.

    Args:
      text: The text to be included in the response.
      is_thought: Whether the text is a thought.
      grounding_metadata: The grounding metadata to include.
      interrupted: Whether this response was interrupted.

    Returns:
      An LlmResponse containing the full text.
    """
    part = types.Part.from_text(text=text)
    if is_thought:
      part.thought = True

    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[part],
        ),
        grounding_metadata=grounding_metadata,
        interrupted=interrupted,
        partial=False,
        live_session_id=self._gemini_session.session_id,
    )

  def _to_generate_content_usage_metadata(
      self, usage_metadata: types.UsageMetadata
  ) -> types.GenerateContentResponseUsageMetadata:
    """Converts live API usage metadata to GenerateContentResponse usage metadata.

    The live API names output tokens `response_token_count`/
    `response_tokens_details`, whereas `GenerateContentResponseUsageMetadata`
    names them `candidates_token_count`/`candidates_tokens_details`.

    Args:
      usage_metadata: The live API usage metadata.

    Returns:
      The converted usage metadata.
    """
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage_metadata.prompt_token_count,
        cached_content_token_count=usage_metadata.cached_content_token_count,
        candidates_token_count=usage_metadata.response_token_count,
        total_token_count=usage_metadata.total_token_count,
        thoughts_token_count=usage_metadata.thoughts_token_count,
        tool_use_prompt_token_count=usage_metadata.tool_use_prompt_token_count,
        prompt_tokens_details=usage_metadata.prompt_tokens_details,
        cache_tokens_details=usage_metadata.cache_tokens_details,
        candidates_tokens_details=usage_metadata.response_tokens_details,
        tool_use_prompt_tokens_details=usage_metadata.tool_use_prompt_tokens_details,
        traffic_type=usage_metadata.traffic_type,
    )

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Yields:
      LlmResponse: The model response.
    """

    text = ''
    is_thought = False
    tool_call_parts = []
    last_grounding_metadata = None
    tool_call_metadata = None
    async with Aclosing(self._gemini_session.receive()) as agen:
      # TODO(b/440101573): Reuse StreamingResponseAggregator to accumulate
      # partial content and emit responses as needed.
      async for message in agen:
        logger.debug('Got LLM Live message: %s', message)
        live_session_id = self._gemini_session.session_id
        if message.usage_metadata:
          # Remap live token usage to GenerateContentResponse usage metadata.
          yield LlmResponse(
              usage_metadata=self._to_generate_content_usage_metadata(
                  message.usage_metadata
              ),
              model_version=self._model_version,
              live_session_id=live_session_id,
          )
        if message.server_content:
          content = message.server_content.model_turn
          grounding_metadata = message.server_content.grounding_metadata
          if grounding_metadata:
            last_grounding_metadata = self._merge_grounding_metadata(
                last_grounding_metadata, grounding_metadata
            )

          # Standalone grounding_metadata event (when content is empty)
          if (
              not (content and content.parts)
              and message.server_content.grounding_metadata
              and not message.server_content.turn_complete
          ):
            yield LlmResponse(
                grounding_metadata=message.server_content.grounding_metadata,
                interrupted=message.server_content.interrupted,
                model_version=self._model_version,
                live_session_id=live_session_id,
                turn_complete_reason=getattr(
                    message.server_content, 'turn_complete_reason', None
                ),
            )

          if content and content.parts:
            llm_response = LlmResponse(
                content=content,
                interrupted=message.server_content.interrupted,
                model_version=self._model_version,
                live_session_id=live_session_id,
                turn_complete_reason=getattr(
                    message.server_content, 'turn_complete_reason', None
                ),
            )
            # grounding_metadata is yielded again at turn_complete,
            # so avoid duplicating it here if turn_complete is true.
            if not message.server_content.turn_complete:
              if message.server_content.grounding_metadata is not None:
                llm_response.grounding_metadata = (
                    message.server_content.grounding_metadata
                )
            if content.parts[0].text:
              current_is_thought = getattr(content.parts[0], 'thought', False)
              if text and current_is_thought != is_thought:
                yield self.__build_full_text_response(text, is_thought)
                text = ''
                is_thought = False

              text += content.parts[0].text
              is_thought = current_is_thought
              llm_response.partial = True
            # don't yield the merged text event when receiving audio data
            elif text and not content.parts[0].inline_data:
              yield self.__build_full_text_response(
                  text, is_thought, last_grounding_metadata
              )
              text = ''
              is_thought = False
              last_grounding_metadata = None
            yield llm_response
          # Note: in some cases, tool_call may arrive before
          # generation_complete, causing transcription to appear after
          # tool_call in the session log.
          if message.server_content.input_transcription:
            # Gemini 3.x Live only sends a single final input
            # transcription
            if self._is_gemini_3_x_live:
              if message.server_content.input_transcription.text:
                yield LlmResponse(
                    input_transcription=types.Transcription(
                        text=message.server_content.input_transcription.text,
                        finished=True,
                    ),
                    partial=False,
                    model_version=self._model_version,
                    live_session_id=live_session_id,
                )
            else:
              if message.server_content.input_transcription.text:
                self._input_transcription_text += (
                    message.server_content.input_transcription.text
                )
                yield LlmResponse(
                    input_transcription=types.Transcription(
                        text=message.server_content.input_transcription.text,
                        finished=False,
                    ),
                    partial=True,
                    model_version=self._model_version,
                    live_session_id=live_session_id,
                )
              # finished=True and partial transcription may happen in the same
              # message.
              if message.server_content.input_transcription.finished:
                yield LlmResponse(
                    input_transcription=types.Transcription(
                        text=self._input_transcription_text,
                        finished=True,
                    ),
                    partial=False,
                    model_version=self._model_version,
                    live_session_id=live_session_id,
                )
                self._input_transcription_text = ''
          if message.server_content.output_transcription:
            if message.server_content.output_transcription.text:
              self._output_transcription_text += (
                  message.server_content.output_transcription.text
              )
              yield LlmResponse(
                  output_transcription=types.Transcription(
                      text=message.server_content.output_transcription.text,
                      finished=False,
                  ),
                  partial=True,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
            if message.server_content.output_transcription.finished:
              yield LlmResponse(
                  output_transcription=types.Transcription(
                      text=self._output_transcription_text,
                      finished=True,
                  ),
                  partial=False,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
              self._output_transcription_text = ''
          # The Gemini API or Vertex AI might not send a transcription finished signal.
          # Instead, we rely on generation_complete, turn_complete or
          # interrupted signals to flush any pending transcriptions.
          if (
              message.server_content.interrupted
              or message.server_content.turn_complete
              or message.server_content.generation_complete
          ):
            if self._input_transcription_text:
              yield LlmResponse(
                  input_transcription=types.Transcription(
                      text=self._input_transcription_text,
                      finished=True,
                  ),
                  partial=False,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
              self._input_transcription_text = ''
            if self._output_transcription_text:
              yield LlmResponse(
                  output_transcription=types.Transcription(
                      text=self._output_transcription_text,
                      finished=True,
                  ),
                  partial=False,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
              self._output_transcription_text = ''
          if message.server_content.turn_complete:
            # Capture final grounding metadata before last_grounding_metadata is cleared in the next block.
            final_grounding_metadata = (
                grounding_metadata
                or last_grounding_metadata
                or (
                    types.GroundingMetadata()
                    if self._is_gemini_3_x_live
                    else None
                )
            )
            if (
                final_grounding_metadata
                and final_grounding_metadata.retrieval_queries
                and not final_grounding_metadata.grounding_chunks
            ):
              logger.warning(
                  'Incomplete grounding_metadata received: retrieval_queries=%s'
                  ' but grounding_chunks is empty. This may indicate a'
                  ' transient issue with the Vertex AI Search backend.',
                  final_grounding_metadata.retrieval_queries,
              )

            if text:
              yield self.__build_full_text_response(
                  text,
                  is_thought,
                  last_grounding_metadata,
                  message.server_content.interrupted,
              )
              text = ''
              is_thought = False
              last_grounding_metadata = None
            if tool_call_parts:
              logger.debug('Returning aggregated tool_call_parts')
              yield LlmResponse(
                  content=types.Content(role='model', parts=tool_call_parts),
                  grounding_metadata=tool_call_metadata,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
              tool_call_parts = []
              if tool_call_metadata is not None:
                last_grounding_metadata = None
              tool_call_metadata = None

            yield LlmResponse(
                turn_complete=True,
                interrupted=message.server_content.interrupted,
                # If last_grounding_metadata was cleared in the full text yield,
                # avoid duplicating it here.
                grounding_metadata=grounding_metadata
                or last_grounding_metadata
                or (
                    types.GroundingMetadata()
                    if self._is_gemini_3_x_live
                    else None
                ),
                model_version=self._model_version,
                live_session_id=live_session_id,
                turn_complete_reason=getattr(
                    message.server_content, 'turn_complete_reason', None
                ),
            )
            last_grounding_metadata = None  # Reset after yielding
            break
          # in case of empty content or parts, we still surface it
          # in case it's an interrupted message, we merge the previous partial
          # text. Other we don't merge. because content can be none when model
          # safety threshold is triggered
          if message.server_content.interrupted:
            if text:
              yield self.__build_full_text_response(
                  text,
                  is_thought,
                  last_grounding_metadata,
                  interrupted=True,
              )
              text = ''
              is_thought = False
              last_grounding_metadata = None
            else:
              yield LlmResponse(
                  interrupted=message.server_content.interrupted,
                  grounding_metadata=last_grounding_metadata,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
              last_grounding_metadata = None
        if message.tool_call:
          logger.debug('Received tool call: %s', message.tool_call)
          if text:
            yield self.__build_full_text_response(
                text, is_thought, last_grounding_metadata
            )
            text = ''
            is_thought = False
            last_grounding_metadata = None
          tool_call_parts.extend([
              types.Part(function_call=function_call)
              for function_call in message.tool_call.function_calls
          ])
          if not self._is_gemini_3_x_live:
            if tool_call_metadata is None:
              tool_call_metadata = last_grounding_metadata
          # Gemini 3.x Live does not emit turn_complete until it receives the
          # tool response, so yield tool calls immediately to avoid
          # deadlocking the conversation. Other models (e.g. 2.5-pro,
          # native-audio) send turn_complete after tool calls, so buffer
          # and merge them into a single response at turn_complete.
          if self._is_gemini_3_x_live and tool_call_parts:
            logger.debug(
                'Yielding tool_call_parts immediately for Gemini 3.x live tool'
                ' call'
            )
            yield LlmResponse(
                content=types.Content(role='model', parts=tool_call_parts),
                grounding_metadata=last_grounding_metadata,
                model_version=self._model_version,
                live_session_id=live_session_id,
            )
            tool_call_parts = []
            last_grounding_metadata = None
        if message.session_resumption_update:
          logger.debug('Received session resumption message: %s', message)
          yield (
              LlmResponse(
                  live_session_resumption_update=message.session_resumption_update,
                  model_version=self._model_version,
                  live_session_id=live_session_id,
              )
          )
        if message.go_away:
          logger.debug('Received GoAway message: %s', message.go_away)
          yield LlmResponse(
              go_away=message.go_away,
              model_version=self._model_version,
              live_session_id=live_session_id,
          )

      if tool_call_parts:
        logger.debug('Exited loop with pending tool_call_parts')
        yield LlmResponse(
            content=types.Content(role='model', parts=tool_call_parts),
            model_version=self._model_version,
            live_session_id=self._gemini_session.session_id,
        )

  async def close(self):
    """Closes the llm server connection."""

    await self._gemini_session.close()
