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

import logging
from typing import AsyncGenerator
from typing import Union

from google.genai import live
from google.genai import types

from ..utils.context_utils import Aclosing
from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

RealtimeInput = Union[types.Blob, types.ActivityStart, types.ActivityEnd]


class GeminiLlmConnection(BaseLlmConnection):
  """The Gemini model connection."""

  def __init__(self, gemini_session: live.AsyncSession):
    self._gemini_session = gemini_session

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

    # We ignore any audio from user during the agent transfer phase
    contents = [
        content
        for content in history
        if content.parts and content.parts[0].text
    ]

    if contents:
      await self._gemini_session.send(
          input=types.LiveClientContent(
              turns=contents,
              turn_complete=contents[-1].role == 'user',
          ),
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

    assert content.parts
    if content.parts[0].function_response:
      # All parts have to be function responses.
      function_responses = [part.function_response for part in content.parts]
      logger.debug('Sending LLM function response: %s', function_responses)
      await self._gemini_session.send(
          input=types.LiveClientToolResponse(
              function_responses=function_responses
          ),
      )
    else:
      logger.debug('Sending LLM new content %s', content)
      await self._gemini_session.send(
          input=types.LiveClientContent(
              turns=[content],
              turn_complete=True,
          )
      )

  async def send_realtime(self, input: RealtimeInput):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    Args:
      input: The input to send to the model.
    """
    if isinstance(input, types.Blob):
      input_blob = input.model_dump()
      logger.debug('Sending LLM Blob: %s', input_blob)
      await self._gemini_session.send(input=input_blob)
    elif isinstance(input, types.ActivityStart):
      logger.debug('Sending LLM activity start signal')
      await self._gemini_session.send_realtime_input(activity_start=input)
    elif isinstance(input, types.ActivityEnd):
      logger.debug('Sending LLM activity end signal')
      await self._gemini_session.send_realtime_input(activity_end=input)
    else:
      raise ValueError('Unsupported input type: %s' % type(input))

  def __build_full_text_response(self, text: str):
    """Builds a full text response.

    The text should not partial and the returned LlmResponse is not be
    partial.

    Args:
      text: The text to be included in the response.

    Returns:
      An LlmResponse containing the full text.
    """
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text=text)],
        ),
    )

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Yields:
      LlmResponse: The model response.
    """

    text = ''
    async with Aclosing(self._gemini_session.receive()) as agen:
      # TODO(b/440101573): Reuse StreamingResponseAggregator to accumulate
      # partial content and emit responses as needed.
      async for message in agen:
        logger.debug('Got LLM Live message: %s', message)
        if message.usage_metadata:
          yield LlmResponse(usage_metadata=message.usage_metadata)
        if message.server_content:
          content = message.server_content.model_turn
          if content and content.parts:
            llm_response = LlmResponse(
                content=content, interrupted=message.server_content.interrupted
            )
            if content.parts[0].text:
              text += content.parts[0].text
              llm_response.partial = True
            # don't yield the merged text event when receiving audio data
            elif text and not content.parts[0].inline_data:
              yield self.__build_full_text_response(text)
              text = ''
            yield llm_response
          if (
              message.server_content.input_transcription
              and message.server_content.input_transcription.text
          ):
            llm_response = LlmResponse(
                input_transcription=message.server_content.input_transcription,
            )
            yield llm_response
          if (
              message.server_content.output_transcription
              and message.server_content.output_transcription.text
          ):
            llm_response = LlmResponse(
                output_transcription=message.server_content.output_transcription
            )
            yield llm_response
          if message.server_content.turn_complete:
            if text:
              yield self.__build_full_text_response(text)
              text = ''
            yield LlmResponse(
                turn_complete=True,
                interrupted=message.server_content.interrupted,
            )
            break
          # in case of empty content or parts, we sill surface it
          # in case it's an interrupted message, we merge the previous partial
          # text. Other we don't merge. because content can be none when model
          # safety threshold is triggered
          if message.server_content.interrupted and text:
            yield self.__build_full_text_response(text)
            text = ''
          yield LlmResponse(interrupted=message.server_content.interrupted)
        if message.tool_call:
          if text:
            yield self.__build_full_text_response(text)
            text = ''
          parts = [
              types.Part(function_call=function_call)
              for function_call in message.tool_call.function_calls
          ]
          yield LlmResponse(content=types.Content(role='model', parts=parts))
        if message.session_resumption_update:
          logger.info('Received session resumption message: %s', message)
          yield (
              LlmResponse(
                  live_session_resumption_update=message.session_resumption_update
              )
          )

  async def close(self):
    """Closes the llm server connection."""

    await self._gemini_session.close()
