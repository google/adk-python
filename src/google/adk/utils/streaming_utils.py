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

from typing import AsyncGenerator
from typing import Optional

from google.genai import types

from ..models.llm_response import LlmResponse


class StreamingResponseAggregator:
  """Aggregates partial streaming responses.

  It aggregates content from partial responses, and generates LlmResponses for
  individual (partial) model responses, as well as for aggregated content.
  """

  def __init__(self):
    self._text = ''
    self._thought_text = ''
    self._usage_metadata = None
    self._response = None

  async def process_response(
      self, response: types.GenerateContentResponse
  ) -> AsyncGenerator[LlmResponse, None]:
    """Processes a single model response.

    Args:
      response: The response to process.

    Yields:
      The generated LlmResponse(s), for the partial response, and the aggregated
      response if needed.
    """
    # results = []
    self._response = response
    llm_response = LlmResponse.create(response)
    self._usage_metadata = llm_response.usage_metadata
    if (
        llm_response.content
        and llm_response.content.parts
        and llm_response.content.parts[0].text
    ):
      part0 = llm_response.content.parts[0]
      if part0.thought:
        self._thought_text += part0.text
      else:
        self._text += part0.text
      llm_response.partial = True
    elif (self._thought_text or self._text) and (
        not llm_response.content
        or not llm_response.content.parts
        # don't yield the merged text event when receiving audio data
        or not llm_response.content.parts[0].inline_data
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=self._thought_text, thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=self._text))
      yield LlmResponse(
          content=types.ModelContent(parts=parts),
          usage_metadata=llm_response.usage_metadata,
      )
      self._thought_text = ''
      self._text = ''
    yield llm_response

  def close(self) -> Optional[LlmResponse]:
    """Generate an aggregated response at the end, if needed.

    This should be called after all the model responses are processed.

    Returns:
      The aggregated LlmResponse.
    """
    if (
        (self._text or self._thought_text)
        and self._response
        and self._response.candidates
    ):
      parts = []
      if self._thought_text:
        parts.append(types.Part(text=self._thought_text, thought=True))
      if self._text:
        parts.append(types.Part.from_text(text=self._text))
      candidate = self._response.candidates[0]
      return LlmResponse(
          content=types.ModelContent(parts=parts),
          error_code=None
          if candidate.finish_reason == types.FinishReason.STOP
          else candidate.finish_reason,
          error_message=None
          if candidate.finish_reason == types.FinishReason.STOP
          else candidate.finish_message,
          usage_metadata=self._usage_metadata,
      )
