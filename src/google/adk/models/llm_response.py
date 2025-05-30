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

from typing import Any
from typing import Optional

from google.genai import types
from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict


class LlmResponse(BaseModel):
  """LLM response class that provides the first candidate response from the

  model if available. Otherwise, returns error code and message.

  Attributes:
    content: The content of the response.
    grounding_metadata: The grounding metadata of the response.
    partial: Indicates whether the text content is part of a unfinished text
      stream. Only used for streaming mode and when the content is plain text.
    turn_complete: Indicates whether the response from the model is complete.
      Only used for streaming mode.
    error_code: Error code if the response is an error. Code varies by model.
    error_message: Error message if the response is an error.
    interrupted: Flag indicating that LLM was interrupted when generating the
      content. Usually it's due to user interruption during a bidi streaming.
    custom_metadata: The custom metadata of the LlmResponse.
  """

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  content: Optional[types.Content] = None
  """The content of the response."""

  grounding_metadata: Optional[types.GroundingMetadata] = None
  """The grounding metadata of the response."""

  partial: Optional[bool] = None
  """Indicates whether the text content is part of a unfinished text stream.

  Only used for streaming mode and when the content is plain text.
  """

  turn_complete: Optional[bool] = None
  """Indicates whether the response from the model is complete.

  Only used for streaming mode.
  """

  error_code: Optional[str] = None
  """Error code if the response is an error. Code varies by model."""

  error_message: Optional[str] = None
  """Error message if the response is an error."""

  interrupted: Optional[bool] = None
  """Flag indicating that LLM was interrupted when generating the content.
  Usually it's due to user interruption during a bidi streaming.
  """

  custom_metadata: Optional[dict[str, Any]] = None
  """The custom metadata of the LlmResponse.

  An optional key-value pair to label an LlmResponse.

  NOTE: the entire dict must be JSON serializable.
  """

  usage_metadata: Optional[types.GenerateContentResponseUsageMetadata] = None
  """The usage metadata of the LlmResponse"""

  @staticmethod
  def create(
      generate_content_response: types.GenerateContentResponse,
  ) -> 'LlmResponse':
    """Creates an LlmResponse from a GenerateContentResponse.

    Args:
      generate_content_response: The GenerateContentResponse to create the
        LlmResponse from.

    Returns:
      The LlmResponse.
    """
    usage_metadata = generate_content_response.usage_metadata
    if generate_content_response.candidates:
      candidate = generate_content_response.candidates[0]
      if candidate.content and candidate.content.parts:
        return LlmResponse(
            content=candidate.content,
            grounding_metadata=candidate.grounding_metadata,
            usage_metadata=usage_metadata,
        )
      else:
        return LlmResponse(
            error_code=candidate.finish_reason,
            error_message=candidate.finish_message,
            usage_metadata=usage_metadata,
        )
    else:
      if generate_content_response.prompt_feedback:
        prompt_feedback = generate_content_response.prompt_feedback
        return LlmResponse(
            error_code=prompt_feedback.block_reason,
            error_message=prompt_feedback.block_reason_message,
            usage_metadata=usage_metadata,
        )
      else:
        return LlmResponse(
            error_code='UNKNOWN_ERROR',
            error_message='Unknown error.',
            usage_metadata=usage_metadata,
        )
