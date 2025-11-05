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

from .cache_metadata import CacheMetadata


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
    input_transcription: Audio transcription of user input.
    output_transcription: Audio transcription of model output.
    avg_logprobs: Average log probability of the generated tokens.
    logprobs_result: Detailed log probabilities for chosen and top candidate tokens.
  """

  model_config = ConfigDict(
      extra='forbid',
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""

  model_version: Optional[str] = None
  """Output only. The model version used to generate the response."""

  content: Optional[types.Content] = None
  """The generative content of the response.

  This should only contain content from the user or the model, and not any
  framework or system-generated data.
  """

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

  finish_reason: Optional[types.FinishReason] = None
  """The finish reason of the response."""

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

  live_session_resumption_update: Optional[
      types.LiveServerSessionResumptionUpdate
  ] = None
  """The session resumption update of the LlmResponse"""

  input_transcription: Optional[types.Transcription] = None
  """Audio transcription of user input."""

  output_transcription: Optional[types.Transcription] = None
  """Audio transcription of model output."""

  avg_logprobs: Optional[float] = None
  """Average log probability of the generated tokens."""

  logprobs_result: Optional[types.LogprobsResult] = None
  """Detailed log probabilities for chosen and top candidate tokens."""

  cache_metadata: Optional[CacheMetadata] = None
  """Context cache metadata if caching was used for this response.

  Contains cache identification, usage tracking, and lifecycle information.
  This field is automatically populated when context caching is enabled.
  """

  citation_metadata: Optional[types.CitationMetadata] = None
  """Citation metadata for the response.

  This field is automatically populated when citation is enabled.
  """

  @staticmethod
  def create(
      generate_content_response: types.GenerateContentResponse,
  ) -> LlmResponse:
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
      if (
          candidate.content and candidate.content.parts
      ) or candidate.finish_reason == types.FinishReason.STOP:
        return LlmResponse(
            content=candidate.content,
            grounding_metadata=candidate.grounding_metadata,
            usage_metadata=usage_metadata,
            finish_reason=candidate.finish_reason,
            citation_metadata=candidate.citation_metadata,
            avg_logprobs=candidate.avg_logprobs,
            logprobs_result=candidate.logprobs_result,
            model_version=generate_content_response.model_version,
        )
      else:
        return LlmResponse(
            error_code=candidate.finish_reason,
            error_message=candidate.finish_message,
            citation_metadata=candidate.citation_metadata,
            usage_metadata=usage_metadata,
            finish_reason=candidate.finish_reason,
            avg_logprobs=candidate.avg_logprobs,
            logprobs_result=candidate.logprobs_result,
            model_version=generate_content_response.model_version,
        )
    else:
      if generate_content_response.prompt_feedback:
        prompt_feedback = generate_content_response.prompt_feedback
        return LlmResponse(
            error_code=prompt_feedback.block_reason,
            error_message=prompt_feedback.block_reason_message,
            usage_metadata=usage_metadata,
            model_version=generate_content_response.model_version,
        )
      else:
        return LlmResponse(
            error_code='UNKNOWN_ERROR',
            error_message='Unknown error.',
            usage_metadata=usage_metadata,
            model_version=generate_content_response.model_version,
        )
