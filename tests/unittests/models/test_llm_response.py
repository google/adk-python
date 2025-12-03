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

"""Tests for LlmResponse, including log probabilities feature."""

from google.adk.models.llm_response import LlmResponse
from google.genai import types


def test_llm_response_create_with_logprobs():
  """Test LlmResponse.create() extracts logprobs from candidate."""
  avg_logprobs = -0.75
  logprobs_result = types.LogprobsResult(
      chosen_candidates=[], top_candidates=[]
  )

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Response text')]),
              finish_reason=types.FinishReason.STOP,
              avg_logprobs=avg_logprobs,
              logprobs_result=logprobs_result,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.avg_logprobs == avg_logprobs
  assert response.logprobs_result == logprobs_result
  assert response.content.parts[0].text == 'Response text'
  assert response.finish_reason == types.FinishReason.STOP


def test_llm_response_create_without_logprobs():
  """Test LlmResponse.create() handles missing logprobs gracefully."""
  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Response text')]),
              finish_reason=types.FinishReason.STOP,
              avg_logprobs=None,
              logprobs_result=None,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.avg_logprobs is None
  assert response.logprobs_result is None
  assert response.content.parts[0].text == 'Response text'


def test_llm_response_create_error_case_with_logprobs():
  """Test LlmResponse.create() includes logprobs in error cases."""
  avg_logprobs = -2.1

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=None,  # No content - error case
              finish_reason=types.FinishReason.SAFETY,
              finish_message='Safety filter triggered',
              avg_logprobs=avg_logprobs,
              logprobs_result=None,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.avg_logprobs == avg_logprobs
  assert response.logprobs_result is None
  assert response.error_code == types.FinishReason.SAFETY
  assert response.error_message == 'Safety filter triggered'


def test_llm_response_create_no_candidates():
  """Test LlmResponse.create() with no candidates."""
  generate_content_response = types.GenerateContentResponse(
      candidates=[],
      prompt_feedback=types.GenerateContentResponsePromptFeedback(
          block_reason=types.BlockedReason.SAFETY,
          block_reason_message='Prompt blocked for safety',
      ),
  )

  response = LlmResponse.create(generate_content_response)

  # No candidates means no logprobs
  assert response.avg_logprobs is None
  assert response.logprobs_result is None
  assert response.error_code == types.BlockedReason.SAFETY
  assert response.error_message == 'Prompt blocked for safety'


def test_llm_response_create_with_concrete_logprobs_result():
  """Test LlmResponse.create() with detailed logprobs_result containing actual token data."""
  # Create realistic logprobs data
  chosen_candidates = [
      types.LogprobsResultCandidate(
          token='The', log_probability=-0.1, token_id=123
      ),
      types.LogprobsResultCandidate(
          token=' capital', log_probability=-0.5, token_id=456
      ),
      types.LogprobsResultCandidate(
          token=' of', log_probability=-0.2, token_id=789
      ),
  ]

  top_candidates = [
      types.LogprobsResultTopCandidates(
          candidates=[
              types.LogprobsResultCandidate(
                  token='The', log_probability=-0.1, token_id=123
              ),
              types.LogprobsResultCandidate(
                  token='A', log_probability=-2.3, token_id=124
              ),
              types.LogprobsResultCandidate(
                  token='This', log_probability=-3.1, token_id=125
              ),
          ]
      ),
      types.LogprobsResultTopCandidates(
          candidates=[
              types.LogprobsResultCandidate(
                  token=' capital', log_probability=-0.5, token_id=456
              ),
              types.LogprobsResultCandidate(
                  token=' city', log_probability=-1.2, token_id=457
              ),
              types.LogprobsResultCandidate(
                  token=' main', log_probability=-2.8, token_id=458
              ),
          ]
      ),
  ]

  avg_logprobs = -0.27  # Average of -0.1, -0.5, -0.2
  logprobs_result = types.LogprobsResult(
      chosen_candidates=chosen_candidates, top_candidates=top_candidates
  )

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(
                  parts=[types.Part(text='The capital of France is Paris.')]
              ),
              finish_reason=types.FinishReason.STOP,
              avg_logprobs=avg_logprobs,
              logprobs_result=logprobs_result,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.avg_logprobs == avg_logprobs
  assert response.logprobs_result is not None

  # Test chosen candidates
  assert len(response.logprobs_result.chosen_candidates) == 3
  assert response.logprobs_result.chosen_candidates[0].token == 'The'
  assert response.logprobs_result.chosen_candidates[0].log_probability == -0.1
  assert response.logprobs_result.chosen_candidates[0].token_id == 123
  assert response.logprobs_result.chosen_candidates[1].token == ' capital'
  assert response.logprobs_result.chosen_candidates[1].log_probability == -0.5
  assert response.logprobs_result.chosen_candidates[1].token_id == 456

  # Test top candidates
  assert len(response.logprobs_result.top_candidates) == 2
  assert (
      len(response.logprobs_result.top_candidates[0].candidates) == 3
  )  # 3 alternatives for first token
  assert response.logprobs_result.top_candidates[0].candidates[0].token == 'The'
  assert (
      response.logprobs_result.top_candidates[0].candidates[0].token_id == 123
  )
  assert response.logprobs_result.top_candidates[0].candidates[1].token == 'A'
  assert (
      response.logprobs_result.top_candidates[0].candidates[1].token_id == 124
  )
  assert (
      response.logprobs_result.top_candidates[0].candidates[2].token == 'This'
  )
  assert (
      response.logprobs_result.top_candidates[0].candidates[2].token_id == 125
  )


def test_llm_response_create_with_partial_logprobs_result():
  """Test LlmResponse.create() with logprobs_result having only chosen_candidates."""
  chosen_candidates = [
      types.LogprobsResultCandidate(
          token='Hello', log_probability=-0.05, token_id=111
      ),
      types.LogprobsResultCandidate(
          token=' world', log_probability=-0.8, token_id=222
      ),
  ]

  logprobs_result = types.LogprobsResult(
      chosen_candidates=chosen_candidates,
      top_candidates=[],  # Empty top candidates
  )

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Hello world')]),
              finish_reason=types.FinishReason.STOP,
              avg_logprobs=-0.425,  # Average of -0.05 and -0.8
              logprobs_result=logprobs_result,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.avg_logprobs == -0.425
  assert response.logprobs_result is not None
  assert len(response.logprobs_result.chosen_candidates) == 2
  assert len(response.logprobs_result.top_candidates) == 0
  assert response.logprobs_result.chosen_candidates[0].token == 'Hello'
  assert response.logprobs_result.chosen_candidates[1].token == ' world'


def test_llm_response_create_with_citation_metadata():
  """Test LlmResponse.create() extracts citation_metadata from candidate."""
  citation_metadata = types.CitationMetadata(
      citations=[
          types.Citation(
              start_index=0,
              end_index=10,
              uri='https://example.com',
          )
      ]
  )

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Response text')]),
              finish_reason=types.FinishReason.STOP,
              citation_metadata=citation_metadata,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.citation_metadata == citation_metadata
  assert response.content.parts[0].text == 'Response text'


def test_llm_response_create_without_citation_metadata():
  """Test LlmResponse.create() handles missing citation_metadata gracefully."""
  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Response text')]),
              finish_reason=types.FinishReason.STOP,
              citation_metadata=None,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.citation_metadata is None
  assert response.content.parts[0].text == 'Response text'


def test_llm_response_create_error_case_with_citation_metadata():
  """Test LlmResponse.create() includes citation_metadata in error cases."""
  citation_metadata = types.CitationMetadata(
      citations=[
          types.Citation(
              start_index=0,
              end_index=10,
              uri='https://example.com',
          )
      ]
  )

  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=None,  # No content - blocked case
              finish_reason=types.FinishReason.RECITATION,
              finish_message='Response blocked due to recitation triggered',
              citation_metadata=citation_metadata,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.citation_metadata == citation_metadata
  assert response.error_code == types.FinishReason.RECITATION
  assert (
      response.error_message == 'Response blocked due to recitation triggered'
  )


def test_llm_response_create_empty_content_with_stop_reason():
  """Test LlmResponse.create() with empty content and stop finish reason."""
  generate_content_response = types.GenerateContentResponse(
      candidates=[
          types.Candidate(
              content=types.Content(parts=[]),
              finish_reason=types.FinishReason.STOP,
          )
      ]
  )

  response = LlmResponse.create(generate_content_response)

  assert response.error_code is None
  assert response.content is not None


def test_llm_response_create_includes_model_version():
  """Test LlmResponse.create() includes model version."""
  generate_content_response = types.GenerateContentResponse(
      model_version='gemini-2.0-flash',
      candidates=[
          types.Candidate(
              content=types.Content(parts=[types.Part(text='Response text')]),
              finish_reason=types.FinishReason.STOP,
          )
      ],
  )
  response = LlmResponse.create(generate_content_response)
  assert response.model_version == 'gemini-2.0-flash'
