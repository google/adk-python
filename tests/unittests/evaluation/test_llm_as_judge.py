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

from typing import Optional
from unittest.mock import MagicMock

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge import LlmAsJudge
from google.adk.evaluation.llm_as_judge_utils import get_eval_status
from google.adk.evaluation.llm_as_judge_utils import get_text_from_content
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


class MockLlmAsJudge(LlmAsJudge):

  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    return "formatted prompt"

  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> Optional[float]:
    return 1.0

  def aggregate_per_invocation_samples(
      self,
      per_invocation_samples: list[PerInvocationResult],
  ) -> PerInvocationResult:
    return per_invocation_samples[0]

  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    return EvaluationResult(
        overall_score=1.0, overall_eval_status=EvalStatus.PASSED
    )


@pytest.fixture
def mock_llm_as_judge():
  return MockLlmAsJudge(
      EvalMetric(
          metric_name="test_metric",
          threshold=0.5,
          judge_model_options=JudgeModelOptions(
              judge_model="gemini-2.5-flash",
              judge_model_config=genai_types.GenerateContentConfig(),
              num_samples=3,
          ),
      ),
  )


def test_get_text_from_content():
  content = genai_types.Content(
      parts=[
          genai_types.Part(text="This is a test text."),
          genai_types.Part(text="This is another test text."),
      ],
      role="model",
  )
  assert (
      get_text_from_content(content)
      == "This is a test text.\nThis is another test text."
  )


def test_get_eval_status():
  assert get_eval_status(score=0.8, threshold=0.8) == EvalStatus.PASSED
  assert get_eval_status(score=0.7, threshold=0.8) == EvalStatus.FAILED
  assert get_eval_status(score=0.8, threshold=0.9) == EvalStatus.FAILED
  assert get_eval_status(score=0.9, threshold=0.8) == EvalStatus.PASSED
  assert get_eval_status(score=None, threshold=0.8) == EvalStatus.NOT_EVALUATED


def test_llm_as_judge_init_missing_judge_model_options():
  with pytest.raises(ValueError):
    MockLlmAsJudge(
        EvalMetric(metric_name="test_metric", threshold=0.8),
    )


def test_llm_as_judge_init_unregistered_model():
  with pytest.raises(ValueError):
    MockLlmAsJudge(
        EvalMetric(
            metric_name="test_metric",
            threshold=0.8,
            judge_model_options=JudgeModelOptions(
                judge_model="unregistered_model",
            ),
        ),
    )


@pytest.fixture
def mock_judge_model():
  mock_judge_model = MagicMock()

  async def mock_generate_content_async(llm_request):
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="auto rater response")],
        )
    )

  mock_judge_model.generate_content_async = mock_generate_content_async
  return mock_judge_model


@pytest.mark.asyncio
async def test_evaluate_invocations_with_mock(
    mock_llm_as_judge, mock_judge_model
):
  mock_llm_as_judge._judge_model = mock_judge_model

  mock_format_auto_rater_prompt = MagicMock(
      wraps=mock_llm_as_judge.format_auto_rater_prompt
  )
  mock_llm_as_judge.format_auto_rater_prompt = mock_format_auto_rater_prompt

  mock_convert_auto_rater_response_to_score = MagicMock(
      wraps=mock_llm_as_judge.convert_auto_rater_response_to_score
  )
  mock_llm_as_judge.convert_auto_rater_response_to_score = (
      mock_convert_auto_rater_response_to_score
  )

  mock_aggregate_per_invocation_samples = MagicMock(
      wraps=mock_llm_as_judge.aggregate_per_invocation_samples
  )
  mock_llm_as_judge.aggregate_per_invocation_samples = (
      mock_aggregate_per_invocation_samples
  )

  mock_aggregate_invocation_results = MagicMock(
      wraps=mock_llm_as_judge.aggregate_invocation_results
  )
  mock_llm_as_judge.aggregate_invocation_results = (
      mock_aggregate_invocation_results
  )

  actual_invocations = [
      Invocation(
          invocation_id="id1",
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="user content 1")],
              role="user",
          ),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="final response 1")],
              role="model",
          ),
      ),
      Invocation(
          invocation_id="id2",
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="user content 2")],
              role="user",
          ),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="final response 2")],
              role="model",
          ),
      ),
  ]
  expected_invocations = [
      Invocation(
          invocation_id="id1",
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="user content 1")],
              role="user",
          ),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="expected response 1")],
              role="model",
          ),
      ),
      Invocation(
          invocation_id="id2",
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="user content 2")],
              role="user",
          ),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="expected response 2")],
              role="model",
          ),
      ),
  ]

  result = await mock_llm_as_judge.evaluate_invocations(
      actual_invocations, expected_invocations
  )

  # Assertions
  assert result.overall_score == 1.0
  assert mock_llm_as_judge.format_auto_rater_prompt.call_count == 2
  assert mock_llm_as_judge.convert_auto_rater_response_to_score.call_count == 6
  assert mock_llm_as_judge.aggregate_invocation_results.call_count == 1
