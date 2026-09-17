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

import asyncio
from typing import Optional

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import LlmAsAJudgeCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge import AutoRaterScore
from google.adk.evaluation.llm_as_judge import LlmAsJudge
from google.adk.evaluation.llm_as_judge_utils import get_eval_status
from google.adk.evaluation.llm_as_judge_utils import get_text_from_content
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pydantic
import pytest


class MockLlmAsJudge(LlmAsJudge):

  def format_auto_rater_prompt(
      self,
      actual_invocation: Invocation,
      expected_invocation: Optional[Invocation],
      rubrics: Optional[list[Rubric]] = None,
  ) -> str:
    return "formatted prompt"

  def convert_auto_rater_response_to_score(
      self,
      llm_response: LlmResponse,
      rubrics: Optional[list[Rubric]] = None,
  ) -> AutoRaterScore:
    return AutoRaterScore(score=1.0)

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


class PerInvocationReportingLlmAsJudge(MockLlmAsJudge):
  """Surfaces the per-invocation results the base class graded."""

  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    return EvaluationResult(per_invocation_results=per_invocation_results)


@pytest.fixture
def mock_llm_as_judge():
  return MockLlmAsJudge(
      eval_metric=EvalMetric(
          metric_name="test_metric",
          threshold=0.5,
          criterion=LlmAsAJudgeCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash",
                  judge_model_config=genai_types.GenerateContentConfig(),
                  num_samples=3,
              ),
          ),
      ),
      criterion_type=LlmAsAJudgeCriterion,
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


def test_llm_as_judge_init_missing_criterion():
  with pytest.raises(ValueError):
    MockLlmAsJudge(
        EvalMetric(metric_name="test_metric", threshold=0.8),
        criterion_type=LlmAsAJudgeCriterion,
    )


def test_llm_as_judge_init_unregistered_model():
  with pytest.raises(ValueError):
    MockLlmAsJudge(
        EvalMetric(
            metric_name="test_metric",
            threshold=0.8,
            criterion=LlmAsAJudgeCriterion(
                threshold=0.5,
                judge_model_options=JudgeModelOptions(
                    judge_model="unregistered_model",
                    judge_model_config=genai_types.GenerateContentConfig(),
                    num_samples=3,
                ),
            ),
        ),
        criterion_type=LlmAsAJudgeCriterion,
    )


@pytest.fixture
def mock_judge_model(mocker):
  mock_judge_model = mocker.MagicMock()

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
    mock_llm_as_judge, mock_judge_model, mocker
):
  mock_llm_as_judge._judge_model = mock_judge_model

  mock_format_auto_rater_prompt = mocker.MagicMock(
      wraps=mock_llm_as_judge.format_auto_rater_prompt
  )
  mock_llm_as_judge.format_auto_rater_prompt = mock_format_auto_rater_prompt

  mock_convert_auto_rater_response_to_score = mocker.MagicMock(
      wraps=mock_llm_as_judge.convert_auto_rater_response_to_score
  )
  mock_llm_as_judge.convert_auto_rater_response_to_score = (
      mock_convert_auto_rater_response_to_score
  )

  mock_aggregate_per_invocation_samples = mocker.MagicMock(
      wraps=mock_llm_as_judge.aggregate_per_invocation_samples
  )
  mock_llm_as_judge.aggregate_per_invocation_samples = (
      mock_aggregate_per_invocation_samples
  )

  mock_aggregate_invocation_results = mocker.MagicMock(
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


@pytest.mark.asyncio
async def test_evaluate_invocations_grades_criterion_only_metric(
    mock_judge_model,
):
  # A metric configured with just a criterion carries no deprecated threshold,
  # and must still be graded against the criterion's own threshold.
  judge = PerInvocationReportingLlmAsJudge(
      eval_metric=EvalMetric(
          metric_name="test_metric",
          criterion=LlmAsAJudgeCriterion(
              threshold=0.5,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash",
                  judge_model_config=genai_types.GenerateContentConfig(),
                  num_samples=1,
              ),
          ),
      ),
      criterion_type=LlmAsAJudgeCriterion,
  )
  judge._judge_model = mock_judge_model
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
      )
  ]

  result = await judge.evaluate_invocations(actual_invocations)

  # The auto-rater scores 1.0, which clears the criterion's 0.5 threshold.
  assert [r.eval_status for r in result.per_invocation_results] == [
      EvalStatus.PASSED
  ]


@pytest.mark.asyncio
async def test_evaluate_invocations_parallelism_limit(
    mock_llm_as_judge, mocker
):
  mock_llm_as_judge._judge_model_options.parallelism_limit = 2

  active_calls = 0
  max_active_calls = 0

  async def mock_generate_content_async(llm_request):
    nonlocal active_calls, max_active_calls
    active_calls += 1
    max_active_calls = max(max_active_calls, active_calls)
    await asyncio.sleep(0.1)
    active_calls -= 1
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="auto rater response")],
        )
    )

  mock_judge_model = mocker.MagicMock()
  mock_judge_model.generate_content_async = mock_generate_content_async
  mock_llm_as_judge._judge_model = mock_judge_model

  actual_invocations = [
      Invocation(
          invocation_id="id1",
          user_content=genai_types.Content(parts=[genai_types.Part(text="u1")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="r1")]
          ),
      ),
      Invocation(
          invocation_id="id2",
          user_content=genai_types.Content(parts=[genai_types.Part(text="u2")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="r2")]
          ),
      ),
  ]

  mock_llm_as_judge._judge_model_options.num_samples = 3

  await mock_llm_as_judge.evaluate_invocations(
      actual_invocations, actual_invocations
  )

  assert max_active_calls == 2


@pytest.mark.asyncio
async def test_evaluate_invocations_sample_failure(mock_llm_as_judge, mocker):
  call_count = 0

  async def mock_generate_content_async(llm_request):
    nonlocal call_count
    call_count += 1
    if call_count == 1:
      raise RuntimeError("Simulated LLM failure")
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="auto rater response")],
        )
    )

  mock_judge_model = mocker.MagicMock()
  mock_judge_model.generate_content_async = mock_generate_content_async
  mock_llm_as_judge._judge_model = mock_judge_model

  mock_aggregate_per_invocation_samples = mocker.MagicMock(
      wraps=mock_llm_as_judge.aggregate_per_invocation_samples
  )
  mock_llm_as_judge.aggregate_per_invocation_samples = (
      mock_aggregate_per_invocation_samples
  )

  def mock_aggregate_invocation_results(per_invocation_results):
    return EvaluationResult(
        per_invocation_results=per_invocation_results,
    )

  mock_llm_as_judge.aggregate_invocation_results = (
      mock_aggregate_invocation_results
  )

  actual_invocations = [
      Invocation(
          invocation_id="id1",
          user_content=genai_types.Content(parts=[genai_types.Part(text="u1")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="r1")]
          ),
      ),
      Invocation(
          invocation_id="id2",
          user_content=genai_types.Content(parts=[genai_types.Part(text="u2")]),
          final_response=genai_types.Content(
              parts=[genai_types.Part(text="r2")]
          ),
      ),
  ]

  mock_llm_as_judge._judge_model_options.num_samples = 2

  result = await mock_llm_as_judge.evaluate_invocations(
      actual_invocations, actual_invocations
  )

  assert len(result.per_invocation_results) == 2
  assert (
      result.per_invocation_results[0].eval_status == EvalStatus.NOT_EVALUATED
  )
  assert result.per_invocation_results[0].score is None
  assert result.per_invocation_results[1].eval_status == EvalStatus.PASSED
  assert result.per_invocation_results[1].score == 1.0
  assert mock_aggregate_per_invocation_samples.call_count == 1


@pytest.mark.parametrize("invalid_limit", [0, -1])
def test_judge_model_options_invalid_parallelism_limit(invalid_limit):
  with pytest.raises(pydantic.ValidationError):
    JudgeModelOptions(parallelism_limit=invalid_limit)
