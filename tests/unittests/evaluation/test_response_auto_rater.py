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

from unittest.mock import MagicMock

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.response_auto_rater import get_eval_status
from google.adk.evaluation.response_auto_rater import get_text_from_content
from google.adk.evaluation.response_auto_rater import ResponseAutoRater
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


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


def test_response_auto_rater_init_missing_judge_model_name():
  with pytest.raises(ValueError):
    ResponseAutoRater(
        EvalMetric(metric_name="test_metric", threshold=0.8),
        "test prompt template: {prompt}",
    )


def test_response_auto_rater_init_missing_judge_model_config():
  with pytest.raises(ValueError):
    ResponseAutoRater(
        EvalMetric(
            metric_name="test_metric", judge_model="test model", threshold=0.8
        ),
        "test prompt template: {prompt}",
    )


@pytest.fixture
def mock_judge_model():
  mock_judge_model = MagicMock()

  async def mock_generate_content_async(llm_request):
    yield LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text=f"auto rater response")],
        )
    )

  mock_judge_model.generate_content_async = mock_generate_content_async
  return mock_judge_model


@pytest.mark.asyncio
async def test_evaluate_invocations_with_mock(mock_judge_model):
  # Create a dummy ResponseAutoRater instance
  auto_rater = ResponseAutoRater(
      eval_metric=EvalMetric(
          metric_name="test_metric",
          threshold=0.5,
          judge_model="gemini-2.5-flash",
          judge_model_config=genai_types.GenerateContentConfig(),
      ),
      auto_rater_prompt_template="test_prompt",
      auto_rater_num_samples=3,
  )
  auto_rater._judge_model = mock_judge_model
  auto_rater.format_auto_rater_prompt = MagicMock(
      return_value="formatted prompt"
  )
  auto_rater.convert_auto_rater_response_to_score = MagicMock(return_value=1.0)
  auto_rater.aggregate_invocation_results = MagicMock(
      return_value=EvaluationResult(
          overall_score=1.0, overall_eval_status=EvalStatus.PASSED
      )
  )

  # Create dummy invocations
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

  result = await auto_rater.evaluate_invocations(
      actual_invocations, expected_invocations
  )

  # Assertions
  assert result.overall_score == 1.0
  assert auto_rater.convert_auto_rater_response_to_score.call_count == 6
  assert auto_rater.aggregate_invocation_results.call_count == 1
