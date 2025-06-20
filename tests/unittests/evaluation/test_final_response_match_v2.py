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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.final_response_match_v2 import ResponseMatchV2Evaluator
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types


def create_test_template() -> str:
  return """
This is a test template.

{{
  "Function API spec": {function_api_spec},
  "User prompt": {prompt},
  "Agent response": {response},
  "Reference response": {golden_response},
}}

The answer should be a json alone which follows the json structure below:
{{
  "is_the_agent_response_valid": [valid or invalid],
  "reasoning":
}}
"""


def _create_test_evaluator_gemini(threshold: float) -> ResponseMatchV2Evaluator:
  return ResponseMatchV2Evaluator(
      EvalMetric(
          metric_name="final_response_match_v2",
          threshold=threshold,
          judge_model="gemini-2.5-flash",
          judge_model_config=genai_types.GenerateContentConfig(),
      ),
      create_test_template(),
      auto_rater_num_samples=3,
  )


def _create_test_invocations(
    candidate: str, reference: str
) -> tuple[Invocation, Invocation]:
  """Returns tuple of (actual_invocation, expected_invocation)."""
  actual_invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")],
          role="user",
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=candidate)],
          role="model",
      ),
  )
  expected_invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="This is a test query.")],
          role="user",
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=reference)],
          role="model",
      ),
  )
  return actual_invocation, expected_invocation


def test_format_auto_rater_prompt():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  actual_invocation, expected_invocation = _create_test_invocations(
      "candidate text", "reference text"
  )
  prompt = evaluator.format_auto_rater_prompt(
      actual_invocation, expected_invocation
  )
  assert prompt == """
This is a test template.

{
  "Function API spec": None,
  "User prompt": This is a test query.,
  "Agent response": candidate text,
  "Reference response": reference text,
}

The answer should be a json alone which follows the json structure below:
{
  "is_the_agent_response_valid": [valid or invalid],
  "reasoning":
}
"""


def test_convert_auto_rater_response_to_score_valid():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  auto_rater_response = """```json
{
  "is_the_agent_response_valid": "valid",
  "reasoning": "The response is valid."
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score == 1.0


def test_convert_auto_rater_response_to_score_invalid():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  auto_rater_response = """```json
{
  "is_the_agent_response_valid": "invalid",
  "reasoning": "The response is invalid."
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score == 0.0


def test_convert_auto_rater_response_to_score_invalid_json():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="invalid json")],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score is None


def test_convert_auto_rater_response_to_score_missing_key():
  evaluator = _create_test_evaluator_gemini(threshold=0.8)
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="{}")],
          role="model",
      )
  )
  score = evaluator.convert_auto_rater_response_to_score(llm_response)
  assert score is None


def test_aggregate_invocation_results():
  evaluator = _create_test_evaluator_gemini(threshold=0.5)
  actual_invocation, expected_invocation = _create_test_invocations(
      "candidate text", "reference text"
  )

  per_invocation_results = [
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=None,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=100.0,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      PerInvocationResult(
          actual_invocation=actual_invocation,
          expected_invocation=expected_invocation,
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
  ]

  aggregated_result = evaluator.aggregate_invocation_results(
      per_invocation_results
  )
  assert aggregated_result.overall_score == 0.5
  assert aggregated_result.overall_eval_status == EvalStatus.PASSED
