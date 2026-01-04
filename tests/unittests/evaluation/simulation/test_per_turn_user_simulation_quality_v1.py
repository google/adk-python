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

from google.adk.evaluation.eval_case import ConversationScenario
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import LlmBackedUserSimulatorCriterion
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge import AutoRaterScore
from google.adk.evaluation.llm_as_judge_utils import Label
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import _format_conversation_history
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import _parse_llm_response
from google.adk.evaluation.simulation.per_turn_user_simulator_quality_v1 import PerTurnUserSimulatorQualityV1
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": True
      }
    ],
    "is_valid_undefined_key": True
  }
  ```""",
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": True
      }
    ],
    "is_valid": "undefined label",
  }
  ```""",
    ],
)
def test_parse_llm_response_label_not_found(response_text):
  label = _parse_llm_response(response_text)
  assert label == Label.NOT_FOUND


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": True
      }
    ],
    "is_valid": True
  }
  ```""",
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": True
      }
    ],
    "is_valid": "true"
  }
  ```""",
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": True
      }
    ],
    "is_valid": "valid"
  }
  ```""",
    ],
)
def test_parse_llm_response_label_valid(response_text):
  label = _parse_llm_response(response_text)
  assert label == Label.VALID


@pytest.mark.parametrize(
    "response_text",
    [
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": False
      }
    ],
    "is_valid": False
  }
  ```""",
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": False
      }
    ],
    "is_valid": "false",
  }
  ```""",
        """```json
  {
    "criteria": [
      {
        "name": "TEST_NAME",
        "reasoning": "test_resonining",
        "passes": False
      }
    ],
    "is_valid": "invalid",
  }
  ```""",
    ],
)
def test_parse_llm_response_label_invalid(response_text):
  label = _parse_llm_response(response_text)
  assert label == Label.INVALID


def create_test_template() -> str:
  return """This is a test template with stop signal: `{stop_signal}`.

# Conversation Plan
{conversation_plan}

# Conversation History
{conversation_history}

# Generated User Response
{generated_user_response}
""".strip()


def _create_test_evaluator(
    threshold: float = 1.0, stop_signal: str = "test stop signal"
) -> PerTurnUserSimulatorQualityV1:
  evaluator = PerTurnUserSimulatorQualityV1(
      EvalMetric(
          metric_name="test_per_turn_user_simulator_quality_v1",
          threshold=threshold,
          criterion=LlmBackedUserSimulatorCriterion(
              threshold=threshold,
              stop_signal=stop_signal,
              judge_model_options=JudgeModelOptions(
                  judge_model="gemini-2.5-flash",
                  judge_model_config=genai_types.GenerateContentConfig(),
                  num_samples=3,
              ),
          ),
      ),
  )
  evaluator._prompt_template = create_test_template()
  return evaluator


def _create_test_conversation_scenario(
    conversation_plan: str = "test conversation plan",
    starting_prompt: str = "test starting prompt",
) -> ConversationScenario:
  """Returns a ConversationScenario."""
  return ConversationScenario(
      starting_prompt=starting_prompt,
      conversation_plan=conversation_plan,
  )


def _create_test_invocation(
    invocation_id: str,
    user_content: str = "user content",
    model_content: str = "model content",
) -> Invocation:
  return Invocation(
      invocation_id=invocation_id,
      user_content=genai_types.Content(
          parts=[genai_types.Part(text=user_content)],
          role="user",
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text=model_content)],
          role="model",
      ),
  )


def _create_test_invocations(
    conversation_history: list[str],
) -> list[Invocation]:
  conversation_length = len(conversation_history)

  assert conversation_length % 2 == 0

  invocations = []
  for i in range(conversation_length // 2):
    user_message = conversation_history[2 * i]
    model_message = conversation_history[2 * i + 1]

    invocations.append(
        _create_test_invocation(
            "turn {i}", user_content=user_message, model_content=model_message
        )
    )

  return invocations


def test_format_llm_prompt():
  evaluator = _create_test_evaluator(stop_signal="test stop signal")

  starting_prompt = "first user prompt."
  conversation_scenario = _create_test_conversation_scenario(
      conversation_plan="test conversation plan.",
      starting_prompt=starting_prompt,
  )
  invocation_history = _create_test_invocations([
      starting_prompt,
      "first agent response.",
      "second user prompt.",
      "second agent response.",
      "third user prompt.",
      "third agent response.",
  ])

  prompt = evaluator._format_llm_prompt(
      invocation=invocation_history[-1],
      conversation_scenario=conversation_scenario,
      previous_invocations=invocation_history[:-1],
  )

  assert (
      prompt == """This is a test template with stop signal: `test stop signal`.

# Conversation Plan
test conversation plan.

# Conversation History
user: first user prompt.

model: first agent response.

user: second user prompt.

model: second agent response.

# Generated User Response
third user prompt.
""".strip()
  )


def test_convert_llm_response_to_score_pass():
  evaluator = _create_test_evaluator()
  auto_rater_response = """```json
{
  "is_valid": True,
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  auto_rater_score = evaluator._convert_llm_response_to_score(llm_response)
  assert auto_rater_score == AutoRaterScore(score=1.0)


def test_convert_llm_response_to_score_failure():
  evaluator = _create_test_evaluator()
  auto_rater_response = """```json
{
  "is_valid": False,
}
```"""
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text=auto_rater_response)],
          role="model",
      )
  )
  auto_rater_score = evaluator._convert_llm_response_to_score(llm_response)
  assert auto_rater_score == AutoRaterScore(score=0.0)


def test_convert_llm_response_to_score_invalid_json():
  evaluator = _create_test_evaluator()
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="invalid json")],
          role="model",
      )
  )
  auto_rater_score = evaluator._convert_llm_response_to_score(llm_response)
  assert auto_rater_score == AutoRaterScore()


def test_convert_llm_response_to_score_missing_key():
  evaluator = _create_test_evaluator()
  llm_response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="{}")],
          role="model",
      )
  )
  auto_rater_score = evaluator._convert_llm_response_to_score(llm_response)
  assert auto_rater_score == AutoRaterScore()


def test_aggregate_samples_not_evaluated():
  evaluator = _create_test_evaluator()
  samples = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=None,
          eval_status=EvalStatus.NOT_EVALUATED,
      ),
  ]

  aggregation = evaluator._aggregate_samples(samples)
  assert aggregation == samples[0]


def test_aggregate_samples_pass():
  evaluator = _create_test_evaluator()
  # The majority of results should be positive.
  samples = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
  ]

  aggregation_result = evaluator._aggregate_samples(samples)

  assert aggregation_result.score == 1.0
  assert aggregation_result.eval_status == EvalStatus.PASSED


def test_aggregate_samples_failure():
  evaluator = _create_test_evaluator()

  # The majority of results should be negative.
  samples = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
  ]

  aggregation_result = evaluator._aggregate_samples(samples)

  assert aggregation_result.score == 0.0
  assert aggregation_result.eval_status == EvalStatus.FAILED


def test_format_conversation_history():
  conversation_history = [
      "first user prompt.",
      "first agent response.",
      "second user prompt.",
      "second agent response.",
  ]
  invocation_history = _create_test_invocations(conversation_history)
  formatted_history = _format_conversation_history(invocation_history)
  assert formatted_history == """user: first user prompt.

model: first agent response.

user: second user prompt.

model: second agent response."""


def test_evaluate_first_turn_pass():
  evaluator = _create_test_evaluator(
      threshold=0.8, stop_signal="test stop signal"
  )
  conversation_scenario = _create_test_conversation_scenario(
      conversation_plan="plan",
      starting_prompt="test starting prompt",
  )
  invocation = _create_test_invocation("1", user_content="test starting prompt")

  result = evaluator._evaluate_first_turn(invocation, conversation_scenario)

  assert result.score == 1.0
  assert result.eval_status == EvalStatus.PASSED


def test_evaluate_first_turn_failure():
  evaluator = _create_test_evaluator(
      threshold=1.0, stop_signal="test stop signal"
  )
  conversation_scenario = _create_test_conversation_scenario(
      conversation_plan="plan",
      starting_prompt="test starting prompt",
  )
  invocation = _create_test_invocation("1", "wrong starting prompt")

  result = evaluator._evaluate_first_turn(invocation, conversation_scenario)

  assert result.score == 0.0
  assert result.eval_status == EvalStatus.FAILED


def test_aggregate_conversation_results_all_pass_produces_pass():
  evaluator = _create_test_evaluator()
  results = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("4"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  ]
  aggregation = evaluator._aggregate_conversation_results(results)
  assert aggregation.overall_score == 1.0
  assert aggregation.overall_eval_status == EvalStatus.PASSED


def test_aggregate_conversation_results_percentage_above_threshold_produces_pass():
  evaluator = _create_test_evaluator(threshold=0.7)
  results = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=0.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("4"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  ]
  aggregation = evaluator._aggregate_conversation_results(results)
  assert aggregation.overall_score == 0.75
  assert aggregation.overall_eval_status == EvalStatus.PASSED


def test_aggregate_conversation_results_all_failures_produces_failure():
  evaluator = _create_test_evaluator()
  results = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("4"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
  ]
  aggregation = evaluator._aggregate_conversation_results(results)
  assert aggregation.overall_score == 0.0
  assert aggregation.overall_eval_status == EvalStatus.FAILED


def test_aggregate_conversation_percentage_below_threshold_produces_failure():
  evaluator = _create_test_evaluator(threshold=1.0)
  results = [
      PerInvocationResult(
          actual_invocation=_create_test_invocation("1"),
          score=0.0,
          eval_status=EvalStatus.FAILED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("2"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("3"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
      PerInvocationResult(
          actual_invocation=_create_test_invocation("4"),
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  ]
  aggregation = evaluator._aggregate_conversation_results(results)
  assert aggregation.overall_score == 0.75
  assert aggregation.overall_eval_status == EvalStatus.FAILED


@pytest.mark.asyncio
async def test_evaluate_invocations_all_pass():
  evaluator = _create_test_evaluator()

  async def sample_llm_valid(*args, **kwargs):
    return AutoRaterScore(score=1.0)

  evaluator._sample_llm = sample_llm_valid
  starting_prompt = "first user prompt."
  conversation_scenario = _create_test_conversation_scenario(
      starting_prompt=starting_prompt
  )
  invocations = _create_test_invocations(
      [starting_prompt, "model 1.", "user 2.", "model 2."]
  )
  result = await evaluator.evaluate_invocations(
      actual_invocations=invocations,
      expected_invocations=None,
      conversation_scenario=conversation_scenario,
  )

  assert result.overall_score == 1.0
  assert result.overall_eval_status == EvalStatus.PASSED
  assert len(result.per_invocation_results) == 2
  assert result.per_invocation_results[0].score == 1.0
  assert result.per_invocation_results[1].score == 1.0
