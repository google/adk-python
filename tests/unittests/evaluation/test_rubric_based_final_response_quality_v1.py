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

from google.adk.evaluation.app_details import AgentDetails
from google.adk.evaluation.app_details import AppDetails
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge_utils import get_average_rubric_score
from google.adk.evaluation.rubric_based_final_response_quality_v1 import _parse_auto_rater_response
from google.adk.evaluation.rubric_based_final_response_quality_v1 import RubricBasedFinalResponseQualityV1Evaluator
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


@pytest.fixture
def evaluator() -> RubricBasedFinalResponseQualityV1Evaluator:
  """Returns a RubricBasedFinalResponseQualityV1Evaluator."""
  rubrics = [
      Rubric(
          rubric_id="1",
          rubric_content=RubricContent(text_property="Is the response good?"),
      ),
      Rubric(
          rubric_id="2",
          rubric_content=RubricContent(text_property="Is the response bad?"),
      ),
  ]
  judge_model_options = JudgeModelOptions(
      judge_model_config=None,
      num_samples=3,
  )
  criterion = RubricsBasedCriterion(
      threshold=0.5, rubrics=rubrics, judge_model_options=judge_model_options
  )
  metric = EvalMetric(
      metric_name=PrebuiltMetrics.RUBRIC_BASED_FINAL_RESPONSE_QUALITY_V1.value,
      threshold=0.5,
      criterion=criterion,
  )
  return RubricBasedFinalResponseQualityV1Evaluator(metric)


def _create_per_invocation_result(
    rubric_scores: list[RubricScore],
) -> PerInvocationResult:
  """Helper to create a PerInvocationResult."""
  return PerInvocationResult(
      actual_invocation=Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="part_1")]
          )
      ),
      expected_invocation=Invocation(
          user_content=genai_types.Content(
              parts=[genai_types.Part(text="part_2")]
          )
      ),
      score=get_average_rubric_score(rubric_scores),
      rubric_scores=rubric_scores,
      eval_status=EvalStatus.NOT_EVALUATED,
  )


def test_format_auto_rater_prompt_with_basic_invocation(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with a basic invocation."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "User input here." in prompt
  assert "Final agent response." in prompt
  assert "Is the response good?" in prompt
  assert "Is the response bad?" in prompt
  assert "<developer_instructions>\n  \n  </developer_instructions>" in prompt
  assert (
      "<available_tools>\n  Agent has no tools.\n  </available_tools>" in prompt
  )
  assert (
      "<response_steps>\n  No intermediate steps were taken.\n "
      " </response_steps>"
  ) in prompt


def test_format_auto_rater_prompt_with_app_details(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with app_details in invocation."""
  tool = genai_types.Tool(
      function_declarations=[
          genai_types.FunctionDeclaration(
              name="test_func", description="A test function."
          )
      ]
  )
  app_details = AppDetails(
      agent_details={
          "agent1": AgentDetails(
              name="agent1",
              instructions="This is an agent instruction.",
              tool_declarations=[tool],
          )
      },
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      app_details=app_details,
      intermediate_data=InvocationEvents(
          invocation_events=[InvocationEvent(author="agent1", content=None)]
      ),
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "This is an agent instruction." in prompt
  assert '"name": "test_func"' in prompt
  assert '"description": "A test function."' in prompt


def test_format_auto_rater_prompt_with_intermediate_data(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with intermediate_data in invocation."""
  tool_call = genai_types.FunctionCall(
      name="test_func", args={"arg1": "val1"}, id="call1"
  )
  tool_response = genai_types.FunctionResponse(
      name="test_func", response={"result": "ok"}, id="call1"
  )
  intermediate_data = IntermediateData(
      tool_uses=[tool_call], tool_responses=[tool_response]
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      intermediate_data=intermediate_data,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"step": 0' in prompt
  assert '"tool_call":' in prompt
  assert '"name": "test_func"' in prompt
  assert '"tool_response":' in prompt
  assert '"result": "ok"' in prompt


def test_format_auto_rater_prompt_with_app_details_no_tools(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with app_details but no tools."""
  app_details = AppDetails(
      agent_details={
          "agent1": AgentDetails(name="agent1", tool_declarations=[])
      },
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      app_details=app_details,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert '"tool_declarations": {\n    "agent1": []\n  }' in prompt


def test_format_auto_rater_prompt_with_intermediate_data_no_tools(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests format_auto_rater_prompt with intermediate_data but no tool calls."""
  intermediate_data = IntermediateData(tool_uses=[], tool_responses=[])
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="Final agent response.")]
      ),
      intermediate_data=intermediate_data,
  )
  prompt = evaluator.format_auto_rater_prompt(invocation, None)

  assert "No intermediate steps were taken." in prompt


def test_parse_auto_rater_response_with_empty_string():
  """Tests _parse_auto_rater_response with an empty string."""
  assert _parse_auto_rater_response("") == []


def test_parse_auto_rater_response_with_malformed_string():
  """Tests _parse_auto_rater_response with a malformed string."""
  response = "This is just some random text without the expected format."
  assert _parse_auto_rater_response(response) == []


def test_parse_auto_rater_response_with_single_yes_verdict():
  """Tests _parse_auto_rater_response with a single 'yes' verdict."""
  response = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: yes
    """
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 1
  assert parsed[0].property_text == "Is the response good?"
  assert parsed[0].rationale == "It was good."
  assert parsed[0].score == 1.0


def test_parse_auto_rater_response_with_single_no_verdict():
  """Tests _parse_auto_rater_response with a single 'no' verdict."""
  response = """
    Property: Is the response bad?
    Rationale: It was bad.
    Verdict: no
    """
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 1
  assert parsed[0].property_text == "Is the response bad?"
  assert parsed[0].rationale == "It was bad."
  assert parsed[0].score == 0.0


def test_parse_auto_rater_response_with_invalid_verdict():
  """Tests _parse_auto_rater_response with an invalid verdict."""
  response = """
    Property: Is it unclear?
    Rationale: I cannot tell.
    Verdict: maybe
    """
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 1
  assert parsed[0].property_text == "Is it unclear?"
  assert parsed[0].rationale == "I cannot tell."
  assert parsed[0].score is None


def test_parse_auto_rater_response_with_multiple_verdicts():
  """Tests _parse_auto_rater_response with multiple verdicts."""
  response = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: yes

    Property: Is the response bad?
    Rationale: It was not bad.
    Verdict: no
    """
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 2
  assert parsed[0].property_text == "Is the response good?"
  assert parsed[0].rationale == "It was good."
  assert parsed[0].score == 1.0
  assert parsed[1].property_text == "Is the response bad?"
  assert parsed[1].rationale == "It was not bad."
  assert parsed[1].score == 0.0


def test_parse_auto_rater_response_with_incomplete_entry():
  """Tests _parse_auto_rater_response with an incomplete entry."""
  response = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: yes

    Property: Is the response bad?
    Rationale: It was not bad.
    """  # Missing Verdict
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 1  # zip will only create one item
  assert parsed[0].property_text == "Is the response good?"


def test_parse_auto_rater_response_with_case_insensitive_verdict():
  """Tests _parse_auto_rater_response is case-insensitive for verdicts."""
  response = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: Yes
    Property: Is the response bad?
    Rationale: It was bad.
    Verdict: NO
    """
  parsed = _parse_auto_rater_response(response)
  assert len(parsed) == 2
  assert parsed[0].score == 1.0
  assert parsed[1].score == 0.0


def test_convert_auto_rater_response_to_score_with_empty_response(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests convert_auto_rater_response_to_score with an empty response."""
  response = LlmResponse(
      content=genai_types.Content(parts=[genai_types.Part(text="")])
  )
  auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
  assert auto_rater_score.score is None
  assert auto_rater_score.rubric_scores == []


def test_convert_auto_rater_response_to_score_with_malformed_response(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests convert_auto_rater_response_to_score with a malformed response."""
  response = LlmResponse(
      content=genai_types.Content(
          parts=[genai_types.Part(text="This is not a valid format.")]
      )
  )
  auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
  assert auto_rater_score.score is None
  assert auto_rater_score.rubric_scores == []


def test_convert_auto_rater_response_to_score_with_mixed_verdicts(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests convert_auto_rater_response_to_score with mixed verdicts."""
  response_text = """
  Property: Is the response good?
  Rationale: It was good.
  Verdict: yes
  Property: Is the response bad?
  Rationale: It was bad.
  Verdict: no
  """
  response = LlmResponse(
      content=genai_types.Content(parts=[genai_types.Part(text=response_text)])
  )
  auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
  assert auto_rater_score.score == 0.5
  assert len(auto_rater_score.rubric_scores) == 2
  assert auto_rater_score.rubric_scores[0].score == 1.0
  assert auto_rater_score.rubric_scores[1].score == 0.0


def test_convert_auto_rater_response_to_score_with_invalid_verdict(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests convert_auto_rater_response_to_score with an invalid verdict."""
  response_text = """
  Property: Is the response good?
  Rationale: It was good.
  Verdict: yes
  Property: Is the response bad?
  Rationale: I cannot tell.
  Verdict: invalid
  """
  response = LlmResponse(
      content=genai_types.Content(parts=[genai_types.Part(text=response_text)])
  )
  auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
  assert auto_rater_score.score == 1.0
  assert len(auto_rater_score.rubric_scores) == 2
  assert auto_rater_score.rubric_scores[0].score == 1.0
  assert auto_rater_score.rubric_scores[1].score is None


def test_convert_auto_rater_response_to_score_with_unknown_property(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests convert_auto_rater_response_to_score with an unknown property."""
  response_text = """
  Property: Is the response amazing?
  Rationale: It was amazing.
  Verdict: yes
  """
  response = LlmResponse(
      content=genai_types.Content(parts=[genai_types.Part(text=response_text)])
  )
  auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
  assert auto_rater_score.score is None
  assert len(auto_rater_score.rubric_scores) == 0


def test_aggregate_per_invocation_samples_with_no_rubric_scores(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation when samples have no rubric scores."""
  samples = [
      _create_per_invocation_result([]),
      _create_per_invocation_result([]),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score is None
  assert result.rubric_scores == []


def test_aggregate_per_invocation_samples_with_majority_positive(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation with a majority of positive scores."""
  samples = [
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score == 1.0
  assert len(result.rubric_scores) == 1
  assert result.rubric_scores[0].rubric_id == "1"
  assert result.rubric_scores[0].score == 1.0


def test_aggregate_per_invocation_samples_with_majority_negative(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation with a majority of negative scores."""
  samples = [
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score == 0.0
  assert len(result.rubric_scores) == 1
  assert result.rubric_scores[0].rubric_id == "1"
  assert result.rubric_scores[0].score == 0.0


def test_aggregate_per_invocation_samples_with_tie_verdicts(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation with a tie, where negative should win."""
  samples = [
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score == 0.0
  assert len(result.rubric_scores) == 1
  assert result.rubric_scores[0].rubric_id == "1"
  assert result.rubric_scores[0].score == 0.0


def test_aggregate_per_invocation_samples_with_all_none_scores(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation when all samples have a score of None."""
  samples = [
      _create_per_invocation_result(
          [RubricScore(rubric_id="1", score=None, rationale="r1")]
      ),
      _create_per_invocation_result(
          [RubricScore(rubric_id="1", score=None, rationale="r2")]
      ),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score is None
  assert len(result.rubric_scores) == 1
  assert result.rubric_scores[0].rubric_id == "1"
  assert result.rubric_scores[0].score is None
  assert result.rubric_scores[0].rationale == "r1"


def test_aggregate_per_invocation_samples_with_multiple_rubrics(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregation with multiple rubrics."""
  samples = [
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=1.0),
          RubricScore(rubric_id="2", score=0.0),
      ]),
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=1.0),
          RubricScore(rubric_id="2", score=0.0),
      ]),
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=0.0),
          RubricScore(rubric_id="2", score=1.0),
      ]),
  ]
  result = evaluator.aggregate_per_invocation_samples(samples)
  assert result.score == 0.5
  assert len(result.rubric_scores) == 2
  rubric1_score = next(
      (s for s in result.rubric_scores if s.rubric_id == "1"), None
  )
  rubric2_score = next(
      (s for s in result.rubric_scores if s.rubric_id == "2"), None
  )
  assert rubric1_score is not None
  assert rubric1_score.score == 1.0
  assert rubric2_score is not None
  assert rubric2_score.score == 0.0


def test_aggregate_invocation_results_with_empty_list(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with an empty list."""
  result = evaluator.aggregate_invocation_results([])
  assert isinstance(result, EvaluationResult)
  assert result.overall_score is None
  assert result.overall_rubric_scores == []
  assert result.per_invocation_results == []


def test_aggregate_invocation_results_with_no_rubric_scores(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with samples that have no rubric scores."""
  invocations = [
      _create_per_invocation_result([]),
      _create_per_invocation_result([]),
  ]
  result = evaluator.aggregate_invocation_results(invocations)
  assert result.overall_score is None
  assert result.overall_rubric_scores == []
  assert result.per_invocation_results == invocations


def test_aggregate_invocation_results_with_single_invocation(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with a single invocation result."""
  invocations = [
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=1.0),
          RubricScore(rubric_id="2", score=0.0),
      ])
  ]
  result = evaluator.aggregate_invocation_results(invocations)
  assert result.overall_score == 0.5
  assert len(result.overall_rubric_scores) == 2
  rubric1_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "1"
  )
  rubric2_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "2"
  )
  assert rubric1_score.score == 1.0
  assert rubric2_score.score == 0.0


def test_aggregate_invocation_results_with_multiple_invocations_single_rubric(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with multiple invocations for a single rubric."""
  invocations = [
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
      _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
  ]
  result = evaluator.aggregate_invocation_results(invocations)
  assert result.overall_score == pytest.approx(2 / 3)
  assert len(result.overall_rubric_scores) == 1
  assert result.overall_rubric_scores[0].rubric_id == "1"
  assert result.overall_rubric_scores[0].score == pytest.approx(2 / 3)


def test_aggregate_invocation_results_with_multiple_invocations_and_rubrics(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with multiple invocations and rubrics."""
  invocations = [
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=1.0),
          RubricScore(rubric_id="2", score=0.0),
      ]),
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=0.0),
          RubricScore(rubric_id="2", score=1.0),
      ]),
  ]
  result = evaluator.aggregate_invocation_results(invocations)
  assert result.overall_score == 0.5
  assert len(result.overall_rubric_scores) == 2
  rubric1_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "1"
  )
  rubric2_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "2"
  )
  assert rubric1_score.score == 0.5
  assert rubric2_score.score == 0.5


def test_aggregate_invocation_results_with_none_scores(
    evaluator: RubricBasedFinalResponseQualityV1Evaluator,
):
  """Tests aggregate_invocation_results with some None scores."""
  invocations = [
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=1.0),
          RubricScore(rubric_id="2", score=None),
      ]),
      _create_per_invocation_result([
          RubricScore(rubric_id="1", score=0.0),
          RubricScore(rubric_id="2", score=1.0),
      ]),
  ]
  result = evaluator.aggregate_invocation_results(invocations)
  assert result.overall_score == pytest.approx(2 / 3)
  assert len(result.overall_rubric_scores) == 2
  rubric1_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "1"
  )
  rubric2_score = next(
      s for s in result.overall_rubric_scores if s.rubric_id == "2"
  )
  assert rubric1_score.score == 0.5
  assert rubric2_score.score == 1.0
