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

from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import JudgeModelOptions
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.llm_as_judge_utils import get_average_rubric_score
from google.adk.evaluation.rubric_based_evaluator import DefaultAutoRaterResponseParser
from google.adk.evaluation.rubric_based_evaluator import MajorityVotePerInvocationResultsAggregator
from google.adk.evaluation.rubric_based_evaluator import MeanInvocationResultsSummarizer
from google.adk.evaluation.rubric_based_evaluator import RubricBasedEvaluator
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
import pytest


class FakeRubricBasedEvaluator(RubricBasedEvaluator):
  """A fake implementation of RubricBasedEvaluator intended for testing."""

  def __init__(
      self,
      eval_metric: EvalMetric,
      rubric_type: str | None = None,
  ):
    super().__init__(
        eval_metric,
        criterion_type=RubricsBasedCriterion,
        rubric_type=rubric_type,
    )

  def format_auto_rater_prompt(
      self, actual: Invocation, expected: Invocation
  ) -> str:
    return "fake response"


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


class TestDefaultAutoRaterResponseParser:
  """Test cases for DefaultAutoRaterResponseParser."""

  def test_parse_auto_rater_response_with_empty_string(self):
    """Tests _parse_auto_rater_response with an empty string."""
    assert DefaultAutoRaterResponseParser().parse("") == []

  def test_parse_auto_rater_response_with_malformed_string(self):
    """Tests _parse_auto_rater_response with a malformed string."""
    response = "This is just some random text without the expected format."
    assert DefaultAutoRaterResponseParser().parse(response) == []

  def test_parse_auto_rater_response_with_single_yes_verdict(self):
    """Tests _parse_auto_rater_response with a single 'yes' verdict."""
    response = """
      Property: Is the response good?
      Rationale: It was good.
      Verdict: yes
      """
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 1
    assert parsed[0].property_text == "Is the response good?"
    assert parsed[0].rationale == "It was good."
    assert parsed[0].score == 1.0

  def test_parse_auto_rater_response_with_single_no_verdict(self):
    """Tests _parse_auto_rater_response with a single 'no' verdict."""
    response = """
      Property: Is the response bad?
      Rationale: It was bad.
      Verdict: no
      """
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 1
    assert parsed[0].property_text == "Is the response bad?"
    assert parsed[0].rationale == "It was bad."
    assert parsed[0].score == 0.0

  def test_parse_auto_rater_response_with_invalid_verdict(self):
    """Tests _parse_auto_rater_response with an invalid verdict."""
    response = """
      Property: Is it unclear?
      Rationale: I cannot tell.
      Verdict: maybe
      """
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 1
    assert parsed[0].property_text == "Is it unclear?"
    assert parsed[0].rationale == "I cannot tell."
    assert parsed[0].score is None

  def test_parse_auto_rater_response_with_multiple_verdicts(self):
    """Tests _parse_auto_rater_response with multiple verdicts."""
    response = """
      Property: Is the response good?
      Rationale: It was good.
      Verdict: yes

      Property: Is the response bad?
      Rationale: It was not bad.
      Verdict: no
      """
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 2
    assert parsed[0].property_text == "Is the response good?"
    assert parsed[0].rationale == "It was good."
    assert parsed[0].score == 1.0
    assert parsed[1].property_text == "Is the response bad?"
    assert parsed[1].rationale == "It was not bad."
    assert parsed[1].score == 0.0

  def test_parse_auto_rater_response_with_incomplete_entry(self):
    """Tests _parse_auto_rater_response with an incomplete entry."""
    response = """
      Property: Is the response good?
      Rationale: It was good.
      Verdict: yes

      Property: Is the response bad?
      Rationale: It was not bad.
      """  # Missing Verdict
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 1  # zip will only create one item
    assert parsed[0].property_text == "Is the response good?"

  def test_parse_auto_rater_response_with_case_insensitive_verdict(self):
    """Tests _parse_auto_rater_response is case-insensitive for verdicts."""
    response = """
      Property: Is the response good?
      Rationale: It was good.
      Verdict: Yes
      Property: Is the response bad?
      Rationale: It was bad.
      Verdict: NO
      """
    parsed = DefaultAutoRaterResponseParser().parse(response)
    assert len(parsed) == 2
    assert parsed[0].score == 1.0
    assert parsed[1].score == 0.0


class TestMajorityVotePerInvocationResultsAggregator:

  def test_aggregate_per_invocation_samples_with_no_rubric_scores(
      self,
  ):
    """Tests aggregation when samples have no rubric scores."""
    samples = [
        _create_per_invocation_result([]),
        _create_per_invocation_result([]),
    ]

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

    assert result.score is None
    assert result.rubric_scores == []

  def test_aggregate_per_invocation_samples_with_majority_positive(
      self,
  ):
    """Tests aggregation with a majority of positive scores."""
    samples = [
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
    ]

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

    assert result.score == 1.0
    assert len(result.rubric_scores) == 1
    assert result.rubric_scores[0].rubric_id == "1"
    assert result.rubric_scores[0].score == 1.0

  def test_aggregate_per_invocation_samples_with_majority_negative(
      self,
  ):
    """Tests aggregation with a majority of negative scores."""
    samples = [
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
    ]

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

    assert result.score == 0.0
    assert len(result.rubric_scores) == 1
    assert result.rubric_scores[0].rubric_id == "1"
    assert result.rubric_scores[0].score == 0.0

  def test_aggregate_per_invocation_samples_with_tie_verdicts(
      self,
  ):
    """Tests aggregation with a tie, where negative should win."""
    samples = [
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
    ]

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

    assert result.score == 0.0
    assert len(result.rubric_scores) == 1
    assert result.rubric_scores[0].rubric_id == "1"
    assert result.rubric_scores[0].score == 0.0

  def test_aggregate_per_invocation_samples_with_all_none_scores(
      self,
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

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

    assert result.score is None
    assert len(result.rubric_scores) == 1
    assert result.rubric_scores[0].rubric_id == "1"
    assert result.rubric_scores[0].score is None
    assert result.rubric_scores[0].rationale == "r1"

  def test_aggregate_per_invocation_samples_with_multiple_rubrics(
      self,
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

    result = MajorityVotePerInvocationResultsAggregator().aggregate(
        samples, threshold=0.5
    )

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


class TestMeanInvocationResultsSummarizer:
  """Test cases for MeanInvocationResultsSummarizer."""

  def test_summarize_with_empty_list(
      self,
  ):
    """Tests aggregate_invocation_results with an empty list."""
    result = MeanInvocationResultsSummarizer().summarize([], threshold=0.5)
    assert result.overall_score is None
    assert result.overall_rubric_scores == []
    assert result.per_invocation_results == []

  def test_summarize_with_no_rubric_scores(
      self,
  ):
    """Tests aggregate_invocation_results with samples that have no rubric scores."""
    invocations = [
        _create_per_invocation_result([]),
        _create_per_invocation_result([]),
    ]
    result = MeanInvocationResultsSummarizer().summarize(
        invocations, threshold=0.5
    )
    assert result.overall_score is None
    assert result.overall_rubric_scores == []
    assert result.per_invocation_results == invocations

  def test_summarize_with_single_invocation(
      self,
  ):
    """Tests aggregate_invocation_results with a single invocation result."""
    invocations = [
        _create_per_invocation_result([
            RubricScore(rubric_id="1", score=1.0),
            RubricScore(rubric_id="2", score=0.0),
        ])
    ]
    result = MeanInvocationResultsSummarizer().summarize(
        invocations, threshold=0.5
    )
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

  def test_summarize_with_multiple_invocations_single_rubric(
      self,
  ):
    """Tests aggregate_invocation_results with multiple invocations for a single rubric."""
    invocations = [
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=0.0)]),
        _create_per_invocation_result([RubricScore(rubric_id="1", score=1.0)]),
    ]
    result = MeanInvocationResultsSummarizer().summarize(
        invocations, threshold=0.5
    )
    assert result.overall_score == pytest.approx(2 / 3)
    assert len(result.overall_rubric_scores) == 1
    assert result.overall_rubric_scores[0].rubric_id == "1"
    assert result.overall_rubric_scores[0].score == pytest.approx(2 / 3)

  def test_summarize_with_multiple_invocations_and_rubrics(
      self,
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
    result = MeanInvocationResultsSummarizer().summarize(
        invocations, threshold=0.5
    )
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

  def test_summarize_with_none_scores(
      self,
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
    result = MeanInvocationResultsSummarizer().summarize(
        invocations, threshold=0.5
    )
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


class TestRubricBasedEvaluator:
  """Tests for RubricBasedEvaluator."""

  @pytest.fixture
  def evaluator(self) -> FakeRubricBasedEvaluator:
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
    return FakeRubricBasedEvaluator(metric)

  def test_convert_auto_rater_response_to_score_with_empty_response(
      self,
      evaluator: RubricBasedEvaluator,
  ):
    """Tests convert_auto_rater_response_to_score with an empty response."""
    evaluator.create_effective_rubrics_list(None)
    response = LlmResponse(
        content=genai_types.Content(parts=[genai_types.Part(text="")])
    )
    auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
    assert auto_rater_score.score is None
    assert auto_rater_score.rubric_scores == []

  def test_convert_auto_rater_response_to_score_with_malformed_response(
      self,
      evaluator: RubricBasedEvaluator,
  ):
    """Tests convert_auto_rater_response_to_score with a malformed response."""
    evaluator.create_effective_rubrics_list(None)
    response = LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text="This is not a valid format.")]
        )
    )
    auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
    assert auto_rater_score.score is None
    assert auto_rater_score.rubric_scores == []

  def test_convert_auto_rater_response_to_score_with_mixed_verdicts(
      self,
      evaluator: RubricBasedEvaluator,
  ):
    """Tests convert_auto_rater_response_to_score with mixed verdicts."""
    evaluator.create_effective_rubrics_list(None)
    response_text = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: yes
    Property: Is the response bad?
    Rationale: It was bad.
    Verdict: no
    """
    response = LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text=response_text)]
        )
    )
    auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
    assert auto_rater_score.score == 0.5
    assert len(auto_rater_score.rubric_scores) == 2
    assert auto_rater_score.rubric_scores[0].score == 1.0
    assert auto_rater_score.rubric_scores[1].score == 0.0

  def test_convert_auto_rater_response_to_score_with_invalid_verdict(
      self,
      evaluator: RubricBasedEvaluator,
  ):
    """Tests convert_auto_rater_response_to_score with an invalid verdict."""
    evaluator.create_effective_rubrics_list(None)
    response_text = """
    Property: Is the response good?
    Rationale: It was good.
    Verdict: yes
    Property: Is the response bad?
    Rationale: I cannot tell.
    Verdict: invalid
    """
    response = LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text=response_text)]
        )
    )
    auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
    assert auto_rater_score.score == 1.0
    assert len(auto_rater_score.rubric_scores) == 2
    assert auto_rater_score.rubric_scores[0].score == 1.0
    assert auto_rater_score.rubric_scores[1].score is None

  def test_convert_auto_rater_response_to_score_with_unknown_property(
      self,
      evaluator: RubricBasedEvaluator,
  ):
    """Tests convert_auto_rater_response_to_score with an unknown property."""
    evaluator.create_effective_rubrics_list(None)
    response_text = """
    Property: Is the response amazing?
    Rationale: It was amazing.
    Verdict: yes
    """
    response = LlmResponse(
        content=genai_types.Content(
            parts=[genai_types.Part(text=response_text)]
        )
    )
    auto_rater_score = evaluator.convert_auto_rater_response_to_score(response)
    assert auto_rater_score.score is None
    assert auto_rater_score.rubric_scores == []

  def test_create_effective_rubrics_list_with_invocation_rubrics(
      self, evaluator: RubricBasedEvaluator
  ):
    invocation_rubrics = [
        Rubric(
            rubric_id="3",
            rubric_content=RubricContent(text_property="Invocation rubric"),
        )
    ]
    evaluator.create_effective_rubrics_list(invocation_rubrics)
    effective_rubrics = evaluator.get_effective_rubrics_list()
    assert len(effective_rubrics) == 3
    assert {r.rubric_id for r in effective_rubrics} == {"1", "2", "3"}

  def test_create_effective_rubrics_list_with_duplicate_invocation_rubric_id(
      self, evaluator: RubricBasedEvaluator
  ):
    invocation_rubrics = [
        Rubric(
            rubric_id="1",
            rubric_content=RubricContent(text_property="Invocation rubric"),
        )
    ]
    with pytest.raises(
        ValueError, match="Rubric with rubric_id '1' already exists."
    ):
      evaluator.create_effective_rubrics_list(invocation_rubrics)

  def test_create_effective_rubrics_list_with_no_invocation_rubrics(
      self, evaluator: RubricBasedEvaluator
  ):
    evaluator.create_effective_rubrics_list(None)
    effective_rubrics = evaluator.get_effective_rubrics_list()
    assert len(effective_rubrics) == 2
    assert {r.rubric_id for r in effective_rubrics} == {"1", "2"}

  def test_get_effective_rubrics_list_before_creation_raises_error(
      self, evaluator: RubricBasedEvaluator
  ):
    with pytest.raises(
        ValueError, match="Effective rubrics list not initialized."
    ):
      evaluator.get_effective_rubrics_list()

  def test_create_effective_rubrics_list_multiple_calls(
      self, evaluator: RubricBasedEvaluator
  ):
    invocation_rubrics1 = [
        Rubric(
            rubric_id="3",
            rubric_content=RubricContent(text_property="Invocation rubric 1"),
        )
    ]
    evaluator.create_effective_rubrics_list(invocation_rubrics1)
    effective_rubrics1 = evaluator.get_effective_rubrics_list()
    assert len(effective_rubrics1) == 3
    assert {r.rubric_id for r in effective_rubrics1} == {"1", "2", "3"}

    invocation_rubrics2 = [
        Rubric(
            rubric_id="4",
            rubric_content=RubricContent(text_property="Invocation rubric 2"),
        )
    ]
    evaluator.create_effective_rubrics_list(invocation_rubrics2)
    effective_rubrics2 = evaluator.get_effective_rubrics_list()
    assert len(effective_rubrics2) == 3
    assert {r.rubric_id for r in effective_rubrics2} == {"1", "2", "4"}

  def test_create_effective_rubrics_filters_by_rubric_type(
      self, evaluator: RubricBasedEvaluator
  ):
    evaluator_with_type = FakeRubricBasedEvaluator(
        evaluator._eval_metric, rubric_type="TEST_TYPE"
    )
    invocation_rubrics = [
        Rubric(
            rubric_id="test_type_rubric",
            rubric_content=RubricContent(text_property="Invocation rubric 1"),
            type="TEST_TYPE",
        ),
        Rubric(
            rubric_id="other_type_rubric",
            rubric_content=RubricContent(text_property="Invocation rubric 2"),
            type="OTHER_TYPE",
        ),
    ]
    evaluator_with_type.create_effective_rubrics_list(invocation_rubrics)
    effective_rubrics = evaluator_with_type.get_effective_rubrics_list()
    assert len(effective_rubrics) == 3
    assert {r.rubric_id for r in effective_rubrics} == {
        "1",
        "2",
        "test_type_rubric",
    }
