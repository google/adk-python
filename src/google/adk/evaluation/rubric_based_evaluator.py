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

import abc
import logging
import re
from typing import Optional

from typing_extensions import override

from ..models.llm_response import LlmResponse
from ..utils.feature_decorator import experimental
from .common import EvalBaseModel
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .eval_rubrics import Rubric
from .eval_rubrics import RubricScore
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .llm_as_judge import AutoRaterScore
from .llm_as_judge import LlmAsJudge
from .llm_as_judge_utils import get_average_rubric_score
from .llm_as_judge_utils import get_eval_status
from .llm_as_judge_utils import get_text_from_content

logger = logging.getLogger("google_adk." + __name__)


class RubricResponse(EvalBaseModel):
  """Internal data model to represent a rubric's response from the auto-rater."""

  property_text: Optional[str] = None
  rationale: Optional[str] = None
  score: Optional[float] = None


class AutoRaterResponseParser(abc.ABC):
  """An interface for parsing auto rater's response."""

  @abc.abstractmethod
  def parse(self, auto_rater_response: str) -> list[RubricResponse]:
    """Parses the auto rater's response."""
    raise NotImplementedError


_PROPERTY_PATTERN = r"(?<=Property: )(.*)"
_RATIONALE_PATTERN = r"(?<=Rationale: )(.*)"
_VERDICT_PATTERN = r"(?<=Verdict: )(.*)"


class DefaultAutoRaterResponseParser(AutoRaterResponseParser):
  """The default implementation of the AutoRaterResponseParser."""

  def parse(self, auto_rater_response: str) -> list[RubricResponse]:
    """Returns a list of RubricResponse parsed from the AutoRater's response."""
    properties = re.findall(_PROPERTY_PATTERN, auto_rater_response)
    rationales = re.findall(_RATIONALE_PATTERN, auto_rater_response)
    scores = []

    for verdict in re.findall(_VERDICT_PATTERN, auto_rater_response):
      if "yes" in verdict.lower():
        score = 1.0
      elif "no" in verdict.lower():
        score = 0.0
      else:
        score = None

      scores.append(score)

    rubric_responses = []
    for p, r, s in zip(properties, rationales, scores):
      rubric_responses.append(
          RubricResponse(property_text=p.strip(), rationale=r.strip(), score=s)
      )

    return rubric_responses


class PerInvocationResultsAggregator(abc.ABC):
  """An interface for aggregating per invocation samples.

  AutoRaters that are backed by an LLM are known to have certain degree of
  unreliabilty to their responses. In order to counter that we sample the
  autorater more than once for a single invocation.

  The aggregator helps convert those multiple samples into a single result.
  """

  @abc.abstractmethod
  def aggregate(
      self,
      per_invocation_samples: list[PerInvocationResult],
      threshold: float,
  ) -> PerInvocationResult:
    """Aggregates per invocation samples into a single result."""
    raise NotImplementedError


class MajorityVotePerInvocationResultsAggregator(
    PerInvocationResultsAggregator
):
  """Aggregates per invocation samples using majority vote."""

  def aggregate(
      self,
      per_invocation_samples: list[PerInvocationResult],
      threshold: float,
  ) -> PerInvocationResult:
    """Returns a combined result for the invocation using majority vote.

    This method takes all those samples for a single invocation and combines
    them to generate one single result for the invocation.

    This method specifically uses majority vote to aggregate scores for a
    rubric. Take following Invocation and Rubric for example:

      Invocation:
         User: Is it going to be cold in Seattle tomorrow?
         Weather Agent: No, it will be moderately warm as predicted temperature
         for Seattle, WA tomorrow is 88F.

      Rubric: Agent's response was concise and to the point.

      We will sample the AutoRater 5 times, and the AutoRater responds
      with (skipping the rationale field for now):
        Sample 1:
          Verdict: Yes
        Sample 2:
          Verdict: No
        Sample 3:
          Verdict: Yes
        Sample 4:
          Verdict: Yes
        Sample 5:
          Verdict: No

      This method will use majority vote and combine the results of 5 samples
      into one, and it will report "Yes" as the final verdict.
    """
    score_category_by_rubric_id = {}

    # We go over each rubric for each sample, and categorize the rubric into
    # one of the following buckets:
    #  - Bucket 0: No score was generated for the rubric
    #  - Bucket 1: Score was generated and it was positive (1.0)
    #  - Bucket 2: Score was generated and it was negative (0.0)
    for sample in per_invocation_samples:
      if not sample.rubric_scores:
        continue

      for rubric_score in sample.rubric_scores:
        rubric_id = rubric_score.rubric_id
        if rubric_id not in score_category_by_rubric_id:
          score_category_by_rubric_id[rubric_id] = ([], [], [])

        if rubric_score.score is None:  # No score
          score_category_by_rubric_id[rubric_id][0].append(rubric_score)
        elif rubric_score.score == 1.0:  # Positive Result
          score_category_by_rubric_id[rubric_id][1].append(rubric_score)
        else:  # Negative result
          score_category_by_rubric_id[rubric_id][2].append(rubric_score)

    aggregated_rubric_scores = []
    for rubric_id in score_category_by_rubric_id:
      no_scores, positives, negatives = score_category_by_rubric_id[rubric_id]

      if not positives and not negatives:
        # There has to be at least a no score rubric!
        aggregated_rubric_scores.append(no_scores[0])

      # This is where we are taking a majority vote.
      elif len(positives) > len(negatives):
        aggregated_rubric_scores.append(positives[0])
      else:
        aggregated_rubric_scores.append(negatives[0])

    aggregated_overall_score = get_average_rubric_score(
        aggregated_rubric_scores
    )

    return PerInvocationResult(
        actual_invocation=per_invocation_samples[0].actual_invocation,
        expected_invocation=per_invocation_samples[0].expected_invocation,
        score=aggregated_overall_score,
        rubric_scores=aggregated_rubric_scores,
        eval_status=get_eval_status(aggregated_overall_score, threshold),
    )


class InvocationResultsSummarizer(abc.ABC):
  """An interface for summarizing per invocation results."""

  @abc.abstractmethod
  def summarize(
      self, per_invocation_results: list[PerInvocationResult], threshold: float
  ) -> EvaluationResult:
    """Summaries per invocation results into a single result."""
    raise NotImplementedError


class MeanInvocationResultsSummarizer(InvocationResultsSummarizer):
  """Summarizes per invocation results using mean score."""

  def summarize(
      self, per_invocation_results: list[PerInvocationResult], threshold: float
  ) -> EvaluationResult:
    """Summarizes per invocation evaluation results into a single score.

    A single eval case can have multiple invocations and the eval metric is
    assessed for each invocation. But, we do want to summarize and make a
    statement on how the eval case as a whole performed on the metric.

    This method helps us aggregate rubric scores across invocation.

    This method calculates the mean score of a rubric across several
    invocations.
    """

    unaggregated_rubric_scores = []  # Later used to calculate average.

    # Collect rubric scores by id, so that we can calculate average score
    # for each rubric id.
    rubric_scores_by_id = {}
    for sample in per_invocation_results:
      if not sample.rubric_scores:
        continue

      for rubric_score in sample.rubric_scores:
        rubric_id = rubric_score.rubric_id
        if rubric_id not in rubric_scores_by_id:
          rubric_scores_by_id[rubric_id] = []

        rubric_scores_by_id[rubric_id].append(rubric_score)
        unaggregated_rubric_scores.append(rubric_score)

    aggregated_rubric_scores = []
    for rubric_id, rubric_scores in rubric_scores_by_id.items():
      overall_score = get_average_rubric_score(rubric_scores)
      aggregated_rubric_scores.append(
          RubricScore(
              rubric_id=rubric_id,
              score=overall_score,
              # There is no real way for us generate a rationale here, so we
              # make is clear to the consumer of the result.
              rationale=(
                  "This is an aggregated score derived from individual entries."
                  " Please refer to individual entries in each invocation for"
                  " actual rationale from the model."
              ),
          )
      )

    # Use unaggregate rubric score to calculate overall score.
    aggregated_overall_score = get_average_rubric_score(
        unaggregated_rubric_scores
    )
    return EvaluationResult(
        overall_score=aggregated_overall_score,
        overall_eval_status=get_eval_status(
            aggregated_overall_score, threshold
        ),
        per_invocation_results=per_invocation_results,
        overall_rubric_scores=aggregated_rubric_scores,
    )


def _normalize_text(text: str) -> str:
  """Returns a normalized version of the passed in text."""
  if not isinstance(text, str):
    return ""
  return text.lower().strip()


@experimental
class RubricBasedEvaluator(LlmAsJudge):
  """A base class for rubric based evaluators."""

  def __init__(
      self,
      eval_metric: EvalMetric,
      criterion_type: type[BaseCriterion],
      auto_rater_response_parser: AutoRaterResponseParser = (
          DefaultAutoRaterResponseParser()
      ),
      per_invocation_results_aggregator: PerInvocationResultsAggregator = (
          MajorityVotePerInvocationResultsAggregator()
      ),
      invocation_results_summarizer: InvocationResultsSummarizer = (
          MeanInvocationResultsSummarizer()
      ),
  ):
    """Initializes the RubricBasedEvaluator.

    Args:
      eval_metric: The evaluation metric configuration.
      criterion_type: The type of the criterion used for this evaluator.
      auto_rater_response_parser: An object that parses the auto-rater's
        response text and extracts rubric scores.
      per_invocation_results_aggregator: An object that aggregates multiple
        samples for a single invocation into a single result. This is useful in
        cases where the auto-rater is an LLM and multiple samples are generated
        to account for the unreliability of the LLM.
      invocation_results_summarizer: An object that summarizes the results of
        all invocations in an eval case into a single result.
    """
    super().__init__(
        eval_metric,
        criterion_type=criterion_type,
    )
    self._auto_rater_prompt_template = ""
    self._auto_rater_response_parser = auto_rater_response_parser
    self._per_invocation_results_aggregator = per_invocation_results_aggregator
    self._invocation_results_summarizer = invocation_results_summarizer

    assert self._criterion.rubrics, "Rubrics are required."

    self._rubrics: list[Rubric] = self._criterion.rubrics

    self._normalized_rubric_to_id_map = {
        _normalize_text(r.rubric_content.text_property): r.rubric_id
        for r in self._rubrics
    }

  @override
  def convert_auto_rater_response_to_score(
      self, auto_rater_response: LlmResponse
  ) -> AutoRaterScore:
    """Returns an AutoRaterScore generated from AutoRater's response."""
    response_text = get_text_from_content(auto_rater_response.content)
    rubric_responses = self._auto_rater_response_parser.parse(response_text)
    rubric_scores = []

    for rubric_response in rubric_responses:
      normalized_rubric = _normalize_text(rubric_response.property_text)
      rubric_id = self._normalized_rubric_to_id_map.get(normalized_rubric, None)
      if rubric_id:
        rubric_scores.append(
            RubricScore(
                rubric_id=rubric_id,
                rationale=rubric_response.rationale,
                score=rubric_response.score,
            )
        )
      else:
        logger.warning(
            f"Rubric {rubric_response.property_text} not found in the rubrics"
            " provided to the metric."
        )

    aggregated_score = get_average_rubric_score(rubric_scores)
    return AutoRaterScore(score=aggregated_score, rubric_scores=rubric_scores)

  @override
  def aggregate_per_invocation_samples(
      self,
      per_invocation_samples: list[PerInvocationResult],
  ) -> PerInvocationResult:
    """Returns a combined result by aggregating multiple samples for the same invocation.

    AutoRaters that are backed by an LLM are known to have certain degree of
    unreliabilty to their responses. In order to counter that we sample the
    autorater more than once for a single invocation.

    The aggregator helps convert those multiple samples into a single result.
    """
    return self._per_invocation_results_aggregator.aggregate(
        per_invocation_samples, self._eval_metric.threshold
    )

  @override
  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Summarizes per invocation evaluation results into a single score."""
    return self._invocation_results_summarizer.summarize(
        per_invocation_results, self._eval_metric.threshold
    )
