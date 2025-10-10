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

import logging
import re
from typing import ClassVar
from typing import Optional

from typing_extensions import override

from ..models.llm_response import LlmResponse
from ..utils.feature_decorator import experimental
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import EvalStatus
from .eval_metrics import Interval
from .eval_metrics import LlmAsAJudgeCriterion
from .eval_metrics import MetricInfo
from .eval_metrics import MetricValueInfo
from .eval_metrics import PrebuiltMetrics
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .llm_as_judge import AutoRaterScore
from .llm_as_judge import LlmAsJudge
from .llm_as_judge_utils import get_eval_status
from .llm_as_judge_utils import get_text_from_content
from .llm_as_judge_utils import Label

logger = logging.getLogger("google_adk." + __name__)

_FINAL_RESPONSE_MATCH_V2_PROMPT = """You are an expert rater for an AI agent. The AI agent is going to call an API to answer the user query and generate API tool use code based for the choice of the API and API arguments. The ideal model response should be a function call that fulfills user query, or a natural language response hedges or asks users for further clarification if a function call does not apply.
The primary focus of this rating task is to check correctness of the model responses.

The data consists of:
- A user query.
- A model generated response for the prompt. The responses can consist of:
  - Natural language, when the model is asking for clarification, or tells the user it does not possess the requested functionality / option.
  - Code, in the form of one or multiple python function calls, and additional code as needed, for when the model is fulfilling the user request.
You can use the help from a reference response annotated by a human rater. This reference response is of high quality. You can compare the agent's response with the reference response and decide if the agent's response is valid.
Note sometimes the reference response only contains the key entities of the correct answer and you need to be flexible to allow the agent response to contain more information than the reference response, or to present the key entities in a different format or structure or in shorter or longer format.
When the agent response is provided in the form of tables/dataframes or should be best provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response. Likewise, if you have the reference response, then find out the key entities and main components in them and check whether you can retrieve those from the agent response. If the prompt does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.

You should follow the constitutions below very carefully to rate the model response:
- Allow flexibility of format even when reference code only uses one of the possible format, unless API spec or user prompt has explicit format requirement
  - e.g. For state name, allow both abbreviation and full name unless API spec has explicit requirement. e.g. both 'tx' and 'Texas' should be allowed in the agent response even when reference code only uses one of them.
  - e.g. If a reference response list outputs in a list format, the agent response is allowed to use sentence format and vice versa unless user prompt explicitly asks for a specific format.
  - e.g. For numbers, allow flexibility of formatting, e.g. 1000000 vs 1,000,000.
- The model shouldn't assume that it doesn't have access to according data or incapable of answering the question if reference response is able to find a legit answer.
- If the model response contains the correct final answer, rate it as valid even when the model response contains more information than the reference response.
- If the user prompt has csv or other table format data, don't read it yourself. Trust the reference response final answer instead.
- When the validation needs maths, date calculations, do not use your own calculator. Trust the reference response final answer instead.
- Be mindful about unit of numbers. For example, if the reference response says 100 miles, but the model response says 100 km, it is invalid.
- When the agent response or the reference response is provided in the form of tables/dataframes: focus on the key entities and main components requested in the user query and check whether you can retrieve those from the agent response and whether those match the reference response. If the user query does not specify any format instructions and the main items/components are included in the response then tolerate the differences in the formatting of those tables/dataframes.
- When the answer is in numeric format, check whether there are any format requirements in the numeric format, rounding, precision, number of decimals, etc. specified in the user query and the prompt. If there are no such instructions, then tolerate different numerical formats.
- When the answer is in numeric format and there are rounding or precision differences between the agent response and the reference response, if no further instructions are provided evaluate if the rounding strategy or precision in the agent response follows the standards for that entity. For instance, model accuracy scores must be reported with at least two decimal places (e.g., 0.798 → 0.80 is acceptable,  but 0.7 is not).

Below are the inputs:
{{
  "User prompt": {prompt},
  "Agent response": {response},
  "Reference response": {golden_response},
}}

The answer should be a json alone which follows the json structure below:
{{
  "reasoning": [reasoning],
  "is_the_agent_response_valid": [valid or invalid],
}}
Answer with assertiveness:
"""


def _parse_critique(response: str) -> Label:
  """Parses the judge model critique and extracts the final label.

  Args:
    response: model response

  Returns:
    The extracted label, either VALID, INVALID, or NOT_FOUND.
  """
  # Regex matching the label field in the response. The end of the field is
  # identified by either a comma, new line, or an end-bracket.
  label_match_is_response_valid = re.search(
      r'"is_the_agent_response_valid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]',
      response,
  )
  # In case the model names the label field as "is_the_agent_response_*invalid*"
  # instead of "..._*valid*"
  label_match_is_response_invalid = re.search(
      r'"is_the_agent_response_invalid":\s*\[*[\n\s]*"*([^"^\]^\s]*)"*[\n\s]*\]*\s*[,\n\}]',
      response,
  )
  # Remove any trailing whitespace, commas, or end-brackets from the label.
  if label_match_is_response_valid:
    label = label_match_is_response_valid.group(1).strip(r"\s,\}")
    if label in [
        Label.INVALID.value,
        Label.ALMOST.value,
        Label.FALSE.value,
        *Label.PARTIALLY_VALID.value,
    ]:
      label = Label.INVALID
    elif label in [Label.VALID.value, Label.TRUE.value]:
      label = Label.VALID
    else:
      label = Label.NOT_FOUND
  elif label_match_is_response_invalid:
    label = label_match_is_response_invalid.group(1).strip(r"\s,\}")
    label = (
        Label.INVALID
        if label in [Label.TRUE.value, Label.INVALID.value]
        else Label.VALID
    )
  else:
    label = Label.NOT_FOUND
  return label


@experimental
class FinalResponseMatchV2Evaluator(LlmAsJudge):
  """V2 final response match evaluator which uses an LLM to judge responses.

  The evaluator prompts the LLM to output whether the agent final response is
  valid or invalid, hence outputs a score of 0 or 1. Repeated invocation samples
  are aggregated by taking majority vote, and then the overall score is the
  fraction, ranging from 0 to 1, of valid samples. Higher values of overall
  score indicate better final response performance of the agent.
  """

  criterion_type: ClassVar[type[LlmAsAJudgeCriterion]] = LlmAsAJudgeCriterion

  def __init__(
      self,
      eval_metric: EvalMetric,
  ):
    super().__init__(eval_metric, FinalResponseMatchV2Evaluator.criterion_type)
    self._auto_rater_prompt_template = _FINAL_RESPONSE_MATCH_V2_PROMPT

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name=PrebuiltMetrics.FINAL_RESPONSE_MATCH_V2.value,
        description=(
            "This metric evaluates if the agent's final response matches a"
            " golden/expected final response using LLM as a judge. Value range"
            " for this metric is [0,1], with values closer to 1 more desirable."
        ),
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  @override
  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    reference = get_text_from_content(expected_invocation.final_response)
    response = get_text_from_content(actual_invocation.final_response)
    user_prompt = get_text_from_content(expected_invocation.user_content)
    return self._auto_rater_prompt_template.format(
        prompt=user_prompt,
        response=response,
        golden_response=reference,
    )

  @override
  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> AutoRaterScore:
    response_text = get_text_from_content(llm_response.content)
    if response_text is None:
      return AutoRaterScore()
    label = _parse_critique(response_text)
    if label == Label.VALID:
      return AutoRaterScore(score=1.0)
    elif label == Label.INVALID:
      return AutoRaterScore(score=0.0)
    else:
      return AutoRaterScore()

  @override
  def aggregate_per_invocation_samples(
      self,
      per_invocation_samples: list[PerInvocationResult],
  ) -> PerInvocationResult:
    """Aggregates samples of per-invocation results by taking majority vote.

    Only consider results that were successfully evaluated. In the case of a
    tie, consider the result to be invalid.

    Args:
      per_invocation_samples: Samples of per-invocation results to aggregate.

    Returns:
      If there is a majority of valid results, return the first valid result.
      Otherwise, return the first invalid result. If no results were
      successfully evaluated, return the first sample.
    """
    positive_results = []
    negative_results = []
    for result in per_invocation_samples:
      if result.score == 1.0:
        positive_results.append(result)
      elif result.score == 0.0:
        negative_results.append(result)
    # If no results were successfully evaluated, just return the first sample.
    if not positive_results and not negative_results:
      return per_invocation_samples[0]
    elif len(positive_results) > len(negative_results):
      return positive_results[0]
    else:
      return negative_results[0]

  @override
  def aggregate_invocation_results(
      self, per_invocation_results: list[PerInvocationResult]
  ) -> EvaluationResult:
    """Computes the fraction of invocation results that are valid."""
    num_valid = 0
    num_evaluated = 0
    for result in per_invocation_results:
      if result.score is None or result.eval_status == EvalStatus.NOT_EVALUATED:
        continue
      num_evaluated += 1
      num_valid += result.score
    overall_score = num_valid / num_evaluated
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=get_eval_status(
            overall_score, self._criterion.threshold
        ),
        per_invocation_results=per_invocation_results,
    )
