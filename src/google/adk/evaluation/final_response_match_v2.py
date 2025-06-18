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

import json
import logging
import re

from typing_extensions import override

from ..models.llm_response import LlmResponse
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import PerInvocationResult
from .response_auto_rater import get_eval_status
from .response_auto_rater import get_text_from_content
from .response_auto_rater import ResponseAutoRater

logger = logging.getLogger("google_adk." + __name__)


class ResponseMatchV2Evaluator(ResponseAutoRater):
  """AutoRater-based evaluator to judge final response."""

  def __init__(
      self,
      eval_metric: EvalMetric,
      auto_rater_prompt_template: str,
      auto_rater_num_samples: int = 5,
      multi_turn: bool = False,
  ):
    super().__init__(
        eval_metric, auto_rater_prompt_template, auto_rater_num_samples
    )
    self._multi_turn = multi_turn

  @override
  def format_auto_rater_prompt(
      self, actual_invocation: Invocation, expected_invocation: Invocation
  ) -> str:
    reference = get_text_from_content(expected_invocation.final_response)
    response = get_text_from_content(actual_invocation.final_response)
    user_prompt = get_text_from_content(expected_invocation.user_content)
    return self._auto_rater_prompt_template.format(
        function_api_spec="None",
        prompt=user_prompt,
        response=response,
        golden_response=reference,
    )

  @override
  def convert_auto_rater_response_to_score(
      self, llm_response: LlmResponse
  ) -> Optional[float]:
    try:
      response_text = get_text_from_content(llm_response.content).strip()
      match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
      if match:
        response_json_text = match.group(1)
        parsed_response = json.loads(response_json_text)
      else:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as e:
      logger.error("Failed to parse auto rater response: %s", llm_response)
      return None
    if "is_the_agent_response_valid" not in parsed_response:
      logger.error(
          "Auto rater response does not contain key"
          " 'is_the_agent_response_valid': %s",
          llm_response,
      )
      return None
    is_valid = parsed_response["is_the_agent_response_valid"].lower() == "valid"
    return 1.0 if is_valid else 0.0

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
            overall_score, self._eval_metric.threshold
        ),
        per_invocation_results=per_invocation_results,
    )
