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

from typing import Any
from typing import Optional

from google.genai import types as genai_types
import pandas as pd
from tabulate import tabulate
from typing_extensions import deprecated
from typing_extensions import override
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation import MetricPromptTemplateExamples

from .eval_case import IntermediateData
from .eval_case import Invocation
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .evaluator import PerInvocationResult


class ResponseEvaluator(Evaluator):
  """Runs response evaluation for agents."""

  def __init__(self, threshold: float, metric_name: str):
    if "response_evaluation_score" == metric_name:
      self._metric_name = MetricPromptTemplateExamples.Pointwise.COHERENCE
    elif "response_match_score" == metric_name:
      self._metric_name = "rouge_1"
    else:
      raise ValueError(f"`{metric_name}` is not supported.")

    self._threshold = threshold

  @override
  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    total_score = 0.0
    num_invocations = 0
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      prompt = self._get_text(expected.user_content)
      reference = self._get_text(expected.final_response)
      response = self._get_text(actual.final_response)
      actual_tool_use = self._get_tool_use_trajectory(actual.intermediate_data)
      reference_trajectory = self._get_tool_use_trajectory(
          expected.intermediate_data
      )

      eval_case = {
          "prompt": prompt,
          "reference": reference,
          "response": response,
          "actual_tool_user": actual_tool_use,
          "reference_trajectory": reference_trajectory,
      }

      eval_case_result = ResponseEvaluator._perform_eval(
          pd.DataFrame([eval_case]), [self._metric_name]
      )
      score = self._get_score(eval_case_result)
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=score,
              eval_status=self._get_eval_status(score),
          )
      )
      total_score += score
      num_invocations += 1

    if per_invocation_results:
      overall_score = total_score / num_invocations
      return EvaluationResult(
          overall_score=overall_score,
          overall_eval_status=self._get_eval_status(overall_score),
          per_invocation_results=per_invocation_results,
      )

    return EvaluationResult()

  def _get_text(self, content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  def _get_tool_use_trajectory(
      self, intermediate_data: Optional[IntermediateData]
  ) -> list[dict[str, Any]]:
    tool_use_trajectory = []
    if not intermediate_data:
      return tool_use_trajectory

    for function_call in intermediate_data.tool_uses:
      tool_use_trajectory.append({
          "tool_name": function_call.name,
          "tool_input": function_call.args or {},
      })

    return tool_use_trajectory

  def _get_score(self, eval_result) -> float:
    return eval_result.summary_metrics[f"{self._metric_name}/mean"].item()

  def _get_eval_status(self, score: float):
    return EvalStatus.PASSED if score >= self._threshold else EvalStatus.FAILED

  @staticmethod
  @deprecated(
      "This method has been deprecated and will be removed soon. Please use"
      " evaluate_invocations instead."
  )
  def evaluate(
      raw_eval_dataset: list[list[dict[str, Any]]],
      evaluation_criteria: list[str],
      *,
      print_detailed_results: bool = False,
  ):
    r"""Returns the value of requested evaluation metrics.

    Args:
      raw_eval_dataset: The dataset that will be evaluated.
      evaluation_criteria: The evaluation criteria to be used. This method
        support two criteria, `response_evaluation_score` and
        `response_match_score`.
      print_detailed_results: Prints detailed results on the console. This is
        usually helpful during debugging.

    A note on evaluation_criteria:
      `response_match_score`: This metric compares the agents final natural
        language response with the expected final response, stored in the
        "reference" field in test/eval files. We use Rouge metric to compare the
        two responses.

        Value Range: [0, 1]. A score closer to 0 means poor similarity between
          response and reference. A score closer to 1 means strong similarity
          between response and reference.

      `response_evaluation_score`: Uses LLM to evalaute coherence of the
        response, including tool use. This is pointwise metric.

        Value range: [0, 5], where 0 means that the agent's response is not
        coherent, while 5 means it is . High values are good.
    A note on raw_eval_dataset:
      The dataset should be a list session, where each session is represented
      as a list of interaction that need evaluation. Each evaluation is
      represented as a dictionary that is expected to have values for the
      following keys:

        1) query
        2) response
        3) acutal_tool_use
        4) expected_tool_use
        5) reference

      Here is a sample eval_dataset value with one entry:
      [
        [
          {
            "query": "roll a die for me",
            "response": "I rolled a 16 sided die and got 13.\n",
            "expected_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ],
            "acutal_tool_use": [
              {
                "tool_name": "roll_die",
                "tool_input": {
                  "sides": 16
                }
              }
            ],
            "reference": "I rolled a 16 sided die and got 13.\n"
          }
        ]
      ]
    """
    if not raw_eval_dataset:
      raise ValueError("The evaluation dataset is empty.")

    metrics = ResponseEvaluator._get_metrics(
        raw_eval_dataset, evaluation_criteria
    )
    flattened_queries = [
        item for sublist in raw_eval_dataset for item in sublist
    ]
    eval_dataset = pd.DataFrame(flattened_queries).rename(
        columns={"query": "prompt", "expected_tool_use": "reference_trajectory"}
    )

    eval_result = ResponseEvaluator._perform_eval(
        dataset=eval_dataset, metrics=metrics
    )

    if print_detailed_results:
      ResponseEvaluator._print_results(eval_result)
    return eval_result.summary_metrics

  @staticmethod
  def _get_metrics(raw_eval_dataset, criteria):
    metrics = []
    if (
        "response_evaluation_score" in criteria
        and "query" in raw_eval_dataset[0][0]
        and "expected_tool_use" in raw_eval_dataset[0][0]
    ):
      metrics.append(MetricPromptTemplateExamples.Pointwise.COHERENCE)
    if (
        "response_match_score" in criteria
        and "reference" in raw_eval_dataset[0][0]
    ):
      metrics.append("rouge_1")
    return metrics

  @staticmethod
  def _perform_eval(dataset, metrics):
    """This method hides away the call to external service.

    Primarily helps with unit testing.
    """
    eval_task = EvalTask(dataset=dataset, metrics=metrics)

    return eval_task.evaluate()

  @staticmethod
  def _print_results(eval_result):
    print("Evaluation Summary Metrics:", eval_result.summary_metrics)
    print(tabulate(eval_result.metrics_table, headers="keys", tablefmt="grid"))
