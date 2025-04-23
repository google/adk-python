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

from typing import Any

import pandas as pd
from tabulate import tabulate

from .evaluation_constants import EvalConstants


class TrajectoryEvaluator:
  """Evaluates tool use trajectories for accuracy."""

  @staticmethod
  def evaluate(
      eval_dataset: list[list[dict[str, Any]]],
      *,
      print_detailed_results: bool = False,
  ):
    r"""Returns the mean tool use accuracy of the eval dataset.

    Tool use accuracy is calculated by comparing the expected and the actual
    tool use trajectories. An exact match scores a 1, 0 otherwise. The final
    number is an average of these individual scores.

    Value range: [0, 1], where 0 is means none of the too use entries aligned,
    and 1 would mean all of them aligned. Higher value is good.

    Args:
      eval_dataset: The dataset that will be evaluated.
      print_detailed_results: Prints detailed results on the console. This is
        usually helpful during debugging.

    A note on eval_dataset:
      The dataset should be a list session, where each session is represented
      as a list of interaction that need evaluation. Each evaluation is
      represented as a dictionary that is expected to have values for the
      following keys:
        1) query
        2) response
        3) acutal_tool_use
        4) expected_tool_use

      Here is a sample eval_dataset value with one entry:

      [
        [
          {
            "query": "Roll a 16 sided dice for me",
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
            ]
          }
        ]
      ]
    """
    if not eval_dataset:
      raise ValueError("The evaluation dataset is empty.")

    results_df = pd.DataFrame(
        columns=[
            "query",
            "response",
            "actual_tool_use",
            "expected_tool_use",
            "tool_use_accuracy",
        ]
    )
    failures = []

    for conversation in eval_dataset:
      for index, row in enumerate(conversation):
        new_row, failure = TrajectoryEvaluator._evaluate_row(row)
        results_df = pd.concat(
            [results_df, pd.DataFrame([new_row])], ignore_index=True
        )
        if failure:
          failure["turn"] = index + 1
          failures.append(failure)

    TrajectoryEvaluator._report_failures(failures)

    if print_detailed_results:
      TrajectoryEvaluator._print_results(results_df)

    return results_df["tool_use_accuracy"].mean()

  @staticmethod
  def _evaluate_row(row):
    # We don't evaluate the mock tool outputs.
    expected = TrajectoryEvaluator._remove_tool_outputs(
        row["expected_tool_use"]
    )
    actual = row["actual_tool_use"]
    tool_use_accuracy = (
        1.0 if TrajectoryEvaluator.are_tools_equal(actual, expected) else 0.0
    )

    new_row = {
        "query": row["query"],
        "response": row["response"],
        "actual_tool_use": actual,
        "expected_tool_use": expected,
        "tool_use_accuracy": tool_use_accuracy,
    }
    failure = (
        None
        if tool_use_accuracy == 1.0
        else {"query": row["query"], "actual": actual, "expected": expected}
    )
    return new_row, failure

  @staticmethod
  def are_tools_equal(list_a_original, list_b_original):
    # Remove other entries that we don't want to evaluate
    list_a = [
        {"tool_name": tool["tool_name"], "tool_input": tool["tool_input"]}
        for tool in list_a_original
    ]

    list_b = [
        {"tool_name": tool["tool_name"], "tool_input": tool["tool_input"]}
        for tool in list_b_original
    ]

    return list_a == list_b

  @staticmethod
  def _remove_tool_outputs(tool_use_list):
    """Removes 'mock_tool_output' from each dictionary in the list."""
    result = []
    for tool_use in tool_use_list:
      new_tool_use = (
          tool_use.copy()
      )  # Create a copy to avoid modifying the original
      new_tool_use.pop(
          EvalConstants.MOCK_TOOL_OUTPUT, None
      )  # Remove 'tool_output' if it exists
      result.append(new_tool_use)
    return result

  @staticmethod
  def _report_failures(failures):
    if failures:
      print("Failures:")
      for failure in failures:
        print(f"""{{
  "turn": {failure["turn"]},
  "query": '{failure["query"]}',
  "actual": {failure["actual"]},
  "expected_tool_use": {failure["expected"]},
}}
""")

  @staticmethod
  def _print_results(results_df):
    print(tabulate(results_df, headers="keys", tablefmt="grid"))
