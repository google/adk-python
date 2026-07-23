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

from google.adk.cli.cli_eval import pretty_print_eval_result
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultDetails
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_rubrics import RubricScore
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types as genai_types


def test_pretty_print_eval_result_with_empty_criterion_rubrics(capsys):
  """Tests pretty printing falls back to rubric id when criterion rubrics are empty."""
  criterion = RubricsBasedCriterion(threshold=0.5)
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.RUBRIC_BASED_TOOL_USE_QUALITY_V1.value,
      threshold=0.5,
      criterion=criterion,
      score=1.0,
      eval_status=EvalStatus.PASSED,
      details=EvalMetricResultDetails(
          rubric_scores=[
              RubricScore(
                  rubric_id="invocation-rubric",
                  score=1.0,
                  rationale="The correct tool was used.",
              )
          ]
      ),
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      )
  )
  eval_result = EvalCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[metric_result],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=invocation,
              eval_metric_results=[metric_result],
          )
      ],
      session_id="session-id",
  )

  pretty_print_eval_result(eval_result)

  captured = capsys.readouterr()
  assert "Rubric: invocation-rubric" in captured.out
  assert "The correct tool was used." in captured.out
