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

from google.adk.cli.cli_eval import _format_metric_cell
from google.adk.cli.cli_eval import _format_tool_call
from google.adk.cli.cli_eval import pretty_print_eval_result
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultDetails
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import PrebuiltMetrics
from google.adk.evaluation.eval_metrics import RubricsBasedCriterion
from google.adk.evaluation.eval_metrics import TokenUsageDetails
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


def test_pretty_print_eval_result_renders_token_breakdown(capsys):
  """The per-type token counts are printed under the token usage score."""
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
      score=1651.0,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(
          token_usage_details=TokenUsageDetails(
              total_tokens=1651.0,
              input_tokens=1180.0,
              prompt_tokens=1180.0,
              tool_use_tokens=0.0,
              output_tokens=471.0,
              candidates_tokens=343.0,
              reasoning_tokens=128.0,
          )
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
  # Indentation is containment: every count sits under the one it is part of.
  # A count the backend never reported (here `cached`) reads n/a, never 0.
  assert (
      "Token breakdown:\n"
      "  total:            1651\n"
      "    input:          1180\n"
      "      prompt:       1180\n"
      "        cached:     n/a\n"
      "      tool use:     0\n"
      "    output:         471\n"
      "      candidates:   343\n"
      "      reasoning:    128\n"
  ) in captured.out


def test_pretty_print_eval_result_without_token_breakdown(capsys):
  """Metrics that carry no token details do not print the breakdown block."""
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value,
      score=2.0,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(),
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
  assert "Token breakdown:" not in captured.out


def _function_call(**args) -> genai_types.FunctionCall:
  return genai_types.FunctionCall(
      id="call_123", name="set_device_info", args=args
  )


def test_format_tool_call_drops_id_and_streaming_fields():
  """Only the name and args are shown, one arg per line."""
  text = _format_tool_call(_function_call(device_id="device_2", status="OFF"))

  assert text == "set_device_info(\n  device_id='device_2',\n  status='OFF')"


def test_format_tool_call_without_args():
  call = genai_types.FunctionCall(name="list_devices")

  assert _format_tool_call(call) == "list_devices()"


def test_format_metric_cell():
  """Informational cells show the value; pass/fail cells show status (score)."""
  passed = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value,
      score=1.0,
      threshold=1.0,
      eval_status=EvalStatus.PASSED,
  )
  count = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOOL_CALL_COUNT_V1.value,
      score=2.0,
      eval_status=EvalStatus.INFORMATIONAL,
  )
  duration = EvalMetricResult(
      metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value,
      score=14.498,
      eval_status=EvalStatus.INFORMATIONAL,
  )
  missing = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
      score=None,
      eval_status=EvalStatus.INFORMATIONAL,
  )
  tokens = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
      score=1008.0,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(
          token_usage_details=TokenUsageDetails(
              total_tokens=1008.0, input_tokens=844.0, output_tokens=164.0
          )
      ),
  )

  assert _format_metric_cell(passed) == "PASSED (1.0)"
  assert _format_metric_cell(count) == "2"
  assert _format_metric_cell(duration) == "14.50s"
  assert _format_metric_cell(missing) == "n/a"
  assert _format_metric_cell(tokens) == "1008 (in 844, out 164)"


def test_invocation_details_are_vertical(capsys):
  """Each invocation prints one `label: value` line per field, aligned."""
  token_metric = EvalMetricResult(
      metric_name=PrebuiltMetrics.TOKEN_USAGE_V1.value,
      score=1008.0,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(
          token_usage_details=TokenUsageDetails(
              total_tokens=1008.0, input_tokens=844.0, output_tokens=164.0
          )
      ),
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="Turn off device_2.")]
      ),
      intermediate_data=IntermediateData(
          tool_uses=[_function_call(device_id="device_2", status="OFF")]
      ),
  )
  eval_result = EvalCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[token_metric],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=invocation,
              eval_metric_results=[token_metric],
          )
      ],
      session_id="session-id",
  )

  pretty_print_eval_result(eval_result)

  details = capsys.readouterr().out.split("Invocation Details:\n")[1]
  # No expected invocation, so the expected_* fields are left out; an empty
  # response reads (none); continuation lines sit under the value column.
  assert details.startswith(
      "\n"
      "Invocation 1 of 1\n"
      "  prompt:                     Turn off device_2.\n"
      "  actual_response:            (none)\n"
      "  actual_tool_calls:          set_device_info(\n"
      "                                device_id='device_2',\n"
      "                                status='OFF')\n"
      "  token_usage_v1:             1008 (in 844, out 164)\n"
  )


def test_invocation_details_wrap_long_values(capsys):
  """A long value wraps under its own value column, not back to the margin."""
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="word " * 30)]
      ),
  )
  eval_result = EvalCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=invocation, eval_metric_results=[]
          )
      ],
      session_id="session-id",
  )

  pretty_print_eval_result(eval_result)

  lines = capsys.readouterr().out.splitlines()
  prompt_index = next(
      i for i, line in enumerate(lines) if line.startswith("  prompt:")
  )
  assert lines[prompt_index + 1].startswith(" " * 30 + "word")
