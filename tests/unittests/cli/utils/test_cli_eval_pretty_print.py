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
from google.adk.cli.cli_eval import _pretty_print_eval_results_across_runs
from google.adk.cli.cli_eval import pretty_print_eval_result
from google.adk.evaluation._eval_case_result_multi_run_aggregator import MethodResult
from google.adk.evaluation._eval_case_result_multi_run_aggregator import MultiRunCaseResult
from google.adk.evaluation.eval_case import IntermediateData
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import DurationDetails
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
import pytest


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


def test_pretty_print_eval_result_renders_duration_breakdown(capsys):
  """Where the seconds went is printed under the invocation duration score."""
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value,
      score=7.366,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(
          duration_details=DurationDetails(
              total_seconds=7.366,
              model_seconds=3.55,
              tool_seconds=0.003,
              framework_seconds=3.813,
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
  # The parts partition the turn rather than nest, so they are indented under
  # the total only to show that they add back up to it.
  assert (
      "Duration breakdown:\n"
      "  total:            7.366s\n"
      "    model:          3.55s\n"
      "    tool:           0.003s\n"
      "    framework:      3.813s\n"
  ) in captured.out


def test_pretty_print_eval_result_without_duration_breakdown(capsys):
  """A duration that was never split prints the score and no breakdown."""
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value,
      score=7.366,
      eval_status=EvalStatus.INFORMATIONAL,
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

  assert "Duration breakdown:" not in capsys.readouterr().out


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
  duration_split = EvalMetricResult(
      metric_name=PrebuiltMetrics.INVOCATION_DURATION_V1.value,
      score=14.498,
      eval_status=EvalStatus.INFORMATIONAL,
      details=EvalMetricResultDetails(
          duration_details=DurationDetails(
              total_seconds=14.498,
              model_seconds=9.2,
              tool_seconds=0.3,
              framework_seconds=4.998,
          )
      ),
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
  assert _format_metric_cell(duration_split) == (
      "14.50s (model 9.2s, tool 0.3s, framework 4.998s)"
  )
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


def _run_result(
    *, repetition_index: int | None, score: float, eval_status: EvalStatus
) -> EvalCaseResult:
  """Returns the result of one run of an eval case, scored by one metric."""
  metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
      threshold=0.8,
      score=score,
      eval_status=eval_status,
  )
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="User input here.")]
      )
  )
  return EvalCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=eval_status,
      overall_eval_metric_results=[metric_result],
      eval_metric_result_per_invocation=[
          EvalMetricResultPerInvocation(
              actual_invocation=invocation,
              eval_metric_results=[metric_result],
          )
      ],
      session_id=f"session-{repetition_index}",
      repetition_index=repetition_index,
  )


def test_pretty_print_eval_results_across_runs_prints_verdict_then_runs(
    capsys,
):
  """Each case's pooled verdict comes first, then each of its runs."""
  first_run = _run_result(
      repetition_index=0, score=1.0, eval_status=EvalStatus.PASSED
  )
  second_run = _run_result(
      repetition_index=1, score=0.7, eval_status=EvalStatus.FAILED
  )
  pooled_metric_result = EvalMetricResult(
      metric_name=PrebuiltMetrics.RESPONSE_MATCH_SCORE.value,
      threshold=0.8,
      score=0.85,
      eval_status=EvalStatus.PASSED,
  )
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[pooled_metric_result],
      runs=[first_run, second_run],
      metric_methods={
          PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: MethodResult(
              name="mean_over_invocations",
              value=0.85,
              threshold=0.8,
              status=EvalStatus.PASSED,
          )
      },
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=2
  )

  out = capsys.readouterr().out
  verdict_at = out.index(
      "Overall Eval Status across 2 runs (1 of 2 runs passed): PASSED\n"
  )
  pooled_score_at = out.index(
      "Metric: response_match_score, Method: mean_over_invocations,"
      " Value: 0.85, Threshold: 0.8, Status: PASSED\n"
  )
  # Each run's label sits on the line just above the block it labels.
  first_run_at = out.index("Run: 1 of 2\nEval Set Id: eval-set\n")
  second_run_at = out.index("Run: 2 of 2\nEval Set Id: eval-set\n")
  assert verdict_at < pooled_score_at < first_run_at < second_run_at
  # A run that was scored and failed is not a run that crashed.
  assert "without producing any metric results" not in out


def test_pretty_print_eval_results_across_runs_counts_crashed_runs(capsys):
  """The verdict says how many runs crashed before they could be scored.

  A crashed run counts as crashed, not as passed, failed or not evaluated. By
  default it fails the case even though every pooled metric may pass, so
  without the line under the verdict nothing would explain the failure.
  """
  crashed_run = EvalCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id="session-0",
      repetition_index=0,
  )
  scored_run = _run_result(
      repetition_index=1, score=1.0, eval_status=EvalStatus.PASSED
  )
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=scored_run.overall_eval_metric_results,
      runs=[crashed_run, scored_run],
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=2
  )

  out = capsys.readouterr().out
  assert (
      "Overall Eval Status across 2 runs (1 of 2 runs passed, 1 crashed):"
      " FAILED\n"
      "1 of 2 runs failed without producing any metric results.\n"
  ) in out
  assert out.index("runs failed without") < out.index("Run: 1 of 2\n")


@pytest.mark.parametrize(
    ("crashed", "runs", "final_eval_status", "expected"),
    [
        (
            1,
            2,
            EvalStatus.PASSED,
            (
                "1 of 2 runs failed without producing any metric results"
                " (up to 1 allowed).\n"
            ),
        ),
        (
            2,
            3,
            EvalStatus.FAILED,
            (
                "2 of 3 runs failed without producing any metric results, more"
                " than the 1 allowed.\n"
            ),
        ),
    ],
    ids=["within_the_tolerance", "more_than_allowed"],
)
def test_pretty_print_eval_results_across_runs_says_how_many_may_crash(
    capsys, crashed, runs, final_eval_status, expected
):
  """With crashed runs tolerated, the crash line says how many may crash."""
  crashed_runs = [
      EvalCaseResult(
          eval_set_id="eval-set",
          eval_id="eval-id",
          final_eval_status=EvalStatus.FAILED,
          overall_eval_metric_results=[],
          eval_metric_result_per_invocation=[],
          session_id=f"session-{index}",
          repetition_index=index,
      )
      for index in range(crashed)
  ]
  scored_runs = [
      _run_result(
          repetition_index=index, score=1.0, eval_status=EvalStatus.PASSED
      )
      for index in range(crashed, runs)
  ]
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=final_eval_status,
      overall_eval_metric_results=scored_runs[0].overall_eval_metric_results,
      runs=crashed_runs + scored_runs,
      max_crashed_runs=1,
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=runs
  )

  assert expected in capsys.readouterr().out


def test_pretty_print_eval_results_across_runs_counts_undecided_runs(capsys):
  """The verdict says how many runs neither passed nor failed.

  A run with a metric that was not evaluated counts as neither, so without
  this the count of passed runs would leave such a run unexplained.
  """
  passed_run = _run_result(
      repetition_index=0, score=1.0, eval_status=EvalStatus.PASSED
  )
  undecided_run = _run_result(
      repetition_index=1, score=0.0, eval_status=EvalStatus.NOT_EVALUATED
  )
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=passed_run.overall_eval_metric_results,
      runs=[passed_run, undecided_run],
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=2
  )

  assert (
      "Overall Eval Status across 2 runs (1 of 2 runs passed, 1 not"
      " evaluated): PASSED\n"
  ) in capsys.readouterr().out


def test_pretty_print_eval_results_across_runs_appends_a_rules_note(capsys):
  """A metric judged by counting passing runs shows the count in its line."""
  runs = [
      _run_result(repetition_index=0, score=1.0, eval_status=EvalStatus.PASSED),
      _run_result(repetition_index=1, score=0.7, eval_status=EvalStatus.FAILED),
  ]
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.FAILED,
      overall_eval_metric_results=[],
      runs=runs,
      metric_methods={
          PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: MethodResult(
              name="pass_rate",
              value=0.5,
              threshold=0.9,
              status=EvalStatus.FAILED,
              note="1 of 2 runs passed",
          )
      },
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=2
  )

  assert (
      "Metric: response_match_score, Method: pass_rate, Value: 0.5,"
      " Threshold: 0.9, Status: FAILED (1 of 2 runs passed)\n"
  ) in capsys.readouterr().out


def test_pretty_print_eval_results_across_runs_prints_cross_run_criteria(
    capsys,
):
  """Each cross-run criterion is printed after the metrics, before the runs."""
  run = _run_result(
      repetition_index=0, score=1.0, eval_status=EvalStatus.PASSED
  )
  verdict = MultiRunCaseResult(
      eval_set_id="eval-set",
      eval_id="eval-id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=run.overall_eval_metric_results,
      runs=[run],
      metric_methods={
          PrebuiltMetrics.RESPONSE_MATCH_SCORE.value: MethodResult(
              name="mean_over_invocations",
              value=1.0,
              threshold=0.8,
              status=EvalStatus.PASSED,
          )
      },
      cross_run_results=[
          MethodResult(
              name="pass_rate",
              value=1.0,
              threshold=0.8,
              status=EvalStatus.PASSED,
              note="1 of 1 runs passed",
          )
      ],
  )

  _pretty_print_eval_results_across_runs(
      aggregate_eval_results=[verdict], num_runs=1
  )

  out = capsys.readouterr().out
  metric_at = out.index("Metric: response_match_score")
  criterion_at = out.index(
      "Cross-run criterion: pass_rate, Value: 1.0, Threshold: 0.8,"
      " Status: PASSED (1 of 1 runs passed)\n"
  )
  run_at = out.index("Run: 1 of 1\n")
  assert metric_at < criterion_at < run_at
