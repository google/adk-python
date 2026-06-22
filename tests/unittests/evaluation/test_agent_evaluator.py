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

import os

from google.adk.evaluation.agent_evaluator import _EvalMetricResultWithInvocation
from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.evaluator import EvalStatus
from google.genai import types as genai_types
import pandas as pd
import pytest


def _content(text: str) -> genai_types.Content:
  return genai_types.Content(parts=[genai_types.Part(text=text)])


def _make_result_with_invocation(
    metric_name: str,
    score: float,
    threshold: float,
    eval_status: EvalStatus,
    prompt: str,
    expected_response: str,
    actual_response: str,
) -> _EvalMetricResultWithInvocation:
  return _EvalMetricResultWithInvocation(
      actual_invocation=Invocation(
          user_content=_content(prompt),
          final_response=_content(actual_response),
      ),
      expected_invocation=Invocation(
          user_content=_content(prompt),
          final_response=_content(expected_response),
      ),
      eval_metric_result=EvalMetricResult(
          metric_name=metric_name,
          threshold=threshold,
          score=score,
          eval_status=eval_status,
      ),
  )


def test_get_results_as_rows_flattens_metrics_and_invocations():
  eval_metric_results = {
      "response_match_score": [
          _make_result_with_invocation(
              metric_name="response_match_score",
              score=1.0,
              threshold=0.8,
              eval_status=EvalStatus.PASSED,
              prompt="What is 2 + 2?",
              expected_response="4",
              actual_response="4",
          ),
          _make_result_with_invocation(
              metric_name="response_match_score",
              score=0.0,
              threshold=0.8,
              eval_status=EvalStatus.FAILED,
              prompt="Capital of France?",
              expected_response="Paris",
              actual_response="London",
          ),
      ],
  }

  rows = AgentEvaluator._get_results_as_rows(
      eval_set_id="my_eval_set",
      eval_id="my_eval_case",
      eval_metric_results=eval_metric_results,
  )

  assert len(rows) == 2
  first = rows[0]
  assert first["eval_set_id"] == "my_eval_set"
  assert first["eval_id"] == "my_eval_case"
  assert first["metric_name"] == "response_match_score"
  assert first["threshold"] == 0.8
  assert first["score"] == 1.0
  assert first["eval_status"] == "PASSED"
  assert first["prompt"] == "What is 2 + 2?"
  assert first["expected_response"] == "4"
  assert first["actual_response"] == "4"

  # Failing invocation should still be captured.
  assert rows[1]["eval_status"] == "FAILED"
  assert rows[1]["actual_response"] == "London"


def test_get_results_as_rows_handles_missing_expected_invocation():
  result = _EvalMetricResultWithInvocation(
      actual_invocation=Invocation(
          user_content=_content("hi"),
          final_response=_content("hello"),
      ),
      expected_invocation=None,
      eval_metric_result=EvalMetricResult(
          metric_name="safety_v1",
          threshold=0.5,
          score=1.0,
          eval_status=EvalStatus.PASSED,
      ),
  )

  rows = AgentEvaluator._get_results_as_rows(
      eval_set_id="s",
      eval_id="c",
      eval_metric_results={"safety_v1": [result]},
  )

  assert len(rows) == 1
  assert rows[0]["prompt"] == "hi"
  assert rows[0]["expected_response"] == ""
  assert rows[0]["actual_response"] == "hello"


def test_write_results_to_csv_writes_expected_file(tmp_path):
  rows = [
      {
          "eval_set_id": "s",
          "eval_id": "c",
          "metric_name": "response_match_score",
          "threshold": 0.8,
          "score": 1.0,
          "eval_status": "PASSED",
          "prompt": "What is 2 + 2?",
          "expected_response": "4",
          "actual_response": "4",
          "expected_tool_calls": "",
          "actual_tool_calls": "",
      },
  ]
  output_file = os.path.join(str(tmp_path), "nested", "eval_results.csv")

  AgentEvaluator._write_results_to_csv(rows=rows, output_file=output_file)

  # The nested directory should have been created.
  assert os.path.isfile(output_file)

  df = pd.read_csv(output_file)
  assert list(df.columns) == list(rows[0].keys())
  assert len(df) == 1
  assert df.iloc[0]["metric_name"] == "response_match_score"
  assert df.iloc[0]["eval_status"] == "PASSED"
  assert df.iloc[0]["score"] == 1.0


def test_write_results_to_csv_appends_without_duplicate_header(tmp_path):
  output_file = os.path.join(str(tmp_path), "eval_results.csv")

  def _row(eval_id: str, score: float, status: str) -> dict:
    return {
        "eval_set_id": "s",
        "eval_id": eval_id,
        "metric_name": "response_match_score",
        "threshold": 0.8,
        "score": score,
        "eval_status": status,
        "prompt": "p",
        "expected_response": "e",
        "actual_response": "a",
        "expected_tool_calls": "",
        "actual_tool_calls": "",
    }

  AgentEvaluator._write_results_to_csv(
      rows=[_row("case_1", 1.0, "PASSED")], output_file=output_file
  )
  AgentEvaluator._write_results_to_csv(
      rows=[_row("case_2", 0.0, "FAILED")], output_file=output_file
  )

  df = pd.read_csv(output_file)
  # Two appends should accumulate two rows, with the header written only once.
  assert len(df) == 2
  assert sorted(df["eval_id"].tolist()) == ["case_1", "case_2"]
  assert "eval_id" not in df["eval_id"].tolist()


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-v"]))
