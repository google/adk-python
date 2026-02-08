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

"""Unit tests for utilities in cli_eval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from google.adk.cli.cli_eval import aggregate_eval_case_results
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult


def test_get_eval_sets_manager_local(monkeypatch):
  mock_local_manager = mock.MagicMock()
  monkeypatch.setattr(
      "google.adk.evaluation.local_eval_sets_manager.LocalEvalSetsManager",
      lambda *a, **k: mock_local_manager,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(eval_storage_uri=None, agents_dir="some/dir")
  assert manager == mock_local_manager


def test_get_eval_sets_manager_gcs(monkeypatch):
  mock_gcs_manager = mock.MagicMock()
  mock_create_gcs = mock.MagicMock()
  mock_create_gcs.return_value = SimpleNamespace(
      eval_sets_manager=mock_gcs_manager
  )
  monkeypatch.setattr(
      "google.adk.cli.utils.evals.create_gcs_eval_managers_from_uri",
      mock_create_gcs,
  )
  from google.adk.cli.cli_eval import get_eval_sets_manager

  manager = get_eval_sets_manager(
      eval_storage_uri="gs://bucket", agents_dir="some/dir"
  )
  assert manager == mock_gcs_manager
  mock_create_gcs.assert_called_once_with("gs://bucket")


def test_aggregate_eval_case_results_across_runs():
  eval_results = [
      EvalCaseResult(
          eval_set_id="set1",
          eval_id="case1",
          final_eval_status=EvalStatus.PASSED,
          overall_eval_metric_results=[
              EvalMetricResult(
                  metric_name="response_match_score",
                  threshold=0.8,
                  score=1.0,
                  eval_status=EvalStatus.PASSED,
              )
          ],
          eval_metric_result_per_invocation=[],
          session_id="s1",
      ),
      EvalCaseResult(
          eval_set_id="set1",
          eval_id="case1",
          final_eval_status=EvalStatus.FAILED,
          overall_eval_metric_results=[
              EvalMetricResult(
                  metric_name="response_match_score",
                  threshold=0.8,
                  score=0.6,
                  eval_status=EvalStatus.FAILED,
              )
          ],
          eval_metric_result_per_invocation=[],
          session_id="s2",
      ),
  ]

  aggregate_results = aggregate_eval_case_results(eval_results)

  assert len(aggregate_results) == 1
  assert aggregate_results[0].eval_set_id == "set1"
  assert aggregate_results[0].eval_id == "case1"
  assert aggregate_results[0].overall_eval_metric_results[0].score == 0.8
  assert (
      aggregate_results[0].overall_eval_metric_results[0].eval_status
      == EvalStatus.PASSED
  )
  assert aggregate_results[0].final_eval_status == EvalStatus.PASSED
