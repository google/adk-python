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

from google.adk.evaluation.eval_config import _DEFAULT_EVAL_CONFIG
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import get_eval_metrics_from_config
from google.adk.evaluation.eval_config import get_evaluation_criteria_or_default
from google.adk.evaluation.eval_rubrics import Rubric
from google.adk.evaluation.eval_rubrics import RubricContent
import pytest


def test_get_evaluation_criteria_or_default_returns_default():
  assert get_evaluation_criteria_or_default("") == _DEFAULT_EVAL_CONFIG


def test_get_evaluation_criteria_or_default_reads_from_file(mocker):
  mocker.patch("os.path.exists", return_value=True)
  eval_config = EvalConfig(
      criteria={"tool_trajectory_avg_score": 0.5, "response_match_score": 0.5}
  )
  mocker.patch(
      "builtins.open", mocker.mock_open(read_data=eval_config.model_dump_json())
  )
  assert get_evaluation_criteria_or_default("dummy_path") == eval_config


def test_get_evaluation_criteria_or_default_returns_default_if_file_not_found(
    mocker,
):
  mocker.patch("os.path.exists", return_value=False)
  assert (
      get_evaluation_criteria_or_default("dummy_path") == _DEFAULT_EVAL_CONFIG
  )


def test_get_eval_metrics_from_config():
  rubric_1 = Rubric(
      rubric_id="test-rubric",
      rubric_content=RubricContent(text_property="test"),
  )
  eval_config = EvalConfig(
      criteria={
          "tool_trajectory_avg_score": 1.0,
          "response_match_score": 0.8,
          "final_response_match_v2": {
              "threshold": 0.5,
              "judge_model_options": {
                  "judge_model": "gemini-pro",
                  "num_samples": 1,
              },
          },
          "rubric_based_final_response_quality_v1": {
              "threshold": 0.9,
              "judge_model_options": {
                  "judge_model": "gemini-ultra",
                  "num_samples": 1,
              },
              "rubrics": [rubric_1],
          },
      }
  )
  eval_metrics = get_eval_metrics_from_config(eval_config)

  assert len(eval_metrics) == 4
  assert eval_metrics[0].metric_name == "tool_trajectory_avg_score"
  assert eval_metrics[0].threshold == 1.0
  assert eval_metrics[0].criterion.threshold == 1.0
  assert eval_metrics[1].metric_name == "response_match_score"
  assert eval_metrics[1].threshold == 0.8
  assert eval_metrics[1].criterion.threshold == 0.8
  assert eval_metrics[2].metric_name == "final_response_match_v2"
  assert eval_metrics[2].threshold == 0.5
  assert eval_metrics[2].criterion.threshold == 0.5
  assert (
      eval_metrics[2].criterion.judge_model_options["judge_model"]
      == "gemini-pro"
  )
  assert eval_metrics[3].metric_name == "rubric_based_final_response_quality_v1"
  assert eval_metrics[3].threshold == 0.9
  assert eval_metrics[3].criterion.threshold == 0.9
  assert (
      eval_metrics[3].criterion.judge_model_options["judge_model"]
      == "gemini-ultra"
  )
  assert len(eval_metrics[3].criterion.rubrics) == 1
  assert eval_metrics[3].criterion.rubrics[0] == rubric_1


def test_get_eval_metrics_from_config_with_custom_metrics():
  eval_config = EvalConfig(
      criteria={
          "custom_metric_1": 1.0,
          "custom_metric_2": {
              "threshold": 0.5,
          },
      },
      custom_metrics={
          "custom_metric_1": {
              "code_config": {"name": "path/to/custom/metric_1"},
          },
          "custom_metric_2": {
              "code_config": {"name": "path/to/custom/metric_2"},
          },
      },
  )
  eval_metrics = get_eval_metrics_from_config(eval_config)

  assert len(eval_metrics) == 2
  assert eval_metrics[0].metric_name == "custom_metric_1"
  assert eval_metrics[0].threshold == 1.0
  assert eval_metrics[0].criterion.threshold == 1.0
  assert eval_metrics[0].custom_function_path == "path/to/custom/metric_1"
  assert eval_metrics[1].metric_name == "custom_metric_2"
  assert eval_metrics[1].threshold == 0.5
  assert eval_metrics[1].criterion.threshold == 0.5
  assert eval_metrics[1].custom_function_path == "path/to/custom/metric_2"


def test_custom_metric_code_config_with_args_raises_error():
  with pytest.raises(ValueError):
    _ = EvalConfig(
        criteria={"custom_metric": 1.0},
        custom_metrics={
            "custom_metric": {
                "code_config": {"name": "name", "args": [{"value": 1}]},
            }
        },
    )


def test_get_eval_metrics_from_config_empty_criteria():
  eval_config = EvalConfig(criteria={})
  eval_metrics = get_eval_metrics_from_config(eval_config)
  assert not eval_metrics
