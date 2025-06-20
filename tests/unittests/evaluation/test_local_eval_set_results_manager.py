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
import os
import shutil
import tempfile
import time
from unittest.mock import patch

from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation._eval_set_results_manager_utils import _sanitize_eval_set_result_name
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.local_eval_set_results_manager import _ADK_EVAL_HISTORY_DIR
from google.adk.evaluation.local_eval_set_results_manager import _EVAL_SET_RESULT_FILE_EXTENSION
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
import pytest


class TestLocalEvalSetResultsManager:

  @pytest.fixture(autouse=True)
  def setup(self):
    self.temp_dir = tempfile.mkdtemp()
    self.agents_dir = os.path.join(self.temp_dir, "agents")
    os.makedirs(self.agents_dir)
    self.manager = LocalEvalSetResultsManager(self.agents_dir)
    self.app_name = "test_app"
    self.eval_set_id = "test_eval_set"
    self.eval_case_results = [
        EvalCaseResult(
            eval_set_file="test_file",
            eval_set_id=self.eval_set_id,
            eval_id="case1",
            final_eval_status=EvalStatus.PASSED,
            eval_metric_results=[],
            overall_eval_metric_results=[],
            eval_metric_result_per_invocation=[],
            session_id="session1",
        )
    ]
    self.timestamp = time.time()  # Store the timestamp
    self.eval_set_result_id = (
        self.app_name + "_" + self.eval_set_id + "_" + str(self.timestamp)
    )
    self.eval_set_result_name = _sanitize_eval_set_result_name(
        self.eval_set_result_id
    )
    self.eval_set_result = EvalSetResult(
        eval_set_result_id=self.eval_set_result_id,
        eval_set_result_name=self.eval_set_result_name,
        eval_set_id=self.eval_set_id,
        eval_case_results=self.eval_case_results,
        creation_timestamp=self.timestamp,
    )

  def teardown(self):
    shutil.rmtree(self.temp_dir)

  @patch("time.time")
  def test_save_eval_set_result(self, mock_time):
    mock_time.return_value = self.timestamp
    self.manager.save_eval_set_result(
        self.app_name, self.eval_set_id, self.eval_case_results
    )
    eval_history_dir = os.path.join(
        self.agents_dir, self.app_name, _ADK_EVAL_HISTORY_DIR
    )
    expected_file_path = os.path.join(
        eval_history_dir,
        self.eval_set_result_name + _EVAL_SET_RESULT_FILE_EXTENSION,
    )
    assert os.path.exists(expected_file_path)
    with open(expected_file_path, "r") as f:
      actual_eval_set_result_json = json.load(f)

    # need to convert eval_set_result to json
    expected_eval_set_result_json = self.eval_set_result.model_dump_json()
    assert expected_eval_set_result_json == actual_eval_set_result_json

  @patch("time.time")
  def test_get_eval_set_result(self, mock_time):
    mock_time.return_value = self.timestamp
    self.manager.save_eval_set_result(
        self.app_name, self.eval_set_id, self.eval_case_results
    )
    retrieved_result = self.manager.get_eval_set_result(
        self.app_name, self.eval_set_result_name
    )
    assert retrieved_result == self.eval_set_result

  @patch("time.time")
  def test_get_eval_set_result_not_found(self, mock_time):
    mock_time.return_value = self.timestamp

    with pytest.raises(NotFoundError) as e:
      self.manager.get_eval_set_result(self.app_name, "non_existent_id")

  @patch("time.time")
  def test_list_eval_set_results(self, mock_time):
    mock_time.return_value = self.timestamp
    # Save two eval set results for the same app
    self.manager.save_eval_set_result(
        self.app_name, self.eval_set_id, self.eval_case_results
    )
    timestamp2 = time.time() + 1
    mock_time.return_value = timestamp2
    eval_set_result_id2 = (
        self.app_name + "_" + self.eval_set_id + "_" + str(timestamp2)
    )
    eval_set_result_name2 = _sanitize_eval_set_result_name(eval_set_result_id2)
    eval_case_results2 = [
        EvalCaseResult(
            eval_set_file="test_file",
            eval_set_id=self.eval_set_id,
            eval_id="case2",
            final_eval_status=EvalStatus.FAILED,
            eval_metric_results=[],
            overall_eval_metric_results=[],
            eval_metric_result_per_invocation=[],
            session_id="session2",
        )
    ]
    self.manager.save_eval_set_result(
        self.app_name, self.eval_set_id, eval_case_results2
    )

    # Save one eval set result for a different app
    app_name2 = "another_app"
    timestamp3 = time.time() + 2
    mock_time.return_value = timestamp3

    self.manager.save_eval_set_result(
        app_name2, self.eval_set_id, self.eval_case_results
    )

    results = self.manager.list_eval_set_results(self.app_name)
    expected_result = [self.eval_set_result_name, eval_set_result_name2]
    assert set(results) == set(expected_result)

  def test_list_eval_set_results_empty(self):
    # No eval set results saved for the app
    results = self.manager.list_eval_set_results(self.app_name)
    assert results == []
