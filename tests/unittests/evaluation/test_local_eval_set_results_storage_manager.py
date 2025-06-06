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

import json
import os
import tempfile

from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.local_eval_set_results_storage_manager import LocalEvalSetResultsStorageManager
import pytest


@pytest.fixture
def temp_agents_dir():
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


def test_init(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  assert manager._agents_dir == temp_agents_dir


def test_get_eval_set_result_path(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  path = manager.get_eval_set_result_path("test_app", "test_eval_set_result")
  expected_path = os.path.join(
      temp_agents_dir,
      "test_app",
      ".adk/eval_history",
      "test_eval_set_result.evalset_result.json",
  )
  assert path == expected_path


def test_list_eval_set_results(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  app_dir = os.path.join(temp_agents_dir, "test_app")
  eval_history_dir = os.path.join(app_dir, ".adk/eval_history")
  os.makedirs(eval_history_dir, exist_ok=True)
  with open(
      os.path.join(eval_history_dir, "eval_result_1.evalset_result.json"), "w"
  ) as f:
    f.write(
        '{"eval_set_result_id": "eval_result_1", "eval_set_id": "eval_set_1",'
        ' "eval_case_results": []}'
    )
  with open(
      os.path.join(eval_history_dir, "eval_result_2.evalset_result.json"), "w"
  ) as f:
    f.write(
        '{"eval_set_result_id": "eval_result_2", "eval_set_id": "eval_set_2",'
        ' "eval_case_results": []}'
    )
  # add a file that should be ignored
  with open(
      os.path.join(eval_history_dir, "not_eval_set_result.json"), "w"
  ) as f:
    f.write("ignore this")

  eval_results = manager.list_eval_set_results("test_app")
  assert eval_results == ["eval_result_1", "eval_result_2"]


def test_list_eval_set_results_empty(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  app_dir = os.path.join(temp_agents_dir, "test_app")
  eval_history_dir = os.path.join(app_dir, ".adk/eval_history")
  os.makedirs(eval_history_dir, exist_ok=True)

  eval_results = manager.list_eval_set_results("test_app")
  assert eval_results == []


def test_save_eval_set_result(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  eval_set_result = EvalSetResult(
      eval_set_result_id="test_eval_set_result",
      eval_set_id="test_eval_set",
      eval_case_results=[],
  )
  path = os.path.join(
      temp_agents_dir,
      "test_app",
      ".adk/eval_history",
      "test_eval_set_result.evalset_result.json",
  )
  manager.save_eval_set_result(path, eval_set_result)
  with open(path, "r") as f:
    loaded_content = f.read()
  assert loaded_content == eval_set_result.model_dump_json(indent=2)


def test_load_eval_set_result(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  eval_set_result_data = (
      '{"eval_set_result_id": "test_eval_set_result", "eval_set_id":'
      ' "test_eval_set", "eval_case_results": []}'
  )
  path = os.path.join(
      temp_agents_dir,
      "test_app",
      ".adk/eval_history",
      "test_eval_set_result.evalset_result.json",
  )
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    f.write(eval_set_result_data)
  loaded_eval_set_result = manager.load_eval_set_result(path)
  assert loaded_eval_set_result == EvalSetResult.model_validate_json(
      eval_set_result_data
  )


def test_load_eval_set_result_not_found(temp_agents_dir):
  manager = LocalEvalSetResultsStorageManager(temp_agents_dir)
  path = os.path.join(
      temp_agents_dir,
      "test_app",
      ".adk/eval_history",
      "test_eval_set_result.evalset_result.json",
  )
  loaded_eval_set_result = manager.load_eval_set_result(path)
  assert loaded_eval_set_result is None
