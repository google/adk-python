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
import logging
import os

from typing_extensions import override

from ..errors.not_found_error import NotFoundError
from ._eval_set_results_manager_utils import create_eval_set_result
from .eval_result import EvalCaseResult
from .eval_result import EvalSetResult
from .eval_set_results_manager import EvalSetResultsManager

logger = logging.getLogger("google_adk." + __name__)

_ADK_EVAL_HISTORY_DIR = ".adk/eval_history"
_EVAL_SET_RESULT_FILE_EXTENSION = ".evalset_result.json"


class LocalEvalSetResultsManager(EvalSetResultsManager):
  """An EvalSetResult manager that stores eval set results locally on disk."""

  def __init__(self, agents_dir: str):
    self._agents_dir = agents_dir

  @override
  def save_eval_set_result(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case_results: list[EvalCaseResult],
  ) -> None:
    """Creates and saves a new EvalSetResult given eval_case_results."""
    eval_set_result = create_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )
    # Write eval result file, with eval_set_result_name.
    app_eval_history_dir = self._get_eval_history_dir(app_name)
    if not os.path.exists(app_eval_history_dir):
      os.makedirs(app_eval_history_dir)
    # Convert to json and write to file.
    eval_set_result_json = eval_set_result.model_dump_json()
    eval_set_result_file_path = os.path.join(
        app_eval_history_dir,
        eval_set_result.eval_set_result_name + _EVAL_SET_RESULT_FILE_EXTENSION,
    )
    logger.info("Writing eval result to file: %s", eval_set_result_file_path)
    with open(eval_set_result_file_path, "w") as f:
      f.write(json.dumps(eval_set_result_json, indent=2))

  @override
  def get_eval_set_result(
      self, app_name: str, eval_set_result_id: str
  ) -> EvalSetResult:
    """Returns an EvalSetResult identified by app_name and eval_set_result_id."""
    # Load the eval set result file data.
    maybe_eval_result_file_path = (
        os.path.join(
            self._get_eval_history_dir(app_name),
            eval_set_result_id,
        )
        + _EVAL_SET_RESULT_FILE_EXTENSION
    )
    if not os.path.exists(maybe_eval_result_file_path):
      raise NotFoundError(f"Eval set result `{eval_set_result_id}` not found.")
    with open(maybe_eval_result_file_path, "r") as file:
      eval_result_data = json.load(file)
    return EvalSetResult.model_validate_json(eval_result_data)

  @override
  def list_eval_set_results(self, app_name: str) -> list[str]:
    """Returns the eval result ids that belong to the given app_name."""
    app_eval_history_directory = self._get_eval_history_dir(app_name)

    if not os.path.exists(app_eval_history_directory):
      return []

    eval_result_files = [
        file.removesuffix(_EVAL_SET_RESULT_FILE_EXTENSION)
        for file in os.listdir(app_eval_history_directory)
        if file.endswith(_EVAL_SET_RESULT_FILE_EXTENSION)
    ]
    return eval_result_files

  def _get_eval_history_dir(self, app_name: str) -> str:
    return os.path.join(self._agents_dir, app_name, _ADK_EVAL_HISTORY_DIR)
