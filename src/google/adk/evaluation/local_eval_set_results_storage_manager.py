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

import logging
import os
from typing import Optional

from typing_extensions import override

from .eval_result import EvalSetResult
from .eval_set_results_manager import EvalSetResultsStorageManager

logger = logging.getLogger("google_adk." + __name__)

_ADK_EVAL_HISTORY_DIR = ".adk/eval_history"
_EVAL_SET_RESULT_FILE_EXTENSION = ".evalset_result.json"


class LocalEvalSetResultsStorageManager(EvalSetResultsStorageManager):
  """An EvalSetResultsStorageManager that stores eval set results locally on disk."""

  def __init__(self, agents_dir: str):
    self._agents_dir = agents_dir

  def _get_eval_history_dir(self, app_name: str) -> str:
    return os.path.join(self._agents_dir, app_name, _ADK_EVAL_HISTORY_DIR)

  @override
  def get_eval_set_result_path(
      self, app_name: str, eval_set_result_id: str
  ) -> str:
    """Gets the path to the EvalSetResult identified by app_name and eval_set_result_id."""
    return os.path.join(
        self._agents_dir,
        app_name,
        _ADK_EVAL_HISTORY_DIR,
        eval_set_result_id + _EVAL_SET_RESULT_FILE_EXTENSION,
    )

  @override
  def list_eval_set_results(self, app_name: str) -> list[str]:
    """Gets the EvalSetResult id from the given path."""
    app_eval_history_directory = self._get_eval_history_dir(app_name)

    if not os.path.exists(app_eval_history_directory):
      return []

    eval_result_files = [
        file.removesuffix(_EVAL_SET_RESULT_FILE_EXTENSION)
        for file in os.listdir(app_eval_history_directory)
        if file.endswith(_EVAL_SET_RESULT_FILE_EXTENSION)
    ]
    return sorted(eval_result_files)

  @override
  def save_eval_set_result(self, path: str, eval_set_result: EvalSetResult):
    """Writes the EvalSetResult to the given path."""
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Saving EvalSetResult to %s", path)
    with open(path, "w") as f:
      f.write(eval_set_result.model_dump_json(indent=2))

  @override
  def load_eval_set_result(self, path: str) -> Optional[EvalSetResult]:
    """Loads the EvalSetResult from the given path."""
    try:
      with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        return EvalSetResult.model_validate_json(content)
    except FileNotFoundError:
      return None
