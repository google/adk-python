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

from .eval_set import EvalSet
from .eval_sets_manager import EvalSetStorageManager

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class LocalEvalSetStorageManager(EvalSetStorageManager):
  """An EvalSetStorageManager that stores eval sets locally on disk."""

  def __init__(self, agents_dir: str):
    self._agents_dir = agents_dir

  def get_eval_set_path(self, app_name: str, eval_set_id: str) -> str:
    """Gets the path to the EvalSet identified by app_name and eval_set_id."""
    return os.path.join(
        self._agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

  def list_eval_sets(self, app_name: str) -> list[str]:
    """Gets the EvalSet id from the given path."""
    eval_sets_dir = os.path.join(self._agents_dir, app_name)
    eval_sets = []
    for file in os.listdir(eval_sets_dir):
      if file.endswith(_EVAL_SET_FILE_EXTENSION):
        eval_sets.append(
            os.path.basename(file).removesuffix(_EVAL_SET_FILE_EXTENSION)
        )

    return sorted(eval_sets)

  def save_eval_set(self, path: str, eval_set: EvalSet):
    """Writes the EvalSet to the given path."""
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Saving EvalSet to %s", path)
    with open(path, "w") as f:
      f.write(eval_set.model_dump_json(indent=2))

  def load_eval_set(self, path: str) -> EvalSet | None:
    """Loads the EvalSet from the given path."""
    try:
      with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        return EvalSet.model_validate_json(content)
    except FileNotFoundError:
      return None
