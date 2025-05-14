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
import logging
import os
import re
from typing import Any
from typing_extensions import override
from .eval_sets_manager import EvalSetsManager

logger = logging.getLogger(__name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class LocalEvalSetsManager(EvalSetsManager):
  """An EvalSets manager that stores eval sets locally on disk."""

  def __init__(self, agent_dir: str):
    self._agent_dir = agent_dir

  @override
  def get_eval_set(self, app_name: str, eval_set_id: str) -> Any:
    """Returns an EvalSet identified by an app_name and eval_set_id."""
    # Load the eval set file data
    eval_set_file_path = self._get_eval_set_file_path(app_name, eval_set_id)
    with open(eval_set_file_path, "r") as file:
      return json.load(file)  # Load JSON into a list

  @override
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet given the app_name and eval_set_id."""
    self._validate_id(id_name="Eval Set Id", id_value=eval_set_id)

    # Define the file path
    new_eval_set_path = self._get_eval_set_file_path(app_name, eval_set_id)

    logger.info("Creating eval set file `%s`", new_eval_set_path)

    if not os.path.exists(new_eval_set_path):
      # Write the JSON string to the file
      logger.info("Eval set file doesn't exist, we will create a new one.")
      with open(new_eval_set_path, "w") as f:
        empty_content = json.dumps([], indent=2)
        f.write(empty_content)

  @override
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSets that belong to the given app_name."""
    eval_set_file_path = os.path.join(self._agent_dir, app_name)
    eval_sets = []
    for file in os.listdir(eval_set_file_path):
      if file.endswith(_EVAL_SET_FILE_EXTENSION):
        eval_sets.append(
            os.path.basename(file).removesuffix(_EVAL_SET_FILE_EXTENSION)
        )

    return sorted(eval_sets)

  @override
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: Any):
    """Adds the given EvalCase to an existing EvalSet identified by app_name and eval_set_id."""
    eval_case_id = eval_case["name"]
    self._validate_id(id_name="Eval Case Id", id_value=eval_case_id)

    # Load the eval set file data
    eval_set_file_path = self._get_eval_set_file_path(app_name, eval_set_id)
    with open(eval_set_file_path, "r") as file:
      eval_set_data = json.load(file)  # Load JSON into a list

    if [x for x in eval_set_data if x["name"] == eval_case_id]:
      raise ValueError(
          f"Eval id `{eval_case_id}` already exists in `{eval_set_id}`"
          " eval set.",
      )

    eval_set_data.append(eval_case)
    # Serialize the test data to JSON and write to the eval set file.
    with open(eval_set_file_path, "w") as f:
      f.write(json.dumps(eval_set_data, indent=2))

  def _get_eval_set_file_path(self, app_name: str, eval_set_id: str) -> str:
    return os.path.join(
        self._agent_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

  def _validate_id(self, id_name: str, id_value: str):
    pattern = r"^[a-zA-Z0-9_]+$"
    if not bool(re.fullmatch(pattern, id_value)):
      raise ValueError(
          f"Invalid {id_name}. {id_name} should have the `{pattern}` format",
      )
