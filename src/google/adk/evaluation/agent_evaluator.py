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
from os import path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import uuid

from pydantic import ValidationError

from .eval_set import EvalSet
from .evaluation_generator import EvaluationGenerator
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .evaluator import Evaluator
from .local_eval_sets_manager import convert_eval_set_to_pydanctic_schema
from .response_evaluator import ResponseEvaluator
from .trajectory_evaluator import TrajectoryEvaluator

logger = logging.getLogger("google_adk." + __name__)


# Constants for default runs and evaluation criteria
NUM_RUNS = 2
TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = "response_evaluation_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"

ALLOWED_CRITERIA = [
    TOOL_TRAJECTORY_SCORE_KEY,
    RESPONSE_EVALUATION_SCORE_KEY,
    RESPONSE_MATCH_SCORE_KEY,
]


QUERY_COLUMN = "query"
REFERENCE_COLUMN = "reference"
EXPECTED_TOOL_USE_COLUMN = "expected_tool_use"


DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,  # Rouge-1 text match; 0.8 is default.
}


def load_json(file_path: str) -> Union[Dict, List]:
  with open(file_path, "r") as f:
    return json.load(f)


class AgentEvaluator:
  """An evaluator for Agents, mainly intended for helping with test cases."""

  @staticmethod
  def find_config_for_test_file(test_file: str):
    """Find the test_config.json file in the same folder as the test file."""
    test_folder = os.path.dirname(test_file)
    config_path = os.path.join(test_folder, "test_config.json")
    if os.path.exists(config_path):
      config_data = load_json(config_path)
      if "criteria" in config_data and isinstance(
          config_data["criteria"], dict
      ):
        return config_data["criteria"]
      else:
        raise ValueError(
            f"Invalid format for test_config.json at {config_path}. Expected a"
            " 'criteria' dictionary."
        )
    return DEFAULT_CRITERIA

  @staticmethod
  async def evaluate_eval_set(
      agent_module: str,
      eval_set: EvalSet,
      criteria: dict[str, float],
      num_runs=NUM_RUNS,
      agent_name=None,
  ):
    """Evaluates an agent using the given EvalSet.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' in the loaded module.
      eval_set: The eval set.
      criteria: Evauation criterias, a dictionary of metric names to their
        respective thresholds.
      num_runs: Number of times all entries in the eval dataset should be
        assessed.
      agent_name: The name of the agent.
    """
    eval_case_responses_list = await EvaluationGenerator.generate_responses(
        eval_set=eval_set,
        agent_module_path=agent_module,
        repeat_num=num_runs,
        agent_name=agent_name,
    )

    for eval_case_responses in eval_case_responses_list:
      actual_invocations = [
          invocation
          for invocations in eval_case_responses.responses
          for invocation in invocations
      ]
      expected_invocations = (
          eval_case_responses.eval_case.conversation * num_runs
      )

      for metric_name, threshold in criteria.items():
        metric_evaluator = AgentEvaluator._get_metric_evaluator(
            metric_name=metric_name, threshold=threshold
        )

        evaluation_result: EvaluationResult = (
            metric_evaluator.evaluate_invocations(
                actual_invocations=actual_invocations,
                expected_invocations=expected_invocations,
            )
        )

        assert evaluation_result.overall_eval_status == EvalStatus.PASSED, (
            f"{metric_name} for {agent_module} Failed. Expected {threshold},"
            f" but got {evaluation_result.overall_score}."
        )

  @staticmethod
  async def evaluate(
      agent_module: str,
      eval_dataset_file_path_or_dir: str,
      num_runs: int = NUM_RUNS,
      agent_name: Optional[str] = None,
      initial_session_file: Optional[str] = None,
  ):
    """Evaluates an Agent given eval data.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' in the loaded module.
      eval_dataset_file_path_or_dir: The eval data set. This can be either a string representing
        full path to the file containing eval dataset, or a directory that is
        recursively explored for all files that have a `.test.json` suffix.
      num_runs: Number of times all entries in the eval dataset should be
        assessed.
      agent_name: The name of the agent.
      initial_session_file: File that contains initial session state that is
        needed by all the evals in the eval dataset.
    """
    test_files = []
    if isinstance(eval_dataset_file_path_or_dir, str) and os.path.isdir(
        eval_dataset_file_path_or_dir
    ):
      for root, _, files in os.walk(eval_dataset_file_path_or_dir):
        for file in files:
          if file.endswith(".test.json"):
            test_files.append(path.join(root, file))
    else:
      test_files = [eval_dataset_file_path_or_dir]

    initial_session = AgentEvaluator._get_initial_session(initial_session_file)

    for test_file in test_files:
      criteria = AgentEvaluator.find_config_for_test_file(test_file)
      eval_set = AgentEvaluator._load_eval_set_from_file(
          test_file, criteria, initial_session
      )

      await AgentEvaluator.evaluate_eval_set(
          agent_module=agent_module,
          eval_set=eval_set,
          criteria=criteria,
          num_runs=num_runs,
          agent_name=agent_name,
      )

  @staticmethod
  def migrate_eval_data_to_new_schema(
      old_eval_data_file: str,
      new_eval_data_file: str,
      initial_session_file: Optional[str] = None,
  ):
    """A utility for migrating eval data to new schema backed by EvalSet."""
    if not old_eval_data_file or not new_eval_data_file:
      raise ValueError(
          "One of old_eval_data_file or new_eval_data_file is empty."
      )

    criteria = AgentEvaluator.find_config_for_test_file(old_eval_data_file)
    initial_session = AgentEvaluator._get_initial_session(initial_session_file)

    eval_set = AgentEvaluator._get_eval_set_from_old_format(
        old_eval_data_file, criteria, initial_session
    )

    with open(new_eval_data_file, "w") as f:
      f.write(eval_set.model_dump_json(indent=2))

  @staticmethod
  def _load_eval_set_from_file(
      eval_set_file: str,
      criteria: dict[str, float],
      initial_session: dict[str, Any],
  ) -> EvalSet:
    """Loads an EvalSet from the given file."""
    if os.path.isfile(eval_set_file):
      with open(eval_set_file, "r", encoding="utf-8") as f:
        content = f.read()

      try:
        eval_set = EvalSet.model_validate_json(content)
        assert len(initial_session) == 0, (
            "Intial session should be specified as a part of EvalSet file."
            " Explicit initial session is only needed, when specifying data in"
            " the older schema."
        )
        return eval_set
      except ValidationError:
        # We assume that the eval data was specified in the old format
        logger.warning(
            f"Contents of {eval_set_file} appear to be in older format.To avoid"
            " this warning, please update your test files to contain data in"
            " EvalSet schema. You can use `migrate_eval_data_to_new_schema`"
            " for migrating your old test files."
        )

    # If we are here, the data must be specified in the older format.
    return AgentEvaluator._get_eval_set_from_old_format(
        eval_set_file, criteria, initial_session
    )

  @staticmethod
  def _get_eval_set_from_old_format(
      eval_set_file: str,
      criteria: dict[str, float],
      initial_session: dict[str, Any],
  ) -> EvalSet:
    data = AgentEvaluator._load_dataset(eval_set_file)[0]
    AgentEvaluator._validate_input([data], criteria)
    eval_data = {
        "name": eval_set_file,
        "data": data,
        "initial_session": initial_session,
    }
    return convert_eval_set_to_pydanctic_schema(
        eval_set_id=str(uuid.uuid4()), eval_set_in_json_format=[eval_data]
    )

  @staticmethod
  def _get_initial_session(initial_session_file: Optional[str] = None):
    initial_session = {}
    if initial_session_file:
      with open(initial_session_file, "r") as f:
        initial_session = json.loads(f.read())
    return initial_session

  @staticmethod
  def _load_dataset(
      input_data: Union[str, List[str], List[Dict], List[List[Dict]]],
  ) -> List[List[Dict]]:
    def load_json_file(file_path: str) -> List[Dict]:
      data = load_json(file_path)
      if not isinstance(data, list) or not all(
          isinstance(d, dict) for d in data
      ):
        raise ValueError(f"{file_path} must contain a list of dictionaries.")
      return data

    if isinstance(input_data, str):
      if os.path.isdir(input_data):
        test_files = []
        for root, _, files in os.walk(input_data):
          for file in files:
            if file.endswith(".test.json"):
              test_files.append(os.path.join(root, file))
        return [load_json_file(f) for f in test_files]
      elif os.path.isfile(input_data):
        return [load_json_file(input_data)]
      else:
        raise ValueError(f"Input path {input_data} is invalid.")
    elif isinstance(input_data, list):
      if all(isinstance(i, str) and os.path.isfile(i) for i in input_data):
        return [load_json_file(i) for i in input_data]
      raise TypeError("Input list must contain valid file paths.")
    raise TypeError("Invalid input type for dataset loading.")

  @staticmethod
  def _validate_input(eval_dataset, criteria):
    """Validates that the evaluation criteria align with the provided dataset.

    For efficiency, we only use first row to validate input.
    """
    if not eval_dataset:
      raise ValueError("The evaluation dataset is None or empty.")

    for key in criteria:
      if key not in ALLOWED_CRITERIA:
        raise ValueError(
            f"Invalid criteria key: {key}. Expected one of {ALLOWED_CRITERIA}."
        )

    if not eval_dataset:
      raise ValueError("The evaluation dataset is empty.")
    sample = eval_dataset[0]
    first_query = sample[0]

    if not isinstance(sample, list) and not isinstance(first_query, dict):
      raise ValueError(
          "Each evaluation dataset sample must be list of dictionary. But it's"
          f" {eval_dataset}"
      )

    if TOOL_TRAJECTORY_SCORE_KEY in criteria:
      if (
          QUERY_COLUMN not in first_query
          or EXPECTED_TOOL_USE_COLUMN not in first_query
      ):
        raise ValueError(
            f"Samples for {TOOL_TRAJECTORY_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' and '{EXPECTED_TOOL_USE_COLUMN}' keys. The"
            f" sample is {sample}."
        )

    if RESPONSE_EVALUATION_SCORE_KEY in criteria:
      if QUERY_COLUMN not in first_query:
        raise ValueError(
            f"Samples for {RESPONSE_EVALUATION_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' key. The sample is {sample}."
        )

    if RESPONSE_MATCH_SCORE_KEY in criteria:
      if QUERY_COLUMN not in first_query or REFERENCE_COLUMN not in first_query:
        raise ValueError(
            f"Samples for {RESPONSE_MATCH_SCORE_KEY} must include"
            f" '{QUERY_COLUMN}' and '{REFERENCE_COLUMN}' keys. The sample is"
            f" {sample}."
        )

  @staticmethod
  def _get_metric_evaluator(metric_name: str, threshold: float) -> Evaluator:
    if metric_name == TOOL_TRAJECTORY_SCORE_KEY:
      return TrajectoryEvaluator(threshold=threshold)
    elif (
        metric_name == RESPONSE_MATCH_SCORE_KEY
        or metric_name == RESPONSE_EVALUATION_SCORE_KEY
    ):
      return ResponseEvaluator(threshold=threshold, metric_name=metric_name)

    raise ValueError(f"Unsupported eval metric: {metric_name}")
