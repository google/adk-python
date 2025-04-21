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
from os import path
from typing import Dict
from typing import List
from typing import Union

from .evaluation_generator import EvaluationGenerator
from .response_evaluator import ResponseEvaluator
from .trajectory_evaluator import TrajectoryEvaluator

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
  def evaluate(
      agent_module,
      eval_dataset_file_path_or_dir,
      num_runs=NUM_RUNS,
      agent_name=None,
      initial_session_file=None,
  ):
    """Evaluates an Agent given eval data.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' in the loaded module.
      eval_dataset: The eval data set. This can be either a string representing
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

    initial_session_state = {}
    if initial_session_file:
      with open(initial_session_file, "r") as f:
        initial_session_state = json.loads(f.read())["state"]

    for test_file in test_files:
      dataset = AgentEvaluator._load_dataset(test_file)[0]
      criteria = AgentEvaluator.find_config_for_test_file(test_file)

      AgentEvaluator._validate_input([dataset], criteria)

      evaluation_response = AgentEvaluator._generate_responses(
          agent_module,
          [dataset],
          num_runs,
          agent_name=agent_name,
          initial_session={"state": initial_session_state},
      )

      if AgentEvaluator._response_evaluation_required(criteria, [dataset]):
        AgentEvaluator._evaluate_response_scores(
            agent_module, evaluation_response, criteria
        )

      if AgentEvaluator._trajectory_evaluation_required(criteria, [dataset]):
        AgentEvaluator._evaluate_tool_trajectory(
            agent_module, evaluation_response, criteria
        )

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
  def _get_infer_criteria(eval_dataset):
    """Infers evaluation criteria based on the provided dataset.

    Args:
        eval_dataset (list): A list of evaluation samples.

    Returns:
        dict: Inferred evaluation criteria based on dataset fields.
    """
    inferred_criteria = {}
    sample = eval_dataset[0][0]

    if QUERY_COLUMN in sample and EXPECTED_TOOL_USE_COLUMN in sample:
      inferred_criteria[TOOL_TRAJECTORY_SCORE_KEY] = DEFAULT_CRITERIA[
          TOOL_TRAJECTORY_SCORE_KEY
      ]

    if QUERY_COLUMN in sample and REFERENCE_COLUMN in sample:
      inferred_criteria[RESPONSE_MATCH_SCORE_KEY] = DEFAULT_CRITERIA[
          RESPONSE_MATCH_SCORE_KEY
      ]

    return inferred_criteria

  @staticmethod
  def _generate_responses(
      agent_module, eval_dataset, num_runs, agent_name=None, initial_session={}
  ):
    """Generates evaluation responses by running the agent module multiple times."""
    return EvaluationGenerator.generate_responses(
        eval_dataset,
        agent_module,
        repeat_num=num_runs,
        agent_name=agent_name,
        initial_session=initial_session,
    )

  @staticmethod
  def _generate_responses_from_session(eval_dataset, session_path):
    """Generates evaluation responses by running the agent module multiple times."""
    return EvaluationGenerator.generate_responses_from_session(
        session_path, eval_dataset
    )

  @staticmethod
  def _response_evaluation_required(criteria, eval_dataset):
    """Checks if response evaluation are needed."""
    return REFERENCE_COLUMN in eval_dataset[0][0] and any(
        key in criteria
        for key in [RESPONSE_EVALUATION_SCORE_KEY, RESPONSE_MATCH_SCORE_KEY]
    )

  @staticmethod
  def _trajectory_evaluation_required(evaluation_criteria, eval_dataset):
    """Checks if response evaluation are needed."""
    return (
        EXPECTED_TOOL_USE_COLUMN in eval_dataset[0][0]
        and TOOL_TRAJECTORY_SCORE_KEY in evaluation_criteria
    )

  @staticmethod
  def _evaluate_response_scores(agent_module, evaluation_response, criteria):
    """Evaluates response scores and raises an assertion error if they don't meet the criteria."""
    metrics = ResponseEvaluator.evaluate(
        evaluation_response, criteria, print_detailed_results=True
    )

    AgentEvaluator._assert_score(
        metrics,
        "coherence/mean",
        criteria.get(RESPONSE_EVALUATION_SCORE_KEY),
        "Average response evaluation score",
        agent_module,
    )

    AgentEvaluator._assert_score(
        metrics,
        "rouge_1/mean",
        criteria.get(RESPONSE_MATCH_SCORE_KEY),
        "Average response match score",
        agent_module,
    )

  @staticmethod
  def _evaluate_tool_trajectory(agent_module, evaluation_response, criteria):
    """Evaluates tool trajectory scores and raises an assertion error if they don't meet the criteria."""
    score = TrajectoryEvaluator.evaluate(
        evaluation_response, print_detailed_results=True
    )
    AgentEvaluator._assert_score(
        {TOOL_TRAJECTORY_SCORE_KEY: score},
        TOOL_TRAJECTORY_SCORE_KEY,
        criteria[TOOL_TRAJECTORY_SCORE_KEY],
        "Average tool trajectory evaluation score",
        agent_module,
    )

  @staticmethod
  def _assert_score(metrics, metric_key, threshold, description, agent_module):
    """Asserts that a metric meets the specified threshold."""
    if metric_key in metrics:
      actual_score = metrics[metric_key]
      assert actual_score >= threshold, (
          f"{description} for {agent_module} is lower than expected. "
          f"Expected >= {threshold}, but got {actual_score}."
      )
