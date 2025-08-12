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

import importlib
import json
import logging
import os
from os import path
import statistics
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
import uuid

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import ValidationError

from ..agents.base_agent import BaseAgent
from .constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from .eval_case import IntermediateData
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import EvalMetricResult
from .eval_metrics import PrebuiltMetrics
from .eval_result import EvalCaseResult
from .eval_set import EvalSet
from .eval_sets_manager import EvalSetsManager
from .evaluator import EvalStatus
from .in_memory_eval_sets_manager import InMemoryEvalSetsManager
from .local_eval_sets_manager import convert_eval_set_to_pydanctic_schema

logger = logging.getLogger("google_adk." + __name__)


# Constants for default runs and evaluation criteria
NUM_RUNS = 2

TOOL_TRAJECTORY_SCORE_KEY = PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
RESPONSE_MATCH_SCORE_KEY = PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
SAFETY_V1_KEY = PrebuiltMetrics.SAFETY_V1.value

ALLOWED_CRITERIA = [
    TOOL_TRAJECTORY_SCORE_KEY,
    RESPONSE_EVALUATION_SCORE_KEY,
    RESPONSE_MATCH_SCORE_KEY,
    SAFETY_V1_KEY,
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


class _EvalMetricResultWithInvocation(BaseModel):
  """EvalMetricResult along with both actual and expected invocation.

  This is class is intentionally marked as private and is created for
  convenience.
  """

  actual_invocation: Invocation
  expected_invocation: Invocation
  eval_metric_result: EvalMetricResult


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
      num_runs: int = NUM_RUNS,
      agent_name: Optional[str] = None,
      print_detailed_results: bool = True,
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
      agent_name: The name of the agent, if trying to evaluate something other
        than root agent. If left empty or none, then root agent is evaluated.
      print_detailed_results: Whether to print detailed results for each metric
        evaluation.
    """
    agent_for_eval = AgentEvaluator._get_agent_for_eval(
        module_name=agent_module, agent_name=agent_name
    )
    eval_metrics = [
        EvalMetric(metric_name=n, threshold=t) for n, t in criteria.items()
    ]

    # Step 1: Perform evals, basically inferencing and evaluation of metrics
    eval_results_by_eval_id = await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=agent_for_eval,
        eval_set=eval_set,
        eval_metrics=eval_metrics,
        num_runs=num_runs,
    )

    # Step 2: Post-process the results!

    # We keep track of eval case failures, these are not infra failures but eval
    # test failures. We track them and then report them towards the end.
    failures: list[str] = []

    for _, eval_results_per_eval_id in eval_results_by_eval_id.items():
      eval_metric_results = (
          AgentEvaluator._get_eval_metric_results_with_invocation(
              eval_results_per_eval_id
          )
      )
      failures_per_eval_case = AgentEvaluator._process_metrics_and_get_failures(
          eval_metric_results=eval_metric_results,
          print_detailed_results=print_detailed_results,
          agent_module=agent_name,
      )

      failures.extend(failures_per_eval_case)

    assert not failures, (
        "Following are all the test failures. If you looking to get more"
        " details on the failures, then please re-run this test with"
        " `print_details` set to `True`.\n{}".format("\n".join(failures))
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
      eval_dataset_file_path_or_dir: The eval data set. This can be either a
        string representing full path to the file containing eval dataset, or a
        directory that is recursively explored for all files that have a
        `.test.json` suffix.
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
  def _print_details(
      eval_metric_result_with_invocations: list[
          _EvalMetricResultWithInvocation
      ],
      overall_eval_status: EvalStatus,
      overall_score: Optional[float],
      metric_name: str,
      threshold: float,
  ):
    try:
      from pandas import pandas as pd
      from tabulate import tabulate
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e
    print(
        f"Summary: `{overall_eval_status}` for Metric:"
        f" `{metric_name}`. Expected threshold: `{threshold}`, actual value:"
        f" `{overall_score}`."
    )

    data = []
    for per_invocation_result in eval_metric_result_with_invocations:
      data.append({
          "eval_status": per_invocation_result.eval_metric_result.eval_status,
          "score": per_invocation_result.eval_metric_result.score,
          "threshold": threshold,
          "prompt": AgentEvaluator._convert_content_to_text(
              per_invocation_result.expected_invocation.user_content
          ),
          "expected_response": AgentEvaluator._convert_content_to_text(
              per_invocation_result.expected_invocation.final_response
          ),
          "actual_response": AgentEvaluator._convert_content_to_text(
              per_invocation_result.actual_invocation.final_response
          ),
          "expected_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
              per_invocation_result.expected_invocation.intermediate_data
          ),
          "actual_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
              per_invocation_result.actual_invocation.intermediate_data
          ),
      })

    print(tabulate(pd.DataFrame(data), headers="keys", tablefmt="grid"))
    print("\n\n")  # Few empty lines for visual clarity

  @staticmethod
  def _convert_content_to_text(content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  @staticmethod
  def _convert_tool_calls_to_text(
      intermediate_data: Optional[IntermediateData],
  ) -> str:
    if intermediate_data and intermediate_data.tool_uses:
      return "\n".join([str(t) for t in intermediate_data.tool_uses])

    return ""

  @staticmethod
  def _get_agent_for_eval(
      module_name: str, agent_name: Optional[str] = None
  ) -> BaseAgent:
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)
    root_agent = agent_module.agent.root_agent

    agent_for_eval = root_agent
    if agent_name:
      agent_for_eval = root_agent.find_agent(agent_name)
      assert agent_for_eval, f"Sub-Agent `{agent_name}` not found."

    return agent_for_eval

  @staticmethod
  def _get_eval_sets_manager(
      app_name: str, eval_set: EvalSet
  ) -> EvalSetsManager:
    eval_sets_manager = InMemoryEvalSetsManager()

    eval_sets_manager.create_eval_set(
        app_name=app_name, eval_set_id=eval_set.eval_set_id
    )
    for eval_case in eval_set.eval_cases:
      eval_sets_manager.add_eval_case(
          app_name=app_name,
          eval_set_id=eval_set.eval_set_id,
          eval_case=eval_case,
      )

    return eval_sets_manager

  @staticmethod
  async def _get_eval_results_by_eval_id(
      agent_for_eval: BaseAgent,
      eval_set: EvalSet,
      eval_metrics: list[EvalMetric],
      num_runs: int,
  ) -> dict[str, list[EvalCaseResult]]:
    """Returns EvalCaseResults grouped by eval case id.

    The grouping happens because of the "num_runs" argument, where for any value
    greater than 1, we would have generated inferences num_runs times and so
    by extension we would have evaluated metrics on each of those inferences.
    """
    try:
      from .base_eval_service import EvaluateConfig
      from .base_eval_service import EvaluateRequest
      from .base_eval_service import InferenceConfig
      from .base_eval_service import InferenceRequest
      from .local_eval_service import LocalEvalService
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

    # It is okay to pick up this dummy name.
    app_name = "test_app"
    eval_service = LocalEvalService(
        root_agent=agent_for_eval,
        eval_sets_manager=AgentEvaluator._get_eval_sets_manager(
            app_name=app_name, eval_set=eval_set
        ),
    )

    inference_requests = [
        InferenceRequest(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            inference_config=InferenceConfig(),
        )
    ] * num_runs  # Repeat inference request num_runs times.

    # Generate inferences
    inference_results = []
    for inference_request in inference_requests:
      async for inference_result in eval_service.perform_inference(
          inference_request=inference_request
      ):
        inference_results.append(inference_result)

    # Evaluate metrics
    # As we perform more than one run for an eval case, we collect eval results
    # by eval id.
    eval_results_by_eval_id: dict[str, list[EvalCaseResult]] = {}
    evaluate_request = EvaluateRequest(
        inference_results=inference_results,
        evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
    )
    async for eval_result in eval_service.evaluate(
        evaluate_request=evaluate_request
    ):
      eval_id = eval_result.eval_id
      if eval_id not in eval_results_by_eval_id:
        eval_results_by_eval_id[eval_id] = []

      eval_results_by_eval_id[eval_id].append(eval_result)

    return eval_results_by_eval_id

  @staticmethod
  def _get_eval_metric_results_with_invocation(
      eval_results_per_eval_id: list[EvalCaseResult],
  ) -> dict[str, list[_EvalMetricResultWithInvocation]]:
    """Retruns _EvalMetricResultWithInvocation grouped by metric.

    EvalCaseResult contain results for each metric per invocation.

    This method flips it around and returns a structure that groups metric
    results per invocation by eval metric.

    This is a convenience function.
    """
    eval_metric_results: dict[str, list[_EvalMetricResultWithInvocation]] = {}

    # Go over the EvalCaseResult one by one, do note that at this stage all
    # EvalCaseResult belong to the same eval id.
    for eval_case_result in eval_results_per_eval_id:
      # For the given eval_case_result, we go over metric results for each
      # invocation. Do note that a single eval case can have more than one
      # invocation and for each invocation there could be more than on eval
      # metrics that were evaluated.
      for (
          eval_metrics_per_invocation
      ) in eval_case_result.eval_metric_result_per_invocation:
        # Go over each eval_metric_result for an invocation.
        for (
            eval_metric_result
        ) in eval_metrics_per_invocation.eval_metric_results:
          metric_name = eval_metric_result.metric_name
          if metric_name not in eval_metric_results:
            eval_metric_results[metric_name] = []

          actual_invocation = eval_metrics_per_invocation.actual_invocation
          expected_invocation = eval_metrics_per_invocation.expected_invocation

          eval_metric_results[metric_name].append(
              _EvalMetricResultWithInvocation(
                  actual_invocation=actual_invocation,
                  expected_invocation=expected_invocation,
                  eval_metric_result=eval_metric_result,
              )
          )
    return eval_metric_results

  @staticmethod
  def _process_metrics_and_get_failures(
      eval_metric_results: dict[str, list[_EvalMetricResultWithInvocation]],
      print_detailed_results: bool,
      agent_module: str,
  ) -> list[str]:
    """Returns a list of failures based on the score for each invocation."""
    failures: list[str] = []
    for (
        metric_name,
        eval_metric_results_with_invocations,
    ) in eval_metric_results.items():
      threshold = eval_metric_results_with_invocations[
          0
      ].eval_metric_result.threshold
      scores = [
          m.eval_metric_result.score
          for m in eval_metric_results_with_invocations
          if m.eval_metric_result.score
      ]

      if scores:
        overall_score = statistics.mean(scores)
        overall_eval_status = (
            EvalStatus.PASSED
            if overall_score >= threshold
            else EvalStatus.FAILED
        )
      else:
        overall_score = None
        overall_eval_status = EvalStatus.NOT_EVALUATED

      # Gather all the failures.
      if overall_eval_status != EvalStatus.PASSED:
        if print_detailed_results:
          AgentEvaluator._print_details(
              eval_metric_result_with_invocations=eval_metric_results_with_invocations,
              overall_eval_status=overall_eval_status,
              overall_score=overall_score,
              metric_name=metric_name,
              threshold=threshold,
          )
        failures.append(
            f"{metric_name} for {agent_module} Failed. Expected {threshold},"
            f" but got {overall_score}."
        )

    return failures
