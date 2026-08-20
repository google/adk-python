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

from collections.abc import Awaitable
from collections.abc import Mapping
import importlib
import json
import logging
import os
from os import path
import statistics
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Union
import uuid

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import ValidationError

from ..agents.base_agent import BaseAgent
from ..apps.app import App
from ..artifacts.base_artifact_service import BaseArtifactService
from ..utils import _json_utils
from ..utils.context_utils import Aclosing
from .constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from .eval_case import get_all_tool_calls
from .eval_case import IntermediateDataType
from .eval_case import Invocation
from .eval_config import EvalConfig
from .eval_config import get_eval_metrics_from_config
from .eval_config import get_evaluation_criteria_or_default
from .eval_config import LiveModelConfig
from .eval_metrics import _get_metric_threshold
from .eval_metrics import BaseCriterion
from .eval_metrics import EvalMetric
from .eval_metrics import EvalMetricResult
from .eval_metrics import PrebuiltMetrics
from .eval_result import EvalCaseResult
from .eval_set import EvalSet
from .eval_set_results_manager import EvalSetResultsManager
from .eval_sets_manager import EvalSetsManager
from .evaluator import EvalStatus
from .in_memory_eval_sets_manager import InMemoryEvalSetsManager
from .local_eval_sets_manager import convert_eval_set_to_pydantic_schema
from .metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from .metric_evaluator_registry import MetricEvaluatorRegistry
from .metric_evaluator_registry import register_custom_metrics_from_config
from .simulation.user_simulator_provider import UserSimulatorProvider

logger = logging.getLogger("google_adk." + __name__)


# Constants for default runs and evaluation criteria
NUM_RUNS = 2

TOOL_TRAJECTORY_SCORE_KEY = PrebuiltMetrics.TOOL_TRAJECTORY_AVG_SCORE.value
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = PrebuiltMetrics.RESPONSE_EVALUATION_SCORE.value
RESPONSE_MATCH_SCORE_KEY = PrebuiltMetrics.RESPONSE_MATCH_SCORE.value
SAFETY_V1_KEY = PrebuiltMetrics.SAFETY_V1.value


class _AsyncAgentFactory(Protocol):

  def __call__(self) -> Awaitable[tuple[BaseAgent, object]]:
    """Loads a root agent and optional cleanup metadata."""


ALLOWED_CRITERIA = [
    TOOL_TRAJECTORY_SCORE_KEY,
    RESPONSE_EVALUATION_SCORE_KEY,
    RESPONSE_MATCH_SCORE_KEY,
    SAFETY_V1_KEY,
]

QUERY_COLUMN = "query"
REFERENCE_COLUMN = "reference"
EXPECTED_TOOL_USE_COLUMN = "expected_tool_use"


def load_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
  with open(file_path, "r", encoding="utf-8") as f:
    return cast(Union[Dict[str, Any], List[Any]], json.load(f))


class _EvalMetricResultWithInvocation(BaseModel):
  """EvalMetricResult along with both actual and expected invocation.

  This is class is intentionally marked as private and is created for
  convenience.
  """

  actual_invocation: Invocation
  expected_invocation: Invocation | None = None
  eval_metric_result: EvalMetricResult


class AgentEvaluator:
  """An evaluator for Agents, mainly intended for helping with test cases."""

  @staticmethod
  def find_config_for_test_file(test_file: str) -> EvalConfig:
    """Find the test_config.json file in the same folder as the test file."""
    test_folder = os.path.dirname(test_file)
    config_path = os.path.join(test_folder, "test_config.json")
    return get_evaluation_criteria_or_default(config_path)

  @staticmethod
  async def evaluate_eval_set(
      agent_module: str,
      eval_set: EvalSet,
      criteria: Optional[dict[str, float]] = None,
      eval_config: Optional[EvalConfig] = None,
      num_runs: int = NUM_RUNS,
      agent_name: Optional[str] = None,
      print_detailed_results: bool = True,
      artifact_service: Optional[BaseArtifactService] = None,
      output_file: Optional[str] = None,
      app_name: Optional[str] = None,
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
  ) -> None:
    """Evaluates an agent using the given EvalSet.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' or `get_agent_async` in the loaded module.
      eval_set: The eval set.
      criteria: Evaluation criteria, a dictionary of metric names to their
        respective thresholds. This field is deprecated.
      eval_config: The evaluation config.
      num_runs: Number of times all entries in the eval dataset should be
        assessed.
      agent_name: The name of the agent, if trying to evaluate something other
        than root agent. If left empty or none, then root agent is evaluated.
      print_detailed_results: Whether to print detailed results for each metric
        evaluation.
      artifact_service: The artifact service used to load artifacts during eval.
        Pre-load artifacts here and pin each eval case to a session id (via
        `SessionInput.session_id`) to make them reachable. Defaults to an
        in-memory service.
      output_file: If provided, per-invocation evaluation results (for both
        passing and failing metrics) are written to this path as a CSV file.
        Disabled by default. The parent directory is created if it does not
        already exist.
      app_name: The application name used by eval set results manager while
        persisting eval set results.
      eval_set_results_manager: Optional manager used to persist the eval set
        evaluation result as `*.evalset_result.json`.
    """
    if eval_set_results_manager is not None and not app_name:
      raise ValueError(
          "app_name is required when eval_set_results_manager is provided."
      )

    if criteria:
      logger.warning(
          "`criteria` field is deprecated and will be removed in future"
          " iterations. For now, we will automatically map values in `criteria`"
          " to `eval_config`, but you should move to using `eval_config` field."
      )
      base_criteria = {
          k: BaseCriterion(threshold=v) for k, v in criteria.items()
      }
      eval_config = EvalConfig(criteria=base_criteria)

    if eval_config is None:
      raise ValueError("`eval_config` is required.")

    agent_for_eval, app = await AgentEvaluator._get_agent_for_eval(
        module_name=agent_module, agent_name=agent_name
    )
    eval_metrics = get_eval_metrics_from_config(eval_config)

    user_simulator_provider = UserSimulatorProvider(
        user_simulator_config=eval_config.user_simulator_config
    )
    live_model_config = eval_config.live_model_config

    # `eval_set_results_manager`, when provided, is what persists the eval
    # results as `*.evalset_result.json` files (via LocalEvalService), stored
    # under `app_name`. When no manager is given, nothing is saved, so a dummy
    # `app_name` is fine here.
    app_name = app_name or "test_app"

    # A fork, not the default registry itself: the custom metrics of this eval
    # config belong to this run only. Forking (rather than starting from a bare
    # `MetricEvaluatorRegistry()`) keeps evaluators that the caller registered
    # on the default registry resolvable, which is the only way to plug in a
    # custom `Evaluator` subclass since an eval config can only name a scoring
    # function.
    metric_evaluator_registry = register_custom_metrics_from_config(
        eval_config, DEFAULT_METRIC_EVALUATOR_REGISTRY.fork()
    )

    # Step 1: Perform evals, basically inferencing and evaluation of metrics
    eval_results_by_eval_id = await AgentEvaluator._get_eval_results_by_eval_id(
        agent_for_eval=agent_for_eval,
        eval_set=eval_set,
        eval_metrics=eval_metrics,
        num_runs=num_runs,
        user_simulator_provider=user_simulator_provider,
        app_name=app_name,
        live_model_config=live_model_config,
        artifact_service=artifact_service,
        app=app,
        eval_set_results_manager=eval_set_results_manager,
        metric_evaluator_registry=metric_evaluator_registry,
    )

    # Step 2: Post-process the results!

    # We keep track of eval case failures, these are not infra failures but eval
    # test failures. We track them and then report them towards the end.
    failures: list[str] = []

    # Optionally, we collect per-invocation results across all eval cases and
    # metrics so that they can be written out to a CSV file at the end.
    csv_rows: list[dict[str, Any]] = []

    for eval_id, eval_results_per_eval_id in eval_results_by_eval_id.items():
      eval_metric_results = (
          AgentEvaluator._get_eval_metric_results_with_invocation(
              eval_results_per_eval_id
          )
      )
      failures_per_eval_case = AgentEvaluator._process_metrics_and_get_failures(
          eval_metric_results=eval_metric_results,
          print_detailed_results=print_detailed_results,
          agent_module=agent_module,
      )

      failures.extend(failures_per_eval_case)
      failures.extend(
          AgentEvaluator._get_failures_from_final_eval_status(
              eval_id=eval_id,
              eval_results_per_eval_id=eval_results_per_eval_id,
              agent_module=agent_module,
          )
      )

      if output_file:
        csv_rows.extend(
            AgentEvaluator._get_results_as_rows(
                eval_set_id=eval_set.eval_set_id,
                eval_id=eval_id,
                eval_metric_results=eval_metric_results,
            )
        )

    if output_file:
      AgentEvaluator._write_results_to_csv(
          rows=csv_rows, output_file=output_file
      )

    failure_message = "Following are all the test failures."
    if not print_detailed_results:
      failure_message += (
          " If you looking to get more details on the failures, then please"
          " re-run this test with `print_detailed_results` set to `True`."
      )
    failure_message += "\n" + "\n".join(failures)
    assert not failures, failure_message

  @staticmethod
  async def evaluate(
      agent_module: str,
      eval_dataset_file_path_or_dir: str,
      num_runs: int = NUM_RUNS,
      agent_name: Optional[str] = None,
      initial_session_file: Optional[str] = None,
      print_detailed_results: bool = True,
      artifact_service: Optional[BaseArtifactService] = None,
      output_file: Optional[str] = None,
      app_name: Optional[str] = None,
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
  ) -> None:
    """Evaluates an Agent given eval data.

    Args:
      agent_module: The path to python module that contains the definition of
        the agent. There is convention in place here, where the code is going to
        look for 'root_agent' or 'get_agent_async' in the loaded module.
      eval_dataset_file_path_or_dir: The eval data set. This can be either a
        string representing full path to the file containing eval dataset, or a
        directory that is recursively explored for all files that have a
        `.test.json` suffix.
      num_runs: Number of times all entries in the eval dataset should be
        assessed.
      agent_name: The name of the agent.
      initial_session_file: File that contains initial session state that is
        needed by all the evals in the eval dataset.
      print_detailed_results: Whether to print detailed results for each metric
        evaluation.
      artifact_service: The artifact service used to load artifacts during eval.
        Pre-load artifacts here and pin each eval case to a session id (via
        `SessionInput.session_id`) to make them reachable. Defaults to an
        in-memory service.
      output_file: If provided, per-invocation evaluation results are written to
        this path as a CSV file. Disabled by default. When the eval data spans
        multiple test files, results from all of them are appended to the same
        file.
      app_name: The application name used by eval set results manager while
        persisting eval set results.
      eval_set_results_manager: Optional manager used to persist the eval set
        evaluation result as `*.evalset_result.json`.
    """
    if eval_set_results_manager is not None and not app_name:
      raise ValueError(
          "app_name is required when eval_set_results_manager is provided."
      )

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
      eval_config = AgentEvaluator.find_config_for_test_file(test_file)
      eval_set = AgentEvaluator._load_eval_set_from_file(
          test_file, eval_config, initial_session
      )

      await AgentEvaluator.evaluate_eval_set(
          agent_module=agent_module,
          eval_set=eval_set,
          eval_config=eval_config,
          num_runs=num_runs,
          agent_name=agent_name,
          print_detailed_results=print_detailed_results,
          app_name=app_name,
          eval_set_results_manager=eval_set_results_manager,
          artifact_service=artifact_service,
          output_file=output_file,
      )

  @staticmethod
  def migrate_eval_data_to_new_schema(
      old_eval_data_file: str,
      new_eval_data_file: str,
      initial_session_file: Optional[str] = None,
  ) -> None:
    """A utility for migrating eval data to new schema backed by EvalSet."""
    if not old_eval_data_file or not new_eval_data_file:
      raise ValueError(
          "One of old_eval_data_file or new_eval_data_file is empty."
      )

    eval_config = AgentEvaluator.find_config_for_test_file(old_eval_data_file)
    initial_session = AgentEvaluator._get_initial_session(initial_session_file)

    eval_set = AgentEvaluator._get_eval_set_from_old_format(
        old_eval_data_file, eval_config, initial_session
    )

    with open(new_eval_data_file, "w", encoding="utf-8") as f:
      f.write(eval_set.model_dump_json(indent=2))

  @staticmethod
  def _load_eval_set_from_file(
      eval_set_file: str,
      eval_config: EvalConfig,
      initial_session: dict[str, Any],
  ) -> EvalSet:
    """Loads an EvalSet from the given file."""
    if os.path.isfile(eval_set_file):
      with open(eval_set_file, "r", encoding="utf-8") as f:
        content = f.read()

      try:
        eval_set = EvalSet.model_validate_json(content)
        assert len(initial_session) == 0, (
            "Initial session should be specified as a part of EvalSet file."
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
        eval_set_file, eval_config, initial_session
    )

  @staticmethod
  def _get_eval_set_from_old_format(
      eval_set_file: str,
      eval_config: EvalConfig,
      initial_session: dict[str, Any],
  ) -> EvalSet:
    data = AgentEvaluator._load_dataset(eval_set_file)[0]
    AgentEvaluator._validate_input([data], eval_config.criteria)
    eval_data = {
        "name": eval_set_file,
        "data": data,
        "initial_session": initial_session,
    }
    return convert_eval_set_to_pydantic_schema(
        eval_set_id=str(uuid.uuid4()), eval_set_in_json_format=[eval_data]
    )

  @staticmethod
  def _get_initial_session(
      initial_session_file: Optional[str] = None,
  ) -> dict[str, Any]:
    initial_session: dict[str, Any] = {}
    if initial_session_file:
      with open(initial_session_file, "r", encoding="utf-8") as f:
        initial_session = _json_utils.safe_json_loads(
            f.read(), context=f"initial session file {initial_session_file}"
        )
    return initial_session

  @staticmethod
  def _load_dataset(
      input_data: Union[
          str,
          List[str],
          List[Dict[str, Any]],
          List[List[Dict[str, Any]]],
      ],
  ) -> List[List[Dict[str, Any]]]:
    def load_json_file(file_path: str) -> List[Dict[str, Any]]:
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
        return [load_json_file(cast(str, i)) for i in input_data]
      raise TypeError("Input list must contain valid file paths.")
    raise TypeError("Invalid input type for dataset loading.")

  @staticmethod
  def _validate_input(
      eval_dataset: list[list[dict[str, Any]]],
      criteria: Mapping[str, object],
  ) -> None:
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
  ) -> None:
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
              if per_invocation_result.expected_invocation
              else per_invocation_result.actual_invocation.user_content
          ),
          "expected_response": AgentEvaluator._convert_content_to_text(
              per_invocation_result.expected_invocation.final_response
              if per_invocation_result.expected_invocation
              else None
          ),
          "actual_response": AgentEvaluator._convert_content_to_text(
              per_invocation_result.actual_invocation.final_response
          ),
          "expected_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
              per_invocation_result.expected_invocation.intermediate_data
              if per_invocation_result.expected_invocation
              else None
          ),
          "actual_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
              per_invocation_result.actual_invocation.intermediate_data
          ),
      })

    print(
        tabulate(
            pd.DataFrame(data), headers="keys", tablefmt="grid", maxcolwidths=25
        )
    )
    print("\n\n")  # Few empty lines for visual clarity

  @staticmethod
  def _convert_content_to_text(content: Optional[genai_types.Content]) -> str:
    if content and content.parts:
      return "\n".join([p.text for p in content.parts if p.text])

    return ""

  @staticmethod
  def _convert_tool_calls_to_text(
      intermediate_data: Optional[IntermediateDataType],
  ) -> str:
    tool_calls = get_all_tool_calls(intermediate_data)

    return "\n".join([str(t) for t in tool_calls])

  @staticmethod
  async def _get_agent_for_eval(
      module_name: str, agent_name: Optional[str] = None
  ) -> tuple[BaseAgent, Optional[App]]:
    """Returns the (agent_for_eval, app) pair for the given module.

    If the module exposes an `App` instance via `agent.app`, that App is
    returned alongside the agent to evaluate, so `app.plugins`, context-cache,
    and resumability configs participate in the eval run. Otherwise `app` is
    None and only the bare agent is returned. When `agent_name` is provided,
    the returned agent is the corresponding sub-agent, but the App (if any) is
    still surfaced so its application-wide configuration is honored.
    """
    module_path = f"{module_name}"
    agent_module = importlib.import_module(module_path)

    # One of the two things should be satisfied, either the module should have
    # an "agent" as a member in it or the module name itself should end with
    # ".agent".
    if not (hasattr(agent_module, "agent") or module_name.endswith(".agent")):
      raise ValueError(
          f"Module {module_name} does not have a member named `agent` or the"
          " name should endwith `.agent`."
      )

    agent_module_with_agent: object = getattr(
        agent_module, "agent", agent_module
    )
    root_candidate: object = getattr(
        agent_module_with_agent, "root_agent", None
    )
    if root_candidate is None:
      factory_candidate: object = getattr(
          agent_module_with_agent, "get_agent_async", None
      )
      if not callable(factory_candidate):
        raise ValueError(
            f"Module {module_name} does not have a root_agent or"
            " get_agent_async method."
        )
      factory = cast(_AsyncAgentFactory, factory_candidate)
      root_candidate, _ = await factory()

    root_agent = cast(BaseAgent, root_candidate)

    app = getattr(agent_module_with_agent, "app", None)
    if not isinstance(app, App):
      app = None

    agent_for_eval = root_agent
    if agent_name:
      selected_agent = root_agent.find_agent(agent_name)
      if selected_agent is None:
        raise ValueError(f"Sub-Agent {agent_name!r} not found.")
      agent_for_eval = selected_agent

    return agent_for_eval, app

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
      user_simulator_provider: UserSimulatorProvider,
      live_model_config: Optional[LiveModelConfig] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      app: Optional[App] = None,
      *,
      app_name: str,
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
      metric_evaluator_registry: Optional[MetricEvaluatorRegistry] = None,
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

    eval_service = LocalEvalService(
        root_agent=agent_for_eval,
        eval_sets_manager=AgentEvaluator._get_eval_sets_manager(
            app_name=app_name, eval_set=eval_set
        ),
        user_simulator_provider=user_simulator_provider,
        artifact_service=artifact_service,
        app=app,
        eval_set_results_manager=eval_set_results_manager,
        metric_evaluator_registry=metric_evaluator_registry,
    )

    if live_model_config:
      inference_config = InferenceConfig(
          use_live=True,
          live_timeout_seconds=live_model_config.timeout_seconds,
      )
    else:
      inference_config = InferenceConfig(use_live=False)

    inference_requests = [
        InferenceRequest(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            inference_config=inference_config,
        )
    ] * num_runs  # Repeat inference request num_runs times.

    # Generate inferences
    inference_results = []
    for inference_request in inference_requests:
      async with Aclosing(
          eval_service.perform_inference(inference_request=inference_request)
      ) as agen:
        async for inference_result in agen:
          inference_results.append(inference_result)

    # Evaluate metrics
    # As we perform more than one run for an eval case, we collect eval results
    # by eval id.
    eval_results_by_eval_id: dict[str, list[EvalCaseResult]] = {}
    evaluate_request = EvaluateRequest(
        inference_results=inference_results,
        evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
    )
    async with Aclosing(
        eval_service.evaluate(evaluate_request=evaluate_request)
    ) as agen:
      async for eval_result in agen:
        eval_id = eval_result.eval_id
        if eval_id not in eval_results_by_eval_id:
          eval_results_by_eval_id[eval_id] = []

        eval_results_by_eval_id[eval_id].append(eval_result)

    return eval_results_by_eval_id

  @staticmethod
  def _get_eval_metric_results_with_invocation(
      eval_results_per_eval_id: list[EvalCaseResult],
  ) -> dict[str, list[_EvalMetricResultWithInvocation]]:
    """Returns _EvalMetricResultWithInvocation grouped by metric.

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
      if not eval_metric_results_with_invocations:
        continue
      threshold = _get_metric_threshold(
          eval_metric_results_with_invocations[0].eval_metric_result
      )
      scores = [
          m.eval_metric_result.score
          for m in eval_metric_results_with_invocations
          if m.eval_metric_result.score is not None
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

  @staticmethod
  def _get_failures_from_final_eval_status(
      eval_id: str,
      eval_results_per_eval_id: list[EvalCaseResult],
      agent_module: str,
  ) -> list[str]:
    """Returns failures that the per-invocation metric results cannot show.

    A run that produced no metric results at all, for example because
    inferencing raised, leaves `_process_metrics_and_get_failures` with nothing
    to derive a verdict from. The status recorded on the EvalCaseResult is the
    only record that such a run failed, so we honor it here.
    """
    failed_runs = 0
    for eval_case_result in eval_results_per_eval_id:
      if eval_case_result.final_eval_status != EvalStatus.FAILED:
        continue
      per_invocation_results = (
          eval_case_result.eval_metric_result_per_invocation
      )
      if any(r.eval_metric_results for r in per_invocation_results):
        continue
      failed_runs += 1

    if not failed_runs:
      return []

    return [
        f"{eval_id} for {agent_module} Failed. {failed_runs} of"
        f" {len(eval_results_per_eval_id)} runs were recorded as failed without"
        " producing any metric results."
    ]

  @staticmethod
  def _get_results_as_rows(
      eval_set_id: str,
      eval_id: str,
      eval_metric_results: dict[str, list[_EvalMetricResultWithInvocation]],
  ) -> list[dict[str, Any]]:
    """Flattens eval results into one row per metric per invocation.

    The columns mirror the ones used in `_print_details`, with additional
    identifier columns so that rows from different eval cases and metrics can be
    distinguished within a single CSV file.
    """
    rows: list[dict[str, Any]] = []
    for metric_name, results_with_invocations in eval_metric_results.items():
      for result_with_invocation in results_with_invocations:
        eval_metric_result = result_with_invocation.eval_metric_result
        expected_invocation = result_with_invocation.expected_invocation
        actual_invocation = result_with_invocation.actual_invocation
        rows.append({
            "eval_set_id": eval_set_id,
            "eval_id": eval_id,
            "metric_name": metric_name,
            "threshold": eval_metric_result.threshold,
            "score": eval_metric_result.score,
            "eval_status": eval_metric_result.eval_status.name,
            "prompt": AgentEvaluator._convert_content_to_text(
                expected_invocation.user_content
                if expected_invocation
                else actual_invocation.user_content
            ),
            "expected_response": AgentEvaluator._convert_content_to_text(
                expected_invocation.final_response
                if expected_invocation
                else None
            ),
            "actual_response": AgentEvaluator._convert_content_to_text(
                actual_invocation.final_response
            ),
            "expected_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
                expected_invocation.intermediate_data
                if expected_invocation
                else None
            ),
            "actual_tool_calls": AgentEvaluator._convert_tool_calls_to_text(
                actual_invocation.intermediate_data
            ),
        })
    return rows

  @staticmethod
  def _write_results_to_csv(
      rows: list[dict[str, Any]], output_file: str
  ) -> None:
    """Writes eval results to a CSV file.

    Appends rows to the file if it already exists, writing the header only once.
    Creates parent directories if necessary.
    """
    try:
      import pandas as pd
    except ModuleNotFoundError as e:
      raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

    output_dir = os.path.dirname(output_file)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)

    file_exists = os.path.isfile(output_file)
    pd.DataFrame(rows).to_csv(
        output_file, mode="a", header=not file_exists, index=False
    )
    logger.info("Saved eval results to %s", output_file)
