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

from enum import Enum
import importlib.util
import json
import logging
import os
import sys
import traceback
from typing import Any
from typing import Generator
from typing import Optional
import uuid

from pydantic import BaseModel

from ..agents import Agent

logger = logging.getLogger(__name__)


class EvalStatus(Enum):
  PASSED = 1
  FAILED = 2
  NOT_EVALUATED = 3


class EvalMetric(BaseModel):
  metric_name: str
  threshold: float


class EvalMetricResult(BaseModel):
  score: Optional[float]
  eval_status: EvalStatus


class EvalResult(BaseModel):
  eval_set_file: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]]
  session_id: str


MISSING_EVAL_DEPENDENCIES_MESSAGE = (
    "Eval module is not installed, please install via `pip install"
    " google-adk[eval]`."
)
TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = "response_evaluation_score"

EVAL_SESSION_ID_PREFIX = "___eval___session___"
DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,
}


def _import_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _get_agent_module(agent_module_file_path: str):
  file_path = os.path.join(agent_module_file_path, "__init__.py")
  module_name = "agent"
  return _import_from_path(module_name, file_path)


def get_evaluation_criteria_or_default(
    eval_config_file_path: str,
) -> dict[str, float]:
  """Returns evaluation criteria from the config file, if present.

  Otherwise a default one is returned.
  """
  if eval_config_file_path:
    with open(eval_config_file_path, "r", encoding="utf-8") as f:
      config_data = json.load(f)

    if "criteria" in config_data and isinstance(config_data["criteria"], dict):
      evaluation_criteria = config_data["criteria"]
    else:
      raise ValueError(
          f"Invalid format for test_config.json at {eval_config_file_path}."
          " Expected a 'criteria' dictionary."
      )
  else:
    logger.info("No config file supplied. Using default criteria.")
    evaluation_criteria = DEFAULT_CRITERIA

  return evaluation_criteria


def get_root_agent(agent_module_file_path: str) -> Agent:
  """Returns root agent given the agent module."""
  agent_module = _get_agent_module(agent_module_file_path)
  root_agent = agent_module.agent.root_agent
  return root_agent


def try_get_reset_func(agent_module_file_path: str) -> Any:
  """Returns reset function for the agent, if present, given the agent module."""
  agent_module = _get_agent_module(agent_module_file_path)
  reset_func = getattr(agent_module.agent, "reset_data", None)
  return reset_func


def parse_and_get_evals_to_run(
    eval_set_file_path: tuple[str],
) -> dict[str, list[str]]:
  """Returns a dictionary of eval sets to evals that should be run."""
  eval_set_to_evals = {}
  for input_eval_set in eval_set_file_path:
    evals = []
    if ":" not in input_eval_set:
      eval_set_file = input_eval_set
    else:
      eval_set_file = input_eval_set.split(":")[0]
      evals = input_eval_set.split(":")[1].split(",")

    if eval_set_file not in eval_set_to_evals:
      eval_set_to_evals[eval_set_file] = []

    eval_set_to_evals[eval_set_file].extend(evals)

  return eval_set_to_evals


def run_evals(
    eval_set_to_evals: dict[str, list[str]],
    root_agent: Agent,
    reset_func: Optional[Any],
    eval_metrics: list[EvalMetric],
    session_service=None,
    artifact_service=None,
    print_detailed_results=False,
) -> Generator[EvalResult, None, None]:
  try:
    from ..evaluation.agent_evaluator import EvaluationGenerator
    from ..evaluation.response_evaluator import ResponseEvaluator
    from ..evaluation.trajectory_evaluator import TrajectoryEvaluator
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

  """Returns a summary of eval runs."""
  for eval_set_file, evals_to_run in eval_set_to_evals.items():
    with open(eval_set_file, "r", encoding="utf-8") as file:
      eval_items = json.load(file)  # Load JSON into a list

    assert eval_items, f"No eval data found in eval set file: {eval_set_file}"

    for eval_item in eval_items:
      eval_name = eval_item["name"]
      eval_data = eval_item["data"]
      initial_session = eval_item.get("initial_session", {})

      if evals_to_run and eval_name not in evals_to_run:
        continue

      try:
        print(f"Running Eval: {eval_set_file}:{eval_name}")
        session_id = f"{EVAL_SESSION_ID_PREFIX}{str(uuid.uuid4())}"

        scrape_result = EvaluationGenerator._process_query_with_root_agent(
            data=eval_data,
            root_agent=root_agent,
            reset_func=reset_func,
            initial_session=initial_session,
            session_id=session_id,
            session_service=session_service,
            artifact_service=artifact_service,
        )

        eval_metric_results = []
        for eval_metric in eval_metrics:
          eval_metric_result = None
          if eval_metric.metric_name == TOOL_TRAJECTORY_SCORE_KEY:
            score = TrajectoryEvaluator.evaluate(
                [scrape_result], print_detailed_results=print_detailed_results
            )
            eval_metric_result = _get_eval_metric_result(eval_metric, score)
          elif eval_metric.metric_name == RESPONSE_MATCH_SCORE_KEY:
            score = ResponseEvaluator.evaluate(
                [scrape_result],
                [RESPONSE_MATCH_SCORE_KEY],
                print_detailed_results=print_detailed_results,
            )
            eval_metric_result = _get_eval_metric_result(
                eval_metric, score["rouge_1/mean"].item()
            )
          elif eval_metric.metric_name == RESPONSE_EVALUATION_SCORE_KEY:
            score = ResponseEvaluator.evaluate(
                [scrape_result],
                [RESPONSE_EVALUATION_SCORE_KEY],
                print_detailed_results=print_detailed_results,
            )
            eval_metric_result = _get_eval_metric_result(
                eval_metric, score["coherence/mean"].item()
            )
          else:
            logger.warning("`%s` is not supported.", eval_metric.metric_name)
            eval_metric_results.append((
                eval_metric,
                EvalMetricResult(eval_status=EvalStatus.NOT_EVALUATED),
            ))

          eval_metric_results.append((
              eval_metric,
              eval_metric_result,
          ))
          _print_eval_metric_result(eval_metric, eval_metric_result)

        final_eval_status = EvalStatus.NOT_EVALUATED

        # Go over the all the eval statuses and mark the final eval status as
        # passed if all of them pass, otherwise mark the final eval status to
        # failed.
        for eval_metric_result in eval_metric_results:
          eval_status = eval_metric_result[1].eval_status
          if eval_status == EvalStatus.PASSED:
            final_eval_status = EvalStatus.PASSED
          elif eval_status == EvalStatus.NOT_EVALUATED:
            continue
          elif eval_status == EvalStatus.FAILED:
            final_eval_status = EvalStatus.FAILED
            break
          else:
            raise ValueError("Unknown eval status.")

        yield EvalResult(
            eval_set_file=eval_set_file,
            eval_id=eval_name,
            final_eval_status=final_eval_status,
            eval_metric_results=eval_metric_results,
            session_id=session_id,
        )

        if final_eval_status == EvalStatus.PASSED:
          result = "✅ Passed"
        else:
          result = "❌ Failed"

        print(f"Result: {result}\n")

      except Exception as e:
        print(f"Error: {e}")
        logger.info("Error: %s", str(traceback.format_exc()))


def _get_eval_metric_result(eval_metric, score):
  eval_status = (
      EvalStatus.PASSED if score >= eval_metric.threshold else EvalStatus.FAILED
  )
  return EvalMetricResult(score=score, eval_status=eval_status)


def _print_eval_metric_result(eval_metric, eval_metric_result):
  print(
      f"Metric: {eval_metric.metric_name}\tStatus:"
      f" {eval_metric_result.eval_status}\tScore:"
      f" {eval_metric_result.score}\tThreshold: {eval_metric.threshold}"
  )
