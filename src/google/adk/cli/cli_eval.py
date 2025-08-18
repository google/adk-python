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

import importlib.util
import inspect
import json
import logging
import os
import sys
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import uuid

from typing_extensions import deprecated

from ..agents.llm_agent import Agent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..evaluation.base_eval_service import BaseEvalService
from ..evaluation.base_eval_service import EvaluateConfig
from ..evaluation.base_eval_service import EvaluateRequest
from ..evaluation.base_eval_service import InferenceConfig
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.base_eval_service import InferenceResult
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..evaluation.eval_case import EvalCase
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_metrics import JudgeModelOptions
from ..evaluation.eval_result import EvalCaseResult
from ..evaluation.evaluator import EvalStatus
from ..evaluation.evaluator import Evaluator
from ..sessions.base_session_service import BaseSessionService
from ..utils.context_utils import Aclosing

logger = logging.getLogger("google_adk." + __name__)


TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"
SAFETY_V1_KEY = "safety_v1"
FINAL_RESPONSE_MATCH_V2 = "final_response_match_v2"
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
    evals_to_run_info: list[str],
) -> dict[str, list[str]]:
  """Returns a dictionary of eval set info to evals that should be run.

  Args:
    evals_to_run_info: While the structure is quite simple, a list of string,
      each string actually is formatted with the following convention:
      <eval_set_file_path | eval_set_id>:[comma separated eval case ids]
  """
  eval_set_to_evals = {}
  for input_eval_set in evals_to_run_info:
    evals = []
    if ":" not in input_eval_set:
      # We don't have any eval cases specified. This would be the case where the
      # the user wants to run all eval cases in the eval set.
      eval_set = input_eval_set
    else:
      # There are eval cases that we need to parse. The user wants to run
      # specific eval cases from the eval set.
      eval_set = input_eval_set.split(":")[0]
      evals = input_eval_set.split(":")[1].split(",")
      evals = [s for s in evals if s.strip()]

    if eval_set not in eval_set_to_evals:
      eval_set_to_evals[eval_set] = []

    eval_set_to_evals[eval_set].extend(evals)

  return eval_set_to_evals


async def _collect_inferences(
    inference_requests: list[InferenceRequest],
    eval_service: BaseEvalService,
) -> list[InferenceResult]:
  """Simple utility methods to collect inferences from an eval service.

  The method is intentionally kept private to prevent general usage.
  """
  inference_results = []
  for inference_request in inference_requests:
    async with Aclosing(
        eval_service.perform_inference(inference_request=inference_request)
    ) as agen:
      async for inference_result in agen:
        inference_results.append(inference_result)
  return inference_results


async def _collect_eval_results(
    inference_results: list[InferenceResult],
    eval_service: BaseEvalService,
    eval_metrics: list[EvalMetric],
) -> list[EvalCaseResult]:
  """Simple utility methods to collect eval results from an eval service.

  The method is intentionally kept private to prevent general usage.
  """
  eval_results = []
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=eval_metrics),
  )
  async with Aclosing(
      eval_service.evaluate(evaluate_request=evaluate_request)
  ) as agen:
    async for eval_result in agen:
      eval_results.append(eval_result)

  return eval_results


@deprecated(
    "This method is deprecated and will be removed in fututre release. Please"
    " use LocalEvalService to define your custom evals."
)
async def run_evals(
    eval_cases_by_eval_set_id: dict[str, list[EvalCase]],
    root_agent: Agent,
    reset_func: Optional[Any],
    eval_metrics: list[EvalMetric],
    session_service: Optional[BaseSessionService] = None,
    artifact_service: Optional[BaseArtifactService] = None,
) -> AsyncGenerator[EvalCaseResult, None]:
  """Returns a stream of EvalCaseResult for each eval case that was evaluated.

  Args:
    eval_cases_by_eval_set_id: Eval cases categorized by eval set id to which
      they belong.
    root_agent: Agent to use for inferencing.
    reset_func: If present, this will be called before invoking the agent before
      every inferencing step.
    eval_metrics: A list of metrics that should be used during evaluation.
    session_service: The session service to use during inferencing.
    artifact_service: The artifact service to use during inferencing.
  """
  try:
    from ..evaluation.evaluation_generator import EvaluationGenerator
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

  for eval_set_id, eval_cases in eval_cases_by_eval_set_id.items():
    for eval_case in eval_cases:
      eval_name = eval_case.eval_id
      initial_session = eval_case.session_input
      user_id = initial_session.user_id if initial_session else "test_user_id"

      try:
        print(f"Running Eval: {eval_set_id}:{eval_name}")
        session_id = f"{EVAL_SESSION_ID_PREFIX}{str(uuid.uuid4())}"

        inference_result = (
            await EvaluationGenerator._generate_inferences_from_root_agent(
                invocations=eval_case.conversation,
                root_agent=root_agent,
                reset_func=reset_func,
                initial_session=initial_session,
                session_id=session_id,
                session_service=session_service,
                artifact_service=artifact_service,
            )
        )

        # Initialize the per-invocation metric results to an empty list.
        # We will fill this as we evaluate each metric.
        eval_metric_result_per_invocation = []
        for actual, expected in zip(inference_result, eval_case.conversation):
          eval_metric_result_per_invocation.append(
              EvalMetricResultPerInvocation(
                  actual_invocation=actual,
                  expected_invocation=expected,
                  eval_metric_results=[],
              )
          )

        overall_eval_metric_results = []

        for eval_metric in eval_metrics:
          metric_evaluator = _get_evaluator(eval_metric)

          if inspect.iscoroutinefunction(metric_evaluator.evaluate_invocations):
            evaluation_result = await metric_evaluator.evaluate_invocations(
                actual_invocations=inference_result,
                expected_invocations=eval_case.conversation,
            )
          else:
            evaluation_result = metric_evaluator.evaluate_invocations(
                actual_invocations=inference_result,
                expected_invocations=eval_case.conversation,
            )

          overall_eval_metric_results.append(
              EvalMetricResult(
                  metric_name=eval_metric.metric_name,
                  threshold=eval_metric.threshold,
                  score=evaluation_result.overall_score,
                  eval_status=evaluation_result.overall_eval_status,
              )
          )
          for index, per_invocation_result in enumerate(
              evaluation_result.per_invocation_results
          ):
            eval_metric_result_per_invocation[index].eval_metric_results.append(
                EvalMetricResult(
                    metric_name=eval_metric.metric_name,
                    threshold=eval_metric.threshold,
                    score=per_invocation_result.score,
                    eval_status=per_invocation_result.eval_status,
                )
            )

        final_eval_status = EvalStatus.NOT_EVALUATED
        # Go over the all the eval statuses and mark the final eval status as
        # passed if all of them pass, otherwise mark the final eval status to
        # failed.
        for overall_eval_metric_result in overall_eval_metric_results:
          overall_eval_status = overall_eval_metric_result.eval_status
          if overall_eval_status == EvalStatus.PASSED:
            final_eval_status = EvalStatus.PASSED
          elif overall_eval_status == EvalStatus.NOT_EVALUATED:
            continue
          elif overall_eval_status == EvalStatus.FAILED:
            final_eval_status = EvalStatus.FAILED
            break
          else:
            raise ValueError("Unknown eval status.")

        yield EvalCaseResult(
            eval_set_file=eval_set_id,
            eval_set_id=eval_set_id,
            eval_id=eval_name,
            final_eval_status=final_eval_status,
            eval_metric_results=[],
            overall_eval_metric_results=overall_eval_metric_results,
            eval_metric_result_per_invocation=eval_metric_result_per_invocation,
            session_id=session_id,
            user_id=user_id,
        )

        if final_eval_status == EvalStatus.PASSED:
          result = "✅ Passed"
        else:
          result = "❌ Failed"

        print(f"Result: {result}\n")
      except ModuleNotFoundError as e:
        raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e
      except Exception:
        # Catching the general exception, so that we don't block other eval
        # cases.
        logger.exception(f"Eval failed for `{eval_set_id}:{eval_name}`")


def _get_evaluator(eval_metric: EvalMetric) -> Evaluator:
  try:
    from ..evaluation.final_response_match_v2 import FinalResponseMatchV2Evaluator
    from ..evaluation.response_evaluator import ResponseEvaluator
    from ..evaluation.safety_evaluator import SafetyEvaluatorV1
    from ..evaluation.trajectory_evaluator import TrajectoryEvaluator
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e
  if eval_metric.metric_name == TOOL_TRAJECTORY_SCORE_KEY:
    return TrajectoryEvaluator(threshold=eval_metric.threshold)
  elif (
      eval_metric.metric_name == RESPONSE_MATCH_SCORE_KEY
      or eval_metric.metric_name == RESPONSE_EVALUATION_SCORE_KEY
  ):
    return ResponseEvaluator(
        threshold=eval_metric.threshold, metric_name=eval_metric.metric_name
    )
  elif eval_metric.metric_name == SAFETY_V1_KEY:
    return SafetyEvaluatorV1(eval_metric)
  elif eval_metric.metric_name == FINAL_RESPONSE_MATCH_V2:
    eval_metric.judge_model_options = JudgeModelOptions()
    return FinalResponseMatchV2Evaluator(eval_metric)

  raise ValueError(f"Unsupported eval metric: {eval_metric}")
