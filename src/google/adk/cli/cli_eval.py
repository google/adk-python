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
import logging
import os
import sys
from typing import Any
from typing import Optional

import click
from google.genai import types as genai_types

from ..agents.llm_agent import Agent
from ..evaluation.base_eval_service import BaseEvalService
from ..evaluation.base_eval_service import EvaluateConfig
from ..evaluation.base_eval_service import EvaluateRequest
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.base_eval_service import InferenceResult
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..evaluation.eval_case import get_all_tool_calls
from ..evaluation.eval_case import IntermediateDataType
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_result import EvalCaseResult
from ..evaluation.eval_sets_manager import EvalSetsManager
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


def _convert_content_to_text(
    content: Optional[genai_types.Content],
) -> str:
  if content and content.parts:
    return "\n".join([p.text for p in content.parts if p.text])
  return ""


def _convert_tool_calls_to_text(
    intermediate_data: Optional[IntermediateDataType],
) -> str:
  tool_calls = get_all_tool_calls(intermediate_data)
  return "\n".join([str(t) for t in tool_calls])


def pretty_print_eval_result(eval_result: EvalCaseResult):
  """Pretty prints eval result."""
  try:
    import pandas as pd
    from tabulate import tabulate
  except ModuleNotFoundError as e:
    raise ModuleNotFoundError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

  click.echo(f"Eval Set Id: {eval_result.eval_set_id}")
  click.echo(f"Eval Id: {eval_result.eval_id}")
  click.echo(f"Overall Eval Status: {eval_result.final_eval_status.name}")

  for metric_result in eval_result.overall_eval_metric_results:
    click.echo(
        "---------------------------------------------------------------------"
    )
    click.echo(
        f"Metric: {metric_result.metric_name}, "
        f"Status: {metric_result.eval_status.name}, "
        f"Score: {metric_result.score}, "
        f"Threshold: {metric_result.threshold}"
    )
    if metric_result.details and metric_result.details.rubric_scores:
      click.echo("Rubric Scores:")
      rubrics_by_id = {
          r["rubric_id"]: r["rubric_content"]["text_property"]
          for r in metric_result.criterion.rubrics
      }
      for rubric_score in metric_result.details.rubric_scores:
        rubric = rubrics_by_id.get(rubric_score.rubric_id)
        click.echo(
            f"Rubric: {rubric}, "
            f"Score: {rubric_score.score}, "
            f"Reasoning: {rubric_score.rationale}"
        )

  data = []
  for per_invocation_result in eval_result.eval_metric_result_per_invocation:
    actual_invocation = per_invocation_result.actual_invocation
    expected_invocation = per_invocation_result.expected_invocation
    row_data = {
        "prompt": _convert_content_to_text(actual_invocation.user_content),
        "expected_response": (
            _convert_content_to_text(expected_invocation.final_response)
            if expected_invocation
            else None
        ),
        "actual_response": _convert_content_to_text(
            actual_invocation.final_response
        ),
        "expected_tool_calls": (
            _convert_tool_calls_to_text(expected_invocation.intermediate_data)
            if expected_invocation
            else None
        ),
        "actual_tool_calls": _convert_tool_calls_to_text(
            actual_invocation.intermediate_data
        ),
    }
    for metric_result in per_invocation_result.eval_metric_results:
      row_data[metric_result.metric_name] = (
          f"Status: {metric_result.eval_status.name}, "
          f"Score: {metric_result.score}"
      )
      if metric_result.details and metric_result.details.rubric_scores:
        rubrics_by_id = {
            r["rubric_id"]: r["rubric_content"]["text_property"]
            for r in metric_result.criterion.rubrics
        }
        for rubric_score in metric_result.details.rubric_scores:
          rubric = rubrics_by_id.get(rubric_score.rubric_id)
          row_data[f"Rubric: {rubric}"] = (
              f"Reasoning: {rubric_score.rationale}, "
              f"Score: {rubric_score.score}"
          )
    data.append(row_data)
  if data:
    click.echo(
        "---------------------------------------------------------------------"
    )
    click.echo("Invocation Details:")
    df = pd.DataFrame(data)

    # Identify columns where ALL values are exactly None
    columns_to_keep = []
    for col in df.columns:
      # Check if all elements in the column are NOT None
      if not df[col].apply(lambda x: x is None).all():
        columns_to_keep.append(col)

    # Select only the columns to keep
    df_result = df[columns_to_keep]

    for col in df_result.columns:
      if df_result[col].dtype == "object":
        df_result[col] = df_result[col].str.wrap(40)

    click.echo(
        tabulate(df_result, headers="keys", tablefmt="grid", maxcolwidths=25)
    )
    click.echo("\n\n")  # Few empty lines for visual clarity


def get_eval_sets_manager(
    eval_storage_uri: Optional[str], agents_dir: str
) -> EvalSetsManager:
  """Returns an instance of EvalSetsManager."""
  try:
    from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
    from .utils import evals
  except ModuleNotFoundError as mnf:
    raise click.ClickException(MISSING_EVAL_DEPENDENCIES_MESSAGE) from mnf

  if eval_storage_uri:
    gcs_eval_managers = evals.create_gcs_eval_managers_from_uri(
        eval_storage_uri
    )
    return gcs_eval_managers.eval_sets_manager
  else:
    return LocalEvalSetsManager(agents_dir=agents_dir)
