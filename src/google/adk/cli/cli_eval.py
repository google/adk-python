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

import importlib.util
import logging
import os
import sys
import textwrap
from types import ModuleType
from typing import Any
from typing import cast
from typing import Optional

import click
from google.genai import types as genai_types

from ..agents.base_agent import BaseAgent
from ..apps.app import App
from ..evaluation.base_eval_service import BaseEvalService
from ..evaluation.base_eval_service import EvaluateConfig
from ..evaluation.base_eval_service import EvaluateRequest
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.base_eval_service import InferenceResult
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..evaluation.eval_case import get_all_tool_calls
from ..evaluation.eval_case import IntermediateDataType
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_metrics import PrebuiltMetrics
from ..evaluation.eval_metrics import RubricsBasedCriterion
from ..evaluation.eval_metrics import TokenUsageDetails
from ..evaluation.eval_result import EvalCaseResult
from ..evaluation.eval_sets_manager import EvalSetsManager
from ..evaluation.evaluator import EvalStatus
from ..utils.context_utils import Aclosing

logger = logging.getLogger("google_adk." + __name__)


TOOL_TRAJECTORY_SCORE_KEY = "tool_trajectory_avg_score"
RESPONSE_MATCH_SCORE_KEY = "response_match_score"
SAFETY_V1_KEY = "safety_v1"
FINAL_RESPONSE_MATCH_V2 = "final_response_match_v2"
# This evaluation is not very stable.
# This is always optional unless explicitly specified.
RESPONSE_EVALUATION_SCORE_KEY = "response_evaluation_score"

# Must stay in sync with google.adk.evaluation.local_eval_service. The prefix
# is only allowed to contain lowercase letters, digits and hyphens so the
# generated eval session IDs satisfy the custom-session-ID constraints of
# remote backends such as Vertex AI Agent Engine.
EVAL_SESSION_ID_PREFIX = "adk-eval-session-"
# Prefix used through ADK v2.10. Agent Engine rejects it (underscores), but
# sessions created with it may still exist in local/database session stores.
# Safe to remove once those are no longer expected.
_LEGACY_EVAL_SESSION_ID_PREFIX = "___eval___session___"
DEFAULT_CRITERIA = {
    TOOL_TRAJECTORY_SCORE_KEY: 1.0,  # 1-point scale; 1.0 is perfect.
    RESPONSE_MATCH_SCORE_KEY: 0.8,
}


def _import_from_path(module_name: str, file_path: str) -> ModuleType:
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot import module {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def _get_agent_module(agent_module_file_path: str) -> ModuleType:
  file_path = os.path.join(agent_module_file_path, "__init__.py")
  module_name = "agent"
  return _import_from_path(module_name, file_path)


async def get_app_or_root_agent(
    agent_module_file_path: str,
) -> tuple[Optional[App], BaseAgent]:
  """Returns the (app, root_agent) pair for the given agent module.

  If the module exposes an `App` instance via `app`, that App and its
  `root_agent` are returned. Otherwise `app` is None and the root agent is
  resolved the same way as `get_root_agent`. This lets eval flows participate
  in the App's plugin / cache / resumability lifecycle when one is defined,
  while preserving the bare-`root_agent` path for projects that don't use App.
  """
  agent_module = _get_agent_module(agent_module_file_path)
  agent_module_with_agent = getattr(agent_module, "agent", agent_module)
  app = getattr(agent_module_with_agent, "app", None)
  if isinstance(app, App):
    return app, cast(BaseAgent, app.root_agent)
  if hasattr(agent_module_with_agent, "root_agent"):
    return None, cast(BaseAgent, agent_module_with_agent.root_agent)
  elif hasattr(agent_module_with_agent, "get_agent_async"):
    root_agent, _ = await agent_module_with_agent.get_agent_async()
    return None, cast(BaseAgent, root_agent)
  raise ValueError(
      "Agent module should have either `root_agent` or `get_agent_async`."
  )


async def get_root_agent(agent_module_file_path: str) -> BaseAgent:
  """Returns root agent given the agent module.

  Kept for backward compatibility. New callers should prefer
  `get_app_or_root_agent`, which also surfaces the wrapping `App` (if any)
  so plugins, context-cache, and resumability configs are honored.
  """
  _, root_agent = await get_app_or_root_agent(agent_module_file_path)
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
  eval_set_to_evals: dict[str, list[str]] = {}
  for input_eval_set in evals_to_run_info:
    evals = []
    drive_letter = input_eval_set[:1]
    has_windows_drive_prefix = (
        len(input_eval_set) >= 3
        and drive_letter.isascii()
        and drive_letter.isalpha()
        and input_eval_set[1] == ":"
        and input_eval_set[2] in ("\\", "/")
    )
    selector_separator_index = input_eval_set.find(
        ":", 3 if has_windows_drive_prefix else 0
    )
    if selector_separator_index == -1:
      # We don't have any eval cases specified. This would be the case where the
      # the user wants to run all eval cases in the eval set.
      eval_set = input_eval_set
    else:
      # There are eval cases that we need to parse. The user wants to run
      # specific eval cases from the eval set.
      eval_set = input_eval_set[:selector_separator_index]
      selector_list = input_eval_set[selector_separator_index + 1 :]
      evals = selector_list.split(":")[0].split(",")
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


def _format_tool_call(tool_call: genai_types.FunctionCall) -> str:
  """Formats a tool call as `name(` then one `arg=value` per line.

  Only the name and arguments matter when reading a trajectory; the call id and
  streaming fields (`partial_args`, `will_continue`) are noise in a table. One
  argument per line keeps a narrow table column from splitting a value in half.
  """
  args = [f"{k}={v!r}" for k, v in (tool_call.args or {}).items()]
  if not args:
    return f"{tool_call.name}()"
  return f"{tool_call.name}(\n  " + ",\n  ".join(args) + ")"


def _convert_tool_calls_to_text(
    intermediate_data: Optional[IntermediateDataType],
) -> str:
  tool_calls = get_all_tool_calls(intermediate_data)
  return "\n".join([_format_tool_call(t) for t in tool_calls])


def _format_token_count(value: Optional[float]) -> str:
  """Formats a token count, showing unavailable counts as n/a rather than 0."""
  return "n/a" if value is None else f"{value:g}"


# The token counts to print, and how deep to indent each one. Indentation is
# containment: every row is a part of the nearest row above it that is indented
# less, so `cached` reads as a portion of `prompt` rather than an addition to
# it. See `TokenUsageDetails` for the counts themselves.
_TOKEN_BREAKDOWN_ROWS = (
    ("total_tokens", 1),
    ("input_tokens", 2),
    ("prompt_tokens", 3),
    ("cached_tokens", 4),
    ("tool_use_tokens", 3),
    ("output_tokens", 2),
    ("candidates_tokens", 3),
    ("reasoning_tokens", 3),
)

# Wide enough for the longest indented label, so the counts line up in a column.
_TOKEN_BREAKDOWN_LABEL_WIDTH = 20


def _echo_token_usage_details(details: TokenUsageDetails) -> None:
  """Prints the per-type token counts, indented to show what contains what."""
  click.echo("Token breakdown:")
  for field_name, depth in _TOKEN_BREAKDOWN_ROWS:
    # The heading already says these are tokens, so the shared suffix is
    # dropped from the label rather than repeated on all eight rows.
    name = field_name.removesuffix("_tokens").replace("_", " ")
    label = f"{'  ' * depth}{name}:"
    count = _format_token_count(getattr(details, field_name))
    click.echo(f"{label:<{_TOKEN_BREAKDOWN_LABEL_WIDTH}}{count}")


def _format_metric_cell(metric_result: EvalMetricResult) -> str:
  """Formats one metric's result for a cell of the invocation details table.

  Informational metrics never pass or fail, so their cell is just the value
  (with its unit where one applies) instead of repeating
  `Status: INFORMATIONAL` on every row. Metrics that do pass or fail read
  `PASSED (1.0)`.
  """
  score = metric_result.score
  if metric_result.eval_status != EvalStatus.INFORMATIONAL:
    return f"{metric_result.eval_status.name} ({score})"
  if score is None:
    return "n/a"
  if metric_result.metric_name == PrebuiltMetrics.INVOCATION_DURATION_V1.value:
    return f"{score:.2f}s"
  token_details = (
      metric_result.details.token_usage_details
      if metric_result.details
      else None
  )
  if token_details:
    # The full breakdown is printed with the overall metrics; per invocation,
    # the input/output split is enough to see where the tokens went.
    return (
        f"{score:g}"
        f" (in {_format_token_count(token_details.input_tokens)},"
        f" out {_format_token_count(token_details.output_tokens)})"
    )
  return f"{score:g}"


def pretty_print_eval_result(eval_result: EvalCaseResult) -> None:
  """Pretty prints eval result."""
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
    if metric_result.details and metric_result.details.token_usage_details:
      _echo_token_usage_details(metric_result.details.token_usage_details)
    if metric_result.details and metric_result.details.rubric_scores:
      click.echo("Rubric Scores:")
      rubrics = (
          metric_result.criterion.rubrics
          if isinstance(metric_result.criterion, RubricsBasedCriterion)
          else None
      ) or []
      rubrics_by_id = {
          r.rubric_id: r.rubric_content.text_property for r in rubrics
      }
      for rubric_score in metric_result.details.rubric_scores:
        rubric_text = rubrics_by_id.get(rubric_score.rubric_id)
        if not rubric_text:
          rubric_text = rubric_score.rubric_id
        click.echo(
            f"Rubric: {rubric_text}, "
            f"Score: {rubric_score.score}, "
            f"Reasoning: {rubric_score.rationale}"
        )

  invocation_results = eval_result.eval_metric_result_per_invocation
  if not invocation_results:
    return
  click.echo(
      "---------------------------------------------------------------------"
  )
  click.echo("Invocation Details:")
  for index, per_invocation_result in enumerate(invocation_results, start=1):
    click.echo(f"\nInvocation {index} of {len(invocation_results)}")
    for label, value in _invocation_fields(per_invocation_result):
      _echo_field(label, value)
  click.echo("\n")  # A blank line before the next eval case.


def _invocation_fields(
    per_invocation_result: EvalMetricResultPerInvocation,
) -> list[tuple[str, str]]:
  """Returns the (label, value) pairs printed for one invocation.

  Fields with no value (e.g. no expected invocation) are left out.
  """
  actual = per_invocation_result.actual_invocation
  expected = per_invocation_result.expected_invocation
  fields = [
      ("prompt", _convert_content_to_text(actual.user_content)),
      (
          "expected_response",
          _convert_content_to_text(expected.final_response)
          if expected
          else None,
      ),
      ("actual_response", _convert_content_to_text(actual.final_response)),
      (
          "expected_tool_calls",
          _convert_tool_calls_to_text(expected.intermediate_data)
          if expected
          else None,
      ),
      (
          "actual_tool_calls",
          _convert_tool_calls_to_text(actual.intermediate_data),
      ),
  ]
  for metric_result in per_invocation_result.eval_metric_results:
    fields.append(
        (metric_result.metric_name, _format_metric_cell(metric_result))
    )
    if metric_result.details and metric_result.details.rubric_scores:
      rubrics = (
          metric_result.criterion.rubrics
          if isinstance(metric_result.criterion, RubricsBasedCriterion)
          else None
      ) or []
      rubrics_by_id = {
          r.rubric_id: r.rubric_content.text_property for r in rubrics
      }
      for rubric_score in metric_result.details.rubric_scores:
        rubric = rubrics_by_id.get(rubric_score.rubric_id)
        if not rubric:
          rubric = rubric_score.rubric_id
        fields.append((
            "rubric",
            (
                f"{rubric}\nScore: {rubric_score.score}, "
                f"Reasoning: {rubric_score.rationale}"
            ),
        ))
  return [(label, value) for label, value in fields if value is not None]


# Column where values start in the invocation details: just past the longest
# built-in label (`  tool_trajectory_avg_score: `). A longer label, e.g. a custom
# metric name, pushes its own value further right instead of being truncated.
_FIELD_VALUE_COLUMN = 30
# Width at which long values (e.g. responses) wrap.
_FIELD_VALUE_WIDTH = 80


def _echo_field(label: str, value: str) -> None:
  """Prints one `label: value` line, indenting continuation lines under it.

  Each line already in the value (e.g. one tool-call argument per line) is
  wrapped on its own, so an existing layout survives.
  """
  lines = [
      wrapped
      for line in value.splitlines() or ["(none)"]
      for wrapped in textwrap.wrap(line, _FIELD_VALUE_WIDTH) or [""]
  ]
  prefix = f"  {label}: ".ljust(_FIELD_VALUE_COLUMN)
  click.echo(f"{prefix}{lines[0]}")
  for line in lines[1:]:
    click.echo(f"{' ' * len(prefix)}{line}")


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
