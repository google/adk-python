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

import logging
import time
from typing import TYPE_CHECKING

from google.adk import version
from google.adk.telemetry import _adk_attributes
from google.adk.telemetry import _hallucination
from google.adk.telemetry import tracing
from google.adk.telemetry._token_usage import CACHE_READ_INPUT_TOKENS_MEANING
from google.adk.telemetry._token_usage import INPUT_TOKENS_MEANING
from google.adk.telemetry._token_usage import OUTPUT_TOKENS_MEANING
from google.adk.telemetry._token_usage import REASONING_OUTPUT_TOKENS_MEANING
from google.adk.telemetry._token_usage import TokenUsage
from google.adk.telemetry._token_usage import TOOL_INPUT_TOKENS_MEANING
from google.adk.telemetry._token_usage import TOTAL_TOKENS_MEANING
from opentelemetry import metrics
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes

if TYPE_CHECKING:
  from google.adk.models.llm_request import LlmRequest
  from google.adk.models.llm_response import LlmResponse
  from opentelemetry.trace import Span
  from opentelemetry.util.types import AttributeValue

  from .tracing import GenerateContentSpan

logger = logging.getLogger("google_adk." + __name__)

GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_TOOL_VERSION = "gen_ai.tool.version"

# What one datapoint covers, spelled into the description so a reader of the
# metric catalog can tell the two families apart.
_PER_INVOCATION = "one agent invocation"
_PER_WORKFLOW = "one workflow invocation, across every agent that ran in it"

# What the turn was entered at, keying the per-workflow metrics. Named for the
# common case, but a workflow-rooted runner puts the workflow's name here, not
# an agent's. Experimental prefix because upstream has three competing unmerged
# drafts for the same concept.
ADK_ROOT_AGENT_NAME = "adk.experimental.root_agent.name"

meter = metrics.get_meter(
    name="gcp.vertex.agent",
    version=version.__version__,
)


_agent_invocation_duration = meter.create_histogram(
    "gen_ai.invoke_agent.duration",
    unit="s",
    description="Duration of agent invocations.",
    explicit_bucket_boundaries_advisory=[
        0.1,
        0.2,
        0.4,
        0.8,
        1.6,
        3.2,
        6.4,
        12.8,
        25.6,
        51.2,
        102.4,
        204.8,
        409.6,
    ],
)
_workflow_invocation_duration = meter.create_histogram(
    "gen_ai.invoke_workflow.duration",
    unit="s",
    description="Duration of workflow invocations.",
)
_tool_execution_duration = meter.create_histogram(
    "gen_ai.execute_tool.duration",
    unit="s",
    description="Duration of tool executions.",
    explicit_bucket_boundaries_advisory=[
        0.01,
        0.02,
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.28,
        2.56,
        5.12,
        10.24,
        20.48,
        40.96,
        81.92,
    ],
)
_client_operation_duration = (
    gen_ai_metrics.create_gen_ai_client_operation_duration(meter)
)
_client_token_usage = gen_ai_metrics.create_gen_ai_client_token_usage(meter)

# Bounds are upper inclusive, so the leading 0 buckets exact zeros on their own.
# The tail is sized for a workflow rather than a single agent: a coding agent
# routinely spends hundreds of model and tool calls finishing one task, and a
# workflow sums every agent that ran in it.
_CALL_COUNT_BUCKET_BOUNDS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    12,
    16,
    24,
    32,
    64,
    128,
    256,
    512,
]

_invoke_agent_inference_calls = meter.create_histogram(
    "gen_ai.invoke_agent.inference_calls",
    unit="1",
    description="Number of inference (model) calls per agent invocation.",
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)
_invoke_agent_tool_calls = meter.create_histogram(
    "gen_ai.invoke_agent.tool_calls",
    unit="1",
    description="Number of tool calls per agent invocation.",
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)
_invoke_agent_skill_loads = meter.create_histogram(
    "adk.experimental.invoke_agent.skill.loads",
    unit="1",
    description=f"Number of skill loads over {_PER_INVOCATION}.",
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)

# Bounds are upper inclusive, so the leading 0 buckets exact zeros on their own.
_INPUT_TOKEN_BUCKET_BOUNDS = [
    0,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
_OUTPUT_TOKEN_BUCKET_BOUNDS = [
    0,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]


def _create_token_histogram(
    name: str,
    description: str,
    bounds: list[int],
) -> metrics.Histogram:
  """Creates a token histogram, which is any histogram counted in tokens.

  Args:
    name: The metric name.
    description: What the metric measures. Built from the `_token_usage`
      definitions, which own the meaning both the descriptions and the
      arithmetic follow.
    bounds: The advisory bucket boundaries.

  Returns:
    The histogram.
  """
  return meter.create_histogram(
      name,
      unit="{token}",
      description=description,
      explicit_bucket_boundaries_advisory=bounds,
  )


_invoke_agent_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.input_tokens",
    f"{INPUT_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_agent_output_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.output_tokens",
    f"{OUTPUT_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _OUTPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_agent_total_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.total_tokens",
    f"{TOTAL_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_agent_cache_read_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.cache_read.input_tokens",
    f"{CACHE_READ_INPUT_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_agent_reasoning_output_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.reasoning.output_tokens",
    f"{REASONING_OUTPUT_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _OUTPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_agent_tool_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_agent.tool.input_tokens",
    f"{TOOL_INPUT_TOKENS_MEANING} Summed over {_PER_INVOCATION}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)

# ---- Workflow-grain metrics: one datapoint per `invoke_workflow` ----
_invoke_workflow_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.input_tokens",
    f"{INPUT_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_output_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.output_tokens",
    f"{OUTPUT_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _OUTPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_total_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.total_tokens",
    f"{TOTAL_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_cache_read_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.cache_read.input_tokens",
    f"{CACHE_READ_INPUT_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_reasoning_output_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.reasoning.output_tokens",
    f"{REASONING_OUTPUT_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _OUTPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_tool_input_tokens = _create_token_histogram(
    "adk.experimental.invoke_workflow.tool.input_tokens",
    f"{TOOL_INPUT_TOKENS_MEANING} Summed over {_PER_WORKFLOW}.",
    _INPUT_TOKEN_BUCKET_BOUNDS,
)
_invoke_workflow_inference_calls = meter.create_histogram(
    "adk.experimental.invoke_workflow.inference_calls",
    unit="1",
    description=f"Number of inference (model) calls over {_PER_WORKFLOW}.",
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)
_invoke_workflow_tool_calls = meter.create_histogram(
    "adk.experimental.invoke_workflow.tool_calls",
    unit="1",
    description=(
        f"Number of tool calls over {_PER_WORKFLOW}. Includes the"
        " `transfer_to_agent` calls that route between them."
    ),
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)
_invoke_workflow_skill_loads = meter.create_histogram(
    "adk.experimental.invoke_workflow.skill.loads",
    unit="1",
    description=f"Number of skill loads over {_PER_WORKFLOW}.",
    explicit_bucket_boundaries_advisory=_CALL_COUNT_BUCKET_BOUNDS,
)
_skill_script_executions = meter.create_counter(
    "adk.experimental.skill.script.executions",
    unit="1",
    description="Number of skill script executions.",
)
_skill_loads = meter.create_counter(
    "adk.experimental.skill.loads",
    unit="1",
    description=(
        "Number of times a skill was loaded. Counts the attempt,"
        " so a load that resolved no skill is counted too, under its"
        " `error.type`."
    ),
)


def record_agent_invocation_duration(
    agent_name: str,
    elapsed_s: float,
    error: Exception | None = None,
) -> None:
  """Records the duration of the agent invocation."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = tracing.resolve_error_type(error)
  _agent_invocation_duration.record(elapsed_s, attributes=attrs)


def record_workflow_invocation_duration(
    *,
    workflow_name: str,
    elapsed_s: float,
    nested: bool,
    error: BaseException | None = None,
) -> None:
  """Records the duration of a workflow invocation."""
  attrs: dict[str, AttributeValue] = {
      gen_ai_attributes.GEN_AI_OPERATION_NAME: "invoke_workflow",
  }
  # Root workflow omits the attribute entirely; only nested ones emit it.
  if nested:
    attrs["gen_ai.workflow.nested"] = True
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = tracing.resolve_error_type(error)
  if workflow_name:
    attrs["gen_ai.workflow.name"] = workflow_name
  _workflow_invocation_duration.record(elapsed_s, attributes=attrs)


def record_invoke_agent_inference_calls(agent_name: str, count: int) -> None:
  """Records the number of inference (model) calls in an agent invocation."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _invoke_agent_inference_calls.record(count, attributes=attrs)


def record_invoke_agent_tool_calls(agent_name: str, count: int) -> None:
  """Records the number of tool calls in an agent invocation."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _invoke_agent_tool_calls.record(count, attributes=attrs)


def record_invoke_agent_token_usage(
    agent_name: str,
    totals: TokenUsage,
) -> None:
  """Records the token spend accumulated over one agent invocation.

  Args:
    agent_name: The agent whose invocation these totals belong to.
    totals: Token counts summed over the invocation's model calls.
  """
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _invoke_agent_input_tokens.record(totals.input_tokens or 0, attributes=attrs)
  _invoke_agent_output_tokens.record(
      totals.output_tokens or 0, attributes=attrs
  )
  _invoke_agent_total_tokens.record(totals.total_tokens, attributes=attrs)
  _invoke_agent_cache_read_input_tokens.record(
      totals.cache_read_input_tokens or 0, attributes=attrs
  )
  _invoke_agent_reasoning_output_tokens.record(
      totals.reasoning_output_tokens or 0, attributes=attrs
  )
  _invoke_agent_tool_input_tokens.record(
      totals.tool_input_tokens or 0, attributes=attrs
  )


def _invoke_workflow_attrs(
    root_agent_name: str,
    workflow_name: str | None,
    nested: bool = False,
) -> dict[str, AttributeValue]:
  """Builds the attributes shared by every per-workflow metric.

  Both names, because they disagree when a turn enters at a sub-agent:
  `gen_ai.workflow.name` joins to `gen_ai.invoke_workflow.duration`, while the
  root agent name is the per-app total.

  Args:
    root_agent_name: The runner's agent, i.e. which app.
    workflow_name: The workflow this datapoint covers. Dropped when unset.
    nested: Whether another workflow enclosed this one. Omitted from the result
      when false.

  Returns:
    The attributes to record each per-workflow metric under.
  """
  attrs: dict[str, AttributeValue] = {ADK_ROOT_AGENT_NAME: root_agent_name}
  if workflow_name:
    attrs["gen_ai.workflow.name"] = workflow_name
  if nested:
    attrs["gen_ai.workflow.nested"] = True
  return attrs


def record_invoke_workflow_token_usage(
    *,
    root_agent_name: str,
    workflow_name: str | None,
    totals: TokenUsage,
    nested: bool,
) -> None:
  """Records the token spend of one workflow, across every agent in it.

  Carries no agent dimension: that is meaningless on a value spanning a whole
  workflow.

  Args:
    root_agent_name: The runner's agent.
    workflow_name: The workflow this datapoint covers.
    totals: Token counts summed over every model call made inside it.
    nested: Whether another workflow enclosed this one.
  """
  attrs = _invoke_workflow_attrs(root_agent_name, workflow_name, nested)
  _invoke_workflow_input_tokens.record(
      totals.input_tokens or 0, attributes=attrs
  )
  _invoke_workflow_output_tokens.record(
      totals.output_tokens or 0, attributes=attrs
  )
  _invoke_workflow_total_tokens.record(totals.total_tokens, attributes=attrs)
  _invoke_workflow_cache_read_input_tokens.record(
      totals.cache_read_input_tokens or 0, attributes=attrs
  )
  _invoke_workflow_reasoning_output_tokens.record(
      totals.reasoning_output_tokens or 0, attributes=attrs
  )
  _invoke_workflow_tool_input_tokens.record(
      totals.tool_input_tokens or 0, attributes=attrs
  )


def record_invoke_workflow_inference_calls(
    *,
    root_agent_name: str,
    workflow_name: str | None,
    count: int,
    nested: bool,
) -> None:
  """Records the inference (model) calls made across one workflow.

  Args:
    root_agent_name: The runner's agent.
    workflow_name: The workflow this datapoint covers.
    count: Model calls made by every agent that ran inside it.
    nested: Whether another workflow enclosed this one.
  """
  attrs = _invoke_workflow_attrs(root_agent_name, workflow_name, nested)
  _invoke_workflow_inference_calls.record(count, attributes=attrs)


def record_invoke_workflow_tool_calls(
    *,
    root_agent_name: str,
    workflow_name: str | None,
    count: int,
    nested: bool,
) -> None:
  """Records the tool calls made across one workflow.

  Args:
    root_agent_name: The runner's agent.
    workflow_name: The workflow this datapoint covers.
    count: Tool calls made by every agent that ran inside it, including the
      `transfer_to_agent` calls that route between them.
    nested: Whether another workflow enclosed this one.
  """
  attrs = _invoke_workflow_attrs(root_agent_name, workflow_name, nested)
  _invoke_workflow_tool_calls.record(count, attributes=attrs)


def record_tool_execution_duration(
    tool_name: str,
    tool_type: str,
    agent_name: str,
    elapsed_s: float,
    error: Exception | None = None,
    error_type: str | None = None,
) -> None:
  """Records the duration of the tool execution.

  Args:
    tool_name: Name of the tool that ran.
    tool_type: Class name of the tool that ran.
    agent_name: Name of the agent that ran the tool.
    elapsed_s: Duration of the tool execution, in seconds.
    error: The exception raised by the tool, if any.
    error_type: An error type detected from a tool response that reported a
      failure without raising. Ignored when `error` is also set.
  """
  attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
      gen_ai_attributes.GEN_AI_TOOL_TYPE: tool_type,
  }
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = tracing.resolve_error_type(error)
  elif error_type is not None:
    attrs[error_attributes.ERROR_TYPE] = error_type
  _tool_execution_duration.record(elapsed_s, attributes=attrs)


def record_client_operation_duration(
    agent_name: str,
    elapsed_s: float,
    llm_request: LlmRequest,
    responses: list[LlmResponse],
    error: Exception | None = None,
) -> None:
  """Encapsulates the business logic for tracking gen_ai client operation duration."""

  attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_OPERATION_NAME: "generate_content",
      gen_ai_attributes.GEN_AI_PROVIDER_NAME: _get_provider_name(
          llm_request.model
      ),
  }
  if llm_request.model:
    attrs[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = llm_request.model

  if responses:
    response_model = responses[-1].model_version or llm_request.model
    if response_model:
      attrs[gen_ai_attributes.GEN_AI_RESPONSE_MODEL] = response_model

  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = tracing.resolve_error_type(error)

  _client_operation_duration.record(elapsed_s, attributes=attrs)


def record_client_token_usage(
    agent_name: str,
    llm_request: LlmRequest,
    responses: list[LlmResponse],
) -> None:
  """Encapsulates the business logic for tracking gen_ai client token usage."""
  if not responses:
    return

  # The assumption is that token usage in streaming responses is cumulative.
  # The last response chunk contains the total usage for the entire request.
  # Summing them up across all response chunks would result in overcounting.
  last_response = responses[-1]
  if not last_response.usage_metadata:
    logger.warning(
        "Skipping missing token usage metadata for agent %s and model %s",
        agent_name,
        llm_request.model,
    )
    return

  # OTel semconv for `gen_ai.client.token.usage` states that token counts should
  # be categorized under `gen_ai.token.type` as either "input" or "output".
  # We aggregate prompt and tool use tokens for "input", and candidates and
  # thoughts tokens for "output".
  # `cached_content_token_count` is omitted as it's already included in prompt tokens.
  # `total_token_count` is omitted as SemConv expects input/output breakdown.
  token_usage = TokenUsage.from_usage_metadata(last_response.usage_metadata)
  input_token_count = token_usage.input_tokens or 0
  output_token_count = token_usage.output_tokens or 0
  response_model = last_response.model_version or llm_request.model
  base_attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_OPERATION_NAME: "generate_content",
      gen_ai_attributes.GEN_AI_PROVIDER_NAME: _get_provider_name(
          llm_request.model
      ),
  }
  if llm_request.model:
    base_attrs[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = llm_request.model
  if response_model:
    base_attrs[gen_ai_attributes.GEN_AI_RESPONSE_MODEL] = response_model

  if input_token_count > 0:
    input_attrs = base_attrs.copy()
    input_attrs[gen_ai_attributes.GEN_AI_TOKEN_TYPE] = "input"
    _client_token_usage.record(input_token_count, attributes=input_attrs)

  if output_token_count > 0:
    output_attrs = base_attrs.copy()
    output_attrs[gen_ai_attributes.GEN_AI_TOKEN_TYPE] = "output"
    _client_token_usage.record(output_token_count, attributes=output_attrs)


def _get_provider_name(model: str | None) -> str:
  return tracing._resolve_gen_ai_system_name(model)


def get_elapsed_s(
    span: Span | GenerateContentSpan | None,
    fallback_start: float,
) -> float:
  """Guarantees consistent time source for duration calculation.

  Note: This must be called with an ended span.

  Args:
    span (trace.Span | tracing.GenerateContentSpan | None): The ended span to
      extract duration from.
    fallback_start (float): Fallback start time in seconds (monotonic).

  Returns:
    float: Elapsed duration in seconds.
  """
  if span is None:
    return time.monotonic() - fallback_start

  span = span.span if hasattr(span, "span") else span
  start_ns = getattr(span, "start_time", None)
  end_ns = getattr(span, "end_time", None)

  if isinstance(start_ns, int) and isinstance(end_ns, int):
    return (end_ns - start_ns) / 1e9  # Convert ns to s

  # Fallback if span times are missing
  return time.monotonic() - fallback_start


def record_skill_script_execution(
    agent_name: str,
    skill_name: _hallucination.MaybeHallucinated[str],
    script_path: _hallucination.MaybeHallucinated[str],
    script_exit_code: int | None,
) -> None:
  """Records the result of skill's script executions."""
  attrs: dict[str, AttributeValue] = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      _adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME: skill_name.bounded(),
      _adk_attributes.ADK_EXPERIMENTAL_SKILL_SCRIPT_PATH: script_path.bounded(),
  }
  if script_exit_code is not None:
    # As exit codes can be up to 255, to reduce cardinality we only record
    # whether the script failed or not.
    attrs[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SCRIPT_ENDED_WITH_ERROR] = (
        script_exit_code != 0
    )

  _skill_script_executions.add(1, attributes=attrs)


def record_skill_load(
    agent_name: str,
    skill_name: _hallucination.MaybeHallucinated[str],
    error_type: str | None = None,
) -> None:
  """Records one skill load, whether or not it resolved a skill."""
  attrs: dict[str, AttributeValue] = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      _adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME: skill_name.bounded(),
  }
  if error_type is not None:
    attrs[error_attributes.ERROR_TYPE] = error_type

  _skill_loads.add(1, attributes=attrs)


def record_invoke_agent_skill_loads(
    agent_name: str,
    count: int,
) -> None:
  """Records the number of skill loads in an agent invocation.

  Args:
    agent_name: The agent the invocation ran.
    count: Loads made anywhere inside it, whether or not they resolved a skill.
      An invocation that loaded nothing is recorded as the zero it measured.
  """
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _invoke_agent_skill_loads.record(count, attributes=attrs)


def record_invoke_workflow_skill_loads(
    *,
    root_agent_name: str,
    workflow_name: str | None,
    count: int,
    nested: bool,
) -> None:
  """Records the skill loads made across one workflow.

  Args:
    root_agent_name: The runner's agent.
    workflow_name: The workflow this datapoint covers.
    count: Loads made by every agent that ran inside it.
    nested: Whether another workflow enclosed this one.
  """
  attrs = _invoke_workflow_attrs(root_agent_name, workflow_name, nested)
  _invoke_workflow_skill_loads.record(count, attributes=attrs)
