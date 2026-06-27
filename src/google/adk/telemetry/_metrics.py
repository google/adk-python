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
from typing import TYPE_CHECKING

from google.adk import version
from google.adk.telemetry import tracing
from google.adk.telemetry._token_usage import TokenUsage
from google.genai import types
from opentelemetry import metrics
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from google.adk.models.llm_request import LlmRequest
  from google.adk.models.llm_response import LlmResponse

logger = logging.getLogger("google_adk." + __name__)

GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_TOOL_VERSION = "gen_ai.tool.version"

meter = metrics.get_meter(
    name="gcp.vertex.agent",
    version=version.__version__,
)

_agent_invocation_duration = meter.create_histogram(
    "gen_ai.agent.invocation.duration",
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
_tool_execution_duration = meter.create_histogram(
    "gen_ai.tool.execution.duration",
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
_agent_request_size = meter.create_histogram(
    "gen_ai.agent.request.size",
    unit="By",
    description="Size of agent requests.",
    explicit_bucket_boundaries_advisory=[
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ],
)
_agent_response_size = meter.create_histogram(
    "gen_ai.agent.response.size",
    unit="By",
    description="Size of agent responses.",
    explicit_bucket_boundaries_advisory=[
        1,
        4,
        16,
        64,
        256,
        1024,
        4096,
        16384,
        65536,
        262144,
        1048576,
        4194304,
        16777216,
        67108864,
    ],
)
_agent_workflow_steps = meter.create_histogram(
    "gen_ai.agent.workflow.steps",
    unit="1",
    description="Length of agentic workflow (# of events).",
    explicit_bucket_boundaries_advisory=[
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ],
)
_client_operation_duration = (
    gen_ai_metrics.create_gen_ai_client_operation_duration(meter)
)
_client_token_usage = gen_ai_metrics.create_gen_ai_client_token_usage(meter)


def record_agent_invocation_duration(
    agent_name: str,
    elapsed_s: float,
    error: Exception | None = None,
):
  """Records the duration of the agent invocation."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = type(error).__name__
  _agent_invocation_duration.record(elapsed_s, attributes=attrs)


def record_agent_request_size(
    agent_name: str, user_content: types.Content | None
):
  """Records the size of the agent request."""
  size = _get_content_size(user_content)
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _agent_request_size.record(size, attributes=attrs)


def record_agent_response_size(agent_name: str, events: list[Event]):
  """Records the size of the agent response by extracting content from events."""
  response_content: types.Content | None = None
  for event in reversed(events):
    if event.author == agent_name and event.content:
      response_content = event.content
      break

  size = _get_content_size(response_content)
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  _agent_response_size.record(size, attributes=attrs)


def record_agent_workflow_steps(agent_name: str, events: list[Event]):
  """Records the number of steps in the agent workflow by counting the number of events."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  count = sum(1 for event in events if event.author == agent_name)
  _agent_workflow_steps.record(count, attributes=attrs)


def record_tool_execution_duration(
    tool_name: str,
    tool_type: str,
    agent_name: str,
    elapsed_s: float,
    error: Exception | None = None,
):
  """Records the duration of the tool execution."""
  attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
      gen_ai_attributes.GEN_AI_TOOL_TYPE: tool_type,
  }
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = type(error).__name__
  _tool_execution_duration.record(elapsed_s, attributes=attrs)


def record_client_operation_duration(
    agent_name: str,
    elapsed_s: float,
    llm_request: LlmRequest,
    responses: list[LlmResponse],
    error: Exception | None = None,
):
  """Encapsulates the business logic for tracking gen_ai client operation duration."""

  attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_OPERATION_NAME: "generate_content",
      gen_ai_attributes.GEN_AI_PROVIDER_NAME: _get_provider_name(),
  }
  if llm_request.model:
    attrs[gen_ai_attributes.GEN_AI_REQUEST_MODEL] = llm_request.model

  if responses:
    response_model = responses[-1].model_version or llm_request.model
    if response_model:
      attrs[gen_ai_attributes.GEN_AI_RESPONSE_MODEL] = response_model

  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = type(error).__name__

  _client_operation_duration.record(elapsed_s, attributes=attrs)


def record_client_token_usage(
    agent_name: str,
    llm_request: LlmRequest,
    responses: list[LlmResponse],
):
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
  token_usage = TokenUsage(last_response.usage_metadata)
  input_token_count = token_usage.input_token_count or 0
  output_token_count = token_usage.output_token_count or 0
  response_model = last_response.model_version or llm_request.model
  base_attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_OPERATION_NAME: "generate_content",
      gen_ai_attributes.GEN_AI_PROVIDER_NAME: _get_provider_name(),
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


def _get_content_size(
    content: types.Content | None,
) -> int:
  if not content or not content.parts:
    return 0
  size = 0
  for part in content.parts:
    if part.text is not None:
      size += len(part.text.encode("utf-8"))
    if part.inline_data and part.inline_data.data:
      size += len(part.inline_data.data)
  return size


def _get_provider_name() -> str:
  return tracing._guess_gemini_system_name()
