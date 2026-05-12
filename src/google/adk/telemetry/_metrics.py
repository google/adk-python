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

from google.adk import version
from google.adk.events.event import Event
from google.genai import types
from opentelemetry import metrics
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes import error_attributes

logger = logging.getLogger("google_adk." + __name__)

# TODO(b/477553411): add these attributes to Otel semconv.
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"
GEN_AI_TOOL_VERSION = "gen_ai.tool.version"

# Initialize meter
meter = metrics.get_meter(
    name="gcp.vertex.agent",
    version=version.__version__,
    # TODO(b/477553411): set schema version after OTel semconv updates.
)

# Define histograms
_agent_invocation_duration = meter.create_histogram(
    "gen_ai.agent.invocation.duration",
    unit="ms",
    description="Duration of agent invocations.",
)
_tool_execution_duration = meter.create_histogram(
    "gen_ai.tool.execution.duration",
    unit="ms",
    description="Duration of tool executions.",
)
_agent_request_size = meter.create_histogram(
    "gen_ai.agent.request.size",
    unit="By",
    description="Size of agent requests.",
)
_agent_response_size = meter.create_histogram(
    "gen_ai.agent.response.size",
    unit="By",
    description="Size of agent responses.",
)
_agent_workflow_steps = meter.create_histogram(
    "gen_ai.agent.workflow.steps",
    unit="1",
    description="Length of agentic workflow (# of events).",
)


def record_agent_invocation_duration(
    agent_name: str,
    elapsed_ms: float,
    error: Exception | None = None,
):
  """Records the duration of the agent invocation."""
  attrs = {gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name}
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = type(error).__name__
  _agent_invocation_duration.record(elapsed_ms, attributes=attrs)


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
    agent_name: str,
    elapsed_ms: float,
    error: Exception | None = None,
):
  """Records the duration of the tool execution."""
  attrs = {
      gen_ai_attributes.GEN_AI_AGENT_NAME: agent_name,
      gen_ai_attributes.GEN_AI_TOOL_NAME: tool_name,
  }
  if error is not None:
    attrs[error_attributes.ERROR_TYPE] = type(error).__name__
  _tool_execution_duration.record(elapsed_ms, attributes=attrs)


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
