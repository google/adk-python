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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.util.types import Attributes

from ..agents.context import Context
from ..workflow._base_node import BaseNode
from .tracing import tracer

if TYPE_CHECKING:
  from ..events.event import Event
  from ..workflow._workflow import Workflow


@dataclass(frozen=True)
class TelemetryContext:
  """Telemetry specific context tied to the lifetime of the span."""

  otel_context: context_api.Context
  """OTel context holding the current trace span."""

  _associated_event_ids: list[str] = field(default_factory=list)
  """Event IDs added to the event queue within a given node."""

  def add_event(self, event: Event) -> None:
    """Adds an event ID to the associated events list."""
    self._associated_event_ids.append(event.id)


@dataclass
class _SpanMetadata:
  name: str
  attributes: Attributes


@asynccontextmanager
async def start_as_current_node_span(
    context: Context, node: BaseNode
) -> AsyncIterator[TelemetryContext]:
  """Creates a scope-based OpenTelemetry span, representing a node invocation.

  Implements emitting of the following spans:
  - `invoke_agent {agent.name}`
  - `invoke_workflow {workflow.name}`
  - `invoke_node {node.name}`

  invoke_agent spans align with OpenTelemetry Semantic Conventions (semconv) version 1.36 spans for backwards compatibility.
  https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/README.md

  invoke_workflow spans align with semconv version 1.41, because these were not included in any prior releases.
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/README.md

  invoke_node spans are not present in any semconv release.
  We will create a proposal to standardize them.

  Args:
    context: Context in which the span is created.
    node: The node to be invoked inside the created span.

  Yields:
    Context with the started span.
  """

  span_metadata = _span_metadata(context, node)
  if span_metadata is None:
    token = context_api.attach(context.telemetry_context.otel_context)
    try:
      yield TelemetryContext(
          otel_context=context.telemetry_context.otel_context
      )
    finally:
      context_api.detach(token)
    return

  with tracer.start_as_current_span(
      span_metadata.name,
      attributes=span_metadata.attributes,
      context=context.telemetry_context.otel_context,
  ) as span:
    telemetry_context = TelemetryContext(otel_context=context_api.get_current())
    yield telemetry_context

    if span.is_recording() and len(telemetry_context._associated_event_ids) > 0:
      span.set_attribute(
          "gcp.vertex.agent.associated_event_ids",
          telemetry_context._associated_event_ids,
      )


def _span_metadata(context: Context, node: BaseNode) -> _SpanMetadata | None:
  from ..agents.base_agent import BaseAgent
  from ..workflow._workflow import Workflow

  if isinstance(node, BaseAgent):
    return None
  elif isinstance(node, Workflow):
    return _workflow_span_metadata(context, node)
  else:
    return _default_node_span_metadata(context, node)


def _workflow_span_metadata(
    context: Context, workflow: Workflow
) -> _SpanMetadata:
  return _SpanMetadata(
      name=f"invoke_workflow {workflow.name}",
      attributes={
          GEN_AI_OPERATION_NAME: "invoke_workflow",
          "gen_ai.workflow.name": workflow.name,
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
  )


def _default_node_span_metadata(
    context: Context, node: BaseNode
) -> _SpanMetadata:
  return _SpanMetadata(
      name=f"invoke_node {node.name}",
      attributes={
          GEN_AI_OPERATION_NAME: "invoke_node",
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
  )
