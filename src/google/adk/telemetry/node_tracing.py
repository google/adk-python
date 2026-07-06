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
from collections.abc import Iterator
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
import sys
import time
from typing import TYPE_CHECKING

from opentelemetry import context as context_api
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_CONVERSATION_ID
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GEN_AI_OPERATION_NAME
from opentelemetry.trace import Span

from . import _metrics
from ..agents.context import Context
from ..workflow._base_node import BaseNode
from .tracing import tracer

if TYPE_CHECKING:
  from opentelemetry.util.types import AttributeValue

  from ..agents.base_agent import BaseAgent
  from ..events.event import Event
  from ..workflow._workflow import Workflow

# Span/metric attribute flagging that an `invoke_workflow` span is nested
# within another workflow. Only emitted for nested workflows; the root
# (entrypoint) workflow omits it entirely.
GEN_AI_WORKFLOW_NESTED = "gen_ai.workflow.nested"

# OTel-context key recording that an entrypoint workflow is already active. It
# rides along the otel_context propagated to child nodes, so only the first
# workflow invoked within an invocation is treated as the root -- nested
# workflows (incl. agents-as-tool that spin up their own runner) see the key
# already set and report nested=true.
_ENTRYPOINT_WORKFLOW_KEY = context_api.create_key(
    "adk-entrypoint-workflow-active"
)


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


@asynccontextmanager
async def start_as_current_node_span(
    context: Context, node: BaseNode
) -> AsyncIterator[TelemetryContext]:
  """Creates a scope-based OpenTelemetry span, representing a node invocation.

  Implements emitting of the following spans:
  - `invoke_agent {agent.name}`
  - `invoke_workflow {workflow.name}`
  - `invoke_node {node.name}`

  invoke_agent spans align with OpenTelemetry Semantic Conventions (semconv)
  version 1.36 spans for backwards compatibility.
  https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/README.md

  invoke_workflow spans align with semconv version 1.41, because these were not
  included in any prior releases.
  https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/README.md

  invoke_node spans are not present in any semconv release.
  We will create a proposal to standardize them.

  Args:
    context: Context in which the span is created.
    node: The node to be invoked inside the created span.

  Yields:
    Context with the started span.
  """

  from ..agents.base_agent import BaseAgent
  from ..workflow._workflow import Workflow

  if isinstance(node, BaseAgent):
    with _invoke_agent_span(context, node) as tel_ctx:
      yield tel_ctx
  elif isinstance(node, Workflow):
    with _invoke_workflow_span(context, node) as tel_ctx:
      yield tel_ctx
  else:
    with _invoke_node_span(context, node) as tel_ctx:
      yield tel_ctx


@contextmanager
def _invoke_agent_span(
    context: Context, agent: BaseAgent
) -> Iterator[TelemetryContext]:
  """Passes through an agent node; agents emit their own `invoke_agent` span."""
  del agent
  token = context_api.attach(context.telemetry_context.otel_context)
  try:
    yield TelemetryContext(otel_context=context.telemetry_context.otel_context)
  finally:
    context_api.detach(token)


@contextmanager
def _invoke_workflow_span(
    context: Context, workflow: Workflow
) -> Iterator[TelemetryContext]:
  """Opens an `invoke_workflow` span plus its duration metric for ``node``."""
  with _use_invoke_workflow_span(
      workflow.name,
      context.session.id,
      otel_context=context.telemetry_context.otel_context,
  ) as span:
    tel_ctx = TelemetryContext(otel_context=context_api.get_current())
    yield tel_ctx
    _maybe_set_associated_events(span, tel_ctx)


@contextmanager
def _invoke_node_span(
    context: Context, node: BaseNode
) -> Iterator[TelemetryContext]:
  """Opens an `invoke_node` span for a plain node."""
  with tracer.start_as_current_span(
      f"invoke_node {node.name}",
      attributes={
          GEN_AI_OPERATION_NAME: "invoke_node",
          GEN_AI_CONVERSATION_ID: context.session.id,
      },
      context=context.telemetry_context.otel_context,
  ) as span:
    tel_ctx = TelemetryContext(otel_context=context_api.get_current())
    yield tel_ctx
    _maybe_set_associated_events(span, tel_ctx)


def _maybe_set_associated_events(
    span: Span, telemetry_context: TelemetryContext
) -> None:
  """Stamps the node's associated event IDs onto its span, if any."""
  if span.is_recording() and len(telemetry_context._associated_event_ids) > 0:
    span.set_attribute(
        "gcp.vertex.agent.associated_event_ids",
        telemetry_context._associated_event_ids,
    )


@contextmanager
def _use_invoke_workflow_span(
    workflow_name: str,
    conversation_id: str,
    *,
    otel_context: context_api.Context | None = None,
) -> Iterator[Span]:
  """Opens an `invoke_workflow {workflow_name}` span."""
  if otel_context is None:
    otel_context = context_api.get_current()
  # First workflow in the invocation is the root; subsequent ones are nested.
  # The flag rides along the otel_context propagated to child nodes, so nested
  # workflows see it set.
  nested = bool(context_api.get_value(_ENTRYPOINT_WORKFLOW_KEY, otel_context))
  attributes: dict[str, AttributeValue] = {
      GEN_AI_OPERATION_NAME: "invoke_workflow",
      GEN_AI_CONVERSATION_ID: conversation_id,
  }
  # Root workflow omits the attribute entirely; only nested ones emit it.
  if nested:
    attributes[GEN_AI_WORKFLOW_NESTED] = True
  if workflow_name:
    attributes["gen_ai.workflow.name"] = workflow_name

  span_name = (
      f"invoke_workflow {workflow_name}" if workflow_name else "invoke_workflow"
  )

  start_s = time.monotonic()
  workflow_span: Span | None = None
  try:
    with (
        tracer.start_as_current_span(
            name=span_name,
            attributes=attributes,
            context=otel_context,
        ) as span,
        _mark_nested_workflows(),
    ):
      workflow_span = span
      yield span
  finally:
    _metrics.record_workflow_invocation_duration(
        workflow_name=workflow_name,
        elapsed_s=_metrics.get_elapsed_s(workflow_span, start_s),
        nested=nested,
        error=sys.exc_info()[1],
    )


@contextmanager
def _mark_nested_workflows() -> Iterator[None]:
  token = context_api.attach(
      context_api.set_value(_ENTRYPOINT_WORKFLOW_KEY, True)
  )
  try:
    yield
  finally:
    context_api.detach(token)
