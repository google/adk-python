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

"""Per-node span dispatch in ``node_tracing.start_as_current_node_span``.

The full node telemetry shape is asserted end-to-end in
``test_node_functional``; these tests pin the dispatch itself -- which node
kind gets which span -- and the associated-event bookkeeping, whose values
that digest deliberately masks as non-deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._workflow import Workflow
from opentelemetry import context as context_api
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from .functional_test_helpers import install_telemetry

_SESSION_ID = 'some_session'


class _PlainNode(BaseNode):
  """A node that is neither an agent nor a workflow."""

  async def run(self, ctx: Context, node_input: object) -> AsyncGenerator:
    del ctx, node_input
    return
    yield  # pylint: disable=unreachable


@pytest.fixture(name='span_exporter')
def _span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  span_exporter = InMemorySpanExporter()
  install_telemetry(
      monkeypatch,
      span_exporter,
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )
  return span_exporter


def _context() -> Context:
  session = Session(app_name='test_app', user_id='test_user', id=_SESSION_ID)
  return Context(
      InvocationContext(
          invocation_id='test_invocation_id',
          session=session,
          session_service=InMemorySessionService(),
      )
  )


def _event(event_id: str) -> Event:
  event = Event(author='some_node')
  event.id = event_id
  return event


@pytest.mark.asyncio
async def test_plain_node_gets_an_invoke_node_span(
    span_exporter: InMemorySpanExporter,
):
  """A node that is neither an agent nor a workflow gets its own span kind."""
  async with node_tracing.start_as_current_node_span(
      _context(), _PlainNode(name='some_node')
  ):
    pass

  (span,) = span_exporter.get_finished_spans()
  assert span.name == 'invoke_node some_node'
  assert dict(span.attributes) == {
      'gen_ai.operation.name': 'invoke_node',
      'gen_ai.conversation.id': _SESSION_ID,
  }


@pytest.mark.asyncio
async def test_workflow_node_gets_an_invoke_workflow_span(
    span_exporter: InMemorySpanExporter,
):
  """A workflow node opens the semconv workflow span, named after itself.

  As the first workflow in the invocation it is the root, so the nested flag is
  omitted rather than set to false.
  """
  async with node_tracing.start_as_current_node_span(
      _context(), Workflow(name='some_workflow')
  ):
    pass

  (span,) = span_exporter.get_finished_spans()
  assert span.name == 'invoke_workflow some_workflow'
  assert dict(span.attributes) == {
      'gen_ai.operation.name': 'invoke_workflow',
      'gen_ai.conversation.id': _SESSION_ID,
      'gen_ai.workflow.name': 'some_workflow',
  }


@pytest.mark.asyncio
async def test_agent_node_opens_no_span_of_its_own(
    span_exporter: InMemorySpanExporter,
):
  """Agents emit their own ``invoke_agent`` span from the agent path, so the

  node path must pass through: a span here would duplicate it.
  """
  agent = LlmAgent(name='some_agent', model='not-a-gemini-model')

  async with node_tracing.start_as_current_node_span(_context(), agent):
    pass

  assert span_exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_agent_node_activates_the_context_the_node_carries(
    span_exporter: InMemorySpanExporter,
):
  """The pass-through must activate the OTel context the node carries, not

  leave whatever is current at the call site in place -- that is what puts
  the agent's own span under its parent node's span. The node context is
  built under a span here and entered from outside it, so the two differ.
  """
  agent = LlmAgent(name='some_agent', model='not-a-gemini-model')
  with tracing.tracer.start_as_current_span('parent_node'):
    context = _context()
  carried = context.telemetry_context.otel_context
  assert context_api.get_current() is not carried

  async with node_tracing.start_as_current_node_span(context, agent) as tel_ctx:
    assert context_api.get_current() is carried
    assert tel_ctx.otel_context is carried

  assert context_api.get_current() is not carried


@pytest.mark.asyncio
async def test_node_span_records_the_events_produced_inside_it(
    span_exporter: InMemorySpanExporter,
):
  """The event ids registered during the node are stamped on its span in

  registration order, which is what links a span back to its output.
  """
  async with node_tracing.start_as_current_node_span(
      _context(), _PlainNode(name='some_node')
  ) as tel_ctx:
    tel_ctx.add_event(_event('event-1'))
    tel_ctx.add_event(_event('event-2'))

  (span,) = span_exporter.get_finished_spans()
  assert span.attributes['gcp.vertex.agent.associated_event_ids'] == (
      'event-1',
      'event-2',
  )


@pytest.mark.asyncio
async def test_node_span_omits_associated_events_when_there_are_none(
    span_exporter: InMemorySpanExporter,
):
  """A node that produced nothing omits the attribute rather than recording

  an empty list, so consumers can tell 'no events' from 'not instrumented'.
  """
  async with node_tracing.start_as_current_node_span(
      _context(), _PlainNode(name='some_node')
  ):
    pass

  (span,) = span_exporter.get_finished_spans()
  assert 'gcp.vertex.agent.associated_event_ids' not in span.attributes
