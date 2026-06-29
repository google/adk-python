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

from typing import TYPE_CHECKING

from google.adk.telemetry import tracing
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from .functional_node_test_cases import ALL_NODE_CASES
from .functional_test_helpers import aclosing_wrapping_assertions
from .functional_test_helpers import install_telemetry
from .functional_test_helpers import run_node_scenario
from .functional_test_helpers import TelemetryDigest

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from opentelemetry.sdk.trace import ReadableSpan

  from .functional_test_helpers import FunctionalTestCase


@pytest.mark.parametrize('case', ALL_NODE_CASES, ids=lambda c: c.test_id)
@pytest.mark.asyncio
async def test_telemetry_schema(
    case: FunctionalTestCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Tests creation of multiple spans/logs in an E2E runner invocation with a

  workflow.

  Asserts the entire telemetry schema (spans + attributes + per-span logs)
  matches the hand-written expected shape for the given semconv +
  content-capture configuration.
  """
  case.apply_env(monkeypatch)
  span_exporter = InMemorySpanExporter()
  log_exporter = InMemoryLogRecordExporter()
  metric_reader = InMemoryMetricReader()
  install_telemetry(monkeypatch, span_exporter, log_exporter, metric_reader)

  events = await run_node_scenario()
  spans = span_exporter.get_finished_spans()
  digest = TelemetryDigest.build(
      spans, log_exporter.get_finished_logs(), metric_reader.get_metrics_data()
  )

  assert digest == case.expected
  _verify_associated_events(spans, events)


@pytest.mark.asyncio
async def test_async_generators_wrapped_in_aclosing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Asserts each async generator iterated by the scenario is wrapped in ``aclosing``.

  Necessary because instrumentation utilizes contextvars, which run into
  "ContextVar was created in a different Context" errors when a given
  coroutine gets indeterminately suspended.

  Kept as a single non-parametrized test because the underlying
  ``gc.get_referrers`` walk is expensive (~5 seconds per scenario).
  """
  install_telemetry(
      monkeypatch,
      InMemorySpanExporter(),
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )

  with aclosing_wrapping_assertions():
    _ = await run_node_scenario()


def _verify_associated_events(
    spans: tuple[ReadableSpan, ...], events: list[Event]
):
  def _nodelike_name(span: ReadableSpan) -> str:
    for prefix in ['invoke_node ', 'invoke_workflow ', 'invoke_agent ']:
      if span.name.startswith(prefix):
        return span.name.replace(prefix, '')
    return ''

  def _emitting_node_name(event: Event) -> str:
    # Strip out
    # 1. Path except for the last node (everything before "/")
    # 2. Retry count (everything after "@")
    return event.node_info.path.split('/')[-1].split('@')[0]

  events_by_id = {event.id: event for event in events}
  for span in spans:
    if not span.attributes:
      continue
    associated_ids = span.attributes.get(
        'gcp.vertex.agent.associated_event_ids', None
    )
    if associated_ids is None:
      continue
    assert isinstance(associated_ids, tuple)
    assert len(associated_ids) > 0, f'Span name {span.name} emitted no events'
    for event_id in associated_ids:
      event = events_by_id[str(event_id)]
      assert _nodelike_name(span) == _emitting_node_name(event)


@pytest.mark.asyncio
async def test_exception_preserves_attributes(
    monkeypatch: pytest.MonkeyPatch,
):
  """Test when an exception occurs during tool execution, span attributes are still present on spans where they are expected."""

  span_exporter = InMemorySpanExporter()
  install_telemetry(
      monkeypatch,
      span_exporter,
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )

  captured_events: list[Event] = []
  with pytest.raises(ValueError, match='This tool always fails'):
    await run_node_scenario(failing=True, event_sink=captured_events)

  # Assert
  spans = span_exporter.get_finished_spans()
  _verify_associated_events(spans, captured_events)
  spans_by_name = {span.name: span for span in spans}

  assert 'execute_tool some_tool' in spans_by_name
  tool_span = spans_by_name['execute_tool some_tool']

  attrs = dict(tool_span.attributes)
  # Dynamic ID
  tool_call_id = attrs.get('gen_ai.tool.call.id')

  assert dict(tool_span.attributes) == {
      'gen_ai.operation.name': 'execute_tool',
      'gen_ai.tool.name': 'some_tool',
      'gen_ai.tool.description': 'A sample tool.',
      'gen_ai.tool.type': 'FunctionTool',
      'error.type': 'ValueError',
      'gcp.vertex.agent.llm_request': '{}',
      'gcp.vertex.agent.llm_response': '{}',
      'gcp.vertex.agent.tool_call_args': '{"arg1": "val1"}',
      'gen_ai.tool.call.id': tool_call_id,
      'gcp.vertex.agent.tool_response': '{"result": "<not specified>"}',
  }


@pytest.mark.asyncio
async def test_no_generate_content_for_gemini_model_when_already_instrumented(
    monkeypatch: pytest.MonkeyPatch,
):
  """Tests that generate_content span is not created if already instrumented."""

  span_exporter = InMemorySpanExporter()
  install_telemetry(
      monkeypatch,
      span_exporter,
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )

  # Arrange
  monkeypatch.setattr(
      tracing,
      '_instrumented_with_opentelemetry_instrumentation_google_genai',
      lambda: True,
  )
  monkeypatch.setattr(
      tracing,
      '_is_gemini_agent',
      lambda _: True,
  )

  _ = await run_node_scenario()

  # Assert
  spans = span_exporter.get_finished_spans()
  assert not any(span.name.startswith('generate_content') for span in spans)
