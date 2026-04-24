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

import dataclasses
import gc
import sys
from typing import Any
from typing import Sequence

from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from google.adk.tools import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.genai.types import Part
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.export import Metric
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from ..testing_utils import InMemoryRunner
from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner


@pytest.fixture
def test_model() -> BaseLlm:
  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name="some_tool", args={}),
          Part.from_text(text="text response"),
      ]
  )
  return mock_model


@pytest.fixture
def test_agent(test_model: BaseLlm) -> Agent:
  def some_tool():
    pass

  root_agent = Agent(
      name="some_root_agent",
      model=test_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )
  return root_agent


@pytest.fixture
async def test_runner(test_agent: Agent) -> TestInMemoryRunner:
  runner = TestInMemoryRunner(test_agent)
  return runner


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
  tracer_provider = TracerProvider()
  span_exporter = InMemorySpanExporter()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  def do_replace(tracer):
    monkeypatch.setattr(
        tracer, "start_as_current_span", real_tracer.start_as_current_span
    )

  do_replace(tracing.tracer)

  return span_exporter


@pytest.mark.asyncio
async def test_tracer_start_as_current_span(
    test_runner: TestInMemoryRunner,
    span_exporter: InMemorySpanExporter,
):
  """Test creation of multiple spans in an E2E runner invocation.

  Additionally tests if each async generator invoked is wrapped in Aclosing.
  This is necessary because instrumentation utilizes contextvars, which ran into "ContextVar was created in a different Context" errors,
  when a given coroutine gets indeterminately suspended.
  """
  firstiter, finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro):
    nonlocal firstiter
    # Skip check for specific async context managers in tracing.py,
    # as their internal generators are not expected to be Aclosing-wrapped.
    if (
        coro.__name__ == "use_inference_span"
        or coro.__name__ == "_use_native_generate_content_span"
        or coro.__name__ == "record_agent_invocation"
        or coro.__name__ == "record_tool_execution"
    ):
      firstiter(coro)
      return
    assert any(
        isinstance(referrer, Aclosing)
        or isinstance(indirect_referrer, Aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in Python 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), f"Coro `{coro.__name__}` is not wrapped with Aclosing"
    firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, finalizer)

  # Act
  async with Aclosing(test_runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert list(sorted(span.name for span in spans)) == [
      "call_llm",
      "call_llm",
      "execute_tool some_tool",
      "generate_content mock",
      "generate_content mock",
      "invocation",
      "invoke_agent some_root_agent",
  ]


@pytest.mark.asyncio
async def test_exception_preserves_attributes(
    test_model: BaseLlm, span_exporter: InMemorySpanExporter
):
  """Test when an exception occurs during tool execution, span attributes are still present on spans where they are expected."""

  # Arrange
  async def some_tool():
    raise ValueError("This tool always fails")

  test_agent = Agent(
      name="some_root_agent",
      model=test_model,
      tools=[
          FunctionTool(some_tool),
      ],
  )

  test_runner = TestInMemoryRunner(test_agent)

  # Act
  with pytest.raises(ValueError, match="This tool always fails"):
    async with Aclosing(
        test_runner.run_async_with_new_session_agen("")
    ) as agen:
      async for _ in agen:
        pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert len(spans) > 1
  assert all(
      span.attributes is not None and len(span.attributes) > 0
      for span in spans
      if span.name != "invocation"  # not expected to have attributes
  )


@pytest.mark.asyncio
async def test_no_generate_content_for_gemini_model_when_already_instrumented(
    test_runner: TestInMemoryRunner,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
):
  """Tests"""
  # Arrange
  monkeypatch.setattr(
      tracing,
      "_instrumented_with_opentelemetry_instrumentation_google_genai",
      lambda: True,
  )
  monkeypatch.setattr(
      tracing,
      "_is_gemini_agent",
      lambda _: True,
  )

  # Act
  async with Aclosing(test_runner.run_async_with_new_session_agen("")) as agen:
    async for _ in agen:
      pass

  # Assert
  spans = span_exporter.get_finished_spans()
  assert not any(span.name.startswith("generate_content") for span in spans)


def test_instrumented_with_opentelemetry_instrumentation_google_genai():
  instrumentor = GoogleGenAiSdkInstrumentor()

  assert (
      not tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
  )
  try:
    instrumentor.instrument()
    assert (
        tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
    )
  finally:
    instrumentor.uninstrument()
  assert (
      not tracing._instrumented_with_opentelemetry_instrumentation_google_genai()
  )


@dataclasses.dataclass
class MetricPoint:
  attributes: dict[str, Any]
  value: Any = None


def _extract_metrics(
    metrics_list: Sequence[Metric], name: str
) -> list[MetricPoint]:
  m = next((m for m in metrics_list if m.name == name), None)
  if not m:
    return []
  points = []
  for dp in m.data.data_points:
    value = None
    if hasattr(dp, "sum"):
      value = dp.sum
    elif hasattr(dp, "value"):
      value = dp.value
    points.append(MetricPoint(attributes=dp.attributes, value=value))
  return points


def _setup_test_metrics(monkeypatch):
  reader = InMemoryMetricReader()
  provider = MeterProvider(metric_readers=[reader])
  meter = provider.get_meter("test_meter")
  agent_duration_hist = meter.create_histogram(
      "gen_ai.agent.invocation.duration"
  )
  tool_duration_hist = meter.create_histogram("gen_ai.tool.execution.duration")
  request_size_hist = meter.create_histogram("gen_ai.agent.request.size")
  response_size_hist = meter.create_histogram("gen_ai.agent.response.size")
  workflow_steps_hist = meter.create_histogram("gen_ai.agent.workflow.steps")

  monkeypatch.setattr(
      _metrics, "_agent_invocation_duration", agent_duration_hist
  )
  monkeypatch.setattr(_metrics, "_tool_execution_duration", tool_duration_hist)
  monkeypatch.setattr(_metrics, "_agent_request_size", request_size_hist)
  monkeypatch.setattr(_metrics, "_agent_response_size", response_size_hist)
  monkeypatch.setattr(_metrics, "_agent_workflow_steps", workflow_steps_hist)
  return reader


@pytest.mark.asyncio
async def test_metrics(monkeypatch):
  reader = _setup_test_metrics(monkeypatch)

  async def get_current_time():
    return "2026-04-15T14:26:03Z"

  async def generate_random_number():
    return 42

  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name="get_current_time", args={}),
          Part.from_function_call(name="generate_random_number", args={}),
          Part.from_text(text="Both tools executed."),
      ]
  )
  test_agent = Agent(
      name="complex_agent",
      model=mock_model,
      tools=[
          FunctionTool(get_current_time),
          FunctionTool(generate_random_number),
      ],
  )

  runner = InMemoryRunner(root_agent=test_agent)
  await runner.run_async("Run both tools")

  metrics_data = reader.get_metrics_data()
  assert len(metrics_data.resource_metrics) > 0
  scope_metrics = metrics_data.resource_metrics[0].scope_metrics
  assert len(scope_metrics) > 0
  metrics_list = scope_metrics[0].metrics
  got_invocation = _extract_metrics(
      metrics_list, "gen_ai.agent.invocation.duration"
  )
  assert len(got_invocation) == 1
  for p in got_invocation:
    p.value = None
  want_invocation = [
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.input.type": "text",
              "gen_ai.output.type": "text",
          },
          value=None,
      )
  ]
  assert got_invocation == want_invocation
  got_tool_exec = _extract_metrics(
      metrics_list, "gen_ai.tool.execution.duration"
  )
  assert len(got_tool_exec) == 2
  for p in got_tool_exec:
    p.value = None
  want_tool_exec = [
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.tool.name": "generate_random_number",
              "gen_ai.input.type": "text",
              "gen_ai.output.type": "text",
          },
          value=None,
      ),
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.tool.name": "get_current_time",
              "gen_ai.input.type": "text",
              "gen_ai.output.type": "text",
          },
          value=None,
      ),
  ]
  got_tool_exec.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  want_tool_exec.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  assert got_tool_exec == want_tool_exec
  got_steps = _extract_metrics(metrics_list, "gen_ai.agent.workflow.steps")
  assert len(got_steps) == 1
  want_steps = [
      MetricPoint(attributes={"gen_ai.agent.name": "complex_agent"}, value=6)
  ]
  assert got_steps == want_steps


@pytest.mark.asyncio
async def test_metrics_tool_error(monkeypatch):
  reader = _setup_test_metrics(monkeypatch)

  async def get_current_time():
    return "2026-04-15T14:26:03Z"

  async def failing_tool():
    raise ValueError("Tool failed")

  mock_model = MockModel.create(
      responses=[
          Part.from_function_call(name="get_current_time", args={}),
          Part.from_function_call(name="failing_tool", args={}),
          Part.from_text(text="Should not reach here"),
      ]
  )
  test_agent = Agent(
      name="error_agent",
      model=mock_model,
      tools=[FunctionTool(get_current_time), FunctionTool(failing_tool)],
  )

  runner = InMemoryRunner(root_agent=test_agent)
  with pytest.raises(ValueError, match="Tool failed"):
    await runner.run_async("Run tools")

  metrics_data = reader.get_metrics_data()
  metrics_list = metrics_data.resource_metrics[0].scope_metrics[0].metrics

  # Verify Tool Execution Duration
  got = _extract_metrics(metrics_list, "gen_ai.tool.execution.duration")
  assert len(got) == 2
  for p in got:
    p.value = None

  want = [
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "error_agent",
              "gen_ai.tool.name": "failing_tool",
              "gen_ai.input.type": "text",
              "error.type": "ValueError",
          },
          value=None,
      ),
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "error_agent",
              "gen_ai.tool.name": "get_current_time",
              "gen_ai.input.type": "text",
              "gen_ai.output.type": "text",
          },
          value=None,
      ),
  ]

  got.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  want.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  assert got == want
