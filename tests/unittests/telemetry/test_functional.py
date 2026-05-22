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

import dataclasses
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import gc
import json
import sys
from typing import Any
from typing import Sequence
from typing import TYPE_CHECKING

from google.adk.agents.llm_agent import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from google.adk.tools import FunctionTool
from google.adk.utils.context_utils import Aclosing
from google.genai import types
from google.genai.types import Content
from google.genai.types import FinishReason
from google.genai.types import Part
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
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

if TYPE_CHECKING:
  from opentelemetry.sdk._logs import ReadableLogRecord
  from opentelemetry.sdk.trace import ReadableSpan
  from opentelemetry.util.types import AttributeValue


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
        or coro.__name__ == "record_inference_telemetry"
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
  client_duration_hist = meter.create_histogram(
      "gen_ai.client.operation.duration"
  )
  client_token_usage_hist = meter.create_histogram("gen_ai.client.token.usage")

  monkeypatch.setattr(
      _metrics, "_agent_invocation_duration", agent_duration_hist
  )
  monkeypatch.setattr(_metrics, "_tool_execution_duration", tool_duration_hist)
  monkeypatch.setattr(_metrics, "_agent_request_size", request_size_hist)
  monkeypatch.setattr(_metrics, "_agent_response_size", response_size_hist)
  monkeypatch.setattr(_metrics, "_agent_workflow_steps", workflow_steps_hist)
  monkeypatch.setattr(
      _metrics, "_client_operation_duration", client_duration_hist
  )
  monkeypatch.setattr(_metrics, "_client_token_usage", client_token_usage_hist)
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
      ],
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=20,
          tool_use_prompt_token_count=5,
          thoughts_token_count=10,
          total_token_count=45,
      ),
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
          },
          value=None,
      ),
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.tool.name": "get_current_time",
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
      # (tool call + result) x 2 + text response = 5 steps
      MetricPoint(attributes={"gen_ai.agent.name": "complex_agent"}, value=5)
  ]
  assert got_steps == want_steps

  got_client_duration = _extract_metrics(
      metrics_list, "gen_ai.client.operation.duration"
  )
  assert len(got_client_duration) == 1
  for p in got_client_duration:
    p.value = None
  want_client_duration = [
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.operation.name": "generate_content",
              "gen_ai.provider.name": "gemini",
              "gen_ai.request.model": "mock",
              "gen_ai.response.model": "mock",
          },
          value=None,
      )
  ]
  assert got_client_duration == want_client_duration

  got_client_tokens = _extract_metrics(
      metrics_list, "gen_ai.client.token.usage"
  )
  assert len(got_client_tokens) == 2
  want_client_tokens = [
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.operation.name": "generate_content",
              "gen_ai.provider.name": "gemini",
              "gen_ai.request.model": "mock",
              "gen_ai.response.model": "mock",
              "gen_ai.token.type": "input",
          },
          value=45,  # 15 tokens * 3 turns
      ),
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "complex_agent",
              "gen_ai.operation.name": "generate_content",
              "gen_ai.provider.name": "gemini",
              "gen_ai.request.model": "mock",
              "gen_ai.response.model": "mock",
              "gen_ai.token.type": "output",
          },
          value=90,  # 30 tokens * 3 turns
      ),
  ]
  got_client_tokens.sort(
      key=lambda p: p.attributes.get("gen_ai.token.type", "")
  )
  want_client_tokens.sort(
      key=lambda p: p.attributes.get("gen_ai.token.type", "")
  )
  assert got_client_tokens == want_client_tokens


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
              "error.type": "ValueError",
          },
          value=None,
      ),
      MetricPoint(
          attributes={
              "gen_ai.agent.name": "error_agent",
              "gen_ai.tool.name": "get_current_time",
          },
          value=None,
      ),
  ]

  got.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  want.sort(key=lambda p: p.attributes.get("gen_ai.tool.name", ""))
  assert got == want


# ==============================================================================
# TELEMETRY SHAPES TEST HELPERS & EXPECTATIONS
# ==============================================================================

OTEL_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"
GEN_AI_SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
GEN_AI_USER_MESSAGE_EVENT = "gen_ai.user.message"
GEN_AI_CHOICE_EVENT = "gen_ai.choice"
GEN_AI_COMPLETION_DETAILS_EVENT = "gen_ai.client.inference.operation.details"

NON_DETERMINISTIC_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gcp.vertex.agent.event_id",
    "gen_ai.tool.call.id",
    "gcp.vertex.agent.associated_event_ids",
    "gen_ai.conversation.id",
    "gcp.vertex.agent.invocation_id",
    "gcp.vertex.agent.session_id",
})
JSON_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
})
PRESENT = "PRESENT"


@dataclass
class LogDigest:
  event_name: str
  body: object = None
  attributes: dict[str, object] = field(default_factory=dict)

  @classmethod
  def from_log(cls, log: ReadableLogRecord) -> LogDigest:
    attrs: dict[str, object] = {}
    for k, v in (log.log_record.attributes or {}).items():
      if k in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        attrs[k] = PRESENT
      else:
        attrs[k] = _normalize(v)
    return cls(
        event_name=log.log_record.event_name or "",
        body=_normalize(log.log_record.body),
        attributes=attrs,
    )


@dataclass
class SpanDigest:
  name: str
  attributes: dict[str, AttributeValue]
  children: list[SpanDigest] = field(default_factory=list)
  logs: list[LogDigest] = field(default_factory=list)

  @classmethod
  def from_span(cls, span: ReadableSpan) -> SpanDigest:
    determinized_attributes: dict[str, AttributeValue] = {}
    for attr_key, attr_val in (span.attributes or {}).items():
      if attr_key in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        determinized_attributes[attr_key] = PRESENT
      elif attr_key in JSON_ATTRIBUTE_KEYS and isinstance(attr_val, str):
        determinized_attributes[attr_key] = _normalize(json.loads(attr_val))
      else:
        determinized_attributes[attr_key] = _normalize(attr_val)
    return cls(name=span.name, attributes=determinized_attributes)

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...] = (),
  ) -> SpanDigest:
    digest_by_id: dict[int, SpanDigest] = {}
    for span in spans:
      if span.context is None:
        continue
      digest_by_id[span.context.span_id] = cls.from_span(span)
    for log in logs:
      span_id = log.log_record.span_id
      if span_id is None or span_id == 0:
        continue
      digest = digest_by_id.get(span_id)
      if digest is None:
        continue
      digest.logs.append(LogDigest.from_log(log))
    root: SpanDigest | None = None
    for span in spans:
      if span.context is None:
        continue
      digest = digest_by_id[span.context.span_id]
      if span.parent and span.parent.span_id in digest_by_id:
        parent_digest = digest_by_id[span.parent.span_id]
        parent_digest.children.append(digest)
      else:
        if root is not None:
          raise ValueError("Multiple root spans found.")
        root = digest
    for digest in digest_by_id.values():
      digest.children.sort(key=lambda s: s.name)
      digest.logs[:] = sorted_log_digests(digest.logs)
    if root is None:
      raise ValueError("No root span found in the provided spans.")
    return root

  def all_logs(self) -> list[LogDigest]:
    collected: list[LogDigest] = []

    def _walk(node: SpanDigest) -> None:
      collected.extend(node.logs)
      for child in node.children:
        _walk(child)

    _walk(self)
    return sorted_log_digests(collected)


def sorted_log_digests(logs: list[LogDigest]) -> list[LogDigest]:
  return sorted(
      logs,
      key=lambda log: (
          log.event_name,
          json.dumps(log.body, sort_keys=True, default=str),
          json.dumps(log.attributes, sort_keys=True, default=str),
      ),
  )


def _normalize(value: object) -> object:
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, tuple):
    return [_normalize(v) for v in value]
  if isinstance(value, list):
    return [_normalize(v) for v in value]
  if isinstance(value, dict):
    return {k: _normalize(v) for k, v in value.items() if v is not None}
  return value


def install_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
) -> None:
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)
  monkeypatch.setattr(
      tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  logger_provider = LoggerProvider()
  logger_provider.add_log_record_processor(
      SimpleLogRecordProcessor(log_exporter)
  )
  real_logger = logger_provider.get_logger(__name__)
  monkeypatch.setattr(tracing.otel_logger, "emit", real_logger.emit)


USER_PROMPT = "hello"
AGENT_NAME = "some_root_agent"
AGENT_DESCRIPTION = "A sample root agent."
BASE_INSTRUCTION = "you are helpful"
FULL_SYSTEM_INSTRUCTION = (
    f"{BASE_INSTRUCTION}\n\n"
    f'You are an agent. Your internal name is "{AGENT_NAME}".'
    f' The description about you is "{AGENT_DESCRIPTION}".'
)
FINAL_TEXT = "text response"
TOOL_NAME = "some_tool"
TOOL_DESCRIPTION = "A sample tool."
TOOL_ARGS = {"arg1": "val1"}
TOOL_RESULT_PREFIX = "processed "
TOOL_RESULT = f"{TOOL_RESULT_PREFIX}{TOOL_ARGS['arg1']}"


def _make_llm_response(part: Part) -> LlmResponse:
  return LlmResponse(
      content=Content(role="model", parts=[part]),
      finish_reason=FinishReason.STOP,
  )


def build_test_agent() -> Agent:
  mock_model = MockModel.create(
      responses=[
          _make_llm_response(
              Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS)
          ),
          _make_llm_response(Part.from_text(text=FINAL_TEXT)),
      ]
  )

  def some_tool(arg1: str) -> str:
    """A sample tool."""
    return f"{TOOL_RESULT_PREFIX}{arg1}"

  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=mock_model,
      tools=[FunctionTool(some_tool)],
  )


def build_test_runner() -> TestInMemoryRunner:
  return TestInMemoryRunner(agent=build_test_agent())


async def run_agent_scenario(runner: TestInMemoryRunner) -> None:
  async with Aclosing(
      runner.run_async_with_new_session_agen(
          Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")
      )
  ) as agen:
    async for _ in agen:
      pass


@dataclass(frozen=True)
class FunctionalTestCase:
  test_id: str
  semconv_opt_in: str | None
  capture_content: str | None
  expected_root: SpanDigest

  def apply_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
    if self.semconv_opt_in is None:
      monkeypatch.delenv(OTEL_OPT_IN, raising=False)
    else:
      monkeypatch.setenv(OTEL_OPT_IN, self.semconv_opt_in)
    if self.capture_content is None:
      monkeypatch.delenv(CAPTURE_CONTENT, raising=False)
    else:
      monkeypatch.setenv(CAPTURE_CONTENT, self.capture_content)
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")


EXPECTED_STABLE_NO_CAPTURE = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": "<elided>",
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": "<elided>",
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={"content": "<elided>"},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

EXPECTED_STABLE_CAPTURE = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_call": {
                                                    "args": TOOL_ARGS,
                                                    "name": TOOL_NAME,
                                                }
                                            }],
                                            "role": "model",
                                        },
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": FULL_SYSTEM_INSTRUCTION},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": USER_PROMPT}],
                                            "role": "user",
                                        }
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.system": "gemini",
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_CHOICE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": FINAL_TEXT}],
                                            "role": "model",
                                        },
                                        "index": 0,
                                        "finish_reason": "STOP",
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_SYSTEM_MESSAGE_EVENT,
                                    body={"content": FULL_SYSTEM_INSTRUCTION},
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_call": {
                                                    "args": TOOL_ARGS,
                                                    "name": TOOL_NAME,
                                                }
                                            }],
                                            "role": "model",
                                        }
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{
                                                "function_response": {
                                                    "name": TOOL_NAME,
                                                    "response": {
                                                        "result": TOOL_RESULT
                                                    },
                                                }
                                            }],
                                            "role": "user",
                                        }
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                                LogDigest(
                                    event_name=GEN_AI_USER_MESSAGE_EVENT,
                                    body={
                                        "content": {
                                            "parts": [{"text": USER_PROMPT}],
                                            "role": "user",
                                        }
                                    },
                                    attributes={"gen_ai.system": "gemini"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

EXPECTED_EXPERIMENTAL_NO_CONTENT = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [{
                                    "name": TOOL_NAME,
                                    "description": TOOL_DESCRIPTION,
                                    "type": "function",
                                }],
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [{
                                            "name": TOOL_NAME,
                                            "description": TOOL_DESCRIPTION,
                                            "type": "function",
                                        }],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [{
                                    "name": TOOL_NAME,
                                    "description": TOOL_DESCRIPTION,
                                    "type": "function",
                                }],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [{
                                            "name": TOOL_NAME,
                                            "description": TOOL_DESCRIPTION,
                                            "type": "function",
                                        }],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

_TOOL_DEFINITION_FULL = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "parameters": {
        "properties": {"arg1": {"type": "STRING"}},
        "required": ["arg1"],
        "type": "OBJECT",
    },
    "type": "function",
}
_TOOL_DEFINITION_NO_CONTENT = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "type": "function",
}
_SYSTEM_INSTRUCTIONS = [{"content": FULL_SYSTEM_INSTRUCTION, "type": "text"}]
_TURN_1_INPUT_MESSAGES = [{
    "role": "user",
    "parts": [{"content": USER_PROMPT, "type": "text"}],
}]
_TURN_1_OUTPUT_MESSAGES = [{
    "role": "assistant",
    "parts": [{
        "id": f"{TOOL_NAME}_0",
        "name": TOOL_NAME,
        "arguments": TOOL_ARGS,
        "type": "tool_call",
    }],
    "finish_reason": "stop",
}]
_TURN_2_INPUT_MESSAGES = [
    {
        "role": "user",
        "parts": [{"content": USER_PROMPT, "type": "text"}],
    },
    {
        "role": "assistant",
        "parts": [{
            "id": f"{TOOL_NAME}_0",
            "name": TOOL_NAME,
            "arguments": TOOL_ARGS,
            "type": "tool_call",
        }],
    },
    {
        "role": "user",
        "parts": [{
            "id": f"{TOOL_NAME}_0",
            "response": {"result": TOOL_RESULT},
            "type": "tool_call_response",
        }],
    },
]
_TURN_2_OUTPUT_MESSAGES = [{
    "role": "assistant",
    "parts": [{"content": FINAL_TEXT, "type": "text"}],
    "finish_reason": "stop",
}]

EXPECTED_EXPERIMENTAL_SPAN_ONLY = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_1_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_1_OUTPUT_MESSAGES
                                ),
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_NO_CONTENT
                                        ],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_2_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_2_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_NO_CONTENT
                                        ],
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

EXPECTED_EXPERIMENTAL_EVENT_ONLY = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_NO_CONTENT
                                ],
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_1_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_1_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_NO_CONTENT
                                ],
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_2_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_2_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT = SpanDigest(
    name="invocation",
    attributes={},
    children=[
        SpanDigest(
            name="invoke_agent some_root_agent",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.description": AGENT_DESCRIPTION,
                "gen_ai.agent.name": AGENT_NAME,
                "gen_ai.conversation.id": PRESENT,
            },
            children=[
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_1_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_1_OUTPUT_MESSAGES
                                ),
                            },
                            children=[
                                SpanDigest(
                                    name="execute_tool some_tool",
                                    attributes={
                                        "gen_ai.operation.name": "execute_tool",
                                        "gen_ai.tool.description": (
                                            TOOL_DESCRIPTION
                                        ),
                                        "gen_ai.tool.name": TOOL_NAME,
                                        "gen_ai.tool.type": "FunctionTool",
                                        "gcp.vertex.agent.llm_request": "{}",
                                        "gcp.vertex.agent.llm_response": "{}",
                                        "gcp.vertex.agent.tool_call_args": "{}",
                                        "gen_ai.tool.call.id": PRESENT,
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.tool_response": "{}",
                                    },
                                ),
                            ],
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_1_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_1_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                SpanDigest(
                    name="call_llm",
                    attributes={
                        "gen_ai.system": "gcp.vertex.agent",
                        "gen_ai.request.model": "mock",
                        "gcp.vertex.agent.invocation_id": PRESENT,
                        "gcp.vertex.agent.session_id": PRESENT,
                        "gcp.vertex.agent.event_id": PRESENT,
                        "gcp.vertex.agent.llm_request": "{}",
                        "gcp.vertex.agent.llm_response": "{}",
                        "gen_ai.response.finish_reasons": ["stop"],
                    },
                    children=[
                        SpanDigest(
                            name="generate_content mock",
                            attributes={
                                "gen_ai.operation.name": "generate_content",
                                "gen_ai.request.model": "mock",
                                "gen_ai.agent.name": AGENT_NAME,
                                "gen_ai.conversation.id": PRESENT,
                                "user.id": "test_user",
                                "gcp.vertex.agent.event_id": PRESENT,
                                "gcp.vertex.agent.invocation_id": PRESENT,
                                "gen_ai.response.finish_reasons": ["stop"],
                                "gen_ai.input.messages": _TURN_2_INPUT_MESSAGES,
                                "gen_ai.system_instructions": (
                                    _SYSTEM_INSTRUCTIONS
                                ),
                                "gen_ai.tool.definitions": [
                                    _TOOL_DEFINITION_FULL
                                ],
                                "gen_ai.output.messages": (
                                    _TURN_2_OUTPUT_MESSAGES
                                ),
                            },
                            logs=[
                                LogDigest(
                                    event_name=GEN_AI_COMPLETION_DETAILS_EVENT,
                                    body=None,
                                    attributes={
                                        "gen_ai.agent.name": AGENT_NAME,
                                        "gen_ai.conversation.id": PRESENT,
                                        "user.id": "test_user",
                                        "gcp.vertex.agent.event_id": PRESENT,
                                        "gcp.vertex.agent.invocation_id": (
                                            PRESENT
                                        ),
                                        "gen_ai.response.finish_reasons": [
                                            "stop"
                                        ],
                                        "gen_ai.input.messages": (
                                            _TURN_2_INPUT_MESSAGES
                                        ),
                                        "gen_ai.system_instructions": (
                                            _SYSTEM_INSTRUCTIONS
                                        ),
                                        "gen_ai.tool.definitions": [
                                            _TOOL_DEFINITION_FULL
                                        ],
                                        "gen_ai.output.messages": (
                                            _TURN_2_OUTPUT_MESSAGES
                                        ),
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

ALL_CASES: list[FunctionalTestCase] = [
    FunctionalTestCase(
        test_id="stable-no-capture",
        semconv_opt_in=None,
        capture_content="false",
        expected_root=EXPECTED_STABLE_NO_CAPTURE,
    ),
    FunctionalTestCase(
        test_id="stable-capture",
        semconv_opt_in=None,
        capture_content="true",
        expected_root=EXPECTED_STABLE_CAPTURE,
    ),
    FunctionalTestCase(
        test_id="experimental-no-content",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="no_content",
        expected_root=EXPECTED_EXPERIMENTAL_NO_CONTENT,
    ),
    FunctionalTestCase(
        test_id="experimental-span-only",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="span_only",
        expected_root=EXPECTED_EXPERIMENTAL_SPAN_ONLY,
    ),
    FunctionalTestCase(
        test_id="experimental-event-only",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="event_only",
        expected_root=EXPECTED_EXPERIMENTAL_EVENT_ONLY,
    ),
    FunctionalTestCase(
        test_id="experimental-span-and-event",
        semconv_opt_in=EXPERIMENTAL_OPT_IN,
        capture_content="span_and_event",
        expected_root=EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=[c.test_id for c in ALL_CASES],
)
async def test_telemetry_shapes(
    case: FunctionalTestCase,
    monkeypatch: pytest.MonkeyPatch,
):
  # Arrange
  case.apply_env(monkeypatch)
  span_exporter = InMemorySpanExporter()
  log_exporter = InMemoryLogRecordExporter()
  install_telemetry(monkeypatch, span_exporter, log_exporter)
  runner = build_test_runner()

  # Act
  await run_agent_scenario(runner)

  # Assert
  spans = span_exporter.get_finished_spans()
  logs = log_exporter.get_finished_logs()
  root_digest = SpanDigest.build(tuple(spans), tuple(logs))

  assert root_digest == case.expected_root
