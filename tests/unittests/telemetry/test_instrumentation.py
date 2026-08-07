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

# pylint: disable=protected-access

import time
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow._workflow import Workflow
from google.genai import types
from opentelemetry import trace
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import pytest

from .functional_test_helpers import install_telemetry


def test_get_elapsed_s_span_none():
  """Tests fallback when span is None."""
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(None, start_time)
  assert elapsed == 2.0  # 12 - 10


def test_get_elapsed_s_span_valid():
  """Tests duration calculation with valid span times."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000  # 1s in ns
  mock_span.end_time = 2000000000  # 2s in ns
  elapsed = _metrics.get_elapsed_s(mock_span, time.monotonic())
  assert elapsed == 1.0  # (2 - 1) s


def test_get_elapsed_s_span_missing_start():
  """Tests fallback when start_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  del mock_span.start_time
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_missing_end():
  """Tests fallback when end_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  del mock_span.end_time
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_non_int_start():
  """Tests fallback when start_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000.0
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_non_int_end():
  """Tests fallback when end_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  mock_span.end_time = 2000000000.0
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


@pytest.mark.asyncio
async def test_record_tool_execution_forwards_detected_error_type():
  """A failure detected in the tool response reaches the duration metric."""
  tool = mock.MagicMock()
  tool.name = "sample_tool"
  agent = mock.MagicMock()
  agent.name = "sample_agent"

  with mock.patch.object(
      _metrics, "record_tool_execution_duration"
  ) as mock_record:
    async with _instrumentation.record_tool_execution(
        tool=tool,
        agent=agent,
        function_args={},
        invocation_context=mock.MagicMock(),
    ) as tel_ctx:
      tel_ctx.error_type = "MCP_TOOL_ERROR"

  mock_record.assert_called_once()
  assert mock_record.call_args.kwargs["error"] is None
  assert mock_record.call_args.kwargs["error_type"] == "MCP_TOOL_ERROR"


# ---------------------------------------------------------------------------
# The consolidated span + metric context managers.
#
# These own both a span and the metrics derived from it, so the assertions
# below run against an in-memory span exporter / metric reader rather than
# mocks: a mock cannot show that the span was actually ended, nor that the
# metric attributes and the span attributes agree.
# ---------------------------------------------------------------------------

# Env vars that change what these context managers emit. Cleared per test so
# an ambient value cannot silently rewrite the expected shape.
_TELEMETRY_ENV_VARS = (
    "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN",
    "ADK_TELEMETRY_IGNORE_RUN_CONFIG",
    "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS",
    "OTEL_SEMCONV_STABILITY_OPT_IN",
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


class _Telemetry:
  """Reader over the in-memory span/metric sinks installed for one test."""

  def __init__(
      self,
      span_exporter: InMemorySpanExporter,
      metric_reader: InMemoryMetricReader,
  ):
    self._span_exporter = span_exporter
    self._metric_reader = metric_reader
    self._points = None

  def spans(self):
    """Every span finished so far, in completion order."""
    return list(self._span_exporter.get_finished_spans())

  def only_span(self):
    """The single span the block under test is expected to have produced."""
    spans = self.spans()
    assert len(spans) == 1, [span.name for span in spans]
    return spans[0]

  def points(self, metric_name: str):
    """``(attributes, recorded sum)`` for each point of ``metric_name``."""
    if self._points is None:
      self._points = {}
      data = self._metric_reader.get_metrics_data()
      for resource_metric in data.resource_metrics if data else ():
        for scope_metric in resource_metric.scope_metrics:
          for metric in scope_metric.metrics:
            for point in metric.data.data_points:
              self._points.setdefault(metric.name, []).append(
                  (dict(point.attributes), point.sum)
              )
    return self._points.get(metric_name, [])

  def point_attributes(self, metric_name: str):
    """Just the attribute sets, for metrics whose value is a wall-clock time."""
    return [attributes for attributes, _ in self.points(metric_name)]


@pytest.fixture(name="telemetry")
def _telemetry_fixture(monkeypatch: pytest.MonkeyPatch) -> _Telemetry:
  """Redirects ADK spans and metric histograms into in-memory sinks."""
  for name in _TELEMETRY_ENV_VARS:
    monkeypatch.delenv(name, raising=False)
  # The genai instrumentation library, when active, takes over the inference
  # span; pin it off so the tests exercise ADK's own path.
  monkeypatch.setattr(
      "google.adk.telemetry.tracing._instrumented_with_opentelemetry_instrumentation_google_genai",
      lambda: False,
  )
  span_exporter = InMemorySpanExporter()
  metric_reader = InMemoryMetricReader()
  install_telemetry(
      monkeypatch, span_exporter, InMemoryLogRecordExporter(), metric_reader
  )
  return _Telemetry(span_exporter, metric_reader)


class _EchoTool(BaseTool):
  """A tool that needs no external service to execute."""

  async def run_async(
      self, *, args: dict[str, object], tool_context: ToolContext
  ) -> object:
    return args


def _agent(name: str = "root_agent", description: str = "") -> LlmAgent:
  # A non-Gemini model keeps `_should_emit_native_telemetry` true regardless of
  # whether the genai instrumentation library happens to be installed.
  return LlmAgent(
      name=name, model="not-a-gemini-model", description=description
  )


async def _invocation_context(agent: LlmAgent) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  return InvocationContext(
      invocation_id="test_invocation_id",
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(),
  )


def _function_response_event(
    call_id: str, response: dict[str, object]
) -> Event:
  return Event(
      author="root_agent",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=call_id, name="echo", response=response
                  )
              )
          ],
      ),
  )


# --- record_agent_invocation ----------------------------------------------


@pytest.mark.asyncio
async def test_record_agent_invocation_opens_named_invoke_agent_span(
    telemetry: _Telemetry,
):
  """The span is named after the agent and carries exactly the semconv

  invoke_agent attribute set.
  """
  agent = _agent(description="the root agent")
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent):
    pass

  span = telemetry.only_span()
  assert span.name == "invoke_agent root_agent"
  assert dict(span.attributes) == {
      "gen_ai.operation.name": "invoke_agent",
      "gen_ai.agent.description": "the root agent",
      "gen_ai.agent.name": "root_agent",
      "gen_ai.conversation.id": ctx.session.id,
  }
  assert span.end_time is not None


@pytest.mark.asyncio
async def test_record_agent_invocation_closes_span_and_labels_the_error(
    telemetry: _Telemetry,
):
  """A failing body must still end the span, and the duration metric must be

  attributed to the error rather than silently counted as a success.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)

  with pytest.raises(ValueError, match="agent blew up"):
    async with _instrumentation.record_agent_invocation(ctx, agent):
      raise ValueError("agent blew up")

  span = telemetry.only_span()
  assert span.name == "invoke_agent root_agent"
  assert span.end_time is not None
  assert span.status.status_code is StatusCode.ERROR
  assert telemetry.point_attributes("gen_ai.invoke_agent.duration") == [
      {"gen_ai.agent.name": "root_agent", "error.type": "ValueError"}
  ]


@pytest.mark.asyncio
async def test_record_agent_invocation_flushes_inference_and_tool_counts(
    telemetry: _Telemetry,
):
  """The per-invocation counters are flushed to their own instruments on exit,

  each keyed only by agent name.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as tel_ctx:
    tel_ctx.increment_inference_calls()
    tel_ctx.increment_inference_calls()
    tel_ctx.increment_tool_calls()

  assert telemetry.points("gen_ai.invoke_agent.inference_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 2)
  ]
  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]


@pytest.mark.asyncio
async def test_record_agent_invocation_flushes_counts_even_when_body_fails(
    telemetry: _Telemetry,
):
  """The counters accumulated before a failure are not lost."""
  agent = _agent()
  ctx = await _invocation_context(agent)

  with pytest.raises(ValueError):
    async with _instrumentation.record_agent_invocation(ctx, agent) as tel_ctx:
      tel_ctx.increment_tool_calls()
      raise ValueError("agent blew up")

  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]


@pytest.mark.asyncio
async def test_record_agent_invocation_counts_a_nested_tool_execution(
    telemetry: _Telemetry,
):
  """A tool executed inside the agent block is counted against that agent: the

  two context managers find each other through the OTel context, not through
  an argument.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_agent_invocation(ctx, agent):
    async with _instrumentation.record_tool_execution(tool, agent, {}, ctx):
      pass

  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]


@pytest.mark.asyncio
async def test_record_tool_execution_outside_an_agent_span_counts_nothing(
    telemetry: _Telemetry,
):
  """With no active invoke_agent span there is nothing to count against, and

  the tool call must not blow up looking for one.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_tool_execution(tool, agent, {}, ctx):
    pass

  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == []


# --- record_tool_execution -------------------------------------------------


@pytest.mark.asyncio
async def test_record_tool_execution_opens_named_execute_tool_span(
    telemetry: _Telemetry,
):
  """The span is named after the tool and carries the tool identity, the

  arguments, and the response the caller handed back on the context.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_tool_execution(
      tool, agent, {"text": "hi"}, ctx
  ) as tel_ctx:
    tel_ctx.function_response_event = _function_response_event(
        "call-1", {"out": "hi"}
    )

  span = telemetry.only_span()
  assert span.name == "execute_tool echo"
  attributes = dict(span.attributes)
  assert attributes["gen_ai.operation.name"] == "execute_tool"
  assert attributes["gen_ai.tool.name"] == "echo"
  assert attributes["gen_ai.tool.description"] == "echoes its input"
  assert attributes["gen_ai.tool.type"] == "_EchoTool"
  assert attributes["gen_ai.agent.name"] == "root_agent"
  assert attributes["gen_ai.tool.call.id"] == "call-1"
  assert attributes["gcp.vertex.agent.tool_call_args"] == '{"text": "hi"}'
  assert attributes["gcp.vertex.agent.tool_response"] == '{"out": "hi"}'
  assert "error.type" not in attributes
  assert span.end_time is not None


@pytest.mark.asyncio
async def test_record_tool_execution_records_duration_keyed_by_tool_and_agent(
    telemetry: _Telemetry,
):
  """The duration instrument is dimensioned by agent, tool name and tool

  class -- the class, not the instance name, is what distinguishes tool
  kinds.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_tool_execution(tool, agent, {}, ctx):
    pass

  assert telemetry.point_attributes("gen_ai.execute_tool.duration") == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.tool.name": "echo",
      "gen_ai.tool.type": "_EchoTool",
  }]


@pytest.mark.asyncio
async def test_record_tool_execution_failure_labels_error_and_drops_response(
    telemetry: _Telemetry,
):
  """When the tool raises, the span and the metric both carry the error type,

  and any response event left on the context is discarded: it did not come
  from a completed call, so stamping it would report a success that never
  happened.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  with pytest.raises(ValueError, match="tool blew up"):
    async with _instrumentation.record_tool_execution(
        tool, agent, {}, ctx
    ) as tel_ctx:
      tel_ctx.function_response_event = _function_response_event(
          "call-1", {"out": "hi"}
      )
      raise ValueError("tool blew up")

  span = telemetry.only_span()
  attributes = dict(span.attributes)
  assert span.end_time is not None
  assert attributes["error.type"] == "ValueError"
  assert attributes["gen_ai.tool.call.id"] == "<not specified>"
  assert "gcp.vertex.agent.event_id" not in attributes
  assert telemetry.point_attributes("gen_ai.execute_tool.duration") == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.tool.name": "echo",
      "gen_ai.tool.type": "_EchoTool",
      "error.type": "ValueError",
  }]


@pytest.mark.asyncio
async def test_record_tool_execution_reported_error_labels_span_and_metric(
    telemetry: _Telemetry,
):
  """A tool that reports an error instead of raising labels both signals.

  Setting ``error_type`` on the context is the only signal available when no
  exception propagates out of the call, so the span and the duration metric
  have to agree. A metric that recorded the call as a success would hide the
  failure from any error-rate view built on it.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_tool_execution(
      tool, agent, {}, ctx
  ) as tel_ctx:
    tel_ctx.error_type = "HTTP_ERROR"

  assert dict(telemetry.only_span().attributes)["error.type"] == "HTTP_ERROR"
  assert telemetry.point_attributes("gen_ai.execute_tool.duration") == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.tool.name": "echo",
      "gen_ai.tool.type": "_EchoTool",
      "error.type": "HTTP_ERROR",
  }]


# --- record_inference_telemetry + TelemetryContext.record_llm_response ------


def _llm_response(**overrides) -> LlmResponse:
  defaults = dict(
      content=types.Content(role="model", parts=[types.Part(text="yo")]),
      finish_reason=types.FinishReason.STOP,
      model_version="some-model-001",
      usage_metadata=types.GenerateContentResponseUsageMetadata(
          prompt_token_count=10,
          candidates_token_count=4,
          thoughts_token_count=1,
      ),
  )
  defaults.update(overrides)
  return LlmResponse(**defaults)


@pytest.mark.asyncio
async def test_record_inference_telemetry_opens_generate_content_span(
    telemetry: _Telemetry,
):
  """The inference span is named for the requested model and carries the

  result recorded through the yielded context.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(
      model="some-model",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      llm_request, ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(ctx, _llm_response())

  span = telemetry.only_span()
  assert span.name == "generate_content some-model"
  attributes = dict(span.attributes)
  assert attributes["gen_ai.operation.name"] == "generate_content"
  assert attributes["gen_ai.request.model"] == "some-model"
  assert attributes["gen_ai.agent.name"] == "root_agent"
  assert attributes["gcp.vertex.agent.event_id"] == "event-1"
  assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
  # input = prompt + tool-use tokens; output = candidates + thoughts tokens.
  assert attributes["gen_ai.usage.input_tokens"] == 10
  assert attributes["gen_ai.usage.output_tokens"] == 5
  assert span.end_time is not None


@pytest.mark.asyncio
async def test_record_inference_telemetry_records_token_usage_per_direction(
    telemetry: _Telemetry,
):
  """Token usage is reported as one point per direction, sharing the same

  request/response model dimensions.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(model="some-model")
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      llm_request, ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(ctx, _llm_response())

  shared = {
      "gen_ai.agent.name": "root_agent",
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gemini",
      "gen_ai.request.model": "some-model",
      "gen_ai.response.model": "some-model-001",
  }
  by_direction = {
      attributes["gen_ai.token.type"]: (attributes, value)
      for attributes, value in telemetry.points("gen_ai.client.token.usage")
  }
  assert by_direction == {
      "input": (shared | {"gen_ai.token.type": "input"}, 10),
      "output": (shared | {"gen_ai.token.type": "output"}, 5),
  }
  assert telemetry.point_attributes("gen_ai.client.operation.duration") == [
      shared
  ]


@pytest.mark.asyncio
async def test_record_inference_telemetry_without_a_response_skips_token_usage(
    telemetry: _Telemetry,
):
  """No response means no usage metadata to report; the operation duration is

  still recorded so the call is not invisible.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(model="some-model")
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      llm_request, ctx, model_response_event
  ):
    pass

  assert telemetry.points("gen_ai.client.token.usage") == []
  assert telemetry.point_attributes("gen_ai.client.operation.duration") == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gemini",
      "gen_ai.request.model": "some-model",
  }]


@pytest.mark.asyncio
async def test_record_inference_telemetry_failure_labels_operation_duration(
    telemetry: _Telemetry,
):
  """A failing inference is attributed to the error on the duration metric."""
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(model="some-model")
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  with pytest.raises(ValueError, match="model blew up"):
    async with _instrumentation.record_inference_telemetry(
        llm_request, ctx, model_response_event
    ):
      raise ValueError("model blew up")

  assert telemetry.point_attributes("gen_ai.client.operation.duration") == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gemini",
      "gen_ai.request.model": "some-model",
      "error.type": "ValueError",
  }]


@pytest.mark.asyncio
async def test_record_llm_response_keeps_every_response_in_arrival_order(
    telemetry: _Telemetry,
):
  """Token usage is read off the last response on the assumption that

  streaming usage is cumulative, so both retention and order matter.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tel_ctx = _instrumentation.TelemetryContext()
  first = _llm_response(partial=True, finish_reason=None)
  second = _llm_response()

  with tracing.tracer.start_as_current_span("test_span") as span:
    tel_ctx.span = span
    tel_ctx.record_llm_response(ctx, first)
    tel_ctx.record_llm_response(ctx, second)

  assert tel_ctx.llm_responses == [first, second]


@pytest.mark.asyncio
async def test_record_llm_response_traces_the_result_onto_the_carried_span(
    telemetry: _Telemetry,
):
  """Recording a response also stamps its outcome on the span the context is

  carrying, which is how the inference span learns its finish reason.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tel_ctx = _instrumentation.TelemetryContext()

  with tracing.tracer.start_as_current_span("test_span") as span:
    tel_ctx.span = span
    tel_ctx.record_llm_response(ctx, _llm_response())

  attributes = dict(telemetry.only_span().attributes)
  assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
  assert attributes["gen_ai.usage.input_tokens"] == 10
  assert attributes["gen_ai.usage.output_tokens"] == 5


# --- record_invocation -----------------------------------------------------


def test_record_invocation_legacy_schema_emits_the_invocation_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Schema v1 keeps the bare, attribute-free ``invocation`` span."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")

  with _instrumentation.record_invocation(_agent(), "conversation-1"):
    pass

  span = telemetry.only_span()
  assert span.name == "invocation"
  assert dict(span.attributes or {}) == {}
  assert telemetry.point_attributes("gen_ai.invoke_workflow.duration") == []


def test_record_invocation_semconv_schema_emits_entrypoint_workflow_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Schema v2 replaces it with an entrypoint ``invoke_workflow`` span named

  for the entrypoint, plus a matching duration metric. Being the root, it
  omits the nested flag entirely on both.
  """
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")

  with _instrumentation.record_invocation(_agent(), "conversation-1"):
    pass

  span = telemetry.only_span()
  assert span.name == "invoke_workflow root_agent"
  assert dict(span.attributes) == {
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.conversation.id": "conversation-1",
      "gen_ai.workflow.name": "root_agent",
  }
  assert telemetry.point_attributes("gen_ai.invoke_workflow.duration") == [{
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.workflow.name": "root_agent",
  }]


def test_record_invocation_without_an_entrypoint_omits_the_workflow_name(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """With nothing to name the entrypoint after, the span falls back to the

  bare operation name rather than a name with an empty suffix.
  """
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")

  with _instrumentation.record_invocation(None, "conversation-1"):
    pass

  span = telemetry.only_span()
  assert span.name == "invoke_workflow"
  assert "gen_ai.workflow.name" not in span.attributes


def test_record_invocation_defers_to_a_workflow_entrypoints_own_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A workflow entrypoint opens its own ``invoke_workflow`` span when the

  node runs, so opening one here too would double-count the invocation.
  """
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")

  with _instrumentation.record_invocation(Workflow(name="my_workflow"), "c-1"):
    pass

  assert telemetry.spans() == []
  assert telemetry.point_attributes("gen_ai.invoke_workflow.duration") == []
