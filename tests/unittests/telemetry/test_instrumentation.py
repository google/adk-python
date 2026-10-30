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

import asyncio
import contextlib
import dataclasses
import gc
import logging
import time
from typing import AsyncGenerator
from typing import Iterator
from unittest import mock

from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.events.event import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry import _hallucination
from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode
import pytest

from .functional.scenarios.telemetry_setup import install_telemetry


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


@pytest.mark.asyncio
async def test_record_skill_load_reaches_the_enclosing_tool_execution():
  """A skill load is reported to the tool execution that wraps it."""
  tool = mock.MagicMock()
  tool.name = "load_skill"
  agent = mock.MagicMock()
  agent.name = "sample_agent"

  async with _instrumentation.record_tool_execution(
      tool=tool,
      agent=agent,
      function_args={},
      invocation_context=mock.MagicMock(),
  ) as tel_ctx:
    skill_telemetry = _instrumentation.track_skill_load(
        _hallucination.MaybeHallucinated("sample_skill")
    )
    skill_telemetry.skill = mock.MagicMock()
    skill_telemetry.skill_name = _hallucination.ConfirmedNotHallucinated(
        skill_telemetry.skill_name.maybe_hallucinated_value
    )

  assert isinstance(
      tel_ctx.skill_telemetry, _instrumentation.SkillLoadTelemetry
  )
  assert tel_ctx.skill_telemetry == skill_telemetry


@pytest.mark.asyncio
async def test_record_skill_resource_load_reaches_the_enclosing_tool_execution():
  """A skill resource load is reported to the tool execution that wraps it."""
  tool = mock.MagicMock()
  tool.name = "load_skill_resource"
  agent = mock.MagicMock()
  agent.name = "sample_agent"

  async with _instrumentation.record_tool_execution(
      tool=tool,
      agent=agent,
      function_args={},
      invocation_context=mock.MagicMock(),
  ) as tel_ctx:
    skill_telemetry = _instrumentation.track_skill_resource_load(
        _hallucination.MaybeHallucinated("sample_skill"),
        _hallucination.MaybeHallucinated("sample_path"),
    )
    skill_telemetry.skill_name = _hallucination.ConfirmedNotHallucinated(
        skill_telemetry.skill_name.maybe_hallucinated_value
    )
    skill_telemetry.resource_path = _hallucination.ConfirmedNotHallucinated(
        skill_telemetry.resource_path.maybe_hallucinated_value
    )

  assert isinstance(
      tel_ctx.skill_telemetry, _instrumentation.SkillResourceLoadTelemetry
  )
  assert tel_ctx.skill_telemetry == skill_telemetry


@pytest.mark.asyncio
async def test_record_skill_script_execution_reaches_the_enclosing_tool_execution():
  """A skill script run is reported to the tool execution that wraps it."""
  tool = mock.MagicMock()
  tool.name = "run_skill_script"
  agent = mock.MagicMock()
  agent.name = "sample_agent"

  async with _instrumentation.record_tool_execution(
      tool=tool,
      agent=agent,
      function_args={},
      invocation_context=mock.MagicMock(),
  ) as tel_ctx:
    skill_telemetry = _instrumentation.track_skill_script_execution(
        _hallucination.MaybeHallucinated("sample_skill"),
        _hallucination.MaybeHallucinated("scripts/sample.py"),
    )
    # The exit code is only known once the script has run, so the tool keeps
    # the object the tracker handed it and fills this in afterwards.
    skill_telemetry.script_exit_code = 0

  assert isinstance(
      tel_ctx.skill_telemetry, _instrumentation.SkillScriptExecutionTelemetry
  )
  assert tel_ctx.skill_telemetry == skill_telemetry


def test_record_skill_load_outside_tool_execution_is_a_noop():
  """Callers never depend on a tool execution (and thus a span) being open."""
  _instrumentation.track_skill_load(
      _hallucination.MaybeHallucinated("sample_skill")
  )

  assert _instrumentation._active_tool_execution_tel_ctx() is None


def test_record_skill_script_execution_outside_tool_execution_is_a_noop():
  """The tool still gets an object to record the exit code on."""
  skill_telemetry = _instrumentation.track_skill_script_execution(
      _hallucination.MaybeHallucinated("sample_skill"),
      _hallucination.MaybeHallucinated("scripts/sample.py"),
  )

  assert _instrumentation._active_tool_execution_tel_ctx() is None
  # Unconfirmed until the tool resolves them: nothing has looked either up yet.
  assert skill_telemetry.skill_name == _hallucination.MaybeHallucinated(
      "sample_skill"
  )
  assert skill_telemetry.script_path == _hallucination.MaybeHallucinated(
      "scripts/sample.py"
  )


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
    "ADK_EXPERIMENTAL_TELEMETRY",
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
    """``(attributes, recorded total)`` for each point of ``metric_name``.

    A histogram point carries its total as ``sum`` and a counter point as
    ``value``; both mean the same thing here.
    """
    if self._points is None:
      self._points = {}
      data = self._metric_reader.get_metrics_data()
      for resource_metric in data.resource_metrics if data else ():
        for scope_metric in resource_metric.scope_metrics:
          for metric in scope_metric.metrics:
            for point in metric.data.data_points:
              total = getattr(point, "sum", None)
              if total is None:
                total = getattr(point, "value", None)
              self._points.setdefault(metric.name, []).append(
                  (dict(point.attributes), total)
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
      "google.adk.telemetry._genai_instrumentor.is_instrumented",
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


async def _invocation_context(
    agent: LlmAgent, run_config: RunConfig | None = None
) -> InvocationContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  return InvocationContext(
      invocation_id="test_invocation_id",
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=run_config or RunConfig(),
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

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
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

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    scope.inference_call_count += 2
    scope.tool_call_count += 1

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
    async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
      scope.tool_call_count += 1
      raise ValueError("agent blew up")

  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]


@pytest.mark.asyncio
async def test_record_agent_invocation_without_the_opt_in_emits_no_tokens(
    telemetry: _Telemetry,
):
  """A run that never opted in sees no token metrics, not even a zero one."""
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      pass

  assert not telemetry.points("adk.experimental.invoke_agent.total_tokens")
  # The call counts are stable, so they report regardless.
  assert telemetry.points("gen_ai.invoke_agent.inference_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 0)
  ]


@pytest.mark.asyncio
async def test_record_agent_invocation_that_reported_no_usage_emits_no_tokens(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """An opted-in invocation no model reported usage for emits no token metrics.

  Nothing reported and a reported zero are different answers, and the totals
  sit at zero for both, so the flush goes by whether usage ever arrived.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      pass

  assert not telemetry.points("adk.experimental.invoke_agent.total_tokens")


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

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
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


@pytest.mark.asyncio
async def test_record_agent_invocation_reads_nothing_off_the_context(
    telemetry: _Telemetry,
):
  """An invocation reads only what the span needs off the context.

  Callers drive agents with their own lightweight context objects, and this
  runs on every invocation, including ones that call no tool and no model. A
  new read here, `run_config` say, would take down the run that telemetry only
  observes.
  """
  agent = _agent()

  class _Session:
    id = "session-id"

  class _StandInContext:
    """What a caller's hand-rolled context looks like: a session, no more."""

    session = _Session()

  async with _instrumentation.record_agent_invocation(_StandInContext(), agent):
    pass

  assert telemetry.only_span().name == "invoke_agent root_agent"
  assert telemetry.points("gen_ai.invoke_agent.inference_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 0)
  ]


# --- record_tool_execution -------------------------------------------------


@pytest.mark.asyncio
async def test_record_tool_execution_opens_named_execute_tool_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The span is named after the tool and carries the tool identity, the

  arguments, and the response the caller handed back on the context.
  """
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("experimental", "expected_source"),
    [(False, None), (True, "before_tool_callback")],
)
async def test_record_tool_execution_response_source_needs_experimental(
    telemetry: _Telemetry,
    monkeypatch: pytest.MonkeyPatch,
    experimental: bool,
    expected_source: str | None,
):
  """The response source is experimental telemetry, off by default."""
  if experimental:
    monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)
  tool = _EchoTool(name="echo", description="echoes its input")

  async with _instrumentation.record_tool_execution(
      tool, agent, {}, ctx
  ) as tel_ctx:
    tel_ctx.function_response_event = _function_response_event(
        "call-1", {"out": "hi"}
    )
    tel_ctx.response_source = "before_tool_callback"

  attributes = dict(telemetry.only_span().attributes)
  assert attributes.get("adk.experimental.response.source") == expected_source


# --- skill script execution, on the execute_tool span ----------------------
#
# The exit code is only known after the script has run, so the tool fills it
# in on the object the tracker handed it and the tool execution block stamps
# both signals as it closes. These run through that block rather than calling
# the tracing helper directly: the point is that the span and the counter end
# up agreeing about the same run.

_SKILL_SCRIPT_EXECUTIONS = "adk.experimental.skill.script.executions"


def _script_tool() -> _EchoTool:
  return _EchoTool(name="run_skill_script", description="runs a skill script")


def _loaded_skill(uri: str | None = "file:/skills/sample") -> mock.MagicMock:
  """A stand-in for a loaded skill: its source URI, description and metadata."""
  skill = mock.MagicMock()
  skill._uri = uri
  skill.description = "A sample skill."
  skill.frontmatter.metadata = {}
  return skill


@pytest.mark.asyncio
async def test_skill_script_execution_stamps_the_span_and_counts_the_run(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A script that exited cleanly is described on the span and counted once."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _script_tool(), agent, {}, ctx
  ):
    skill_telemetry = _instrumentation.track_skill_script_execution(
        _hallucination.ConfirmedNotHallucinated("sample_skill"),
        _hallucination.ConfirmedNotHallucinated("scripts/sample.py"),
    )
    skill_telemetry.skill = _loaded_skill()
    skill_telemetry.script_exit_code = 0

  span = telemetry.only_span()
  attributes = dict(span.attributes)
  assert attributes["gen_ai.skill.name"] == "sample_skill"
  assert attributes["gen_ai.skill.resource.name"] == "scripts/sample.py"
  assert attributes["gen_ai.skill.description"] == "A sample skill."
  assert attributes["process.exit.code"] == 0
  assert attributes["gen_ai.skill.source.uri"] == "file:/skills/sample"
  assert span.status.status_code is StatusCode.UNSET
  assert "error.type" not in attributes
  assert telemetry.points(_SKILL_SCRIPT_EXECUTIONS) == [(
      {
          "gen_ai.agent.name": "root_agent",
          "gen_ai.skill.name": "sample_skill",
          "gen_ai.skill.resource.name": "scripts/sample.py",
          "adk.experimental.skill.script.ended_with_error": False,
      },
      1,
  )]


@pytest.mark.asyncio
async def test_skill_script_failure_fails_the_span_and_flags_the_count(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A non-zero exit is a failure even though the tool call itself returned.

  Nothing raised -- the tool handed back a result describing the failure -- so
  the exit code is the only thing that can mark the span failed.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _script_tool(), agent, {}, ctx
  ):
    skill_telemetry = _instrumentation.track_skill_script_execution(
        _hallucination.ConfirmedNotHallucinated("sample_skill"),
        _hallucination.ConfirmedNotHallucinated("scripts/sample.py"),
    )
    skill_telemetry.skill = _loaded_skill()
    skill_telemetry.script_exit_code = 3

  span = telemetry.only_span()
  attributes = dict(span.attributes)
  assert span.status.status_code is StatusCode.ERROR
  assert attributes["error.type"] == "SKILL_SCRIPT_EXECUTION_ERROR"
  # The span keeps the code the metric collapses into a flag.
  assert attributes["process.exit.code"] == 3
  assert telemetry.point_attributes(_SKILL_SCRIPT_EXECUTIONS) == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.skill.name": "sample_skill",
      "gen_ai.skill.resource.name": "scripts/sample.py",
      "adk.experimental.skill.script.ended_with_error": True,
  }]


@pytest.mark.asyncio
async def test_skill_script_execution_that_never_ran_reports_no_exit_code(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The tracker runs before the script does, so the code can stay unknown.

  An unknown script, an unsupported extension or a registry failure all end
  the tool call before anything executes. Reporting a zero there would read as
  a clean run, and reporting a non-zero would invent a failure the script
  never had, so the attribute is simply absent -- the tool call's own
  ``error.type`` already says what went wrong.

  Nothing resolved the skill or the script either, so the counter takes the
  placeholder for both: a name that named nothing is the model's invention,
  and the series must not grow one label per invention.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _script_tool(), agent, {}, ctx
  ):
    _instrumentation.track_skill_script_execution(
        _hallucination.MaybeHallucinated("sample_skill"),
        _hallucination.MaybeHallucinated("scripts/sample.py"),
    )

  span = telemetry.only_span()
  attributes = dict(span.attributes)
  assert "process.exit.code" not in attributes
  assert span.status.status_code is StatusCode.UNSET
  assert "error.type" not in attributes
  assert "gen_ai.skill.description" not in attributes
  assert "gen_ai.skill.source.uri" not in attributes
  # The span keeps what the model actually asked for.
  assert attributes["gen_ai.skill.name"] == "sample_skill"
  assert attributes["gen_ai.skill.resource.name"] == "scripts/sample.py"
  # The attempt still happened, so it is still counted -- just without a
  # verdict on how it ended, and without the unconfirmed names.
  assert telemetry.point_attributes(_SKILL_SCRIPT_EXECUTIONS) == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.skill.name": "<hallucinated>",
      "gen_ai.skill.resource.name": "<hallucinated>",
  }]


# --- skill loads, on the execute_tool span and per invocation --------------
#
# A load produces three metrics: one count per load, dimensioned by the skill
# and by the error it hit, and the number of loads made over the whole agent
# invocation and over the whole workflow enclosing it. All three run through
# the tool execution block, for the same reason the script tests do.

_SKILL_LOADS = "adk.experimental.skill.loads"
_INVOKE_AGENT_SKILL_LOADS = "adk.experimental.invoke_agent.skill.loads"
_INVOKE_WORKFLOW_SKILL_LOADS = "adk.experimental.invoke_workflow.skill.loads"


def _load_tool() -> _EchoTool:
  return _EchoTool(name="load_skill", description="loads a skill")


def _resource_tool() -> _EchoTool:
  return _EchoTool(
      name="load_skill_resource", description="loads a skill resource"
  )


@pytest.mark.asyncio
async def test_skill_load_stamps_the_span_and_counts_the_load(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A load that resolved a skill is stamped on the span and counted."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _load_tool(), agent, {}, ctx
  ):
    skill_telemetry = _instrumentation.track_skill_load(
        _hallucination.ConfirmedNotHallucinated("sample_skill")
    )
    skill_telemetry.skill = _loaded_skill()

  attributes = dict(telemetry.only_span().attributes)
  assert attributes["gen_ai.skill.name"] == "sample_skill"
  assert attributes["gen_ai.skill.description"] == "A sample skill."
  assert attributes["gen_ai.skill.source.uri"] == "file:/skills/sample"
  assert telemetry.points(_SKILL_LOADS) == [(
      {
          "gen_ai.agent.name": "root_agent",
          "gen_ai.skill.name": "sample_skill",
      },
      1,
  )]


@pytest.mark.asyncio
async def test_skill_load_that_resolved_nothing_is_counted_with_its_error(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """An unknown skill name still counts as an attempted load.

  The load rate is only readable next to the loads that failed -- but the
  count carries the placeholder name, since a name that named nothing is the
  model's invention.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _load_tool(), agent, {}, ctx
  ) as tel_ctx:
    _instrumentation.track_skill_load(
        _hallucination.MaybeHallucinated("sample_skill")
    )
    tel_ctx.error_type = "SKILL_NOT_FOUND"

  # The span keeps what the model actually asked for.
  span_attributes = dict(telemetry.only_span().attributes)
  assert span_attributes["gen_ai.skill.name"] == "sample_skill"
  assert telemetry.point_attributes(_SKILL_LOADS) == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.skill.name": "<hallucinated>",
      "error.type": "SKILL_NOT_FOUND",
  }]


@pytest.mark.asyncio
async def test_record_agent_invocation_flushes_the_skill_loads_it_made(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Loads made anywhere under the agent land in its per-invocation total."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      for _ in range(5):
        async with _instrumentation.record_tool_execution(
            _load_tool(), agent, {}, ctx
        ):
          tel_ctx = _instrumentation.track_skill_load(
              _hallucination.ConfirmedNotHallucinated("sample_skill")
          )
          tel_ctx.skill = _loaded_skill()

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == [
      ({"gen_ai.agent.name": "root_agent"}, 5)
  ]


@pytest.mark.asyncio
async def test_invoke_agent_skill_loads_count_the_loads_that_failed_too(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The total counts loads attempted, not loads that resolved a skill.

  A load that named nothing was still a load attempted, and dropping it here
  would make the total disagree with the per-load counter beside it -- which
  is where the failures are read, split by the error they hit.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      for name in ("first_skill", "second_skill"):
        async with _instrumentation.record_tool_execution(
            _load_tool(), agent, {}, ctx
        ):
          tel_ctx = _instrumentation.track_skill_load(
              _hallucination.ConfirmedNotHallucinated(name)
          )
          tel_ctx.skill = _loaded_skill()
      async with _instrumentation.record_tool_execution(
          _load_tool(), agent, {}, ctx
      ) as tel_ctx:
        _instrumentation.track_skill_load(
            _hallucination.MaybeHallucinated("nonexistent_skill")
        )
        tel_ctx.error_type = "SKILL_NOT_FOUND"

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == [
      ({"gen_ai.agent.name": "root_agent"}, 3)
  ]
  # The error stays on the per-load counter, which is dimensioned for it.
  assert telemetry.point_attributes(_SKILL_LOADS) == [
      {
          "gen_ai.agent.name": "root_agent",
          "gen_ai.skill.name": "first_skill",
      },
      {
          "gen_ai.agent.name": "root_agent",
          "gen_ai.skill.name": "second_skill",
      },
      {
          "gen_ai.agent.name": "root_agent",
          "gen_ai.skill.name": "<hallucinated>",
          "error.type": "SKILL_NOT_FOUND",
      },
  ]


@pytest.mark.asyncio
async def test_a_skill_load_that_raised_is_counted_as_a_failed_load(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A load tool that raises failed just as loudly as one reporting a dict.

  Which of the two a failure arrives as is the tool's implementation detail,
  and the span and the tool duration metric already read both. Leaving the
  raised ones out would drop them from the load total entirely.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      with pytest.raises(ValueError):
        async with _instrumentation.record_tool_execution(
            _load_tool(), agent, {}, ctx
        ):
          _instrumentation.track_skill_load(
              _hallucination.MaybeHallucinated("sample_skill")
          )
          raise ValueError("the skill registry was unreachable")

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]
  # The per-load counter names the exception, the same label the span took.
  assert telemetry.point_attributes(_SKILL_LOADS) == [{
      "gen_ai.agent.name": "root_agent",
      "gen_ai.skill.name": "<hallucinated>",
      "error.type": "ValueError",
  }]


@pytest.mark.asyncio
async def test_invoke_agent_skill_loads_is_zero_when_no_skill_was_loaded(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """An invocation that loaded nothing reports zero rather than nothing.

  Zero is a measurement that was taken -- the invocation ran and reached for
  no skill. Dropping the point would sum the loads over only the invocations
  that used skills, so loads-per-invocation would read as though every
  invocation did.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      pass

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == [
      ({"gen_ai.agent.name": "root_agent"}, 0)
  ]


@pytest.mark.asyncio
async def test_the_other_skill_tools_are_not_counted_as_skill_loads(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Only loads count towards the load total, not the other skill tools.

  A script that exited non-zero and a resource that could not be read both
  act on a skill already loaded, so counting them here would report loads
  that never happened.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      async with _instrumentation.record_tool_execution(
          _script_tool(), agent, {}, ctx
      ):
        skill_telemetry = _instrumentation.track_skill_script_execution(
            _hallucination.ConfirmedNotHallucinated("sample_skill"),
            _hallucination.ConfirmedNotHallucinated("scripts/sample.py"),
        )
        skill_telemetry.skill = _loaded_skill()
        skill_telemetry.script_exit_code = 3
      async with _instrumentation.record_tool_execution(
          _resource_tool(), agent, {}, ctx
      ) as tel_ctx:
        _instrumentation.track_skill_resource_load(
            _hallucination.MaybeHallucinated("sample_skill"),
            _hallucination.MaybeHallucinated("refs/none.md"),
        )
        tel_ctx.error_type = "SKILL_RESOURCE_NOT_FOUND"

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == [
      ({"gen_ai.agent.name": "root_agent"}, 0)
  ]


@pytest.mark.asyncio
async def test_invoke_agent_skill_loads_is_not_recorded_without_the_opt_in(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Without the opt-in the loads are never counted, so there is no total.

  Recording one anyway would report every invocation as having loaded no
  skills, which is a measurement nobody took: the count is missing, not zero.
  """
  monkeypatch.delenv("ADK_EXPERIMENTAL_TELEMETRY", raising=False)
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    with _instrumentation._using(scope.otel_context):
      async with _instrumentation.record_tool_execution(
          _load_tool(), agent, {}, ctx
      ) as tel_ctx:
        _instrumentation.track_skill_load(
            _hallucination.MaybeHallucinated("nonexistent_skill")
        )
        tel_ctx.error_type = "SKILL_NOT_FOUND"

  assert telemetry.points(_INVOKE_AGENT_SKILL_LOADS) == []
  # The stable per-invocation counts are unaffected by the opt-in.
  assert telemetry.points("gen_ai.invoke_agent.tool_calls") == [
      ({"gen_ai.agent.name": "root_agent"}, 1)
  ]


@pytest.mark.asyncio
async def test_invoke_workflow_skill_loads_cover_the_whole_workflow(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The workflow total spans every agent the turn routed through.

  The per-invocation totals split the same loads by agent, so the workflow
  point is what answers how many a whole turn made -- one number, whichever
  agents ended up serving it.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  with node_tracing._use_invoke_workflow_span(
      "my_workflow", "conversation-1"
  ) as (_, workflow_context):
    with _instrumentation._using(workflow_context):
      async with _instrumentation.record_agent_invocation(ctx, agent):
        for name in ("first_skill", "second_skill"):
          async with _instrumentation.record_tool_execution(
              _load_tool(), agent, {}, ctx
          ):
            tel_ctx = _instrumentation.track_skill_load(
                _hallucination.ConfirmedNotHallucinated(name)
            )
            tel_ctx.skill = _loaded_skill()

  assert telemetry.points(_INVOKE_WORKFLOW_SKILL_LOADS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      2,
  )]


@pytest.mark.asyncio
async def test_invoke_workflow_skill_loads_count_into_every_enclosing_workflow(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A nested workflow's loads count for it and for the ones around it.

  The outer totals stay inclusive, the same way tokens and call counts do, so
  a workflow's number is what ran under it rather than what ran directly in
  it.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  with node_tracing._use_invoke_workflow_span("outer", "conversation-1") as (
      _,
      outer_context,
  ):
    with _instrumentation._using(outer_context):
      async with _instrumentation.record_agent_invocation(ctx, agent):
        async with _instrumentation.record_tool_execution(
            _load_tool(), agent, {}, ctx
        ):
          tel_ctx = _instrumentation.track_skill_load(
              _hallucination.ConfirmedNotHallucinated("outer_skill")
          )
          tel_ctx.skill = _loaded_skill()
        with node_tracing._use_invoke_workflow_span(
            "inner", "conversation-1"
        ) as (_, inner_context):
          with _instrumentation._using(inner_context):
            async with _instrumentation.record_tool_execution(
                _load_tool(), agent, {}, ctx
            ):
              tel_ctx = _instrumentation.track_skill_load(
                  _hallucination.ConfirmedNotHallucinated("inner_skill")
              )
              tel_ctx.skill = _loaded_skill()

  assert telemetry.points(_INVOKE_WORKFLOW_SKILL_LOADS) == [
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "inner",
              "gen_ai.workflow.nested": True,
          },
          1,
      ),
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "outer",
          },
          2,
      ),
  ]


@pytest.mark.asyncio
async def test_invoke_workflow_skill_loads_is_zero_when_no_skill_was_loaded(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Zero is recorded here for the same reason it is per invocation."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")

  with node_tracing._use_invoke_workflow_span(
      "my_workflow", "conversation-1"
  ) as (_, workflow_context):
    with _instrumentation._using(workflow_context):
      pass

  assert telemetry.points(_INVOKE_WORKFLOW_SKILL_LOADS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      0,
  )]


@pytest.mark.asyncio
async def test_invoke_workflow_skill_loads_is_not_recorded_without_the_opt_in(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The whole per-workflow block is experimental, this metric included."""
  monkeypatch.delenv("ADK_EXPERIMENTAL_TELEMETRY", raising=False)
  agent = _agent()
  ctx = await _invocation_context(agent)

  with node_tracing._use_invoke_workflow_span(
      "my_workflow", "conversation-1"
  ) as (_, workflow_context):
    with _instrumentation._using(workflow_context):
      async with _instrumentation.record_tool_execution(
          _load_tool(), agent, {}, ctx
      ):
        tel_ctx = _instrumentation.track_skill_load(
            _hallucination.ConfirmedNotHallucinated("sample_skill")
        )
        tel_ctx.skill = _loaded_skill()

  assert telemetry.points(_INVOKE_WORKFLOW_SKILL_LOADS) == []


# --- conversation compactions, per workflow --------------------------------

_INVOKE_WORKFLOW_COMPACTIONS = "adk.experimental.invoke_workflow.compactions"


@pytest.mark.asyncio
async def test_compaction_during_an_agent_counts_for_the_workflow(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A compaction an agent made belongs to the turn around it."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  with node_tracing._use_invoke_workflow_span(
      "my_workflow", "conversation-1"
  ) as (_, workflow_context):
    with _instrumentation._using(workflow_context):
      async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
        with _instrumentation._using(scope.otel_context):
          _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      1,
  )]


@pytest.mark.asyncio
async def test_compaction_after_the_agent_finished_counts_for_the_workflow(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """The runner's post-invocation pass lands outside every agent scope.

  It still belongs to the turn, which is the whole reason the count is kept
  per workflow: there is no agent left to attribute it to.
  """
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  with node_tracing._use_invoke_workflow_span(
      "my_workflow", "conversation-1"
  ) as (_, workflow_context):
    with _instrumentation._using(workflow_context):
      async with _instrumentation.record_agent_invocation(ctx, agent):
        pass
      _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      1,
  )]


@pytest.mark.asyncio
async def test_compaction_counts_into_every_enclosing_workflow(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Outer totals stay inclusive, as they do for loads and call counts."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")

  with node_tracing._use_invoke_workflow_span("outer", "conversation-1") as (
      _,
      outer_context,
  ):
    with _instrumentation._using(outer_context):
      _instrumentation.count_compaction()
      with node_tracing._use_invoke_workflow_span(
          "inner", "conversation-1"
      ) as (_, inner_context):
        with _instrumentation._using(inner_context):
          _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "inner",
              "gen_ai.workflow.nested": True,
          },
          1,
      ),
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "outer",
          },
          2,
      ),
  ]


# --- record_inference_telemetry + record_llm_response ----------------------


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
      "gen_ai.provider.name": "gcp.gemini",
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
      "gen_ai.provider.name": "gcp.gemini",
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
      "gen_ai.provider.name": "gcp.gemini",
      "gen_ai.request.model": "some-model",
      "error.type": "ValueError",
  }]


@pytest.mark.asyncio
async def test_record_llm_response_keeps_every_response_in_arrival_order(
    telemetry: _Telemetry,
):
  """Token usage is read off the newest response that reports any, so both

  retention and order matter.
  """
  agent = _agent()
  ctx = await _invocation_context(agent)
  tel_ctx = _instrumentation.InferenceScope(
      llm_request=LlmRequest(), user_id="u"
  )
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
  tel_ctx = _instrumentation.InferenceScope(
      llm_request=LlmRequest(), user_id="u"
  )

  with tracing.tracer.start_as_current_span("test_span") as span:
    tel_ctx.span = span
    tel_ctx.record_llm_response(ctx, _llm_response())

  attributes = dict(telemetry.only_span().attributes)
  assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
  assert attributes["gen_ai.usage.input_tokens"] == 10
  assert attributes["gen_ai.usage.output_tokens"] == 5


_TIME_TO_FIRST_CHUNK = "gen_ai.client.inference.time_to_first_chunk"
_TIME_PER_OUTPUT_CHUNK = "gen_ai.client.inference.time_per_output_chunk"


@pytest.mark.asyncio
async def test_record_inference_telemetry_times_each_streamed_chunk(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Times the first chunk from the span's start, each later one from the last.

  The aggregate the model yields after the last chunk is no chunk.
  """
  clock = mock.Mock(return_value=0)
  monkeypatch.setattr(
      _instrumentation,
      "time",
      mock.Mock(time_ns=clock, monotonic=time.monotonic),
  )
  agent = _agent()
  ctx = await _invocation_context(
      agent, RunConfig(streaming_mode=StreamingMode.SSE)
  )
  llm_request = LlmRequest(model="some-model")
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"
  unfinished_chunk = _llm_response(
      partial=True, finish_reason=None, usage_metadata=None
  )
  # Chunks 0.5s, 0.75s and 1.25s after the span started. The aggregate would
  # be timed at 10s if it were counted.
  responses = [
      (500_000_000, unfinished_chunk),
      (750_000_000, unfinished_chunk),
      (1_250_000_000, _llm_response(partial=True)),
      (10_000_000_000, _llm_response()),
  ]

  async with _instrumentation.record_inference_telemetry(
      llm_request, ctx, model_response_event
  ) as tel_ctx:
    span_start_ns = tel_ctx.span.span.start_time
    for since_span_start_ns, response in responses:
      clock.return_value = span_start_ns + since_span_start_ns
      tel_ctx.record_llm_response(ctx, response)

  attributes = {
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gcp.gemini",
      "gen_ai.request.model": "some-model",
      "gen_ai.response.model": "some-model-001",
  }
  assert telemetry.points(_TIME_TO_FIRST_CHUNK) == [(attributes, 0.5)]
  assert telemetry.points(_TIME_PER_OUTPUT_CHUNK) == [(attributes, 0.75)]


@pytest.mark.asyncio
async def test_first_chunk_is_timed_from_span_open_if_span_is_sampled_out(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A sampled-out span has no start time, so timing starts as it opens."""
  # The span opens at 10s, the first chunk arrives at 10.5s.
  clock = mock.Mock(side_effect=[10_000_000_000, 10_500_000_000])
  monkeypatch.setattr(
      _instrumentation,
      "time",
      mock.Mock(time_ns=clock, monotonic=time.monotonic),
  )
  sampled_out_tracer = TracerProvider(sampler=ALWAYS_OFF).get_tracer(__name__)
  monkeypatch.setattr(
      tracing.tracer,
      "start_as_current_span",
      sampled_out_tracer.start_as_current_span,
  )
  agent = _agent()
  ctx = await _invocation_context(
      agent, RunConfig(streaming_mode=StreamingMode.SSE)
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      LlmRequest(model="some-model"), ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(ctx, _llm_response(partial=True))

  attributes = {
      "gen_ai.operation.name": "generate_content",
      "gen_ai.provider.name": "gcp.gemini",
      "gen_ai.request.model": "some-model",
      "gen_ai.response.model": "some-model-001",
  }
  assert telemetry.points(_TIME_TO_FIRST_CHUNK) == [(attributes, 0.5)]


@pytest.mark.parametrize(
    "run_config",
    [
        RunConfig(),
        RunConfig(streaming_mode=StreamingMode.SSE, support_cfc=True),
    ],
    ids=["not_streamed", "live"],
)
@pytest.mark.asyncio
async def test_record_inference_telemetry_times_no_chunks_unless_streamed(
    telemetry: _Telemetry, run_config: RunConfig
):
  """Times nothing for a call that is not a streamed request.

  A whole answer has no chunks, and a live exchange no request to time the
  first one from.
  """
  agent = _agent()
  ctx = await _invocation_context(agent, run_config)
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      LlmRequest(model="some-model"), ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(ctx, _llm_response())

  assert not telemetry.points(_TIME_TO_FIRST_CHUNK)
  assert not telemetry.points(_TIME_PER_OUTPUT_CHUNK)


@pytest.mark.asyncio
async def test_chunk_timing_is_left_to_the_genai_instrumentor(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Times nothing for a Gemini call the google-genai instrumentor wraps.

  The instrumentor times the same chunks, so ADK recording them too would
  report every streamed call twice.
  """
  monkeypatch.setattr(
      "google.adk.telemetry._genai_instrumentor.is_instrumented",
      lambda: True,
  )
  # ADK delegates only the experimental semconv to the instrumentor.
  monkeypatch.setenv(
      "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
  )
  monkeypatch.setattr(tracing, "_is_gemini_agent", lambda _: True)
  agent = _agent()
  ctx = await _invocation_context(
      agent, RunConfig(streaming_mode=StreamingMode.SSE)
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  async with _instrumentation.record_inference_telemetry(
      LlmRequest(model="some-model"), ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(
        ctx, _llm_response(partial=True, finish_reason=None)
    )
    tel_ctx.record_llm_response(ctx, _llm_response(partial=True))

  assert not telemetry.points(_TIME_TO_FIRST_CHUNK)
  assert not telemetry.points(_TIME_PER_OUTPUT_CHUNK)


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


def test_record_invocation_opens_the_entrypoint_span_for_a_workflow_too(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """A workflow entrypoint gets the runner's span like any other entrypoint."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")

  with _instrumentation.record_invocation(Workflow(name="my_workflow"), "c-1"):
    pass

  span = telemetry.only_span()
  assert span.name == "invoke_workflow my_workflow"
  assert dict(span.attributes) == {
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.conversation.id": "c-1",
      "gen_ai.workflow.name": "my_workflow",
  }


def _workflow_spans(telemetry: _Telemetry):
  return [
      span
      for span in telemetry.spans()
      if span.name.startswith("invoke_workflow")
  ]


def _root_context(
    ic: InvocationContext,
    entrypoint_scope: _instrumentation._WorkflowScope | None,
) -> Context:
  """Builds the context the runner starts its root node under."""
  context = Context(ic)
  context._telemetry_context = dataclasses.replace(
      context.telemetry_context, _entrypoint_scope=entrypoint_scope
  )
  return context


async def _run_once(runner: InMemoryRunner) -> None:
  """Runs one turn of ``runner`` in a new session."""
  session = await runner.session_service.create_session(
      app_name=runner.app_name, user_id="test_user"
  )
  async for _ in runner.run_async(
      user_id="test_user",
      session_id=session.id,
      new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
  ):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoint_emits_one_invoke_workflow_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """The turn has one root `invoke_workflow`, the runner's."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      _,
      entrypoint_scope,
  ):
    async with node_tracing.start_as_current_node_span(
        _root_context(ic, entrypoint_scope), workflow
    ):
      pass

  (span,) = _workflow_spans(telemetry)
  assert span.name == "invoke_workflow my_workflow"
  assert "gen_ai.workflow.nested" not in span.attributes
  assert telemetry.point_attributes("gen_ai.invoke_workflow.duration") == [{
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.workflow.name": "my_workflow",
  }]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoint_runs_in_the_context_the_runner_hands_it(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """Whatever its own context happened to capture."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())
  # Built outside the runner's span, so what it captured is not the runner's.
  root_context = Context(ic)

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      _,
      entrypoint_scope,
  ):
    root_context._telemetry_context = dataclasses.replace(
        root_context.telemetry_context, _entrypoint_scope=entrypoint_scope
    )
    async with node_tracing.start_as_current_node_span(root_context, workflow):
      with tracing.tracer.start_as_current_span("a child span"):
        pass

  (workflow_span,) = _workflow_spans(telemetry)
  (child,) = [span for span in telemetry.spans() if span.name == "a child span"]
  assert child.parent.span_id == workflow_span.context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoint_counts_the_runners_post_invocation_work(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """The runner compacts after the root node, into the same one datapoint."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      invocation_context,
      entrypoint_scope,
  ):
    async with node_tracing.start_as_current_node_span(
        _root_context(ic, entrypoint_scope), workflow
    ):
      _instrumentation.count_compaction()
    with _instrumentation._using(invocation_context):
      _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      2,
  )]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_in_the_entrypoint_is_nested_even_with_its_name(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """Only the entrypoint's root node runs in the runner's span."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      _,
      entrypoint_scope,
  ):
    async with node_tracing.start_as_current_node_span(
        _root_context(ic, entrypoint_scope), workflow
    ) as root_telemetry:
      parent = Context(ic)
      parent._telemetry_context = root_telemetry
      async with node_tracing.start_as_current_node_span(
          parent, Workflow(name="my_workflow")
      ):
        pass

  inner, outer = _workflow_spans(telemetry)
  assert inner.attributes["gen_ai.workflow.nested"]
  assert inner.parent.span_id == outer.context.span_id
  assert "gen_ai.workflow.nested" not in outer.attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoint_started_inside_a_workflow_is_nested(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """A runner started from inside a workflow nests its entrypoint in it."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  outer, inner = Workflow(name="outer"), Workflow(name="inner")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(outer, "conversation-1") as (
      _,
      entrypoint_scope,
  ):
    async with node_tracing.start_as_current_node_span(
        _root_context(ic, entrypoint_scope), outer
    ):
      with _instrumentation.record_invocation(inner, "conversation-1") as (
          _,
          entrypoint_scope,
      ):
        async with node_tracing.start_as_current_node_span(
            _root_context(ic, entrypoint_scope), inner
        ):
          _instrumentation.count_compaction()

  inner_span, outer_span = _workflow_spans(telemetry)
  assert inner_span.attributes["gen_ai.workflow.nested"]
  assert "gen_ai.workflow.nested" not in outer_span.attributes
  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "inner",
              "gen_ai.workflow.nested": True,
          },
          1,
      ),
      (
          {
              "adk.experimental.root_agent.name": "outer",
              "gen_ai.workflow.name": "outer",
          },
          1,
      ),
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_retried_workflow_entrypoint_records_one_datapoint(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """Every attempt of the root node runs in the runner's span."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      _,
      entrypoint_scope,
  ):
    root_context = _root_context(ic, entrypoint_scope)
    for _ in range(2):
      async with node_tracing.start_as_current_node_span(
          root_context, workflow
      ):
        _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      2,
  )]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoint_reports_even_if_its_root_node_never_ran(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """The runner's span reports either way."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")

  with _instrumentation.record_invocation(
      Workflow(name="my_workflow"), "c-1"
  ) as (invocation_context, _):
    with _instrumentation._using(invocation_context):
      _instrumentation.count_compaction()

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [(
      {
          "adk.experimental.root_agent.name": "my_workflow",
          "gen_ai.workflow.name": "my_workflow",
      },
      1,
  )]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_the_runners_post_invocation_spans_nest_in_a_workflow_entrypoint(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """What the runner does after the root node stays inside the turn."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  workflow = Workflow(name="my_workflow")
  ic = await _invocation_context(_agent())

  with _instrumentation.record_invocation(workflow, "conversation-1") as (
      invocation_context,
      entrypoint_scope,
  ):
    async with node_tracing.start_as_current_node_span(
        _root_context(ic, entrypoint_scope), workflow
    ):
      pass
    with _instrumentation._using(invocation_context):
      with tracing.tracer.start_as_current_span(
          "compact_events sliding_window"
      ):
        pass

  (workflow_span,) = _workflow_spans(telemetry)
  (compaction_span,) = [
      span
      for span in telemetry.spans()
      if span.name == "compact_events sliding_window"
  ]
  assert compaction_span.parent.span_id == workflow_span.context.span_id
  assert workflow_span.start_time <= compaction_span.start_time
  assert compaction_span.end_time <= workflow_span.end_time


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_workflow_entrypoints_failure_marks_the_runners_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """The root node's failure is raised again through the runner's span."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)

  async def failing_node(ctx, node_input):
    del ctx, node_input
    raise ValueError("boom")

  runner = InMemoryRunner(
      app_name="test_app",
      node=Workflow(name="my_workflow", edges=[(START, failing_node)]),
  )
  with pytest.raises(ValueError, match="boom"):
    await _run_once(runner)

  (span,) = _workflow_spans(telemetry)
  assert span.name == "invoke_workflow my_workflow"
  assert span.status.status_code is StatusCode.ERROR
  assert telemetry.point_attributes("gen_ai.invoke_workflow.duration") == [{
      "error.type": "ValueError",
      "gen_ai.operation.name": "invoke_workflow",
      "gen_ai.workflow.name": "my_workflow",
  }]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_version", ["1", "2"])
async def test_a_runner_started_by_a_plugin_keeps_its_own_entrypoint_span(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch, schema_version: str
):
  """Even named like the outer entrypoint, it nests rather than taking it over.

  So what that runner does after its root node still counts for its own turn.
  """
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")

  async def compacting_node(ctx, node_input):
    del ctx, node_input
    _instrumentation.count_compaction()

  class _CompactsAfterRun(BasePlugin):

    async def after_run_callback(self, *, invocation_context):
      del invocation_context
      _instrumentation.count_compaction()

  class _RunsANestedRunner(BasePlugin):

    async def before_run_callback(self, *, invocation_context):
      del invocation_context
      await _run_once(
          InMemoryRunner(
              app_name="nested_app",
              node=Workflow(
                  name="my_workflow", edges=[(START, compacting_node)]
              ),
              plugins=[_CompactsAfterRun(name="compacts_after_run")],
          )
      )

  async def idle_node(ctx, node_input):
    del ctx, node_input

  await _run_once(
      InMemoryRunner(
          app_name="test_app",
          node=Workflow(name="my_workflow", edges=[(START, idle_node)]),
          plugins=[_RunsANestedRunner(name="runs_a_nested_runner")],
      )
  )

  assert telemetry.points(_INVOKE_WORKFLOW_COMPACTIONS) == [
      (
          {
              "adk.experimental.root_agent.name": "my_workflow",
              "gen_ai.workflow.name": "my_workflow",
              "gen_ai.workflow.nested": True,
          },
          2,
      ),
      (
          {
              "adk.experimental.root_agent.name": "my_workflow",
              "gen_ai.workflow.name": "my_workflow",
          },
          2,
      ),
  ]


@pytest.mark.asyncio
async def test_record_llm_response_keeps_recording_without_a_finish_reason(
    telemetry: _Telemetry,
):
  """Stopping there truncated the span and its log to it."""
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(
      model="some-model",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"
  merged_fragment = _llm_response(
      content=types.Content(role="model", parts=[types.Part(text="I")]),
      finish_reason=None,
      usage_metadata=None,
  )

  async with _instrumentation.record_inference_telemetry(
      llm_request, ctx, model_response_event
  ) as tel_ctx:
    tel_ctx.record_llm_response(ctx, merged_fragment)
    assert tel_ctx.span.span.end_time is None
    tel_ctx.record_llm_response(ctx, _llm_response())

  attributes = dict(telemetry.only_span().attributes)
  assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
  assert attributes["gen_ai.usage.input_tokens"] == 10
  assert attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.asyncio
async def test_unspecified_finish_reason_does_not_end_the_inference_span(
    telemetry: _Telemetry,
):
  """The proto3 zero value is truthy, so chunk one read as the whole answer."""
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
    # What a backend that does not mark its chunks sends before the last one.
    tel_ctx.record_llm_response(
        ctx,
        _llm_response(
            content=types.Content(role="model", parts=[types.Part(text="I")]),
            finish_reason=types.FinishReason.FINISH_REASON_UNSPECIFIED,
            usage_metadata=None,
        ),
    )
    assert tel_ctx.span.span.end_time is None
    tel_ctx.record_llm_response(ctx, _llm_response())

  attributes = dict(telemetry.only_span().attributes)
  assert attributes["gen_ai.response.finish_reasons"] == ("stop",)
  assert attributes["gen_ai.usage.output_tokens"] == 5


@pytest.mark.asyncio
async def test_inference_span_ends_at_the_finish_reason_not_at_teardown(
    telemetry: _Telemetry,
):
  """Ending at teardown would bill the caller's tool call to the model."""
  agent = _agent()
  ctx = await _invocation_context(agent)
  llm_request = LlmRequest(
      model="some-model",
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )
  model_response_event = mock.MagicMock()
  model_response_event.id = "event-1"

  with tracing.tracer.start_as_current_span("call_llm") as call_llm:
    async with _instrumentation.record_inference_telemetry(
        llm_request, ctx, model_response_event
    ) as tel_ctx:
      tel_ctx.record_llm_response(ctx, _llm_response())
      # Closed on the spot, so the tool the model asked for is timed and
      # parented beside the inference rather than inside it.
      ended_at = tel_ctx.span.span.end_time
      assert ended_at is not None
      with tracing.tracer.start_as_current_span("execute_tool"):
        pass

  spans = {span.name: span for span in telemetry.spans()}
  assert spans["execute_tool"].parent.span_id == (
      call_llm.get_span_context().span_id
  )
  assert spans["generate_content some-model"].end_time == ended_at


@pytest.mark.asyncio
async def test_responses_after_the_finish_reason_still_count_for_metrics(
    telemetry: _Telemetry,
):
  """The span is closed by then, but the spend is still the model's."""
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
    tel_ctx.record_llm_response(ctx, _llm_response())

  assert len(tel_ctx.llm_responses) == 2


# --- stream_in_scope (b/485899016) -----------------------------------------
#
# An async generator has no context of its own: it is resumed, and finalized,
# in the context of whoever drives it. A telemetry scope that held an OTel
# attach token across one of its yields therefore detached a token created in
# another context whenever the consumer closed the stream from a different
# task or dropped it for the async-generator finalizer to close.
# `context_api.detach` swallows that into a "Failed to detach context" log, so
# the symptom was not an exception but a consumer context left with an ended
# span attached to it for good.


@contextlib.contextmanager
def _detach_failures() -> Iterator[list[logging.LogRecord]]:
  """Collects OTel's "Failed to detach context" records."""
  records: list[logging.LogRecord] = []

  class _Collect(logging.Handler):

    def emit(self, record: logging.LogRecord) -> None:
      records.append(record)

  handler = _Collect()
  otel_logger = logging.getLogger("opentelemetry.context")
  otel_logger.addHandler(handler)
  try:
    yield records
  finally:
    otel_logger.removeHandler(handler)


def _counting_scope() -> tuple[context_api.Context, list[str]]:
  """A context carrying a marker, plus the sink spans report themselves to."""
  seen: list[str] = []
  return context_api.set_value("b485899016_scope", seen), seen


async def _emitting_generator(seen: list[str]) -> AsyncGenerator[int, None]:
  """Reports the scope it sees while running, and once more on teardown."""

  def note(where: str) -> None:
    value = context_api.get_value("b485899016_scope")
    seen.append(f"{where}:{'in-scope' if value is not None else 'detached'}")

  try:
    for i in range(100):
      await asyncio.sleep(0)
      note("body")
      yield i
  finally:
    note("teardown")


@pytest.mark.asyncio
async def test_stream_in_scope_survives_a_close_from_another_task():
  """The failing shape from the bug: produced in one task, closed in another."""
  scope_context, seen = _counting_scope()
  agen = _instrumentation.stream_in_scope(
      scope_context, _emitting_generator(seen)
  )

  async def pump() -> None:
    async for i in agen:
      if i == 2:
        return

  with _detach_failures() as failures:
    await asyncio.create_task(pump())
    await asyncio.create_task(agen.aclose())

  assert not failures, [r.getMessage() for r in failures]
  assert "teardown:in-scope" in seen


@pytest.mark.asyncio
async def test_stream_in_scope_survives_the_async_generator_finalizer():
  """Nobody closes the stream; asyncio's finalizer does, on its own task."""
  scope_context, seen = _counting_scope()

  async def pump() -> None:
    agen = _instrumentation.stream_in_scope(
        scope_context, _emitting_generator(seen)
    )
    async for i in agen:
      if i == 2:
        return
    # `agen` goes out of scope without being closed.

  with _detach_failures() as failures:
    await asyncio.create_task(pump())
    gc.collect()
    await asyncio.sleep(0.1)

  assert not failures, [r.getMessage() for r in failures]


@pytest.mark.asyncio
async def test_stream_in_scope_leaves_the_consumers_context_alone():
  """The consumer traces into its own span, not into ADK's."""
  scope_context, seen = _counting_scope()
  before = context_api.get_current()

  async with contextlib.aclosing(
      _instrumentation.stream_in_scope(scope_context, _emitting_generator(seen))
  ) as agen:
    async for i in agen:
      assert context_api.get_value("b485899016_scope") is None
      if i == 2:
        break

  assert context_api.get_current() == before
  # ... while the generator itself ran inside the scope throughout.
  assert [s for s in seen if s.startswith("body:")] == ["body:in-scope"] * 3


@pytest.mark.asyncio
async def test_stream_in_scope_parents_spans_opened_inside_it(
    telemetry: _Telemetry,
):
  """Both the body's spans and its teardown's belong to the scope's span."""
  parent = tracing.tracer.start_span("scope")
  scope_context = trace.set_span_in_context(parent, context_api.get_current())

  async def agen() -> AsyncGenerator[int, None]:
    try:
      with tracing.tracer.start_as_current_span("body"):
        pass
      yield 1
      yield 2
    finally:
      with tracing.tracer.start_as_current_span("teardown"):
        pass

  stream = _instrumentation.stream_in_scope(scope_context, agen())

  async def pump() -> None:
    async for _ in stream:
      return

  # Closed from another task, the shape that used to orphan the teardown span.
  await asyncio.create_task(pump())
  await asyncio.create_task(stream.aclose())
  parent.end()

  by_name = {span.name: span for span in telemetry.spans()}
  assert set(by_name) == {"scope", "body", "teardown"}
  for name in ("body", "teardown"):
    assert by_name[name].parent is not None, f"{name} span was orphaned"
    assert (
        by_name[name].parent.span_id == by_name["scope"].context.span_id
    ), f"{name} span is not parented to the scope"


@pytest.mark.asyncio
async def test_record_agent_invocation_does_not_attach_its_context(
    telemetry: _Telemetry,
):
  """The scope publishes its context; it must not make it current.

  Attaching here is what made the scope unsafe to wrap an async generator in:
  the token outlived every yield, so a close driven from another task detached
  it against the wrong context and left the span attached to the consumer for
  good (b/485899016).
  """
  del telemetry  # Wired up by the fixture; pins the tracer and the env.
  agent = _agent()
  ctx = await _invocation_context(agent)
  before = context_api.get_current()

  async with _instrumentation.record_agent_invocation(ctx, agent) as scope:
    assert context_api.get_current() == before
    # ... but the span is there for `stream_in_scope` to attach.
    assert trace.get_current_span(scope.otel_context) is not trace.INVALID_SPAN

  assert context_api.get_current() == before


def test_record_invocation_does_not_attach_its_context(telemetry: _Telemetry):
  """Same contract for the runner's top-level invocation scope."""
  del telemetry  # Wired up by the fixture; pins the tracer and the env.
  before = context_api.get_current()

  with _instrumentation.record_invocation(
      entrypoint_node=None, conversation_id="conversation-1"
  ) as (invocation_context, _):
    assert context_api.get_current() == before
    assert trace.get_current_span(invocation_context) is not trace.INVALID_SPAN

  assert context_api.get_current() == before


@pytest.mark.asyncio
async def test_stream_in_scope_propagates_an_error_from_the_body():
  """A failing body is not swallowed by the drive loop."""
  scope_context, _ = _counting_scope()

  async def failing() -> AsyncGenerator[int, None]:
    yield 1
    raise ValueError("boom")

  with pytest.raises(ValueError, match="boom"):
    async with contextlib.aclosing(
        _instrumentation.stream_in_scope(scope_context, failing())
    ) as agen:
      async for _ in agen:
        pass


@pytest.mark.asyncio
async def test_stream_in_scope_preserves_inner_streaming_span_across_yields(
    telemetry: _Telemetry,
):
  """Inner span opened inside the generator stays current across yields."""
  del telemetry
  with tracing.tracer.start_as_current_span("caller_root") as caller_span:
    caller_ctx = context_api.get_current()
    observed_inner_spans: list[trace.Span] = []

    async def inner_stream() -> AsyncGenerator[int, None]:
      with tracing.tracer.start_as_current_span(
          "generate_content"
      ) as inner_span:
        yield 1
        observed_inner_spans.append(trace.get_current_span())
        yield 2
        observed_inner_spans.append(trace.get_current_span())
        assert all(s is inner_span for s in observed_inner_spans)

    call_llm_span = tracing.tracer.start_span("call_llm", context=caller_ctx)
    call_llm_ctx = trace.set_span_in_context(call_llm_span, caller_ctx)

    chunks = []
    async with contextlib.aclosing(
        _instrumentation.stream_in_scope(call_llm_ctx, inner_stream())
    ) as agen:
      async for chunk in agen:
        assert trace.get_current_span() is caller_span
        chunks.append(chunk)

    call_llm_span.end()
    assert chunks == [1, 2]
    assert len(observed_inner_spans) == 2
    assert all(
        s is not trace.INVALID_SPAN
        and s is not caller_span
        and s is not call_llm_span
        for s in observed_inner_spans
    )


@pytest.mark.asyncio
async def test_stream_in_scope_propagates_cancellation_into_generator():
  """Consumer-thrown exceptions (like CancelledError) reach the generator body."""
  scope_context, _ = _counting_scope()
  cancelled_in_generator = False

  async def cancellable_generator() -> AsyncGenerator[int, None]:
    nonlocal cancelled_in_generator
    try:
      yield 1
      await asyncio.sleep(10)
    except asyncio.CancelledError:
      cancelled_in_generator = True
      raise

  agen = _instrumentation.stream_in_scope(
      scope_context, cancellable_generator()
  )
  assert await agen.__anext__() == 1
  with pytest.raises(asyncio.CancelledError):
    await agen.athrow(asyncio.CancelledError())
  assert cancelled_in_generator is True


# ---------------------------------------------------------------------------
# Busy interval collection (where a turn's wall-clock time went)
# ---------------------------------------------------------------------------

# Long enough that scheduling jitter cannot be mistaken for the interval, short
# enough to keep these tests quick. Assertions allow half of it either way.
_SLICE = 0.10


def _fake_tool_and_agent():
  """Returns the mocks `record_tool_execution` needs to run a tool."""
  tool = mock.MagicMock()
  tool.name = "sample_tool"
  agent = mock.MagicMock()
  agent.name = "sample_agent"
  return tool, agent


async def _run_tool(seconds: float = _SLICE) -> None:
  """Runs a tool that occupies `seconds`, as the flow would."""
  tool, agent = _fake_tool_and_agent()
  async with _instrumentation.record_tool_execution(
      tool=tool,
      agent=agent,
      function_args={},
      invocation_context=mock.MagicMock(),
  ):
    await asyncio.sleep(seconds)


async def _await_model(seconds: float = _SLICE, chunks: int = 1) -> None:
  """Consumes a model stream whose every chunk takes `seconds` to arrive."""

  async def responses():
    for _ in range(chunks):
      await asyncio.sleep(seconds)
      yield mock.MagicMock(spec=LlmResponse)

  async for _ in _instrumentation.time_model_awaits(responses()):
    pass


def _about(actual: float, expected: float) -> bool:
  """Whether a measured duration is the expected one, allowing for jitter."""
  return abs(actual - expected) < _SLICE / 2


@pytest.mark.asyncio
async def test_a_turn_that_does_nothing_reports_no_busy_time():
  """Nothing in flight is a measured zero, not a missing measurement."""
  with _instrumentation.collect_busy_intervals() as busy:
    pass

  assert busy.model_seconds() == 0.0
  assert busy.tool_seconds() == 0.0
  assert busy.busy_seconds() == 0.0


@pytest.mark.asyncio
async def test_sequential_tools_add_up():
  """Tools that do not overlap occupy the sum of their stretches."""
  with _instrumentation.collect_busy_intervals() as busy:
    await _run_tool()
    await _run_tool()

  assert _about(busy.tool_seconds(), 2 * _SLICE)


@pytest.mark.asyncio
async def test_concurrent_tools_count_their_overlap_once():
  """Two tools in flight together occupy one stretch of wall clock, not two."""
  with _instrumentation.collect_busy_intervals() as busy:
    await asyncio.gather(_run_tool(), _run_tool())

  assert _about(busy.tool_seconds(), _SLICE)


@pytest.mark.asyncio
async def test_a_model_awaited_inside_a_tool_is_model_time_not_tool_time():
  """An `AgentTool` runs model calls inside its own tool interval.

  Counting both in full would attribute the same seconds twice and leave the
  framework remainder negative, so the nested model time is subtracted from
  the tool's.
  """
  tool, agent = _fake_tool_and_agent()

  with _instrumentation.collect_busy_intervals() as busy:
    async with _instrumentation.record_tool_execution(
        tool=tool,
        agent=agent,
        function_args={},
        invocation_context=mock.MagicMock(),
    ):
      await _await_model()
      await asyncio.sleep(_SLICE)

  # The tool spanned both slices; one of them was spent awaiting the model.
  assert _about(busy.model_seconds(), _SLICE)
  assert _about(busy.tool_seconds(), _SLICE)
  # Together they still occupy only the tool's own stretch of wall clock.
  assert _about(busy.busy_seconds(), 2 * _SLICE)


@pytest.mark.asyncio
async def test_model_and_tool_time_sum_to_the_busy_total():
  """What makes the framework figure a remainder the three parts partition."""
  with _instrumentation.collect_busy_intervals() as busy:
    await _await_model()
    await _run_tool()

  assert (
      pytest.approx(busy.model_seconds() + busy.tool_seconds())
      == busy.busy_seconds()
  )


@pytest.mark.asyncio
async def test_every_chunk_of_a_streamed_response_is_timed():
  """Model time is the sum of the waits, not the life of the stream."""
  with _instrumentation.collect_busy_intervals() as busy:
    await _await_model(chunks=3)

  # Three chunk waits plus the pull that ends the stream, which waits too.
  assert _about(busy.model_seconds(), 3 * _SLICE)


@pytest.mark.asyncio
async def test_a_failed_model_call_still_counts_as_model_time():
  """A call the flow recovers from waited on the model like any other.

  Dropping it would quietly move that wait into the framework remainder.
  """

  async def responses():
    await asyncio.sleep(_SLICE)
    raise ValueError("model exploded")
    yield  # pylint: disable=unreachable

  with _instrumentation.collect_busy_intervals() as busy:
    with pytest.raises(ValueError):
      async for _ in _instrumentation.time_model_awaits(responses()):
        pass

  assert _about(busy.model_seconds(), _SLICE)


@pytest.mark.asyncio
async def test_work_outside_the_block_is_not_collected():
  """A collector sees its own turn, not whatever ran before or after it."""
  await _run_tool()

  with _instrumentation.collect_busy_intervals() as busy:
    pass

  await _run_tool()

  assert busy.tool_seconds() == 0.0


@pytest.mark.asyncio
async def test_collection_reaches_concurrently_spawned_tasks():
  """Sub-agents run as tasks; their tool intervals still reach the collector.

  The collector rides the otel context, which a task copies when it is
  created, so work the turn fans out to is covered without being threaded
  through by hand.
  """
  with _instrumentation.collect_busy_intervals() as busy:
    await asyncio.gather(
        asyncio.create_task(_run_tool()),
        asyncio.create_task(_run_tool()),
    )

  assert _about(busy.tool_seconds(), _SLICE)


@pytest.mark.asyncio
async def test_a_run_nobody_is_collecting_for_still_works():
  """The collector is optional; an uninstrumented run just is not measured."""
  await _run_tool()
  await _await_model()

  with _instrumentation.collect_busy_intervals() as busy:
    await _run_tool()

  assert _about(busy.tool_seconds(), _SLICE)
  assert busy.model_seconds() == 0.0
