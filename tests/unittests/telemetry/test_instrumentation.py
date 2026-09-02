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
from google.adk.telemetry import _hallucination
from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
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

from .functional._scenarios import install_telemetry


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
  """A stand-in for a loaded skill; only its source URI is read off it."""
  skill = mock.MagicMock()
  skill._uri = uri
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
  assert attributes["adk.experimental.skill.name"] == "sample_skill"
  assert attributes["adk.experimental.skill.script.path"] == "scripts/sample.py"
  assert attributes["adk.experimental.skill.script.exit_code"] == 0
  assert (
      attributes["adk.experimental.skill.source.uri"] == "file:/skills/sample"
  )
  assert span.status.status_code is StatusCode.UNSET
  assert "error.type" not in attributes
  assert telemetry.points(_SKILL_SCRIPT_EXECUTIONS) == [(
      {
          "gen_ai.agent.name": "root_agent",
          "adk.experimental.skill.name": "sample_skill",
          "adk.experimental.skill.script.path": "scripts/sample.py",
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
  assert attributes["adk.experimental.skill.script.exit_code"] == 3
  assert telemetry.point_attributes(_SKILL_SCRIPT_EXECUTIONS) == [{
      "gen_ai.agent.name": "root_agent",
      "adk.experimental.skill.name": "sample_skill",
      "adk.experimental.skill.script.path": "scripts/sample.py",
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
  assert "adk.experimental.skill.script.exit_code" not in attributes
  assert span.status.status_code is StatusCode.UNSET
  assert "error.type" not in attributes
  # The span keeps what the model actually asked for.
  assert attributes["adk.experimental.skill.name"] == "sample_skill"
  assert attributes["adk.experimental.skill.script.path"] == "scripts/sample.py"
  # The attempt still happened, so it is still counted -- just without a
  # verdict on how it ended, and without the unconfirmed names.
  assert telemetry.point_attributes(_SKILL_SCRIPT_EXECUTIONS) == [{
      "gen_ai.agent.name": "root_agent",
      "adk.experimental.skill.name": "<hallucinated>",
      "adk.experimental.skill.script.path": "<hallucinated>",
  }]


@pytest.mark.asyncio
async def test_skill_script_execution_is_silent_without_the_experimental_opt_in(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """These attributes are experimental, so nothing is emitted by default."""
  monkeypatch.delenv("ADK_EXPERIMENTAL_TELEMETRY", raising=False)
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _script_tool(), agent, {}, ctx
  ):
    skill_telemetry = _instrumentation.track_skill_script_execution(
        _hallucination.ConfirmedNotHallucinated("sample_skill"),
        _hallucination.ConfirmedNotHallucinated("scripts/sample.py"),
    )
    skill_telemetry.script_exit_code = 3

  span = telemetry.only_span()
  assert not [
      key for key in span.attributes if key.startswith("adk.experimental")
  ]
  assert span.status.status_code is StatusCode.UNSET
  assert telemetry.points(_SKILL_SCRIPT_EXECUTIONS) == []


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
  assert attributes["adk.experimental.skill.name"] == "sample_skill"
  assert (
      attributes["adk.experimental.skill.source.uri"] == "file:/skills/sample"
  )
  assert telemetry.points(_SKILL_LOADS) == [(
      {
          "gen_ai.agent.name": "root_agent",
          "adk.experimental.skill.name": "sample_skill",
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
  assert span_attributes["adk.experimental.skill.name"] == "sample_skill"
  assert telemetry.point_attributes(_SKILL_LOADS) == [{
      "gen_ai.agent.name": "root_agent",
      "adk.experimental.skill.name": "<hallucinated>",
      "error.type": "SKILL_NOT_FOUND",
  }]


@pytest.mark.asyncio
async def test_skill_load_is_silent_without_the_experimental_opt_in(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """These attributes are experimental, so nothing is emitted by default."""
  monkeypatch.delenv("ADK_EXPERIMENTAL_TELEMETRY", raising=False)
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_tool_execution(
      _load_tool(), agent, {}, ctx
  ):
    skill_telemetry = _instrumentation.track_skill_load(
        _hallucination.ConfirmedNotHallucinated("sample_skill")
    )
    skill_telemetry.skill = _loaded_skill()

  span = telemetry.only_span()
  assert not [
      key for key in span.attributes if key.startswith("adk.experimental")
  ]
  assert telemetry.points(_SKILL_LOADS) == []


@pytest.mark.asyncio
async def test_record_agent_invocation_flushes_the_skill_loads_it_made(
    telemetry: _Telemetry, monkeypatch: pytest.MonkeyPatch
):
  """Loads made anywhere under the agent land in its per-invocation total."""
  monkeypatch.setenv("ADK_EXPERIMENTAL_TELEMETRY", "true")
  agent = _agent()
  ctx = await _invocation_context(agent)

  async with _instrumentation.record_agent_invocation(ctx, agent):
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

  async with _instrumentation.record_agent_invocation(ctx, agent):
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
          "adk.experimental.skill.name": "first_skill",
      },
      {
          "gen_ai.agent.name": "root_agent",
          "adk.experimental.skill.name": "second_skill",
      },
      {
          "gen_ai.agent.name": "root_agent",
          "adk.experimental.skill.name": "<hallucinated>",
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

  async with _instrumentation.record_agent_invocation(ctx, agent):
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
      "adk.experimental.skill.name": "<hallucinated>",
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

  async with _instrumentation.record_agent_invocation(ctx, agent):
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

  async with _instrumentation.record_agent_invocation(ctx, agent):
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

  async with _instrumentation.record_agent_invocation(ctx, agent):
    async with _instrumentation.record_tool_execution(
        _load_tool(), agent, {}, ctx
    ) as tel_ctx:
      _instrumentation.track_skill_load("nonexistent_skill")
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

  with node_tracing._use_invoke_workflow_span("my_workflow", "conversation-1"):
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

  with node_tracing._use_invoke_workflow_span("outer", "conversation-1"):
    async with _instrumentation.record_agent_invocation(ctx, agent):
      async with _instrumentation.record_tool_execution(
          _load_tool(), agent, {}, ctx
      ):
        tel_ctx = _instrumentation.track_skill_load(
            _hallucination.ConfirmedNotHallucinated("outer_skill")
        )
        tel_ctx.skill = _loaded_skill()
      with node_tracing._use_invoke_workflow_span("inner", "conversation-1"):
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

  with node_tracing._use_invoke_workflow_span("my_workflow", "conversation-1"):
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

  with node_tracing._use_invoke_workflow_span("my_workflow", "conversation-1"):
    async with _instrumentation.record_tool_execution(
        _load_tool(), agent, {}, ctx
    ):
      tel_ctx = _instrumentation.track_skill_load(
          _hallucination.ConfirmedNotHallucinated("sample_skill")
      )
      tel_ctx.skill = _loaded_skill()

  assert telemetry.points(_INVOKE_WORKFLOW_SKILL_LOADS) == []


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
  """Token usage is read off the newest response that reports any, so both

  retention and order matter.
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
