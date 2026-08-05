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

"""Shared infrastructure for the telemetry functional tests.

This module hosts:

* The ``SpanDigest`` / ``LogDigest`` types used to build a deterministic
  comparison shape for in-memory spans + log records.
* ``install_telemetry`` which patches an in-memory tracer + log exporter
  onto ADK's globals.
* The canonical agent / workflow / MCP scenarios shared across the
  ``test_functional.py`` and ``test_node_functional.py`` test suites.
* The ``FunctionalTestCase`` carrier used to parametrize tests, whose
  ``expected`` telemetry is the recording loaded by
  ``functional_test_goldens.py``.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from contextlib import aclosing
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import gc
import inspect
import json
import sys
from types import CodeType
from typing import Literal
from typing import NamedTuple
from typing import TYPE_CHECKING

from google.adk.agents.llm_agent import Agent
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai.types import Content
from google.genai.types import FinishReason
from google.genai.types import Part
from mcp import ClientSession as McpClientSession
from mcp import StdioServerParameters
from mcp.types import ListToolsResult
from mcp.types import PaginatedRequestParams
from mcp.types import Tool as McpTool
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import HistogramDataPoint
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.metrics.export import NumberDataPoint
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import pytest
from typing_extensions import override

if TYPE_CHECKING:
  from google.adk.events.event import Event
  from opentelemetry.sdk.trace import ReadableSpan
  from opentelemetry.sdk._logs import ReadableLogRecord
  from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
  from opentelemetry.sdk.metrics.export import MetricsData
  from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner

# ---------------------------------------------------------------------------
# Env var + semconv constants.
# ---------------------------------------------------------------------------

OTEL_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"
ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"

# Stable semconv event names.
GEN_AI_SYSTEM_MESSAGE_EVENT = "gen_ai.system.message"
GEN_AI_USER_MESSAGE_EVENT = "gen_ai.user.message"
GEN_AI_CHOICE_EVENT = "gen_ai.choice"

# Experimental semconv event name.
GEN_AI_COMPLETION_DETAILS_EVENT = "gen_ai.client.inference.operation.details"

# Difficult to extract, non deterministic attribute keys.
# We check only for their presence, instead of their values.
NON_DETERMINISTIC_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gcp.vertex.agent.event_id",
    "gen_ai.tool.call.id",
    "gcp.vertex.agent.associated_event_ids",
    "gen_ai.conversation.id",
    "gcp.vertex.agent.invocation_id",
    "gcp.vertex.agent.session_id",
})

# Span attribute keys whose values are JSON-serialized strings.
# These are parsed back into Python objects before comparison so that JSON
# property ordering doesn't drive test stability.
JSON_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
})

# Sentinel for a value that cannot be pinned -- a generated id, a wall-clock
# duration, an elided payload. Substituted on both sides of the comparison, so
# such a field is only ever asserted to be present.
PRESENT = "PRESENT"

# Which end-to-end scenario a test case drives.
Scenario = Literal["agent", "node", "mcp"]


# ---------------------------------------------------------------------------
# Digests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogDigest:
  """A deterministic digest of a ``ReadableLogRecord``.

  ``attributes`` and ``body`` are normalized via ``_normalize`` so test
  expectations can be written using plain Python literals (lists/dicts).
  """

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


@dataclass(frozen=True)
class SpanDigest:
  """A deterministic digest of a span in the in-memory span tree.

  In addition to the span's own name + attributes + child spans, each
  digest also carries the ``LogDigest`` records that were emitted while
  the span was the active span (matched by ``log_record.span_id``).

  ``status`` is the span's ``StatusCode`` name, so a tree that expects a
  span to be marked failed says so explicitly. It defaults to ``UNSET``,
  which is what a span that nothing marked carries.
  """

  name: str
  attributes: dict[str, object]
  status: str = "UNSET"
  children: list[SpanDigest] = field(default_factory=list)
  logs: list[LogDigest] = field(default_factory=list)

  @classmethod
  def from_span(cls, span: ReadableSpan) -> SpanDigest:
    """Builds a single ``SpanDigest`` (no children, no logs) from a span.

    Attribute values are normalized so that:
    * Non-deterministic keys collapse to the ``PRESENT`` sentinel.
    * JSON-serialized attribute values are parsed into Python objects.
    * All other values pass through ``_normalize`` (tuples → lists,
      enums → ``.value``, ``None`` dict entries dropped).
    """
    determinized_attributes: dict[str, object] = {}
    for attr_key, attr_val in (span.attributes or {}).items():
      if attr_key in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        determinized_attributes[attr_key] = PRESENT
      elif attr_key in JSON_ATTRIBUTE_KEYS and isinstance(attr_val, str):
        determinized_attributes[attr_key] = _normalize(json.loads(attr_val))
      else:
        determinized_attributes[attr_key] = _normalize(attr_val)
    return cls(
        name=span.name,
        attributes=determinized_attributes,
        status=span.status.status_code.name,
    )

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...] = (),
  ) -> SpanDigest:
    """Builds the in-memory span tree, attaching logs by span id.

    Used for clear diffs with pytest assertions.
    """
    digest_by_id: dict[int, SpanDigest] = {}
    for span in spans:
      if span.context is None:
        continue
      digest_by_id[span.context.span_id] = cls.from_span(span)

    # Attach each log to its enclosing span (matched by span_id).
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

    # Sort for deterministic comparisons.
    for digest in digest_by_id.values():
      digest.children.sort(key=lambda s: s.name)
      digest.logs[:] = sorted_log_digests(digest.logs)

    if root is None:
      raise ValueError("No root span found in the provided spans.")
    return root

  def all_logs(self) -> list[LogDigest]:
    """Returns all log digests in the tree, sorted deterministically."""
    collected: list[LogDigest] = []

    def _walk(node: SpanDigest) -> None:
      collected.extend(node.logs)
      for child in node.children:
        _walk(child)

    _walk(self)
    return sorted_log_digests(collected)


def sorted_log_digests(logs: list[LogDigest]) -> list[LogDigest]:
  """Returns ``logs`` sorted in a stable, content-derived order."""
  return sorted(
      logs,
      key=lambda log: (
          log.event_name,
          json.dumps(log.body, sort_keys=True, default=str),
          json.dumps(log.attributes, sort_keys=True, default=str),
      ),
  )


@dataclass(frozen=True)
class MetricPoint:
  """A single recorded metric data point."""

  attributes: dict[str, object]
  value: object

  def __hash__(self) -> int:
    return hash((self.sort_key(), self.value))

  def sort_key(self) -> str:
    return json.dumps(self.attributes, sort_keys=True, default=str)


class HistogramSpec(NamedTuple):
  """Locates one ADK metric histogram so a test can redirect it.

  ``module`` is the module holding the histogram, ``attr`` the global on it to
  monkeypatch, and ``metric_name`` the instrument name it is recreated under.
  """

  module: object
  attr: str
  metric_name: str


# Histograms recorded by ADK. Each test redirects these onto an in-memory
# reader so the recorded points can be asserted.
_PATCHED_HISTOGRAMS: tuple[HistogramSpec, ...] = (
    HistogramSpec(
        module=_metrics,
        attr="_agent_invocation_duration",
        metric_name="gen_ai.invoke_agent.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_tool_execution_duration",
        metric_name="gen_ai.execute_tool.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_client_operation_duration",
        metric_name="gen_ai.client.operation.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_client_token_usage",
        metric_name="gen_ai.client.token.usage",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_workflow_invocation_duration",
        metric_name="gen_ai.invoke_workflow.duration",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_inference_calls",
        metric_name="gen_ai.invoke_agent.inference_calls",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_tool_calls",
        metric_name="gen_ai.invoke_agent.tool_calls",
    ),
)


def _grouped_metric_points(
    metrics_data: MetricsData,
) -> dict[str, list[MetricPoint]]:
  """Groups every recorded point by metric name.

  Both the names and the points within a group are sorted, so the result is
  independent of recording order and can be compared (and serialized) as
  plain lists.
  """
  grouped: dict[str, set[MetricPoint]] = {}
  for resource_metric in metrics_data.resource_metrics:
    for scope_metric in resource_metric.scope_metrics:
      for metric in scope_metric.metrics:
        for dp in metric.data.data_points:
          # Sum histograms expose ``.sum``; gauge / counter points expose
          # ``.value``. isinstance (not hasattr) keeps the typing precise.
          if isinstance(dp, HistogramDataPoint):
            value = dp.sum
          elif isinstance(dp, NumberDataPoint):
            value = dp.value
          else:
            value = PRESENT
          # ``*.duration`` histograms record wall-clock timings, which are
          # non-deterministic; replace them so expectations need not pin a
          # timing.
          if metric.name.endswith(".duration"):
            value = PRESENT
          grouped.setdefault(metric.name, set()).add(
              MetricPoint(attributes=dict(dp.attributes), value=value)
          )
  return {
      name: sorted(points, key=MetricPoint.sort_key)
      for name, points in sorted(grouped.items())
  }


@dataclass(frozen=True)
class TelemetryDigest:
  """The full telemetry surface produced by one scenario run.

  Bundles the root span tree (with per-span logs attached) and every recorded
  metric point grouped by metric name. Everything is sorted as it is built,
  so a digest is fully deterministic and round-trips through plain JSON:
  ``build`` produces the actual one; ``functional_test_goldens.load_golden``
  the recorded one.
  """

  root_span: SpanDigest
  metric_points: dict[str, list[MetricPoint]]

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...],
      metrics_data: MetricsData,
  ) -> TelemetryDigest:
    """Builds the actual digest from in-memory spans, logs and metrics."""
    return cls(
        root_span=SpanDigest.build(spans, logs),
        metric_points=_grouped_metric_points(metrics_data),
    )


def _normalize(value: object) -> object:
  """Normalizes a value for stable equality.

  * Tuples become lists (OTel coerces sequences to tuples on attributes).
  * Enums become their ``.value``.
  * Dict entries whose value is ``None`` are dropped (these are inserted by
    pydantic ``model_dump`` for unset fields and would dominate diffs).
  """
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, tuple):
    return [_normalize(v) for v in value]
  if isinstance(value, list):
    return [_normalize(v) for v in value]
  if isinstance(value, dict):
    return {k: _normalize(v) for k, v in value.items() if v is not None}
  return value


# ---------------------------------------------------------------------------
# Telemetry plumbing.
# ---------------------------------------------------------------------------


def install_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
  """Installs an in-memory tracer + log exporter + metric reader.

  Spans, logs and metric points emitted by ADK during the test are written
  into the provided exporters / reader. All three MUST be passed in so each
  test makes the choice of sink explicit (e.g. ``InMemoryLogRecordExporter``
  vs ``WebUILogExporter``).
  """
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  monkeypatch.setattr(
      tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  monkeypatch.setattr(
      tracing.tracer,
      "start_span",
      real_tracer.start_span,
  )
  monkeypatch.setattr(
      node_tracing.tracer,
      "start_as_current_span",
      real_tracer.start_as_current_span,
  )
  monkeypatch.setattr(
      node_tracing.tracer,
      "start_span",
      real_tracer.start_span,
  )

  logger_provider = LoggerProvider()
  logger_provider.add_log_record_processor(
      SimpleLogRecordProcessor(log_exporter)
  )
  real_logger = logger_provider.get_logger(__name__)
  monkeypatch.setattr(tracing.otel_logger, "emit", real_logger.emit)

  meter_provider = MeterProvider(metric_readers=[metric_reader])
  meter = meter_provider.get_meter("functional_test_meter")
  for spec in _PATCHED_HISTOGRAMS:
    monkeypatch.setattr(
        spec.module, spec.attr, meter.create_histogram(spec.metric_name)
    )


# ---------------------------------------------------------------------------
# Canonical agent / tool / mock-LLM scenario.
# ---------------------------------------------------------------------------

USER_PROMPT = "hello"
AGENT_NAME = "some_root_agent"
AGENT_DESCRIPTION = "A sample root agent."
BASE_INSTRUCTION = "you are helpful"
# ADK auto-appends agent identity info to the system instruction when the
# agent is invoked as the root of an InMemoryRunner directly.
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

# The node scenario uses a workflow node whose output drives the agent's
# input. The workflow itself wraps the same agent.
WORKFLOW_NAME = "my_workflow"
# The root workflow invokes a nested workflow whose sole node produces the
# input for the agent. The nested workflow exercises the `gen_ai.workflow.nested`
# span attribute + metric dimension (only nested workflows carry it).
NESTED_WORKFLOW_NAME = "my_nested_workflow"
NODE_NAME = "some_node"
NODE_RESULT = "some result"
NODE_USER_ID = "some_user"
NODE_APP_NAME = "some_app"


def _make_llm_response(part: Part) -> LlmResponse:
  return LlmResponse(
      content=Content(role="model", parts=[part]),
      finish_reason=FinishReason.STOP,
  )


def build_test_agent(
    *, failing: bool = False, model_exception: Exception | None = None
) -> Agent:
  """Builds the canonical 1-tool, 2-LLM-turn agent.

  If ``model_exception`` is provided, the mock model raises it instead of
  returning any response, exercising the inference-failure telemetry path.
  """
  # When the model is meant to raise, leave the responses empty so the mock
  # never yields; otherwise it returns the canonical 2-turn conversation.
  mock_model = MockModel.create(
      responses=(
          []
          if model_exception is not None
          else [
              _make_llm_response(
                  Part.from_function_call(name=TOOL_NAME, args=TOOL_ARGS)
              ),
              _make_llm_response(Part.from_text(text=FINAL_TEXT)),
          ]
      ),
      error=model_exception,
  )

  def some_tool(arg1: str) -> str:
    """A sample tool."""
    if failing:
      raise ValueError("This tool always fails")

    return f"{TOOL_RESULT_PREFIX}{arg1}"

  return Agent(
      name=AGENT_NAME,
      description=AGENT_DESCRIPTION,
      instruction=BASE_INSTRUCTION,
      model=mock_model,
      tools=[FunctionTool(some_tool)],
  )


def build_test_runner(
    *, failing: bool = False, model_exception: Exception | None = None
) -> TestInMemoryRunner:
  """Builds a runner around the canonical agent (no workflow wrapper)."""
  return TestInMemoryRunner(
      node=build_test_agent(failing=failing, model_exception=model_exception)
  )


def build_test_workflow(
    *, failing: bool = False, model_exception: Exception | None = None
) -> Workflow:
  """Builds the canonical Workflow: a nested workflow feeding the agent."""
  test_agent = build_test_agent(
      failing=failing, model_exception=model_exception
  )

  async def some_node(ctx, node_input):
    return NODE_RESULT

  # Trivial workflow to test o11y of nested workflows
  nested_workflow = Workflow(
      name=NESTED_WORKFLOW_NAME,
      edges=[(START, some_node)],
  )

  return Workflow(
      name=WORKFLOW_NAME,
      edges=[(START, nested_workflow, test_agent)],
  )


async def run_node_scenario(
    *, failing: bool = False, event_sink: list[Event] | None = None
) -> list[Event]:
  """Runs the workflow scenario to completion, draining the event stream.

  If ``event_sink`` is provided, collected events are appended to it as they
  are drained. This lets callers inspect the events that were emitted before
  an exception propagates (e.g. when ``failing=True``).
  """
  workflow = build_test_workflow(failing=failing)
  runner = InMemoryRunner(app_name=NODE_APP_NAME, node=workflow)
  session = await runner.session_service.create_session(
      app_name=NODE_APP_NAME, user_id=NODE_USER_ID
  )
  content = Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")

  collected_events: list[Event] = event_sink if event_sink is not None else []

  async with aclosing(
      runner.run_async(
          user_id=NODE_USER_ID,
          session_id=session.id,
          new_message=content,
      )
  ) as agen:
    async for event in agen:
      collected_events.append(event)

  return collected_events


async def run_agent_scenario(runner: TestInMemoryRunner) -> None:
  """Runs the non-node scenario to completion, draining the event stream."""
  async with aclosing(
      runner.run_async_with_new_session_agen(
          Content(parts=[Part.from_text(text=USER_PROMPT)], role="user")
      )
  ) as agen:
    async for _ in agen:
      pass


# ---------------------------------------------------------------------------
# MCP scenario.
#
# A ``FakeMcpSession`` substitutes the live ``McpClientSession`` so the
# scenario doesn't need a running MCP server. ``McpToolset.create_session`` is
# patched to hand it out instead of dialing ``StdioServerParameters``.
# ---------------------------------------------------------------------------

MCP_TOOL_NAME = "mcp_echo"
MCP_TOOL_DESCRIPTION = "Echoes back its input."


class FakeMcpSession(McpClientSession):
  """Minimal ``McpClientSession`` stand-in with a counted ``list_tools()``.

  Subclasses ``McpClientSession`` (and skips its real ``__init__``) so that
  every ``isinstance(x, McpClientSession)`` check in ADK and in the MCP
  Python client passes, without needing to wire up the underlying anyio
  memory streams + peer process.
  """

  def __init__(  # pyright: ignore[reportMissingSuperCall]
      self, *, tools: list[McpTool] | None = None
  ) -> None:
    # Deliberately skip ``McpClientSession.__init__``: the real one wants
    # live anyio streams + a peer process. ``isinstance`` checks still
    # succeed, which is all ADK's MCP plumbing requires.
    self._tools: list[McpTool] = (
        tools if tools is not None else [_default_mcp_tool()]
    )
    self.list_tools_call_count: int = 0

  @override
  async def list_tools(
      self,
      cursor: str | None = None,
      *,
      params: PaginatedRequestParams | None = None,
  ) -> ListToolsResult:
    self.list_tools_call_count += 1
    return ListToolsResult(tools=list(self._tools))


def _default_mcp_tool() -> McpTool:
  return McpTool(
      name=MCP_TOOL_NAME,
      description=MCP_TOOL_DESCRIPTION,
      inputSchema={
          "type": "object",
          "properties": {"text": {"type": "string"}},
          "required": ["text"],
      },
  )


def build_mcp_test_runner(
    monkeypatch: pytest.MonkeyPatch, fake_session: FakeMcpSession
) -> TestInMemoryRunner:
  """Builds a single-turn agent runner whose only tool source is MCP.

  Patches the toolset's ``MCPSessionManager`` so ``create_session`` returns
  ``fake_session`` (no socket / subprocess) and ``close`` is a no-op.
  Single-turn (one ``Part.from_text`` response) so an assertion on
  ``fake_session.list_tools_call_count`` is unambiguous: exactly one agent
  invocation is performed.
  """
  toolset = McpToolset(
      connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(command="unused-by-test"),
      )
  )

  async def _create_session(*_args, **_kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return fake_session

  async def _close(*_args, **_kwargs):  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return None

  monkeypatch.setattr(
      toolset._mcp_session_manager, "create_session", _create_session  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]
  )
  monkeypatch.setattr(toolset._mcp_session_manager, "close", _close)  # pyright: ignore[reportPrivateUsage, reportUnknownArgumentType]

  mock_model = MockModel.create(responses=[Part.from_text(text=FINAL_TEXT)])
  return TestInMemoryRunner(
      node=Agent(
          name=AGENT_NAME,
          description=AGENT_DESCRIPTION,
          instruction=BASE_INSTRUCTION,
          model=mock_model,
          tools=[toolset],
      )
  )


# ---------------------------------------------------------------------------
# Parametrization carrier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalTestCase:
  """One row of the (semconv, capture-content, schema-version) matrix."""

  test_id: str
  scenario: Scenario
  semconv_opt_in: str | None
  capture_content: str | None
  schema_version: Literal[1, 2]
  # When set, the mock model raises this instead of responding, and the
  # scenario is expected to propagate it (inference-failure telemetry path).
  model_exception: Exception | None = None
  # When true, the tool raises instead of returning, and the scenario is
  # expected to propagate it (tool-failure telemetry path).
  tool_fails: bool = False

  @property
  def expects_failure(self) -> bool:
    """Whether the scenario is expected to propagate an exception."""
    return self.model_exception is not None or self.tool_fails

  @property
  def expected(self) -> TelemetryDigest:
    """The telemetry recorded for this case under ``functional_goldens/``."""
    # Imported here: the goldens module needs the digest types defined above.
    from .functional_test_goldens import load_golden  # pylint: disable=g-import-not-at-top

    return load_golden(self.scenario, self.test_id)

  def apply_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Applies the per-case env vars for semconv + content capture.

    Always pins ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`` so the tool
    span attributes remain deterministic across all cases.
    """
    if self.semconv_opt_in is None:
      monkeypatch.delenv(OTEL_OPT_IN, raising=False)
    else:
      monkeypatch.setenv(OTEL_OPT_IN, self.semconv_opt_in)
    if self.capture_content is None:
      monkeypatch.delenv(CAPTURE_CONTENT, raising=False)
    else:
      monkeypatch.setenv(CAPTURE_CONTENT, self.capture_content)
    monkeypatch.setenv(
        ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN, str(self.schema_version)
    )
    monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")


# ---------------------------------------------------------------------------
# aclosing wrapping assertions.
# ---------------------------------------------------------------------------


@contextmanager
def aclosing_wrapping_assertions() -> Iterator[None]:
  """Context manager that asserts every async generator is wrapped in ``aclosing``.

  The check uses ``gc.get_referrers`` on every async generator first
  iterated within the block, which is expensive (~5 seconds per
  scenario). Run this once per scenario rather than per parametrized
  test case.

  On exit the original ``sys`` async-gen hooks are restored.
  """
  prev_firstiter, prev_finalizer = sys.get_asyncgen_hooks()

  def wrapped_firstiter(coro: AsyncGenerator[object, object]):
    if _is_async_context_manager():
      if prev_firstiter:
        prev_firstiter(coro)
      return

    assert any(
        isinstance(referrer, aclosing)
        or isinstance(indirect_referrer, aclosing)
        for referrer in gc.get_referrers(coro)
        # Some coroutines have a layer of indirection in Python 3.10
        for indirect_referrer in gc.get_referrers(referrer)
    ), _no_aclosing_assertion_error(coro)

    if prev_firstiter:
      prev_firstiter(coro)

  sys.set_asyncgen_hooks(wrapped_firstiter, prev_finalizer)
  try:
    yield
  finally:
    sys.set_asyncgen_hooks(prev_firstiter, prev_finalizer)


def _no_aclosing_assertion_error(coro: AsyncGenerator[object, object]) -> str:
  first_iter_loc = ""
  definition_loc = ""

  if (f := inspect.currentframe()) and (f := f.f_back) and (f := f.f_back):
    first_iter_loc = f'file "{f.f_code.co_filename}" line "{f.f_lineno}"'
  if (ag_code := getattr(coro, "ag_code", None)) and isinstance(
      ag_code, CodeType
  ):
    definition_loc = (
        f'file "{ag_code.co_filename}" line "{ag_code.co_firstlineno}"'
    )

  header_str = f'Async generator "{coro.__name__}" is not wrapped in aclosing'
  first_iter_str = (
      f"first iterated in {first_iter_loc}" if first_iter_loc else ""
  )
  definition_str = f"defined in {definition_loc}" if definition_loc else ""
  instruction_str = """
Wrap the iteration in the following code snippet before iterating:

async with contextlib.aclosing(...) as agen:
  async for ... as agen:
     ...
"""

  return "\n".join(
      part
      for part in [
          header_str,
          first_iter_str,
          definition_str,
          instruction_str,
      ]
      if part
  )


def _is_async_context_manager() -> bool:
  """Checks if this function was invoked by contextlib.asynccontextmanager."""
  frame = inspect.currentframe()
  while frame:
    if (
        frame.f_code.co_name == "__aenter__"
        and "contextlib" in frame.f_code.co_filename
    ):
      return True
    frame = frame.f_back
  return False
