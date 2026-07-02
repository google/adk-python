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

from google.adk.agents.llm_agent import Agent
from google.adk.telemetry import tracing
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai.types import Part
from mcp import ClientSession as McpClientSession
from mcp import StdioServerParameters
from mcp.types import ListToolsResult
from mcp.types import PaginatedRequestParams
from mcp.types import Tool as McpTool
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
from typing_extensions import override

from ..testing_utils import MockModel
from ..testing_utils import TestInMemoryRunner
from .functional_test_cases import ALL_CASES
from .functional_test_cases import EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT_WITH_MCP
from .functional_test_helpers import aclosing_wrapping_assertions
from .functional_test_helpers import build_test_runner
from .functional_test_helpers import CAPTURE_CONTENT
from .functional_test_helpers import EXPERIMENTAL_OPT_IN
from .functional_test_helpers import FunctionalTestCase
from .functional_test_helpers import install_telemetry
from .functional_test_helpers import OTEL_OPT_IN
from .functional_test_helpers import run_agent_scenario
from .functional_test_helpers import SpanDigest
from .functional_test_helpers import TelemetryDigest


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.test_id)
@pytest.mark.asyncio
async def test_telemetry_schema(
    case: FunctionalTestCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Tests creation of spans/logs/metrics in an E2E runner invocation.

  Asserts the entire telemetry schema (spans + attributes + per-span logs +
  recorded metric points) matches the hand-written expected shape for the
  given semconv + content-capture configuration.
  """
  case.apply_env(monkeypatch)

  span_exporter = InMemorySpanExporter()
  log_exporter = InMemoryLogRecordExporter()
  metric_reader = InMemoryMetricReader()
  install_telemetry(monkeypatch, span_exporter, log_exporter, metric_reader)

  await run_agent_scenario(build_test_runner())

  digest = TelemetryDigest.build(
      span_exporter.get_finished_spans(),
      log_exporter.get_finished_logs(),
      metric_reader.get_metrics_data(),
  )
  assert digest == case.expected


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
    await run_agent_scenario(build_test_runner())


@pytest.mark.asyncio
async def test_exception_preserves_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Test when an exception occurs during tool execution, span attributes are still present on spans where they are expected."""

  span_exporter = InMemorySpanExporter()
  install_telemetry(
      monkeypatch,
      span_exporter,
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )

  with pytest.raises(ValueError, match="This tool always fails"):
    _ = await run_agent_scenario(build_test_runner(failing=True))

  spans = span_exporter.get_finished_spans()

  assert len(spans) > 1
  assert all(
      span.attributes is not None and len(span.attributes) > 0
      for span in spans
      if span.name != "invocation"  # not expected to have attributes
  )


@pytest.mark.asyncio
async def test_no_generate_content_for_gemini_model_when_already_instrumented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Tests that generate_content span is not created if already instrumented."""
  span_exporter = InMemorySpanExporter()
  install_telemetry(
      monkeypatch,
      span_exporter,
      InMemoryLogRecordExporter(),
      InMemoryMetricReader(),
  )

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

  _ = await run_agent_scenario(build_test_runner())

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


# ---------------------------------------------------------------------------
# MCP integration: telemetry adds zero ``list_tools()`` calls of its own.
#
# The standard ADK ↔ MCP integration path is:
#
#   Agent(tools=[McpToolset(...)])
#     → McpToolset.get_tools()  ─ calls list_tools() ONCE, caches MCPTool list
#     → BaseLlmFlow loop calls each MCPTool.process_llm_request, which
#       materializes the tool's FunctionDeclaration into
#       llm_request.config.tools.
#
# By the time the experimental semconv builder reads
# ``llm_request.config.tools``, MCP tools are ALREADY ``types.Tool``
# entries with ``function_declarations``. Because the builder is fully
# synchronous (it never calls ``list_tools()`` itself), the MCP server is
# queried EXACTLY ONCE per agent invocation regardless of which semconv
# (or capture mode) is active. These tests pin that contract AND verify
# the resolved tool definitions surface intact in the experimental
# telemetry.
#
# A ``_FakeMcpSession`` substitutes the live ``McpClientSession`` so the
# test doesn't need a running MCP server. ``McpToolset.create_session``
# is patched to hand it out instead of dialing ``StdioServerParameters``.
# ---------------------------------------------------------------------------


class _FakeMcpSession(McpClientSession):
  """Minimal ``McpClientSession`` stand-in with a counted ``list_tools()``.

  Subclasses ``McpClientSession`` (and skips its real ``__init__``) so that
  every ``isinstance(x, McpClientSession)`` check in ADK and in the MCP
  Python client passes, without needing to wire up the underlying anyio
  memory streams + peer process.
  """

  def __init__(  # pyright: ignore[reportMissingSuperCall]
      self, *, tools: list[McpTool]
  ) -> None:
    # Deliberately skip ``McpClientSession.__init__``: the real one wants
    # live anyio streams + a peer process. ``isinstance`` checks still
    # succeed, which is all ADK's MCP plumbing requires.
    self._tools: list[McpTool] = tools
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


def _make_fake_mcp_toolset(
    monkeypatch: pytest.MonkeyPatch, fake_session: _FakeMcpSession
) -> McpToolset:
  """Returns an ``McpToolset`` whose session manager hands out ``fake_session``.

  Patches the toolset's ``MCPSessionManager`` so:
    * ``create_session`` returns the fake (no socket / subprocess).
    * ``close`` is a no-op (the fake holds no resources).

  Connection params are nominally a stdio command but never actually
  invoked because ``create_session`` is overridden.
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
  return toolset


def _build_mcp_test_runner(toolset: McpToolset) -> TestInMemoryRunner:
  """Builds a single-turn agent runner whose only tool source is ``toolset``.

  Single-turn (one ``Part.from_text`` response) so the assertion on
  ``list_tools_call_count`` is unambiguous: exactly one agent invocation
  is performed.
  """
  mock_model = MockModel.create(
      responses=[Part.from_text(text="text response")]
  )
  test_agent = Agent(
      name="some_root_agent",
      description="A sample root agent.",
      instruction="you are helpful",
      model=mock_model,
      tools=[toolset],
  )
  return TestInMemoryRunner(node=test_agent)


@pytest.mark.asyncio
async def test_mcp_list_tools_called_once_under_experimental_semconv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Experimental semconv: exactly one ``list_tools()`` call per invocation.

  By the time the experimental semconv builder inspects
  ``llm_request.config.tools``, ``McpToolset`` has already materialized
  each MCP tool into a ``FunctionDeclaration`` — so the synchronous
  builder never has to (and never does) talk to the MCP server. The
  MCP-resolved tool definition still surfaces in the experimental
  telemetry intact, sourced from the ``FunctionDeclaration`` rather than
  from a fresh ``list_tools()`` call.
  """
  monkeypatch.setenv(OTEL_OPT_IN, EXPERIMENTAL_OPT_IN)
  monkeypatch.setenv(CAPTURE_CONTENT, "span_and_event")
  monkeypatch.setenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")

  span_exporter = InMemorySpanExporter()
  log_exporter = InMemoryLogRecordExporter()
  install_telemetry(
      monkeypatch, span_exporter, log_exporter, InMemoryMetricReader()
  )

  fake_session = _FakeMcpSession(
      tools=[
          McpTool(
              name="mcp_echo",
              description="Echoes back its input.",
              inputSchema={
                  "type": "object",
                  "properties": {"text": {"type": "string"}},
                  "required": ["text"],
              },
          )
      ]
  )
  toolset = _make_fake_mcp_toolset(monkeypatch, fake_session)

  await run_agent_scenario(_build_mcp_test_runner(toolset))

  assert fake_session.list_tools_call_count == 1

  digest = SpanDigest.build(
      span_exporter.get_finished_spans(),
      log_exporter.get_finished_logs(),
  )
  assert digest == EXPECTED_EXPERIMENTAL_SPAN_AND_EVENT_WITH_MCP
