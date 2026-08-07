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

from google.adk.telemetry import tracing
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE
from .functional_test_helpers import aclosing_wrapping_assertions
from .functional_test_helpers import build_mcp_test_runner
from .functional_test_helpers import build_test_runner
from .functional_test_helpers import CAPTURE_CONTENT
from .functional_test_helpers import EXPERIMENTAL_OPT_IN
from .functional_test_helpers import FakeMcpSession
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
  recorded metric points) matches the shape recorded for the given semconv +
  content-capture configuration in ``functional_goldens/``.
  """
  case.apply_env(monkeypatch)

  span_exporter = InMemorySpanExporter()
  log_exporter = InMemoryLogRecordExporter()
  metric_reader = InMemoryMetricReader()
  install_telemetry(monkeypatch, span_exporter, log_exporter, metric_reader)

  if case.model_exception is not None:
    # The mock raises before responding; the scenario must propagate it.
    with pytest.raises(Exception):  # noqa: B017 -- exact type varies per case.
      await run_agent_scenario(
          build_test_runner(model_exception=case.model_exception)
      )
  elif case.tool_fails:
    # The tool raises while the model is fine; the scenario must propagate it.
    with pytest.raises(ValueError, match="This tool always fails"):
      await run_agent_scenario(build_test_runner(failing=True))
  else:
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


def test_instrumented_detection_normalizes_windows_path_separators(
    monkeypatch: pytest.MonkeyPatch,
):
  """Backslash-separated instrumentation paths are matched on Windows."""
  windows_path = r"C:\pkg\opentelemetry\instrumentation\google_genai\patch.py"

  class _FakeCode:
    co_filename = windows_path

  class _FakeInstrumentedFunction:
    __code__ = _FakeCode
    __wrapped__ = object()

  monkeypatch.setattr(
      tracing.Models, "generate_content", _FakeInstrumentedFunction
  )

  assert tracing._instrumented_with_opentelemetry_instrumentation_google_genai()


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
# (or capture mode) is active. This test pins that contract; the recorded
# ``mcp`` golden pins that the resolved tool definitions surface intact in
# the experimental telemetry.
# ---------------------------------------------------------------------------


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

  fake_session = FakeMcpSession()

  await run_agent_scenario(build_mcp_test_runner(monkeypatch, fake_session))

  assert fake_session.list_tools_call_count == 1

  digest = SpanDigest.build(
      span_exporter.get_finished_spans(),
      log_exporter.get_finished_logs(),
  )
  assert digest == MCP_CASE.expected.root_span
