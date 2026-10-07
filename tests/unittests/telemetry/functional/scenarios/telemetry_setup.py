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

"""The telemetry the functional tests record into.

``install_telemetry`` points ADK's telemetry globals at in-memory exporters,
and the env vars the instrumentations are configured by are named here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from google.adk.telemetry import _metrics
from google.adk.telemetry import node_tracing
from google.adk.telemetry import tracing
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

# ---------------------------------------------------------------------------
# Env var + semconv constants.
# ---------------------------------------------------------------------------

OTEL_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
CAPTURE_CONTENT = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
EXPERIMENTAL_OPT_IN = "gen_ai_latest_experimental"
ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN = "ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN"
ADK_EXPERIMENTAL_TELEMETRY = "ADK_EXPERIMENTAL_TELEMETRY"


# ---------------------------------------------------------------------------
# Telemetry plumbing.
# ---------------------------------------------------------------------------


class HistogramSpec(NamedTuple):
  """Locates one ADK metric histogram so a test can redirect it.

  ``module`` is the module holding the histogram, ``attr`` the global on it to
  monkeypatch, and ``metric_name`` the instrument name it is recreated under.
  """

  module: object
  attr: str
  metric_name: str


CounterSpec = HistogramSpec

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
    # Per-agent token spend, recorded once per agent invocation.
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_input_tokens",
        metric_name="adk.experimental.invoke_agent.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_output_tokens",
        metric_name="adk.experimental.invoke_agent.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_total_tokens",
        metric_name="adk.experimental.invoke_agent.total_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_cache_read_input_tokens",
        metric_name="adk.experimental.invoke_agent.cache_read.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_reasoning_output_tokens",
        metric_name="adk.experimental.invoke_agent.reasoning.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_tool_input_tokens",
        metric_name="adk.experimental.invoke_agent.tool.input_tokens",
    ),
    # The same spend summed over the whole turn, dropping the agent key.
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_input_tokens",
        metric_name="adk.experimental.invoke_workflow.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_output_tokens",
        metric_name="adk.experimental.invoke_workflow.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_total_tokens",
        metric_name="adk.experimental.invoke_workflow.total_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_cache_read_input_tokens",
        metric_name="adk.experimental.invoke_workflow.cache_read.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_reasoning_output_tokens",
        metric_name="adk.experimental.invoke_workflow.reasoning.output_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_tool_input_tokens",
        metric_name="adk.experimental.invoke_workflow.tool.input_tokens",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_inference_calls",
        metric_name="adk.experimental.invoke_workflow.inference_calls",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_tool_calls",
        metric_name="adk.experimental.invoke_workflow.tool_calls",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_agent_skill_loads",
        metric_name="adk.experimental.invoke_agent.skill.loads",
    ),
    HistogramSpec(
        module=_metrics,
        attr="_invoke_workflow_skill_loads",
        metric_name="adk.experimental.invoke_workflow.skill.loads",
    ),
)

_PATCHED_COUNTERS: tuple[CounterSpec, ...] = (
    CounterSpec(
        module=_metrics,
        attr="_skill_script_executions",
        metric_name="adk.experimental.skill.script.executions",
    ),
    CounterSpec(
        module=_metrics,
        attr="_skill_loads",
        metric_name="adk.experimental.skill.loads",
    ),
)


@dataclass(frozen=True)
class TelemetryProviders:
  """The in-memory providers ``install_telemetry`` wired up.

  ADK reads its globals, so it needs no provider; the OTel google-genai
  instrumentor takes them as ``instrument()`` kwargs.
  """

  tracer_provider: TracerProvider
  logger_provider: LoggerProvider
  meter_provider: MeterProvider


def install_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    span_exporter: InMemorySpanExporter,
    log_exporter: InMemoryLogRecordExporter,
    metric_reader: InMemoryMetricReader,
) -> TelemetryProviders:
  """Installs an in-memory tracer + log exporter + metric reader.

  Spans, logs and metric points emitted by ADK during the test are written
  into the provided exporters / reader. All three MUST be passed in so each
  test makes the choice of sink explicit (e.g. ``InMemoryLogRecordExporter``
  vs ``WebUILogExporter``).

  Returns the providers behind them, for instrumentations that are configured
  with providers rather than by patching ADK's globals.
  """
  tracer_provider = TracerProvider()
  tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
  real_tracer = tracer_provider.get_tracer(__name__)

  for module in (tracing, node_tracing):
    monkeypatch.setattr(
        module.tracer,
        "start_as_current_span",
        real_tracer.start_as_current_span,
    )
    monkeypatch.setattr(module.tracer, "start_span", real_tracer.start_span)

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

  for spec in _PATCHED_COUNTERS:
    monkeypatch.setattr(
        spec.module, spec.attr, meter.create_counter(spec.metric_name)
    )

  return TelemetryProviders(
      tracer_provider=tracer_provider,
      logger_provider=logger_provider,
      meter_provider=meter_provider,
  )
