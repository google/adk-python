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

"""Exactly one of ADK and the google-genai instrumentor records an inference."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import functools

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.telemetry import _genai_instrumentor
from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from google.adk.telemetry.context import TelemetryConfig
from google.genai import types
from google.genai.models import AsyncModels
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.instrumentation.google_genai import version as google_genai_version
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

_MODEL = "gemini-2.0-flash"
_ADK_SCOPE = "gcp.vertex.agent"
_INSTRUMENTOR_SCOPE = "opentelemetry.instrumentation.google_genai"

_RESPONSE = types.GenerateContentResponse(
    candidates=[
        types.Candidate(
            content=types.Content(
                role="model", parts=[types.Part(text="hello")]
            ),
            finish_reason=types.FinishReason.STOP,
        )
    ]
)


async def _sdk_stub(
    self: AsyncModels, *, model: str, contents: object, config: object = None
) -> types.GenerateContentResponse:
  del self, model, contents, config  # Unused.
  return _RESPONSE


@pytest.fixture(name="provider")
def _provider_fixture(monkeypatch: pytest.MonkeyPatch) -> TracerProvider:
  provider = TracerProvider()
  monkeypatch.setattr(tracing, "tracer", provider.get_tracer(_ADK_SCOPE))
  monkeypatch.setenv("GOOGLE_API_KEY", "fake-api-key-for-tests")
  monkeypatch.setattr(AsyncModels, "generate_content", _sdk_stub)
  return provider


@contextmanager
def _instrumented(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> Iterator[InMemorySpanExporter]:
  """Installs the real instrumentor over whatever the SDK method is now."""
  exporter = InMemorySpanExporter()
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  # `instrument()` sets this; delenv first so it is restored afterwards.
  monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
  instrumentor = GoogleGenAiSdkInstrumentor()
  instrumentor.instrument(tracer_provider=provider)
  try:
    yield exporter
  finally:
    instrumentor.uninstrument()


async def _run_inference(telemetry_config: TelemetryConfig) -> None:
  agent = LlmAgent(name="root_agent", model=_MODEL)
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name="test_app", user_id="test_user"
  )
  invocation_context = InvocationContext(
      invocation_id="test_invocation_id",
      agent=agent,
      session=session,
      session_service=session_service,
      run_config=RunConfig(telemetry=telemetry_config),
  )
  llm_request = LlmRequest(
      model=_MODEL,
      contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
  )
  async with _instrumentation.record_inference_telemetry(
      llm_request, invocation_context, Event(author="root_agent")
  ) as scope:
    async for response in Gemini(model=_MODEL).generate_content_async(
        llm_request
    ):
      scope.record_llm_response(invocation_context, response)


def _inference_span_scopes(exporter: InMemorySpanExporter) -> list[str]:
  return [
      span.instrumentation_scope.name
      for span in exporter.get_finished_spans()
      if span.name.startswith("generate_content")
  ]


_EXPERIMENTAL = "gen_ai_latest_experimental"


@pytest.mark.parametrize(
    "schema_version,semconv_env,per_request_semconv,expected_scopes",
    [
        ("1", None, None, [_ADK_SCOPE]),
        ("2", _EXPERIMENTAL, None, [_INSTRUMENTOR_SCOPE]),
        ("2", _EXPERIMENTAL, "stable", [_ADK_SCOPE]),
        # < 1.0 records the stable semconv unless opted in, so ADK records.
        (
            "2",
            None,
            None,
            [
                _ADK_SCOPE
                if _genai_instrumentor._has_stable_mode()
                else _INSTRUMENTOR_SCOPE
            ],
        ),
    ],
    ids=["v1", "v2", "v2-per-request-stable", "v2-no-opt-in"],
)
@pytest.mark.asyncio
async def test_one_recorder_per_inference(
    monkeypatch: pytest.MonkeyPatch,
    provider: TracerProvider,
    schema_version: str,
    semconv_env: str | None,
    per_request_semconv: str | None,
    expected_scopes: list[str],
) -> None:
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  if semconv_env is None:
    monkeypatch.delenv("OTEL_SEMCONV_STABILITY_OPT_IN", raising=False)
  else:
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", semconv_env)
  config = TelemetryConfig(genai_semconv_stability_opt_in=per_request_semconv)

  with _instrumented(monkeypatch, provider) as exporter:
    await _run_inference(config)

    assert _inference_span_scopes(exporter) == expected_scopes


@pytest.mark.asyncio
async def test_skipping_the_instrumentor_keeps_a_wrapper_beneath_it(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
  calls: list[str] = []

  @functools.wraps(_sdk_stub)
  async def foreign(self: AsyncModels, **kwargs: object) -> object:
    calls.append("foreign")
    return await _sdk_stub(self, **kwargs)

  monkeypatch.setattr(AsyncModels, "generate_content", foreign)

  with _instrumented(monkeypatch, provider) as exporter:
    await _run_inference(TelemetryConfig())

    assert (_inference_span_scopes(exporter), calls) == (
        [_ADK_SCOPE],
        ["foreign"],
    )


@pytest.mark.asyncio
async def test_adk_records_inference_tokens_when_it_skips_the_instrumentor(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  """An instrumentor that may record the token metrics itself is skipped too."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")
  monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", _EXPERIMENTAL)
  monkeypatch.setattr(google_genai_version, "__version__", "1.0b0")
  response = _RESPONSE.model_copy(
      update={
          "usage_metadata": types.GenerateContentResponseUsageMetadata(
              prompt_token_count=10, candidates_token_count=5
          )
      }
  )

  async def sdk_stub_with_usage(
      self: AsyncModels, *, model: str, contents: object, config: object = None
  ) -> types.GenerateContentResponse:
    del self, model, contents, config  # Unused.
    return response

  monkeypatch.setattr(AsyncModels, "generate_content", sdk_stub_with_usage)
  metric_reader = InMemoryMetricReader()
  meter = MeterProvider(metric_readers=[metric_reader]).get_meter(_ADK_SCOPE)
  monkeypatch.setattr(
      _metrics,
      "_inference_operation_input_tokens",
      meter.create_histogram("gen_ai.client.inference.operation.input_tokens"),
  )

  with _instrumented(monkeypatch, provider):
    await _run_inference(
        TelemetryConfig(genai_semconv_stability_opt_in="stable")
    )

  metrics_data = metric_reader.get_metrics_data()
  [point] = [
      point
      for resource_metrics in metrics_data.resource_metrics
      for scope_metrics in resource_metrics.scope_metrics
      for metric in scope_metrics.metrics
      for point in metric.data.data_points
  ]
  assert (point.count, point.sum) == (1, 10)
