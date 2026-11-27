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
import sys
import types as std_types

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
from google.genai.models import Models
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.dependencies import get_dependency_conflicts
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
import wrapt

try:
  from opentelemetry.util.genai.invocation import CLIENT_INFERENCE_CONTEXT_KEY

  del CLIENT_INFERENCE_CONTEXT_KEY
  _UTIL_GENAI_SUPPRESSES = True
except ImportError:
  _UTIL_GENAI_SUPPRESSES = False

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
  instrumentor = GoogleGenAiSdkInstrumentor()
  if conflict := get_dependency_conflicts(
      instrumentor.instrumentation_dependencies()
  ):
    pytest.skip(str(conflict))
  exporter = InMemorySpanExporter()
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  # `instrument()` sets this; delenv first so it is restored afterwards.
  monkeypatch.delenv("OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT", raising=False)
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
  # Anything but ADK is the instrumentor: under 1.0b0, util-genai names the
  # tracer after itself.
  return [
      _ADK_SCOPE
      if span.instrumentation_scope.name == _ADK_SCOPE
      else _INSTRUMENTOR_SCOPE
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


def _passthrough(wrapped, instance, args, kwargs):
  del instance  # Unused.
  return wrapped(*args, **kwargs)


@pytest.mark.asyncio
async def test_skipping_an_instrumentor_without_the_co_filename_stamp(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  """1.0b0 wraps with wrapt but predates stamping the SDK function."""
  del provider  # Unused; it stubs the SDK.
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
  calls: list[str] = []

  def instrumentor_wrapper(wrapped, instance, args, kwargs):
    calls.append("instrumentor")
    return _passthrough(wrapped, instance, args, kwargs)

  instrumentor_wrapper.__code__ = instrumentor_wrapper.__code__.replace(
      co_filename=f"/site-packages/{_genai_instrumentor._MODULE_PATH}/x.py"
  )
  for cls in (Models, AsyncModels):
    monkeypatch.setattr(
        cls,
        "generate_content",
        wrapt.FunctionWrapper(
            cls.__dict__["generate_content"], instrumentor_wrapper
        ),
    )

  await _run_inference(TelemetryConfig())

  assert (_genai_instrumentor.is_instrumented(), calls) == (True, [])


@pytest.mark.asyncio
async def test_skipping_a_pre_1_0_instrumentor_keeps_a_wrapt_wrapper_beneath_it(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  """< 1.0 replaces the SDK function outright, here an unbound wrapt proxy."""
  del provider  # Unused; it stubs the SDK.
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
  calls: list[str] = []

  def foreign(wrapped, instance, args, kwargs):
    calls.append("foreign")
    return _passthrough(wrapped, instance, args, kwargs)

  beneath = wrapt.FunctionWrapper(
      AsyncModels.__dict__["generate_content"], foreign
  )

  @functools.wraps(beneath)
  async def instrumentor(self: AsyncModels, **kwargs: object) -> object:
    calls.append("instrumentor")
    return await beneath(self, **kwargs)

  instrumentor.__code__ = instrumentor.__code__.replace(
      co_filename=f"/site-packages/{_genai_instrumentor._MODULE_PATH}/x.py"
  )
  monkeypatch.setattr(AsyncModels, "generate_content", instrumentor)

  await _run_inference(TelemetryConfig())

  assert calls == ["foreign"]


@pytest.mark.asyncio
async def test_skipping_the_instrumentor_keeps_a_wrapt_wrapper_above_it(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")
  calls: list[str] = []

  def foreign(wrapped, instance, args, kwargs):
    calls.append("foreign")
    return _passthrough(wrapped, instance, args, kwargs)

  with _instrumented(monkeypatch, provider) as exporter:
    # Undone before the instrumentor is, which restores what it found.
    with pytest.MonkeyPatch.context() as wrapped_again:
      wrapped_again.setattr(
          AsyncModels,
          "generate_content",
          wrapt.FunctionWrapper(
              AsyncModels.__dict__["generate_content"], foreign
          ),
      )
      await _run_inference(TelemetryConfig())

    assert (_inference_span_scopes(exporter), calls) == (
        [_ADK_SCOPE],
        ["foreign"],
    )


@pytest.mark.skipif(
    not _UTIL_GENAI_SUPPRESSES, reason="util-genai < 1.3b0 can't suppress"
)
@pytest.mark.asyncio
async def test_the_instrumentor_stands_down_beneath_any_wrapper(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  """A plain function over it hides it from ADK; util-genai still suppresses it."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "1")

  with _instrumented(monkeypatch, provider) as exporter:
    instrumented = AsyncModels.generate_content

    async def foreign(self: AsyncModels, **kwargs: object) -> object:
      return await instrumented(self, **kwargs)

    with pytest.MonkeyPatch.context() as wrapped_again:
      wrapped_again.setattr(AsyncModels, "generate_content", foreign)
      await _run_inference(TelemetryConfig())

    assert _inference_span_scopes(exporter) == [_ADK_SCOPE]


@pytest.mark.parametrize("schema_version", ["1", "2"])
@pytest.mark.asyncio
async def test_inference_span_marks_and_clears_util_genai_context(
    monkeypatch: pytest.MonkeyPatch,
    provider: TracerProvider,
    schema_version: str,
) -> None:
  """ADK's inference span marks the OTel context for util-genai >= 1.3b0."""
  del provider  # Unused; it stubs the SDK.
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", schema_version)
  context_key = "test-client-inference-context-key"

  class _FakeInferenceData:
    pass

  fake_invocation = std_types.ModuleType("opentelemetry.util.genai.invocation")
  setattr(fake_invocation, "CLIENT_INFERENCE_CONTEXT_KEY", context_key)
  setattr(fake_invocation, "InferenceData", _FakeInferenceData)
  monkeypatch.setitem(
      sys.modules, "opentelemetry.util.genai.invocation", fake_invocation
  )

  observed: list[object] = []

  async def observing_stub(
      self: AsyncModels, *, model: str, contents: object, config: object = None
  ) -> types.GenerateContentResponse:
    del self, model, contents, config  # Unused.
    observed.append(otel_context.get_value(context_key))
    return _RESPONSE

  monkeypatch.setattr(AsyncModels, "generate_content", observing_stub)

  await _run_inference(TelemetryConfig())

  assert len(observed) == 1
  assert isinstance(observed[0], _FakeInferenceData)
  assert otel_context.get_value(context_key) is None


@pytest.mark.asyncio
async def test_adk_records_inference_tokens_when_it_skips_the_instrumentor(
    monkeypatch: pytest.MonkeyPatch, provider: TracerProvider
) -> None:
  """An instrumentor that may record the token metrics itself is skipped too."""
  monkeypatch.setenv("ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN", "2")
  monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", _EXPERIMENTAL)
  # A util-genai that records these metrics, so only ADK's own span emits them.
  real_version = _genai_instrumentor.metadata.version
  monkeypatch.setattr(
      _genai_instrumentor.metadata,
      "version",
      lambda name: "1.4b0"
      if name == "opentelemetry-util-genai"
      else real_version(name),
  )
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
