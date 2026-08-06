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

"""Unit tests for Agent Engine telemetry.

Covers trace context propagated from request headers and the request-path
metric flushing middleware. The middleware drives the request-driven metric
reader from the request lifecycle: a fire-and-forget collect at request start
and an awaited drain collect after the response body has streamed. Traces and
logs are not flushed here (the Agent Engine AdkApp does that). These tests use
a spy reader; no real time, no network.
"""

# pylint: disable=protected-access,redefined-outer-name
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import inspect
from types import SimpleNamespace
from unittest import mock

import fastapi
from google.adk.telemetry import _agent_engine
from google.adk.telemetry._agent_engine import get_propagated_context
from google.adk.telemetry._agent_engine import TopSpanProcessor
from opentelemetry import baggage
from opentelemetry import context
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

_AE_TRACEPARENT_HEADER = "Google-Agent-Engine-Traceparent"
_TRACEPARENT_HEADER = "traceparent"
_SUPPORT_ID_ATTRIBUTE = "supportID"
_SUPPORT_ID_VALUE = "support-id-value"
_TOP_SPAN = "invocation"
_CHILD_SPAN = "child"

_TRACE_ID_HEX = "4bf92f3577b34da6a3ce929d0e0e4736"
_REMOTE_SPAN_ID_HEX = "00f067aa0ba902b7"
_WELL_FORMED_TRACEPARENT = f"00-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}-01"

# Values the trace context propagator refuses, either because they do not
# match the wire format or because the ids they carry are not usable.
_REJECTED_TRACEPARENT_VALUES = [
    "x",
    "00-abc-zz-01",
    "",
    "00",
    "-",
    f"00-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}",
    f'00-{"0" * 32}-{_REMOTE_SPAN_ID_HEX}-01',
    f"ff-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}-01",
]


def _request(**headers: str) -> fastapi.Request:
  """Builds a minimal request carrying the given headers."""
  return fastapi.Request({
      "type": "http",
      "method": "POST",
      "path": "/",
      "headers": [
          (name.lower().encode(), value.encode())
          for name, value in headers.items()
      ],
  })


def _record_spans(ctx: context.Context) -> dict[str, ReadableSpan]:
  """Traces a child span under a top span with ctx attached, keyed by name."""
  exporter = InMemorySpanExporter()
  provider = TracerProvider(shutdown_on_exit=False)
  provider.add_span_processor(TopSpanProcessor())
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  tracer = provider.get_tracer(__name__)

  token = context.attach(ctx)
  try:
    with tracer.start_as_current_span(_TOP_SPAN):
      with tracer.start_as_current_span(_CHILD_SPAN):
        pass
  finally:
    context.detach(token)

  return {span.name: span for span in exporter.get_finished_spans()}


@pytest.mark.parametrize("header_value", _REJECTED_TRACEPARENT_VALUES)
def test_rejected_header_still_produces_child_spans(header_value):
  """A caller-supplied header must not be able to break span creation."""
  spans = _record_spans(
      get_propagated_context(_request(**{_AE_TRACEPARENT_HEADER: header_value}))
  )

  assert set(spans) == {_TOP_SPAN, _CHILD_SPAN}


@pytest.mark.parametrize("header_value", _REJECTED_TRACEPARENT_VALUES)
def test_rejected_header_is_not_stored_in_baggage(header_value):
  """Only a header the propagator accepted is worth carrying in baggage."""
  ctx = get_propagated_context(
      _request(**{_AE_TRACEPARENT_HEADER: header_value})
  )

  assert _TRACEPARENT_HEADER not in baggage.get_all(context=ctx)


@pytest.mark.parametrize("baggage_value", _REJECTED_TRACEPARENT_VALUES)
def test_rejected_value_in_baggage_still_produces_child_spans(baggage_value):
  """The processor runs on every span, so it cannot trust baggage contents."""
  spans = _record_spans(baggage.set_baggage(_TRACEPARENT_HEADER, baggage_value))

  assert set(spans) == {_TOP_SPAN, _CHILD_SPAN}


def test_well_formed_header_is_stored_in_baggage():
  """The top span check reads the accepted header back out of baggage."""
  ctx = get_propagated_context(
      _request(**{_AE_TRACEPARENT_HEADER: _WELL_FORMED_TRACEPARENT})
  )

  assert (
      baggage.get_all(context=ctx)[_TRACEPARENT_HEADER]
      == _WELL_FORMED_TRACEPARENT
  )


def test_well_formed_header_marks_first_span_as_top_span():
  """This is the propagation the rejected-header guards must not break."""
  spans = _record_spans(
      get_propagated_context(
          _request(**{
              _AE_TRACEPARENT_HEADER: _WELL_FORMED_TRACEPARENT,
              _TRACEPARENT_HEADER: _SUPPORT_ID_VALUE,
          })
      )
  )

  assert spans[_TOP_SPAN].parent.span_id == int(_REMOTE_SPAN_ID_HEX, 16)
  assert spans[_TOP_SPAN].attributes[_SUPPORT_ID_ATTRIBUTE] == _SUPPORT_ID_VALUE
  assert _SUPPORT_ID_ATTRIBUTE not in spans[_CHILD_SPAN].attributes


def test_first_span_is_parentless_when_header_is_rejected():
  """Rejecting the header leaves the first span parentless, still the top."""
  spans = _record_spans(
      get_propagated_context(
          _request(**{
              _AE_TRACEPARENT_HEADER: "x",
              _TRACEPARENT_HEADER: _SUPPORT_ID_VALUE,
          })
      )
  )

  assert spans[_TOP_SPAN].parent is None
  assert spans[_TOP_SPAN].attributes[_SUPPORT_ID_ATTRIBUTE] == _SUPPORT_ID_VALUE
  assert _SUPPORT_ID_ATTRIBUTE not in spans[_CHILD_SPAN].attributes


class _SpyReader:
  """Records the order of hook/submit calls made by the middleware."""

  def __init__(self) -> None:
    self.events: list[str] = []

  def note_request_start(self) -> bool:
    self.events.append("start")
    return True

  def note_request_end(self) -> bool:
    self.events.append("end")
    return True

  def note_generate_content_start(self) -> bool:
    self.events.append("generate_content")
    return False

  def submit_collect(self) -> None:
    self.events.append("submit")
    return None


class _FakeResponse:
  """A minimal ASGI-ish response exposing a consumable body_iterator."""

  def __init__(self, chunks: list[bytes]):
    async def _gen() -> AsyncIterator[bytes]:
      for chunk in chunks:
        yield chunk

    self.body_iterator: AsyncIterator[bytes] = _gen()


def test_middleware_glue() -> None:
  """note_request_start precedes call_next; end drain only after the body."""
  spy = _SpyReader()
  dispatch = _agent_engine._metrics_flushing_dispatch(spy)

  async def _drive() -> _SpyReader:
    response = _FakeResponse([b"a", b"b"])

    async def call_next(request: object) -> _FakeResponse:
      del request
      spy.events.append("call_next")
      return response

    wrapped = await dispatch(object(), call_next)

    # Body not consumed yet: request end drain must not have fired.
    assert spy.events == ["start", "submit", "call_next"]

    consumed = [chunk async for chunk in wrapped.body_iterator]
    assert consumed == [b"a", b"b"]
    return spy

  result = asyncio.run(_drive())
  assert "end" in result.events
  assert (
      result.events.index("start")
      < result.events.index("call_next")
      < result.events.index("end")
  )


def test_metrics_drained_on_request_end() -> None:
  """The reader is drained (end + submit) after the body streams, once."""
  spy = _SpyReader()
  dispatch = _agent_engine._metrics_flushing_dispatch(spy)

  async def _drive() -> None:
    response = _FakeResponse([b"x"])

    async def call_next(request: object) -> _FakeResponse:
      del request
      return response

    wrapped = await dispatch(object(), call_next)
    # Before the body is consumed, no request-end drain.
    assert "end" not in spy.events
    _ = [chunk async for chunk in wrapped.body_iterator]

  asyncio.run(_drive())
  assert spy.events.count("end") == 1


def test_drain_failure_does_not_break_response() -> None:
  """A reader that raises on drain never breaks the draining response."""

  class _BoomReader(_SpyReader):

    def note_request_end(self) -> bool:
      raise RuntimeError("boom")

  spy = _BoomReader()
  dispatch = _agent_engine._metrics_flushing_dispatch(spy)

  async def _drive() -> list[bytes]:
    response = _FakeResponse([b"a", b"b"])

    async def call_next(request: object) -> _FakeResponse:
      del request
      return response

    wrapped = await dispatch(object(), call_next)
    return [chunk async for chunk in wrapped.body_iterator]

  consumed = asyncio.run(_drive())
  assert consumed == [b"a", b"b"]  # body streamed despite drain failure.


def test_call_next_exception_drains_and_reraises() -> None:
  """If call_next raises, note_request_end still runs (no in_flight leak)."""
  spy = _SpyReader()
  dispatch = _agent_engine._metrics_flushing_dispatch(spy)

  async def _drive() -> None:
    async def call_next(request: object) -> _FakeResponse:
      del request
      raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
      await dispatch(object(), call_next)

  asyncio.run(_drive())
  # start balanced by end even though the body iterator never installed.
  assert spy.events == ["start", "submit", "end", "submit"]


@pytest.mark.parametrize(
    "otel_to_cloud, metrics_state, expected_middleware",
    [
        # GCP telemetry setup never ran: the reader is on no MeterProvider.
        (False, "state", 0),
        # Not on Agent Engine (or setup failed): nothing to drive.
        (True, None, 0),
        (True, "state", 1),
    ],
)
def test_maybe_install_request_metrics_middleware(
    otel_to_cloud: bool,
    metrics_state: str | None,
    expected_middleware: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The middleware is installed only with both a reader and cloud telemetry."""
  state = (
      SimpleNamespace(reader=_SpyReader(), span_processor=None)
      if metrics_state
      else None
  )
  monkeypatch.setattr(
      "google.adk.telemetry._agent_engine._get_agent_engine_metrics_setup",
      lambda: state,
  )
  app = fastapi.FastAPI()

  _agent_engine.maybe_install_request_metrics_middleware(
      app, otel_to_cloud=otel_to_cloud
  )

  assert len(app.user_middleware) == expected_middleware


@pytest.fixture(autouse=True)
def _clear_agent_engine_metrics_cache():
  """The memoized agent-engine metrics builder must not leak across tests."""
  _agent_engine._get_agent_engine_metrics_setup.cache_clear()
  yield
  _agent_engine._get_agent_engine_metrics_setup.cache_clear()


def test_agent_engine_metrics_skipped_off_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Without GOOGLE_CLOUD_AGENT_ENGINE_ID, no metric state is built."""
  monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)

  assert _agent_engine._get_agent_engine_metrics_setup() is None


def test_agent_engine_metrics_skipped_when_meter_provider_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """If a real MeterProvider is already installed, defer to it (return None)."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "123")
  monkeypatch.setattr(
      "opentelemetry.metrics.get_meter_provider",
      lambda: MeterProvider(),
  )

  assert _agent_engine._get_agent_engine_metrics_setup() is None


def test_agent_engine_metrics_built_on_agent_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """On Agent Engine (no MeterProvider yet), the metric state is built."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "123")
  monkeypatch.setattr(
      "opentelemetry.metrics.get_meter_provider",
      lambda: mock.MagicMock(),  # not an SDK MeterProvider.
  )
  fake_state = mock.MagicMock(name="metrics_state")
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda **_: mock.MagicMock(name="exporter"),
  )
  monkeypatch.setattr(
      "google.adk.telemetry._agent_engine_metric_exporter.build_request_driven_metrics",
      lambda exporter: fake_state,
  )

  assert _agent_engine._get_agent_engine_metrics_setup() is fake_state


def test_agent_engine_metrics_memoized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The result is cached: the 'already installed' check runs only once."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "123")
  monkeypatch.setattr(
      "opentelemetry.metrics.get_meter_provider",
      lambda: mock.MagicMock(),
  )
  fake_state = mock.MagicMock(name="metrics_state")
  calls = {"n": 0}

  def _build(exporter):
    del exporter
    calls["n"] += 1
    return fake_state

  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda **_: mock.MagicMock(name="exporter"),
  )
  monkeypatch.setattr(
      "google.adk.telemetry._agent_engine_metric_exporter.build_request_driven_metrics",
      _build,
  )

  first = _agent_engine._get_agent_engine_metrics_setup()
  second = _agent_engine._get_agent_engine_metrics_setup()

  assert first is second is fake_state
  assert calls["n"] == 1


def test_agent_engine_metrics_none_when_exporter_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A missing GCP metric exporter yields None, not an error."""
  monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "123")
  monkeypatch.setattr(
      "opentelemetry.metrics.get_meter_provider",
      lambda: mock.MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud._get_gcp_otlp_metric_exporter",
      lambda **_: None,
  )

  assert _agent_engine._get_agent_engine_metrics_setup() is None


def test_agent_engine_metrics_builder_takes_no_args() -> None:
  """@functools.cache keys on args, so the one cache entry shared between the
  exporter-setup and middleware-install call sites only holds if the builder is
  nullary. Guard against a param sneaking in and silently breaking export."""
  sig = inspect.signature(
      _agent_engine._get_agent_engine_metrics_setup.__wrapped__
  )
  assert not sig.parameters
