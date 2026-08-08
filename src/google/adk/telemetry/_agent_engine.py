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

import asyncio
import contextlib
import functools
import logging
import os
from types import ModuleType
from typing import cast
from typing import TYPE_CHECKING

from opentelemetry import baggage
from opentelemetry import context
from opentelemetry import metrics
from opentelemetry.sdk import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.trace.propagation import tracecontext
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
  from collections.abc import AsyncGenerator
  from collections.abc import AsyncIterator
  from typing import Mapping
  from typing import Optional

  import fastapi
  from starlette.middleware.base import DispatchFunction
  from starlette.middleware.base import RequestResponseEndpoint
  from starlette.requests import Request
  from starlette.responses import Response
  from starlette.responses import StreamingResponse

  from ._agent_engine_metric_exporter import _RequestDrivenMetricReader
  from ._agent_engine_metric_exporter import MetricsState

logger = logging.getLogger("google_adk." + __name__)

_GOOGLE_AE_TRACEPARENT_HEADER = "Google-Agent-Engine-Traceparent"
_TRACEPARENT_BAGGAGE_KEY = "traceparent"
_GOOGLE_TRACEPARENT_HEADER = "traceparent"
_GOOGLE_TRACEPARENT_BAGGAGE_KEY = "google_traceparent"
_GOOGLE_TRACEPARENT_SUPPORT_ATTRIBUTE_KEY = "supportID"


def get_propagated_context(request: fastapi.Request) -> context.Context:
  """Propagates context from the request headers."""
  ctx = context.get_current()

  if _GOOGLE_TRACEPARENT_HEADER in request.headers:
    original_traceparent = request.headers[_GOOGLE_TRACEPARENT_HEADER]
    ctx = baggage.set_baggage(
        _GOOGLE_TRACEPARENT_BAGGAGE_KEY,
        original_traceparent,
        context=ctx,
    )

  if _GOOGLE_AE_TRACEPARENT_HEADER in request.headers:
    ae_traceparent = request.headers[_GOOGLE_AE_TRACEPARENT_HEADER]
    extracted_ctx = tracecontext.TraceContextTextMapPropagator().extract(
        carrier={"traceparent": ae_traceparent}, context=ctx
    )
    # extract() returns the context unchanged when it rejects the header;
    # testing the extracted span for validity instead would false-accept,
    # since ctx usually already carries a valid span.
    if extracted_ctx is not ctx:
      ctx = baggage.set_baggage(
          _TRACEPARENT_BAGGAGE_KEY, ae_traceparent, context=extracted_ctx
      )

  return ctx


class TopSpanProcessor(trace.SpanProcessor):
  """Top span processor."""

  def on_start(
      self, span: trace.Span, parent_context: Optional[context.Context] = None
  ) -> None:
    """Adds support ID to the top span."""
    baggage_items = baggage.get_all(context=parent_context)
    baggage_trace_header = baggage_items.get(_GOOGLE_TRACEPARENT_BAGGAGE_KEY)
    if self._is_top_span(span, baggage_items) and isinstance(
        baggage_trace_header, str
    ):
      span.set_attribute(
          _GOOGLE_TRACEPARENT_SUPPORT_ATTRIBUTE_KEY, baggage_trace_header
      )

  def on_end(self, span: trace.ReadableSpan) -> None:
    pass

  def shutdown(self) -> None:
    pass

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True

  def _is_top_span(
      self, span: trace.Span, baggage_items: Mapping[str, object]
  ) -> bool:
    """Returns true if the span is a top span.

    Args:
      span: The span to check.
      baggage_items: The baggage items that carry the context.

    Top span (e.g. "Invocation" span) is defined as the first span generated in
    trace generation.
    Top span could have an empty parent or the parent could be the span
    provided by traceparent propagation.
    """
    if span.parent is None or span.parent.span_id == 0:
      return True
    if _TRACEPARENT_BAGGAGE_KEY not in baggage_items:
      return False
    traceparent_parts = str(baggage_items[_TRACEPARENT_BAGGAGE_KEY]).split("-")
    if len(traceparent_parts) < 3:
      return False
    try:
      parent_span_id = int(traceparent_parts[2], 16)
    except ValueError:
      return False
    return span.parent.span_id == parent_span_id


def _metrics_flushing_dispatch(
    reader: _RequestDrivenMetricReader,
) -> DispatchFunction:
  """Returns the dispatch that drives `reader` from the request lifecycle.

  Collection has to happen while a request is in flight: see the module
  docstring of ``_agent_engine_metric_exporter`` for why. Traces and logs are
  not flushed here -- on Agent Engine the ``AdkApp`` in
  ``vertexai.agent_engines`` already force-flushes them per request.
  """

  async def dispatch(
      request: Request, call_next: RequestResponseEndpoint
  ) -> Response:
    # Never let a metrics failure break the request it rides on.
    try:
      if reader.note_request_start():
        _ = reader.submit_collect()  # point 2 -- fire-and-forget, no await.
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception("Metrics request-start hook failed")
    # BaseHTTPMiddleware always hands back a streaming response, so it exposes
    # body_iterator (the base Response type does not).
    try:
      response = cast("StreamingResponse", await call_next(request))
    except BaseException:
      # call_next failed: the draining body iterator below is never installed,
      # so balance note_request_start here or _in_flight leaks for good.
      await _drain_metrics(reader)
      raise
    # BaseHTTPMiddleware's body_iterator is an async generator (supports
    # aclose); the public type is only an AsyncIterable, so narrow it here.
    original_iterator = cast(
        "AsyncGenerator[bytes, None]", response.body_iterator
    )

    async def iterate_then_flush() -> AsyncIterator[bytes]:
      try:
        async with contextlib.aclosing(original_iterator) as body:
          async for chunk in body:
            yield chunk
      finally:
        # Drain metrics after the body streams (while the request still has
        # CPU) and before the connection closes, so the export completes
        # without adding latency to the response.
        await _drain_metrics(reader)

    response.body_iterator = iterate_then_flush()
    return response

  return dispatch


async def _drain_metrics(reader: _RequestDrivenMetricReader) -> None:
  """Drain collect on request end: awaited so the export completes before close."""
  try:
    if reader.note_request_end():
      future = reader.submit_collect()
      if future is not None:
        await asyncio.wrap_future(future)
  except Exception:  # pylint: disable=broad-exception-caught
    logger.exception("Failed to flush metrics on request end")


def telemetry_user_agent_headers() -> dict[str, str] | None:
  """Returns the Vertex Agent Engine User-Agent header, if telemetry is on."""
  if not os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY"):
    return None
  from google.cloud.aiplatform import version as aip_version

  otlp_http_version: ModuleType | None
  try:
    from opentelemetry.exporter.otlp.proto.http import version as _otlp_version

    otlp_http_version = _otlp_version
  except (ImportError, AttributeError):
    otlp_http_version = None

  user_agent = f"Vertex-Agent-Engine/{aip_version.__version__}"
  if otlp_http_version:
    user_agent += f" OTel-OTLP-Exporter-Python/{otlp_http_version.__version__}"
  return {"User-Agent": user_agent}


@functools.cache
def _get_agent_engine_metrics_setup() -> MetricsState | None:
  """Builds the request-driven metric state on Agent Engine, memoized.

  Returns the reader plus the span processor that drives it (to wire into the
  OTel providers and later drain from the request path), or None when:

    1. a ``MeterProvider`` is already installed by something else -- our reader
       would not land on the active provider, so we defer to that setup; or
    2. we are not running on Agent Engine
       (``GOOGLE_CLOUD_AGENT_ENGINE_ID`` unset); or
    3. the GCP metric exporter is unavailable / setup fails.

  Cached so the "already installed" check is evaluated once -- before ADK sets
  its own ``MeterProvider`` in ``maybe_set_otel_providers`` -- and the same
  handles are returned to ``get_gcp_exporters`` (which wires them onto the
  providers) and to the middleware install below.
  """
  if not os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
    return None
  if isinstance(metrics.get_meter_provider(), MeterProvider):
    logger.warning(
        "A MeterProvider is already installed; skipping request-driven metric"
        " export. On Agent Engine's request-billed runtime metrics may be"
        " dropped between requests."
    )
    return None
  try:
    from ._agent_engine_metric_exporter import build_request_driven_metrics
    from .google_cloud import _get_gcp_otlp_metric_exporter

    exporter = _get_gcp_otlp_metric_exporter()
    if exporter is None:
      return None
    return build_request_driven_metrics(exporter)
  except Exception:  # pylint: disable=broad-exception-caught
    logger.warning(
        "Failed to set up request-driven metric export on Agent Engine.",
        exc_info=True,
    )
    return None


def maybe_install_request_metrics_middleware(
    app: fastapi.FastAPI, *, otel_to_cloud: bool
) -> None:
  """Installs the request-path metric flushing middleware, if applicable.

  On Agent Engine's request-billed runtime CPU is throttled the instant a
  request ends, starving background metric export. The GCP exporter setup
  builds a request-driven metric reader there (wired onto the MeterProvider);
  drive it from the request path here. No-op off Agent Engine.

  Args:
    app: The app to install the middleware on.
    otel_to_cloud: Whether to setup telemetry export to GCP.
  """
  if not otel_to_cloud:
    return

  metrics_state = _get_agent_engine_metrics_setup()
  if metrics_state is None:
    return
  app.add_middleware(
      BaseHTTPMiddleware,
      dispatch=_metrics_flushing_dispatch(metrics_state.reader),
  )
