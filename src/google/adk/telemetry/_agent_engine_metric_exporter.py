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

r"""Request-driven, sleepless metric export.

A bespoke OpenTelemetry metric reader (plus the span processor that drives it)
that exports metrics from an agent running on the Vertex AI Agent Runtime
(request-based billing) *without adding latency to any request* and dropping
data only in a rare, well-defined tail case (see below).

The FastAPI/Starlette middleware that also drives this reader from the request
lifecycle -- and flushes traces and logs on the same request path -- lives in
``google.adk.telemetry._agent_engine``.

The problem
-----------
The Agent Runtime throttles CPU the instant a request finishes (request-based
billing). A normal metric pipeline exports on a background timer thread; between
requests that thread gets no CPU, so its periodic export is starved and metric
points are dropped.

The residual loss case
----------------------
This is not fully lossless. A request shorter than the floor that drains right
after a collect -- and is the *last* request before the process goes idle --
loses its points: a collect now is muted by the floor (I2), and the next
collect never comes. Illustrated at the end of the timeline section.

Three constraints shape the solution:

- I1 -- export only while serving. A collect+export must run while a request is
  in flight (the only time CPU is guaranteed).
- I2 -- never collect more often than the floor. Two collects closer than the
  floor (5s default) are rejected by the collection path.
- I3 -- never collect too rarely. A single export carries at most ~200 points,
  so collect at least once per configured period.

Design
------
There is no background ticker and no ``time.sleep``. The reader subclasses the
base ``MetricReader`` directly, so no daemon thread is started; ``collect()`` is
invoked only from the request lifecycle (middleware) and from
``generate_content`` span starts (span processor). The configured export period
is demoted from a hard schedule to a guidepost grid -- hint times used to decide
whether an event-driven collect is warranted. Every collect runs on the request
path, so it always has CPU (I1); the floor is enforced by skipping a collect
that would land too soon (I2); guideposts force a collect during sustained load
so points can't pile up (I3).

Timeline legend::

  [====]   a request being served (holds CPU until its connection closes)
  C        a collect: drain the SDK aggregation + export, run on the request
  path
  P        a guidepost: a periodic "would-be" collect time -- a hint, never a
           real scheduled call
  x        a collect skipped because it would violate the floor (min spacing)
  FLOOR    minimum spacing between two collects (default 3s)
  PERIOD   spacing of the guidepost grid (default 60s)

Baseline -- a request collects when it drains to zero::

  req1  [==========]
                    +- in_flight 1->0  ->  C

Overlap batches into one collect::

  req1  [======]
  req2     [========]
  req3        [==========]
                         +- in_flight -> 0 only here  ->  C  (one collect)

Point 2 -- under continuous overlap, a guidepost fires at the next start::

  req1  [====]
  req2    [=======]
  req3      [==========]
  req4             [==========]
                 P +- guidepost crossed while req3 in flight; req4 arrives -> C

Point 3 -- a guidepost too close to the last collect is muted (grid += PERIOD).

Point 4 -- a lone very long request uses its own inference spans: once
1.5xPERIOD has elapsed since the last collect, the next ``generate_content``
span start collects.

Loss case -- the last request is too short to collect, and nothing follows::

  req1  [==========]
                    +- in_flight 1->0  ->  C
  req2               [=]   (drains < FLOOR after C)
                       +- in_flight 1->0, but a collect here is muted by the
                          floor (I2); its points wait for the next collect
                          ... process goes idle, no later request  ->  C never
                          comes, req2's points are lost

Threading model
---------------
The blocking export never runs on the event loop. The reader owns a
single-worker ``ThreadPoolExecutor``; ``submit_collect()`` runs the collect on
it. The drain collect is the only awaited path (via ``asyncio.wrap_future``),
after the response body has streamed, so the export completes before the
connection closes. Start/``generate_content`` collects are fire-and-forget. All
shared state is guarded by one ``threading.Lock``.

Besides the tail loss case above, metrics-on / tracing-off leaves point 4
inactive; a lone ultra-long request under that config can accumulate. Both are
rare and accepted.
"""

# This module deliberately mirrors OTel SDK internals (the private aggregation /
# temporality preferences, the instrumentation-suppression key, and the
# _receive_metrics contract), so private access is expected throughout.
# pyright: reportPrivateUsage=false
from __future__ import annotations

from collections.abc import Callable
import concurrent.futures
import dataclasses
import logging
import os
import threading
import time

from opentelemetry import context as otel_context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.context.context import Context
from opentelemetry.sdk.environment_variables import OTEL_METRIC_EXPORT_INTERVAL
from opentelemetry.sdk.environment_variables import OTEL_METRIC_EXPORT_TIMEOUT
from opentelemetry.sdk.metrics import export as metrics_export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import Span
from opentelemetry.sdk.trace import SpanProcessor

logger = logging.getLogger("google_adk." + __name__)

# Env-var name for the hard floor on collect spacing (I2), in milliseconds.
GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS = (
    "GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS"
)

# The single minimum spacing between two metric exports to
# telemetry.googleapis.com, shared by every ADK metric reader: the floor (I2)
# for the request-driven reader here, and the export interval of the periodic
# reader in `google_cloud`.
#
# The backend currently accepts points sent more frequently than this, but that
# is only there to absorb drift (a reader firing slightly early); it is not a
# supported rate and must not be relied on. Exporting faster than this interval
# risks points being rejected or throttled -- keep new readers at or above it.
MIN_EXPORT_INTERVAL_MS = 5000.0
# Semantic-convention attribute carrying the GenAI operation name.
_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"


def _env_float(name: str, default: float) -> float:
  """Reads a float env var, falling back to a default on missing/invalid."""
  raw = os.environ.get(name)
  if raw is None:
    return default
  try:
    return float(raw)
  except ValueError:
    logger.warning(
        "Found invalid value for %s=%r, using default %s", name, raw, default
    )
    return default


def _floor_seconds() -> float:
  """Returns the min collect spacing in seconds (from env, default 5.0s)."""
  return (
      _env_float(
          GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS,
          MIN_EXPORT_INTERVAL_MS,
      )
      / 1000.0
  )


class _RequestDrivenMetricReader(metrics_export.MetricReader):
  """A `MetricReader` whose collects are driven by the request lifecycle."""

  def __init__(
      self,
      exporter: metrics_export.MetricExporter,
      *,
      export_interval_millis: float | None = None,
      export_timeout_millis: float | None = None,
      floor_millis: float | None = None,
      now: Callable[[], float] = time.monotonic,
  ):
    # Defer temporality/aggregation to the wrapped exporter, exactly as the
    # SDK's PeriodicExportingMetricReader does.
    super().__init__(
        preferred_temporality=exporter._preferred_temporality,  # pylint: disable=protected-access
        preferred_aggregation=exporter._preferred_aggregation,  # pylint: disable=protected-access
    )
    self._exporter: metrics_export.MetricExporter = exporter
    # Held whenever calling the wrapped exporter, matching the SDK's contract
    # that MetricExporter.export() is never called concurrently.
    self._export_lock: threading.Lock = threading.Lock()
    self._now: Callable[[], float] = now

    if export_interval_millis is None:
      export_interval_millis = _env_float(OTEL_METRIC_EXPORT_INTERVAL, 60000.0)
    if export_timeout_millis is None:
      export_timeout_millis = _env_float(OTEL_METRIC_EXPORT_TIMEOUT, 30000.0)
    self._export_timeout_millis: float = export_timeout_millis
    self._period_s: float = export_interval_millis / 1000.0
    self._floor_s: float = (
        (floor_millis / 1000.0)
        if floor_millis is not None
        else _floor_seconds()
    )

    # All fields below are guarded by _lock.
    self._lock: threading.Lock = threading.Lock()
    self._in_flight: int = 0
    self._last_collect: float | None = None
    # Start of the current busy period (stamped when in-flight goes 0 -> 1). The
    # point-4 "overdue" reference for the current stretch of activity, so a
    # collect from a long-past busy period can't make a short request look
    # overdue (see _overdue_15).
    self._busy_start: float | None = None
    self._collecting: bool = False
    self._next_due: float = self._now() + self._period_s
    self._shutdown: bool = False

    # One worker => collects are serialized (the >= floor timestamp guarantee
    # relies on this).
    self._executor: concurrent.futures.ThreadPoolExecutor = (
        concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="metrics-collect"
        )
    )
    # Keep references to in-flight collect futures so they aren't GC'd.
    self._inflight_collects: set[concurrent.futures.Future[None]] = set()

  # --- Decision helpers (all called under _lock). -------------------------

  def _due(self, now: float) -> bool:
    return now >= self._next_due

  def _floor_ok(self, now: float) -> bool:
    return (
        self._last_collect is None
        or (now - self._last_collect) >= self._floor_s
    )

  def _overdue_15(self, now: float) -> bool:
    # Point 4 keeps a lone long-running request from piling up between collects.
    # "Overdue" is measured over the *current* busy period: from the last
    # collect in it, or -- if none yet -- from when the period began
    # (_busy_start). A collect from an earlier busy period is ignored, so a
    # short request arriving after a long idle gap does not look overdue; and a
    # fresh period is not overdue until 1.5*PERIOD into it. This stops point 4
    # firing on a short request's inference span, where it would stamp the floor
    # and then mute the request-end drain that carries the request's points.
    if self._busy_start is None:
      return False
    ref = self._busy_start
    if self._last_collect is not None:
      ref = max(ref, self._last_collect)
    return (now - ref) >= 1.5 * self._period_s

  def _arm(self, now: float) -> None:
    """Commits to a collect: marks it in flight and advances the guidepost grid.

    Does NOT stamp _last_collect -- that is done by the worker at the actual
    collect, so I2 constrains real collect spacing rather than decision spacing.

    Args:
      now: The current monotonic time, in seconds.
    """
    self._collecting = True
    # Advance the guidepost grid to the first tick strictly after `now`.
    if now >= self._next_due:
      missed = (now - self._next_due) // self._period_s + 1
      self._next_due += missed * self._period_s

  # --- Hooks: fast, synchronous, return "should I collect now?". ----------

  def note_request_start(self) -> bool:
    """Middleware, on request enter. _in_flight is always maintained."""
    with self._lock:
      now = self._now()
      overlap = self._in_flight >= 1
      if not overlap:
        self._busy_start = now  # a fresh busy period; reset the point-4 ref.
      self._in_flight += 1
      if self._collecting:
        return False
      if overlap and self._due(now):
        if self._floor_ok(now):  # point 2 -- collect at the just-started req.
          self._arm(now)
          return True
        # point 3 -- guidepost too close to last collect -> mute it.
        self._next_due += self._period_s
      return False

  def note_request_end(self) -> bool:
    """Middleware, after the response body is fully sent."""
    with self._lock:
      now = self._now()
      self._in_flight -= 1
      if self._in_flight < 0:
        self._in_flight = 0
      if self._in_flight == 0 and not self._collecting and self._floor_ok(now):
        self._arm(now)  # baseline force-flush (not gated on a guidepost).
        return True
      return False

  def note_generate_content_start(self) -> bool:
    """Span processor, on a generate_content span start (point 4)."""
    with self._lock:
      now = self._now()
      if (
          not self._collecting
          and self._in_flight >= 1
          and self._overdue_15(now)
          and self._floor_ok(now)
      ):
        self._arm(now)
        return True
      return False

  # --- Collect execution. -------------------------------------------------

  def collect_now(self) -> None:
    """Runs a committed collect (on the single-worker executor)."""
    with self._lock:
      self._last_collect = self._now()  # actual collect time.
    try:
      self.collect(timeout_millis=self._export_timeout_millis)
    except Exception:  # pylint: disable=broad-exception-caught
      # Runs on the executor; a fire-and-forget collect has no one to observe
      # its Future, so swallow-and-log rather than let the failure vanish.
      logger.exception("Exception during request-driven metric collect")
    finally:
      with self._lock:
        self._collecting = False

  def submit_collect(self) -> concurrent.futures.Future[None] | None:
    """Schedules a collect on the executor; returns its Future (or None).

    Returns None (and clears the _collecting guard) if the work can't be
    scheduled, e.g. during shutdown, so the guard can't wedge shut.

    Returns:
      The scheduled collect's Future, or None if it could not be scheduled.
    """
    with self._lock:
      shutting_down = self._shutdown
    if shutting_down:
      with self._lock:
        self._collecting = False
      return None
    try:
      future = self._executor.submit(self.collect_now)
    except RuntimeError:
      with self._lock:
        self._collecting = False
      return None
    self._inflight_collects.add(future)
    future.add_done_callback(self._inflight_collects.discard)
    return future

  def _receive_metrics(
      self,
      metrics_data: metrics_export.MetricsData,
      timeout_millis: float = 10_000,
      **kwargs: object,
  ) -> None:
    del kwargs  # unused
    token = otel_context.attach(
        otel_context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
      with self._export_lock:
        _ = self._exporter.export(metrics_data, timeout_millis=timeout_millis)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception("Exception while exporting metrics")
    finally:
      otel_context.detach(token)

  def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
    with self._lock:
      if self._shutdown:
        return
      self._shutdown = True
    # Drain any in-flight collects, then a best-effort final collect.
    self._executor.shutdown(wait=True)
    try:
      self.collect(timeout_millis=self._export_timeout_millis)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception("Exception during final metric collect on shutdown")
    self._exporter.shutdown(timeout_millis=timeout_millis, **kwargs)


@dataclasses.dataclass(frozen=True)
class MetricsState:
  """The metric-export handles the app wires up.

  The middleware that drives the reader lives separately, in
  ``google.adk.telemetry._agent_engine``; build it from ``reader``.
  """

  reader: _RequestDrivenMetricReader  # installed on the MeterProvider
  span_processor: SpanProcessor  # installed on the TracerProvider


def build_request_driven_metrics(
    exporter: metrics_export.MetricExporter,
) -> MetricsState:
  """Builds the request-driven reader and the span processor that drives it.

  Unlike a plain periodic reader, the returned reader collects only when driven
  from the request lifecycle. This does NOT set any global provider: the caller
  installs ``reader`` on a ``MeterProvider`` and ``span_processor`` on the
  ``TracerProvider`` (see ``google.adk.telemetry.google_cloud``), and drives the reader
  from the request path via ``google.adk.telemetry._agent_engine``.

  The export timeout comes from the OTel env the SDK reads
  (``OTEL_METRIC_EXPORT_TIMEOUT`` default 30000); the collect floor comes from
  ``GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS`` (default
  ``MIN_EXPORT_INTERVAL_MS``). Each falls back to its default on a missing/invalid value.

  Args:
    exporter: The metric exporter the reader drains into on each collect.

  Returns:
    The reader and the span processor that drive request-based metric export.
  """
  reader = _RequestDrivenMetricReader(exporter)
  return MetricsState(
      reader=reader,
      span_processor=_metrics_flushing_span_processor(reader),
  )


def _metrics_flushing_span_processor(
    reader: _RequestDrivenMetricReader,
    *,
    operation: str = "generate_content",
) -> SpanProcessor:
  """Returns a span processor that collects on `generate_content` span starts."""

  class _MetricsFlushingSpanProcessor(SpanProcessor):
    """Fires a fire-and-forget collect on each matching span start (point 4)."""

    def on_start(
        self, span: Span, parent_context: Context | None = None
    ) -> None:
      del parent_context  # unused
      # on_start runs inside span creation (ADK's inference path); a metrics
      # failure must not break the span it observes.
      try:
        attributes = span.attributes or {}
        name = span.name or ""
        if attributes.get(
            _GEN_AI_OPERATION_NAME
        ) == operation or name.startswith(operation):
          if reader.note_generate_content_start():
            _ = reader.submit_collect()  # fire-and-forget.
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Metrics span-start hook failed")

    def on_end(self, span: ReadableSpan) -> None:
      del span  # unused

    def shutdown(self) -> None:
      pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
      return True

  return _MetricsFlushingSpanProcessor()
