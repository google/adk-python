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

"""Unit tests for the reader in `_agent_engine_metric_exporter`.

Deterministic: no real time, no network. A fake monotonic clock is injected into
the reader; collects are driven inline (call a `note_*` hook and, when it
returns
True, run the collect synchronously so the fake clock stamps the export). The
scenarios mirror the diagrams in the module docstring (baseline drain, overlap
batching, the four guidepost points, the sub-floor skip). Two blanket invariants
are asserted over every scenario:

  I2 -- consecutive collects are >= FLOOR apart.
  I1 -- every collect lands inside some [request_start, request_end] window.
"""

# pylint: disable=protected-access,redefined-outer-name
# This is a unit test of a private module, so private access is expected.
# pyright: reportPrivateUsage=false

from google.adk.telemetry import _agent_engine_metric_exporter as _metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.metrics.export import MetricsData
import pytest

_PERIOD_S = 10.0
_FLOOR_S = 3.0


class _RecordingExporter(MetricExporter):
  """Records the fake-clock time of every export."""

  def __init__(self, clock: list[float]):
    super().__init__()
    self._clock: list[float] = clock
    self.times: list[float] = []

  def export(
      self,
      metrics_data: MetricsData,
      timeout_millis: float = 10_000,
      **kwargs: object,
  ) -> MetricExportResult:
    del metrics_data, timeout_millis, kwargs  # unused
    self.times.append(self._clock[0])
    return MetricExportResult.SUCCESS

  def force_flush(self, timeout_millis: float = 10_000) -> bool:
    del timeout_millis  # unused
    return True

  def shutdown(self, timeout_millis: float = 30_000, **kwargs: object) -> None:
    del timeout_millis, kwargs  # unused


class _Harness:
  """Drives a reader with a fake clock and records collects + request windows."""

  def __init__(self, period_s: float = _PERIOD_S, floor_s: float = _FLOOR_S):
    self.t: list[float] = [0.0]
    self.exporter: _RecordingExporter = _RecordingExporter(self.t)
    self.reader: _metrics._RequestDrivenMetricReader = (
        _metrics._RequestDrivenMetricReader(
            self.exporter,
            export_interval_millis=period_s * 1000.0,
            floor_millis=floor_s * 1000.0,
            now=lambda: self.t[0],
        )
    )
    self.meter_provider: MeterProvider = MeterProvider(
        metric_readers=[self.reader]
    )
    # A cumulative counter with a recorded value so every collect has data to
    # export (an empty collect exports nothing).
    self.meter_provider.get_meter("test").create_counter("c").add(1)
    self._open: dict[str, float] = {}
    self.windows: list[tuple[float, float]] = []

  def at(self, when: float) -> "_Harness":
    self.t[0] = float(when)
    return self

  def start(self, rid: str) -> None:
    self._open[rid] = self.t[0]
    if self.reader.note_request_start():
      self.reader.collect_now()

  def end(self, rid: str) -> None:
    self.windows.append((self._open.pop(rid), self.t[0]))
    if self.reader.note_request_end():
      self.reader.collect_now()

  def generate_content(self) -> None:
    if self.reader.note_generate_content_start():
      self.reader.collect_now()

  @property
  def collects(self) -> list[float]:
    return list(self.exporter.times)

  def close(self) -> None:
    self.meter_provider.shutdown()


# --- Scenario builders (each returns a driven harness). --------------------


def _scenario_baseline_drain() -> _Harness:
  """an isolated request collects when it drains to zero."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(5).end("r1")
  assert h.collects == [5.0]
  return h


def _scenario_overlap_batched() -> _Harness:
  """A burst of overlapping requests produces a single collect."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(1).start("r2")
  h.at(2).start("r3")
  h.at(3).end("r1")
  h.at(4).end("r2")
  h.at(5).end("r3")
  assert h.collects == [5.0]  # one collect for all three.
  return h


def _scenario_guidepost_consumed_by_drain() -> _Harness:
  """A guidepost inside a lone request is swept by its drain."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(12).end("r1")  # crosses the guidepost at 10, but only drains here.
  assert h.collects == [12.0]  # the guidepost never fired on its own.
  return h


def _scenario_guidepost_fires_at_start() -> _Harness:
  """Under continuous overlap, a guidepost fires at next start."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(2).start("r2")
  h.at(4).start("r3")
  h.at(11).start("r4")  # guidepost (10) crossed, overlap -> collect at start.
  h.at(12).end("r1")
  h.at(13).end("r2")
  h.at(14).end("r3")
  h.at(16).end("r4")  # baseline drain collect.
  assert h.collects == [11.0, 16.0]
  return h


def _scenario_guidepost_muted() -> _Harness:
  """A guidepost within FLOOR of the last collect is muted."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(9).end("r1")  # drain collect at 9.
  h.at(9).start("r2")
  h.at(10).start("r3")  # guidepost due, but 10-9 < FLOOR -> muted, no collect.
  h.at(11).end("r2")
  h.at(12).end("r3")  # next collect is this drain.
  assert h.collects == [9.0, 12.0]
  return h


def _scenario_generate_content_backstop() -> _Harness:
  """A lone long request collects off its generate_content spans."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(5).generate_content()  # 5s into busy period (<1.5*PERIOD) -> no collect.
  h.at(10).generate_content()  # 10s into busy period (<15) -> no collect.
  h.at(21).generate_content()  # 21s into busy period (>=15) -> collect at 21.
  h.at(30).generate_content()  # 9s since last collect -> no collect.
  h.at(37).generate_content()  # 16s since last collect (>=15) -> collect at 37.
  h.at(40).end("r1")  # drain collect at 40 (>= FLOOR after 37).
  assert h.collects == [21.0, 37.0, 40.0]
  return h


def _scenario_short_first_request_not_preempted() -> _Harness:
  """A short first request's drain carries its points; no premature gen collect.

  Regression for the empty-metrics bug: point 4 used to treat "no collect yet"
  as overdue, so the first inference span of the very first request fired a
  collect *before* the request's metrics were recorded. That collect stamped the
  floor and muted the request-end drain (< FLOOR later) that carries the points,
  so nothing useful was ever exported. The collect must land at the drain (t=4),
  not at the generation (t=2).

  Returns:
    The driven harness, for the shared invariant checks.
  """
  h = _Harness()
  h.at(0).start("r1")
  h.at(2).generate_content()  # first span of a short first req -> no collect.
  h.at(4).end("r1")  # drain (would be muted if a collect had fired at t=2).
  assert h.collects == [4.0]
  return h


def _scenario_subfloor_skip() -> _Harness:
  """A sub-floor request draining right after a collect is skipped."""
  h = _Harness()
  h.at(0).start("r1")
  h.at(5).end("r1")  # collect at 5.
  h.at(6).start("r2")
  h.at(6.5).end("r2")  # 6.5-5 < FLOOR -> skipped; its points ride the next.
  h.at(9).start("r3")
  h.at(9).end("r3")  # 9-5 >= FLOOR -> collect at 9 (sweeps r2's points).
  assert h.collects == [5.0, 9.0]
  return h


_SCENARIOS = {
    "baseline_drain": _scenario_baseline_drain,
    "overlap_batched": _scenario_overlap_batched,
    "guidepost_consumed_by_drain": _scenario_guidepost_consumed_by_drain,
    "guidepost_fires_at_start": _scenario_guidepost_fires_at_start,
    "guidepost_muted": _scenario_guidepost_muted,
    "generate_content_backstop": _scenario_generate_content_backstop,
    "short_first_request_not_preempted": (
        _scenario_short_first_request_not_preempted
    ),
    "subfloor_skip": _scenario_subfloor_skip,
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_scenario_invariants(name: str) -> None:
  """Every scenario honors I1 (in-flight) and I2 (floor spacing)."""
  h = _SCENARIOS[name]()
  try:
    collects = h.collects
    assert collects, "scenario produced no collects"

    # I2: consecutive collects are >= FLOOR apart.
    for a, b in zip(collects, collects[1:]):
      assert b - a >= _FLOOR_S, f"{name}: floor violated: {collects}"

    # I1: each collect lands inside some [start, end] request window.
    for c in collects:
      assert any(
          start <= c <= end for start, end in h.windows
      ), f"{name}: collect {c} outside all windows {h.windows}"
  finally:
    h.close()


def test_floor_seconds_default(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.delenv(
      _metrics.GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS,
      raising=False,
  )
  assert _metrics._floor_seconds() == _metrics.MIN_EXPORT_INTERVAL_MS / 1000.0


def test_floor_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(
      _metrics.GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS,
      "1500",
  )
  assert _metrics._floor_seconds() == 1.5


def test_floor_seconds_invalid_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv(
      _metrics.GOOGLE_CLOUD_AGENT_ENGINE_METRICS_COLLECTION_INTERVAL_FLOOR_MS,
      "not-a-number",
  )
  assert _metrics._floor_seconds() == _metrics.MIN_EXPORT_INTERVAL_MS / 1000.0
