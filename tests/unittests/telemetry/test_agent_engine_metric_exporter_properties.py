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

"""Property-based tests for the request-driven metric reader (I1, I2 & I4).

Deterministic like ``test_agent_engine_metric_exporter.py`` (fake clock, collects
driven inline), but instead of a handful of hand-written scenarios this file lets
Hypothesis explore synthetic request workloads and asserts the *hard* invariants
from ``_agent_engine_metric_exporter``:

  I1 -- export only while serving: every collect lands inside some
        [request_start, request_end] window.
  I2 -- never collect more often than the floor: consecutive collects are >=
  FLOOR
        (3 s) apart.
  I4 -- no lost points on drain: every request end is flushed by a collect at or
        before the moment in-flight returns to zero (the end of its busy
        period).
        Holds because no collect can land in the last FLOOR seconds of a busy
        period, so its final drain is never floor-blocked. Every in-period
        collect is fired by some request R -- at R's start (point 2) or at a
        generation within R (point 4) -- and the workload keeps both >= FLOOR
        (plus a margin) before R's own end: requests are > FLOOR long (so a
        start collect is > FLOOR before R's end) and every request's
        generations land >= FLOOR before its end. Since R is in flight until its
        end, the busy period cannot end before then, so the collect is > FLOOR
        before the busy-period end. The generation constraint must apply to
        *every* request, not just long ones: _overdue_15 measures "overdue" from
        the last collect in the current busy period, so under a sustained busy
        period even a short request's generation can fire a point-4 collect.

(I3 -- "an export carries at most ~200 points" -- is deliberately *not*
asserted:
it is not a hard guarantee but a tunable. Under sustained overlap the reader
honors "collect once per period" yet a single drain can still exceed the cap;
the
remedy is to lower the export period, not a code change. So it isn't a bug to
guard against here.)

A workload is parametrized by six knobs (the strategy below):

  1. average number of requests            -> clamped to [0, 1000]
  2. variance of the gap between arrivals
  3. average request length (seconds)      -> clamped to [3.01, 10000]
  4. variance of request length
  5. average concurrency                    -> clamped to [0, 10]
  6. average generations per request        -> clamped to (0, 10]

Arrival rate is derived from concurrency and length via Little's law
(mean_gap = mean_length / mean_concurrency), so knob 5 actually controls
overlap.
Each request performs ``generations`` ``generate_content`` span starts, spread
across its first ``length - FLOOR - margin`` seconds so every request's
generations land >= FLOOR before its own end (the I4 constraint above). On
failure the offending 100 s window is rendered as ASCII art via
``hypothesis.note``.
"""

# pylint: disable=protected-access,redefined-outer-name
# Reuses the fake-clock harness from test_agent_engine_metric_exporter.
# pyright: reportPrivateUsage=false

import dataclasses
import math
import random
from typing import Literal

from hypothesis import given
from hypothesis import note
from hypothesis import settings
from hypothesis import strategies as st

from tests.unittests.telemetry.test_agent_engine_metric_exporter import _Harness

_PERIOD_S = 60.0  # guidepost grid (OTel default export interval).
_FLOOR_S = 3.0  # hard floor on collect spacing (I2).

# Min length sits a margin above FLOOR so a collect is always strictly (not
# exactly) more than FLOOR before its request's end; the margin also absorbs
# float error at that boundary (see I4).
_LEN_MARGIN_S = 0.01
_MIN_LEN, _MAX_LEN = _FLOOR_S + _LEN_MARGIN_S, 10_000.0
_MAX_REQUESTS = 1000
_MAX_CONCURRENCY = 10.0
_MAX_GENERATIONS = 10.0

# Event tie-break order at equal timestamps: start < generation < end.
_ORDER_START, _ORDER_GEN, _ORDER_END = 0, 1, 2


@dataclasses.dataclass(frozen=True)
class _Params:
  n_requests: int  # 1. average/target number of requests, in [0, 1000].
  arrival_variance: float  # 2. variance of the inter-arrival gap.
  mean_length: float  # 3. average request length, in [3, 10000] s.
  length_variance: float  # 4. variance of request length.
  mean_concurrency: float  # 5. average concurrency, in [0, 10].
  mean_generations: float  # 6. average generations per request, in (0, 10].
  seed: int  # draws the concrete workload from the knobs above.


@st.composite
def _params(draw: st.DrawFn) -> _Params:
  return _Params(
      n_requests=draw(st.integers(min_value=0, max_value=_MAX_REQUESTS)),
      arrival_variance=draw(
          st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
      ),
      mean_length=draw(
          st.floats(min_value=_MIN_LEN, max_value=_MAX_LEN, allow_nan=False)
      ),
      length_variance=draw(
          st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False)
      ),
      mean_concurrency=draw(
          st.floats(min_value=0.0, max_value=_MAX_CONCURRENCY, allow_nan=False)
      ),
      mean_generations=draw(
          st.floats(
              min_value=0.0,
              max_value=_MAX_GENERATIONS,
              exclude_min=True,  # (0, 10]
              allow_nan=False,
          )
      ),
      seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
  )


@dataclasses.dataclass(frozen=True)
class _Req:
  rid: str
  start: float
  end: float
  gens: tuple[float, ...]  # generate_content times within [start, end].


@dataclasses.dataclass(frozen=True)
class _Event:
  """A single point on the timeline the harness is driven through."""

  t: float
  order: int  # tie-break at equal `t` (_ORDER_START/_GEN/_END).
  rid: str
  kind: Literal["start", "gen", "end"]


@dataclasses.dataclass(frozen=True)
class _Window:
  """A request's [start, end] in-flight window."""

  start: float
  end: float


@dataclasses.dataclass(frozen=True)
class _Collect:
  """A collect/export that actually ran, and the hook kind that fired it."""

  t: float
  kind: Literal["start", "gen", "end"]


@dataclasses.dataclass(frozen=True)
class _Violation:
  """An invariant breach found in a simulation."""

  invariant: Literal["I1", "I2", "I4"]
  t: float  # the offending collect time.
  message: str


@dataclasses.dataclass(frozen=True)
class _Sim:
  reqs: list[_Req]
  windows: list[_Window]
  collects: list[_Collect]


def _build_requests(p: _Params) -> list[_Req]:
  """Turns the six knobs into a concrete list of requests (seeded, so stable)."""
  rng = random.Random(p.seed)
  # Little's law: concurrency = arrival_rate * service_time, so the mean gap
  # between arrivals is mean_length / mean_concurrency. Concurrency ~0 => the
  # requests barely overlap.
  base_gap = p.mean_length / max(p.mean_concurrency, 1e-3)
  arrival_sd = math.sqrt(p.arrival_variance)
  length_sd = math.sqrt(p.length_variance)

  reqs: list[_Req] = []
  t = 0.0
  for i in range(p.n_requests):
    t += max(0.0, rng.gauss(base_gap, arrival_sd))
    length = min(_MAX_LEN, max(_MIN_LEN, rng.gauss(p.mean_length, length_sd)))
    # Poisson-ish count around the mean; no dedicated variance knob for this.
    n_gen = max(
        0, round(rng.gauss(p.mean_generations, math.sqrt(p.mean_generations)))
    )
    # Place gens >= FLOOR (plus margin) before the request's end, for the I4
    # guarantee (see module docstring). Applies to *every* request, not just
    # those > 1.5*PERIOD: _overdue_15 is relative to the last collect in the
    # busy period, so a short request's gen can fire a point-4 collect too.
    gen_span = max(0.0, length - (_FLOOR_S + _LEN_MARGIN_S))
    gens = tuple(t + (k + 0.5) / n_gen * gen_span for k in range(n_gen))
    reqs.append(_Req(f"r{i}", t, t + length, gens))
  return reqs


def _simulate(p: _Params) -> _Sim:
  """Replays the workload through the real reader and records every collect."""
  reqs = _build_requests(p)

  events: list[_Event] = []
  for r in reqs:
    events.append(_Event(r.start, _ORDER_START, r.rid, "start"))
    events.extend(_Event(g, _ORDER_GEN, r.rid, "gen") for g in r.gens)
    events.append(_Event(r.end, _ORDER_END, r.rid, "end"))
  events.sort(key=lambda e: (e.t, e.order))

  h = _Harness(period_s=_PERIOD_S, floor_s=_FLOOR_S)
  collects: list[_Collect] = []
  try:
    for e in events:
      _ = h.at(e.t)
      before = len(h.collects)
      if e.kind == "start":
        h.start(e.rid)
      elif e.kind == "gen":
        h.generate_content()
      else:
        h.end(e.rid)
      if len(h.collects) > before:  # this hook triggered a collect.
        collects.append(_Collect(e.t, e.kind))
    windows = [_Window(start, end) for start, end in h.windows]
  finally:
    h.close()
  return _Sim(reqs=reqs, windows=windows, collects=collects)


def _busy_periods(windows: list[_Window]) -> list[_Window]:
  """Merges request windows into maximal [start, end] in-flight busy periods.

  Two windows that merely touch (one ends exactly when the next starts) belong
  to the same busy period: at equal timestamps the harness applies the start
  before the end, so in-flight never dips to zero. Hence the inclusive `<=`.

  Args:
    windows: Per-request in-flight windows.

  Returns:
    Maximal merged busy periods, ordered by start.
  """
  if not windows:
    return []
  ordered = sorted(windows, key=lambda w: w.start)
  merged = [ordered[0]]
  for w in ordered[1:]:
    last = merged[-1]
    if w.start <= last.end:  # overlapping or touching -> same busy period.
      merged[-1] = _Window(last.start, max(last.end, w.end))
    else:
      merged.append(w)
  return merged


def _violations(sim: _Sim) -> list[_Violation]:
  """Returns every breach of I1/I2/I4 in `sim`."""
  out: list[_Violation] = []
  times = [c.t for c in sim.collects]

  # I2 -- consecutive collects are >= FLOOR apart.
  for a, b in zip(times, times[1:]):
    if b - a < _FLOOR_S - 1e-9:
      out.append(
          _Violation(
              "I2", b, f"collects {b - a:.3f}s apart (< floor {_FLOOR_S}s)"
          )
      )

  # I1 -- each collect lands inside some request window.
  for t in times:
    if not any(w.start <= t <= w.end for w in sim.windows):
      out.append(
          _Violation(
              "I1", t, f"collect at t={t:.1f}s with no request in flight"
          )
      )

  # I4 -- every request end is flushed before in-flight returns to zero: there
  # is a collect between the request's end and the end of its busy period.
  busy = _busy_periods(sim.windows)
  for r in sim.reqs:
    period = next(w for w in busy if w.start <= r.end <= w.end)
    if not any(r.end <= c.t <= period.end for c in sim.collects):
      out.append(
          _Violation(
              "I4",
              r.end,
              f"{r.rid} ended at {r.end:.1f}s with no collect before in-flight "
              f"hit 0 at {period.end:.1f}s",
          )
      )
  return out


def _render_timeline(
    sim: _Sim, focus_t: float, msg: str, width: int = 100
) -> str:
  """Renders a `width`-second window of the workload as ASCII art.

  The window is centered on the offending collect; requests overlapping it are
  drawn as ``[====]`` bars and the collects row marks every collect ``C`` with
  the offending one as ``X``.

  Args:
    sim: The simulated workload (requests and collects) to render.
    focus_t: The time (seconds) to center the window on.
    msg: The violation message shown in the header.
    width: The window width in seconds.

  Returns:
    The multi-line ASCII rendering of the window.
  """
  w0 = max(0.0, focus_t - width / 2)

  def col(x: float) -> int:
    return int(round(x - w0))

  label_w = 7
  pad = " " * (label_w + 1)
  lines = [
      f"VIOLATION [{msg}]",
      f"window [{w0:.0f}s .. {w0 + width:.0f}s]  1 col = 1s",
  ]

  # Ruler: a tick every 10 s.
  ticks = list(" " * width)
  labels = list(" " * width)
  for c in range(0, width + 1, 10):
    if c < width:
      ticks[c] = "|"
    stamp = f"{w0 + c:.0f}"
    for j, ch in enumerate(stamp):
      if c + j < width:
        labels[c + j] = ch
  lines.append(pad + "".join(ticks))
  lines.append(pad + "".join(labels))

  shown = sorted(
      (r for r in sim.reqs if r.end >= w0 and r.start <= w0 + width),
      key=lambda r: r.start,
  )
  clipped = len(shown) - 30
  for r in shown[:30]:
    row = [" "] * width
    a, b = col(r.start), col(r.end)
    for c in range(max(a, 0), min(b, width - 1) + 1):
      row[c] = "="
    if 0 <= a < width:
      row[a] = "["
    if 0 <= b < width:
      row[b] = "]"
    for g in r.gens:  # mark generations with 'o'.
      gc = col(g)
      if 0 <= gc < width and row[gc] in "=[]":
        row[gc] = "o"
    lines.append(f"{r.rid:>{label_w}}|" + "".join(row))
  if clipped > 0:
    lines.append(f"{'...':>{label_w}}|(+{clipped} more requests in window)")

  crow = ["."] * width
  for c in sim.collects:
    ci = col(c.t)
    if 0 <= ci < width:
      crow[ci] = "C"
  fc = col(focus_t)
  if 0 <= fc < width:
    crow[fc] = "X"
  lines.append(f"{'collect':>{label_w}}|" + "".join(crow))
  lines.append(
      pad
      + "legend: [====] request  o generation  C collect  X violating export"
  )
  return "\n".join(lines)


@settings(max_examples=300, deadline=None)
@given(p=_params())
def test_hard_invariants(p: _Params) -> None:
  """I1/I2/I4 hard invariants always hold.

  I1 (export only while serving), I2 (>= floor spacing), and I4 (no lost points
  on drain) are guarantees the reader enforces structurally: every collect is
  driven from a request hook, the floor gate rejects sub-floor spacing, and
  every busy period ends with a drain collect that flushes the requests that
  ended in it.

  Args:
    p: Hypothesis-generated scenario parameters.
  """
  sim = _simulate(p)
  violations = _violations(sim)
  if violations:
    first = violations[0]
    note(f"params: {p}")
    note(_render_timeline(sim, first.t, f"{first.invariant}: {first.message}"))
  assert not violations, "; ".join(
      f"{v.invariant}: {v.message}" for v in violations
  )
