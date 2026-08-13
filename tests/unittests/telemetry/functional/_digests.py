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

"""A deterministic digest of the telemetry one scenario run produced.

``TelemetryDigest.build`` turns in-memory spans, logs and metric points
into a value that can be compared with ``==`` and stored as JSON: the
span tree with its attributes and per-span logs, plus every metric point
grouped by metric name.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import json
from typing import TYPE_CHECKING

from opentelemetry.sdk.metrics.export import HistogramDataPoint
from opentelemetry.sdk.metrics.export import NumberDataPoint

if TYPE_CHECKING:
  from opentelemetry.sdk._logs import ReadableLogRecord
  from opentelemetry.sdk.metrics.export import MetricsData
  from opentelemetry.sdk.trace import ReadableSpan


# Difficult to extract, non deterministic attribute keys.
# We check only for their presence, instead of their values.
NON_DETERMINISTIC_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gcp.vertex.agent.event_id",
    "gen_ai.tool.call.id",
    "gcp.vertex.agent.associated_event_ids",
    "gen_ai.conversation.id",
    "gcp.vertex.agent.invocation_id",
    "gcp.vertex.agent.session_id",
})

# Span attribute keys whose values are JSON-serialized strings.
# These are parsed back into Python objects before comparison so that JSON
# property ordering doesn't drive test stability.
JSON_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
})

# Sentinel for a value that cannot be pinned -- a generated id, a wall-clock
# duration, an elided payload. Substituted on both sides of the comparison, so
# such a field is only ever asserted to be present.
PRESENT = "PRESENT"

# ---------------------------------------------------------------------------
# Digests.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogDigest:
  """A deterministic digest of a ``ReadableLogRecord``.

  ``attributes`` and ``body`` are normalized via ``_normalize`` so test
  expectations can be written using plain Python literals (lists/dicts).
  """

  event_name: str
  body: object = None
  attributes: dict[str, object] = field(default_factory=dict)

  @classmethod
  def from_log(cls, log: ReadableLogRecord) -> LogDigest:
    attrs: dict[str, object] = {}
    for k, v in (log.log_record.attributes or {}).items():
      if k in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        attrs[k] = PRESENT
      else:
        attrs[k] = _normalize(v)
    return cls(
        event_name=log.log_record.event_name or "",
        body=_normalize(log.log_record.body),
        attributes=attrs,
    )


@dataclass(frozen=True)
class SpanDigest:
  """A deterministic digest of a span in the in-memory span tree.

  In addition to the span's own name + attributes + child spans, each
  digest also carries the ``LogDigest`` records that were emitted while
  the span was the active span (matched by ``log_record.span_id``).

  ``status`` is the span's ``StatusCode`` name, so a tree that expects a
  span to be marked failed says so explicitly. It defaults to ``UNSET``,
  which is what a span that nothing marked carries.
  """

  name: str
  attributes: dict[str, object]
  status: str = "UNSET"
  children: list[SpanDigest] = field(default_factory=list)
  logs: list[LogDigest] = field(default_factory=list)

  @classmethod
  def from_span(cls, span: ReadableSpan) -> SpanDigest:
    """Builds a single ``SpanDigest`` (no children, no logs) from a span.

    Attribute values are normalized so that:
    * Non-deterministic keys collapse to the ``PRESENT`` sentinel.
    * JSON-serialized attribute values are parsed into Python objects.
    * All other values pass through ``_normalize`` (tuples → lists,
      enums → ``.value``, ``None`` dict entries dropped).
    """
    determinized_attributes: dict[str, object] = {}
    for attr_key, attr_val in (span.attributes or {}).items():
      if attr_key in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        determinized_attributes[attr_key] = PRESENT
      elif attr_key in JSON_ATTRIBUTE_KEYS and isinstance(attr_val, str):
        determinized_attributes[attr_key] = _normalize(json.loads(attr_val))
      else:
        determinized_attributes[attr_key] = _normalize(attr_val)
    return cls(
        name=span.name,
        attributes=determinized_attributes,
        status=span.status.status_code.name,
    )

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...] = (),
  ) -> SpanDigest:
    """Builds the in-memory span tree, attaching logs by span id.

    Used for clear diffs with pytest assertions.
    """
    digest_by_id: dict[int, SpanDigest] = {}
    for span in spans:
      if span.context is None:
        continue
      digest_by_id[span.context.span_id] = cls.from_span(span)

    # Attach each log to its enclosing span (matched by span_id).
    for log in logs:
      span_id = log.log_record.span_id
      if span_id is None or span_id == 0:
        continue
      digest = digest_by_id.get(span_id)
      if digest is None:
        continue
      digest.logs.append(LogDigest.from_log(log))

    root: SpanDigest | None = None
    for span in spans:
      if span.context is None:
        continue
      digest = digest_by_id[span.context.span_id]
      if span.parent and span.parent.span_id in digest_by_id:
        parent_digest = digest_by_id[span.parent.span_id]
        parent_digest.children.append(digest)
      else:
        if root is not None:
          raise ValueError("Multiple root spans found.")
        root = digest

    # Sort for deterministic comparisons.
    for digest in digest_by_id.values():
      digest.children.sort(key=lambda s: s.name)
      digest.logs[:] = sorted_log_digests(digest.logs)

    if root is None:
      raise ValueError("No root span found in the provided spans.")
    return root

  def all_logs(self) -> list[LogDigest]:
    """Returns all log digests in the tree, sorted deterministically."""
    collected: list[LogDigest] = []

    def _walk(node: SpanDigest) -> None:
      collected.extend(node.logs)
      for child in node.children:
        _walk(child)

    _walk(self)
    return sorted_log_digests(collected)


def sorted_log_digests(logs: list[LogDigest]) -> list[LogDigest]:
  """Returns ``logs`` sorted in a stable, content-derived order."""
  return sorted(
      logs,
      key=lambda log: (
          log.event_name,
          json.dumps(log.body, sort_keys=True, default=str),
          json.dumps(log.attributes, sort_keys=True, default=str),
      ),
  )


@dataclass(frozen=True)
class MetricPoint:
  """A single recorded metric data point."""

  attributes: dict[str, object]
  value: object

  def __hash__(self) -> int:
    return hash((self.sort_key(), self.value))

  def sort_key(self) -> str:
    return json.dumps(self.attributes, sort_keys=True, default=str)


def _grouped_metric_points(
    metrics_data: MetricsData,
) -> dict[str, list[MetricPoint]]:
  """Groups every recorded point by metric name.

  Both the names and the points within a group are sorted, so the result is
  independent of recording order and can be compared (and serialized) as
  plain lists.
  """
  grouped: dict[str, set[MetricPoint]] = {}
  for resource_metric in metrics_data.resource_metrics:
    for scope_metric in resource_metric.scope_metrics:
      for metric in scope_metric.metrics:
        for dp in metric.data.data_points:
          # Sum histograms expose ``.sum``; gauge / counter points expose
          # ``.value``. isinstance (not hasattr) keeps the typing precise.
          if isinstance(dp, HistogramDataPoint):
            value = dp.sum
          elif isinstance(dp, NumberDataPoint):
            value = dp.value
          else:
            value = PRESENT
          # ``*.duration`` histograms record wall-clock timings, which are
          # non-deterministic; replace them so expectations need not pin a
          # timing.
          if metric.name.endswith(".duration"):
            value = PRESENT
          grouped.setdefault(metric.name, set()).add(
              MetricPoint(attributes=dict(dp.attributes), value=value)
          )
  return {
      name: sorted(points, key=MetricPoint.sort_key)
      for name, points in sorted(grouped.items())
  }


@dataclass(frozen=True)
class TelemetryDigest:
  """The full telemetry surface produced by one scenario run.

  Bundles the root span tree (with per-span logs attached) and every recorded
  metric point grouped by metric name. Everything is sorted as it is built,
  so a digest is fully deterministic and round-trips through plain JSON:
  ``build`` produces the actual one; ``functional_test_goldens.load_golden``
  the recorded one.
  """

  root_span: SpanDigest
  metric_points: dict[str, list[MetricPoint]]

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...],
      metrics_data: MetricsData,
  ) -> TelemetryDigest:
    """Builds the actual digest from in-memory spans, logs and metrics."""
    return cls(
        root_span=SpanDigest.build(spans, logs),
        metric_points=_grouped_metric_points(metrics_data),
    )


def _normalize(value: object) -> object:
  """Normalizes a value for stable equality.

  * Tuples become lists (OTel coerces sequences to tuples on attributes).
  * Enums become their ``.value``.
  * Dict entries whose value is ``None`` are dropped (these are inserted by
    pydantic ``model_dump`` for unset fields and would dominate diffs).
  """
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, tuple):
    return [_normalize(v) for v in value]
  if isinstance(value, list):
    return [_normalize(v) for v in value]
  if isinstance(value, dict):
    return {k: _normalize(v) for k, v in value.items() if v is not None}
  return value
