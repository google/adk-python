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

``TelemetryDigest.build`` turns in-memory spans, logs and metric points into
a value that can be compared with ``==`` and stored as JSON: the span tree
with its attributes and per-span logs, plus every metric point grouped by
metric name.

This is one recording, of one run, under one instrumentation. That two of
them may legitimately differ is not this module's concern: see
``_divergences``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from opentelemetry.sdk._logs import ReadableLogRecord
  from opentelemetry.sdk.metrics.export import MetricsData
  from opentelemetry.sdk.trace import ReadableSpan

# Sentinel for a value that cannot be pinned -- a generated id, a wall-clock
# duration, an elided payload. Substituted on both sides of the comparison, so
# such a field is only ever asserted to be present.
PRESENT = "PRESENT"

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

# Payload fields that carry a generated id rather than anything recorded --
# ``{"id": "adk-<uuid>", "name": "some_tool", ...}`` in the experimental
# shape, ``{"function_call": {"id": ...}}`` in the stable one. ADK fills one
# in when the model doesn't supply one, so it is new on every run. A side
# that omits it entirely still diverges: PRESENT is not missing.
NON_DETERMINISTIC_PAYLOAD_FIELDS: frozenset[str] = frozenset({"id"})

# Span attribute keys whose values are JSON-serialized strings.
# These are parsed back into Python objects before comparison so that JSON
# property ordering doesn't drive test stability.
JSON_ATTRIBUTE_KEYS: frozenset[str] = frozenset({
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "gen_ai.system_instructions",
    "gen_ai.tool.definitions",
})


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
    return cls(
        event_name=log.log_record.event_name or "",
        body=_normalize(log.log_record.body),
        attributes={
            key: (
                PRESENT
                if key in NON_DETERMINISTIC_ATTRIBUTE_KEYS
                else _normalize(value)
            )
            for key, value in (log.log_record.attributes or {}).items()
        },
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
    attributes: dict[str, object] = {}
    for key, value in (span.attributes or {}).items():
      if key in NON_DETERMINISTIC_ATTRIBUTE_KEYS:
        attributes[key] = PRESENT
      elif key in JSON_ATTRIBUTE_KEYS and isinstance(value, str):
        attributes[key] = _normalize(json.loads(value))
      else:
        attributes[key] = _normalize(value)
    return cls(
        name=span.name,
        attributes=attributes,
        status=span.status.status_code.name,
    )

  @classmethod
  def build(
      cls,
      spans: tuple[ReadableSpan, ...],
      logs: tuple[ReadableLogRecord, ...] = (),
  ) -> SpanDigest:
    """Builds the in-memory span tree, attaching logs by span id.

    Children and logs come out in the order they were emitted -- the scenario
    runs single-threaded, so that order is deterministic, and it is the order
    a reader of the golden expects the run to have gone in. The exporter hands
    spans over as they *end*, so the tree is assembled by start time.
    """
    digest_by_id: dict[int, SpanDigest] = {}
    for span in spans:
      if span.context is None:
        continue
      digest_by_id[span.context.span_id] = cls.from_span(span)

    # Attach each log to its enclosing span (matched by span_id).
    for log in sorted(logs, key=lambda log: log.log_record.observed_timestamp):
      span_id = log.log_record.span_id
      if span_id is None or span_id == 0:
        continue
      digest = digest_by_id.get(span_id)
      if digest is None:
        continue
      digest.logs.append(LogDigest.from_log(log))

    root: SpanDigest | None = None
    for span in sorted(spans, key=lambda span: span.start_time or 0):
      if span.context is None:
        continue
      digest = digest_by_id[span.context.span_id]
      if span.parent and span.parent.span_id in digest_by_id:
        digest_by_id[span.parent.span_id].children.append(digest)
      elif root is not None:
        raise ValueError("Multiple root spans found.")
      else:
        root = digest

    if root is None:
      raise ValueError("No root span found in the provided spans.")
    return root


def json_key(value: object) -> str:
  """A value's canonical JSON string, for comparing and ordering by."""
  return json.dumps(value, sort_keys=True, default=str)


@dataclass(frozen=True)
class MetricPoint:
  """A single recorded metric data point."""

  attributes: dict[str, object]
  value: object

  def __hash__(self) -> int:
    return hash((self.sort_key(), json_key(self.value)))

  def sort_key(self) -> str:
    return json_key(self.attributes)


@dataclass(frozen=True)
class TelemetryDigest:
  """The full telemetry surface produced by one scenario run.

  Bundles the root span tree (with per-span logs attached) and every recorded
  metric point grouped by metric name. Everything is sorted as it is built,
  so a digest is fully deterministic and round-trips through plain JSON.
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
    """Builds the digest of one recording, from its in-memory telemetry."""
    return cls(
        root_span=SpanDigest.build(spans, logs),
        metric_points=_grouped_metric_points(metrics_data),
    )


def _grouped_metric_points(
    metrics_data: MetricsData,
) -> dict[str, list[MetricPoint]]:
  """Groups every recorded point by metric name.

  Both the names and the points within a group are sorted, so the result is
  independent of recording order and can be compared (and serialized) as
  plain lists.
  """
  # Imported here: only the recording path needs the SDK's point types.
  from opentelemetry.sdk.metrics.export import HistogramDataPoint  # pylint: disable=g-import-not-at-top
  from opentelemetry.sdk.metrics.export import NumberDataPoint  # pylint: disable=g-import-not-at-top

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


def _normalize(value: object) -> object:
  """Normalizes a value for stable equality.

  * Tuples become lists (OTel coerces sequences to tuples on attributes).
  * Enums become their ``.value``.
  * Dict entries whose value is ``None`` are dropped (these are inserted by
    pydantic ``model_dump`` for unset fields and would dominate diffs).
  * A generated id inside a payload collapses to ``PRESENT``, like the
    attribute-level ones.
  """
  if isinstance(value, Enum):
    return value.value
  if isinstance(value, (tuple, list)):
    return [_normalize(v) for v in value]
  if isinstance(value, dict):
    return {
        key: (
            PRESENT
            if key in NON_DETERMINISTIC_PAYLOAD_FIELDS
            else _normalize(item)
        )
        for key, item in value.items()
        if item is not None
    }
  return value
