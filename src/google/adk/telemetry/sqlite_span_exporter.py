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

"""SQLite-backed OpenTelemetry span exporter for local development."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any
from typing import Iterable
from typing import Optional
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext
from opentelemetry.trace import TraceFlags
from opentelemetry.trace import TraceState

logger = logging.getLogger("google_adk." + __name__)

_CREATE_SPANS_TABLE = """
CREATE TABLE IF NOT EXISTS spans (
  span_id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  parent_span_id TEXT,
  name TEXT NOT NULL,
  start_time_unix_nano INTEGER,
  end_time_unix_nano INTEGER,
  session_id TEXT,
  invocation_id TEXT,
  attributes_json TEXT
);
"""

_CREATE_SESSION_INDEX = """
CREATE INDEX IF NOT EXISTS spans_session_id_idx ON spans(session_id);
"""

_CREATE_TRACE_INDEX = """
CREATE INDEX IF NOT EXISTS spans_trace_id_idx ON spans(trace_id);
"""

_INSERT_SPAN = """
INSERT OR REPLACE INTO spans (
  span_id,
  trace_id,
  parent_span_id,
  name,
  start_time_unix_nano,
  end_time_unix_nano,
  session_id,
  invocation_id,
  attributes_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_DEFAULT_TIMEOUT_SECONDS = 30.0


class SqliteSpanExporter(SpanExporter):
  """Exports spans to a local SQLite database.

  This is intended for local development (e.g. `adk web`) to allow reloading
  traces for older sessions after process restart.
  """

  def __init__(self, *, db_path: str):
    self._db_path = db_path
    self._lock = threading.Lock()
    self._conn: Optional[sqlite3.Connection] = None
    self._ensure_schema()

  def _get_connection(self) -> sqlite3.Connection:
    if self._conn is None:
      self._conn = sqlite3.connect(
          self._db_path,
          timeout=_DEFAULT_TIMEOUT_SECONDS,
          check_same_thread=False,
      )
      self._conn.row_factory = sqlite3.Row
    return self._conn

  def _ensure_schema(self) -> None:
    with self._lock:
      conn = self._get_connection()
      conn.execute(_CREATE_SPANS_TABLE)
      conn.execute(_CREATE_SESSION_INDEX)
      conn.execute(_CREATE_TRACE_INDEX)
      conn.commit()

  def _serialize_attributes(self, attributes: dict[str, Any]) -> str:
    try:
      return json.dumps(
          attributes,
          ensure_ascii=False,
          default=lambda o: "<not serializable>",
      )
    except (TypeError, ValueError) as e:
      logger.debug("Failed to serialize span attributes: %r", e)
      return "{}"

  def _deserialize_attributes(self, attributes_json: Any) -> dict[str, Any]:
    if not attributes_json:
      return {}
    try:
      attributes = json.loads(attributes_json)
    except (json.JSONDecodeError, TypeError) as e:
      logger.debug("Failed to deserialize span attributes: %r", e)
      return {}
    return attributes if isinstance(attributes, dict) else {}

  def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
    try:
      with self._lock:
        conn = self._get_connection()
        rows: list[tuple[Any, ...]] = []
        for span in spans:
          attributes = dict(span.attributes) if span.attributes else {}
          session_id = attributes.get(
              "gcp.vertex.agent.session_id"
          ) or attributes.get("gen_ai.conversation.id")
          invocation_id = attributes.get("gcp.vertex.agent.invocation_id")

          parent_span_id = None
          if span.parent is not None:
            parent_span_id = format(span.parent.span_id, "016x")

          rows.append((
              format(span.context.span_id, "016x"),
              format(span.context.trace_id, "032x"),
              parent_span_id,
              span.name,
              span.start_time,
              span.end_time,
              session_id,
              invocation_id,
              self._serialize_attributes(attributes),
          ))
        conn.executemany(_INSERT_SPAN, rows)
        conn.commit()
      return SpanExportResult.SUCCESS
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Failed to export spans to SQLite: %s", e)
      return SpanExportResult.FAILURE

  def shutdown(self) -> None:
    with self._lock:
      if self._conn is not None:
        self._conn.close()
        self._conn = None

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True

  def _query(self, sql: str, params: Iterable[Any]) -> list[sqlite3.Row]:
    with self._lock:
      conn = self._get_connection()
      cur = conn.execute(sql, tuple(params))
      return list(cur.fetchall())

  def _row_to_readable_span(self, row: sqlite3.Row) -> ReadableSpan:
    trace_id_hex = row["trace_id"]
    span_id_hex = row["span_id"]
    trace_id = int(str(trace_id_hex), 16)
    span_id = int(str(span_id_hex), 16)
    trace_state = TraceState()
    trace_flags = TraceFlags(TraceFlags.SAMPLED)
    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=trace_flags,
        trace_state=trace_state,
    )

    parent: SpanContext | None = None
    parent_span_id_hex = row["parent_span_id"]
    if parent_span_id_hex:
      parent = SpanContext(
          trace_id=trace_id,
          span_id=int(str(parent_span_id_hex), 16),
          is_remote=False,
          trace_flags=trace_flags,
          trace_state=trace_state,
      )

    attributes = self._deserialize_attributes(row["attributes_json"])
    return ReadableSpan(
        name=row["name"] or "",
        context=context,
        parent=parent,
        attributes=attributes,
        start_time=row["start_time_unix_nano"],
        end_time=row["end_time_unix_nano"],
    )

  def get_all_spans_for_session(self, session_id: str) -> list[ReadableSpan]:
    """Returns all spans for a session (full trace trees).

    We first find trace_ids associated with the session, then return all spans
    for those trace_ids. This works even if some spans are missing session_id
    attributes (e.g. parent spans).
    """
    trace_rows = self._query(
        "SELECT DISTINCT trace_id FROM spans WHERE session_id = ?",
        (session_id,),
    )
    trace_ids = [r["trace_id"] for r in trace_rows if r["trace_id"]]
    if not trace_ids:
      return []

    placeholders = ",".join("?" for _ in trace_ids)
    rows = self._query(
        f"SELECT * FROM spans WHERE trace_id IN ({placeholders}) "
        "ORDER BY start_time_unix_nano",
        trace_ids,
    )
    return [self._row_to_readable_span(row) for row in rows]
