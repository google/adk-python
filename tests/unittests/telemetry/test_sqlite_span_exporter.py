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

import json
from pathlib import Path
import sqlite3

from google.adk.telemetry.sqlite_span_exporter import _apply_migrations
from google.adk.telemetry.sqlite_span_exporter import SqliteSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext
from opentelemetry.trace import SpanKind
from opentelemetry.trace import TraceFlags
from opentelemetry.trace import TraceState
import pytest


def _create_span(
    *,
    span_id: int = 0x00000000000ABC12,
    trace_id: int = 0x000000000000000000000000000DEF45,
    parent_span_id: int | None = None,
    name: str = "test_span",
    attributes: dict | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    start_time: int = 1000,
    end_time: int = 2000,
) -> ReadableSpan:
  """Helper to create ReadableSpan instances for testing."""
  context = SpanContext(
      trace_id=trace_id,
      span_id=span_id,
      is_remote=False,
      trace_flags=TraceFlags(TraceFlags.SAMPLED),
      trace_state=TraceState(),
  )

  parent = None
  if parent_span_id is not None:
    parent = SpanContext(
        trace_id=trace_id,
        span_id=parent_span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )

  return ReadableSpan(
      name=name,
      context=context,
      parent=parent,
      kind=kind,
      attributes=attributes or {},
      start_time=start_time,
      end_time=end_time,
  )


def test_export_single_span_returns_success(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  span = _create_span(
      name="test_operation",
      attributes={"gcp.vertex.agent.session_id": "session-123"},
  )

  result = exporter.export([span])

  assert result == SpanExportResult.SUCCESS
  assert db_path.exists()


def test_export_empty_list_returns_success(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  result = exporter.export([])

  assert result == SpanExportResult.SUCCESS


def test_get_all_spans_for_session_returns_matching_spans(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  span1 = _create_span(
      span_id=0x111,
      trace_id=0xAAA111,  # Different trace for session-123
      attributes={"gcp.vertex.agent.session_id": "session-123"},
      name="span1",
  )
  span2 = _create_span(
      span_id=0x222,
      trace_id=0xAAA222,  # Different trace for session-123
      attributes={"gcp.vertex.agent.session_id": "session-123"},
      name="span2",
  )
  span3 = _create_span(
      span_id=0x333,
      trace_id=0xBBB333,  # Different trace for session-456
      attributes={"gcp.vertex.agent.session_id": "session-456"},
      name="span3",
  )

  exporter.export([span1, span2, span3])

  result = exporter.get_all_spans_for_session("session-123")

  assert len(result) == 2
  names = [span.name for span in result]
  assert "span1" in names
  assert "span2" in names
  assert "span3" not in names


def test_get_all_spans_for_session_includes_sibling_spans_without_session_id(
    tmp_path,
):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Parent span without session_id (e.g., invocation span)
  parent_span = _create_span(
      span_id=0x100,
      trace_id=0xAAA,
      name="invocation",
      attributes={},  # No session_id
  )

  # Child span with session_id
  child_span = _create_span(
      span_id=0x200,
      trace_id=0xAAA,  # Same trace
      parent_span_id=0x100,
      name="call_llm",
      attributes={"gcp.vertex.agent.session_id": "session-789"},
  )

  # Sibling span without session_id (should be included)
  sibling_span = _create_span(
      span_id=0x300,
      trace_id=0xAAA,  # Same trace
      parent_span_id=0x100,
      name="tool_call",
      attributes={},  # No session_id
  )

  # Unrelated span with different trace_id (should not be included)
  unrelated_span = _create_span(
      span_id=0x400,
      trace_id=0xBBB,  # Different trace
      name="unrelated",
      attributes={},
  )

  exporter.export([parent_span, child_span, sibling_span, unrelated_span])

  result = exporter.get_all_spans_for_session("session-789")

  assert len(result) == 3
  names = [span.name for span in result]
  assert "invocation" in names
  assert "call_llm" in names
  assert "tool_call" in names
  assert "unrelated" not in names


def test_get_all_spans_for_unknown_session_returns_empty_list(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  span = _create_span(
      attributes={"gcp.vertex.agent.session_id": "session-123"},
  )
  exporter.export([span])

  result = exporter.get_all_spans_for_session("unknown-session")

  assert result == []


def test_round_trip_preserves_span_attributes(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  original_attributes = {
      "gcp.vertex.agent.session_id": "session-123",
      "gcp.vertex.agent.invocation_id": "invocation-456",
      "gen_ai.conversation.id": "conv-789",
      "custom.attribute": "test_value",
      "numeric.value": 42,
      "boolean.value": True,
      "list.value": [1, 2, 3],
      "dict.value": {"nested": "data"},
  }

  original_span = _create_span(
      span_id=0x12345678,
      trace_id=0xABCDEF123456789,
      name="test_operation",
      attributes=original_attributes,
      start_time=1000000,
      end_time=2000000,
  )

  exporter.export([original_span])

  retrieved_spans = exporter.get_all_spans_for_session("session-123")

  assert len(retrieved_spans) == 1
  retrieved = retrieved_spans[0]

  assert retrieved.name == "test_operation"
  assert retrieved.context.span_id == 0x12345678
  assert retrieved.context.trace_id == 0xABCDEF123456789
  assert retrieved.start_time == 1000000
  assert retrieved.end_time == 2000000
  assert retrieved.attributes == original_attributes


def test_span_kinds_survive_exporter_restart(tmp_path: Path) -> None:
  db_path = tmp_path / "test.db"
  kinds = [
      SpanKind.CLIENT,
      SpanKind.SERVER,
      SpanKind.PRODUCER,
      SpanKind.CONSUMER,
  ]
  exporter = SqliteSpanExporter(db_path=str(db_path))
  spans = [
      _create_span(
          span_id=index + 1,
          kind=kind,
          attributes={"gen_ai.conversation.id": "session-kind"},
          start_time=index,
      )
      for index, kind in enumerate(kinds)
  ]
  assert exporter.export(spans) == SpanExportResult.SUCCESS
  exporter.shutdown()

  reopened = SqliteSpanExporter(db_path=str(db_path))
  try:
    restored = reopened.get_all_spans_for_session("session-kind")
    assert [(span.context.span_id, span.kind) for span in restored] == [
        (index + 1, kind) for index, kind in enumerate(kinds)
    ]
  finally:
    reopened.shutdown()


def test_legacy_database_retains_spans_and_accepts_new_kinds(
    tmp_path: Path,
) -> None:
  db_path = tmp_path / "legacy.db"
  with sqlite3.connect(db_path) as conn:
    conn.execute("""
        CREATE TABLE spans (
          span_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL,
          parent_span_id TEXT, name TEXT NOT NULL,
          start_time_unix_nano INTEGER, end_time_unix_nano INTEGER,
          session_id TEXT, invocation_id TEXT, attributes_json TEXT
        )
    """)
    conn.execute(
        "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "0000000000000001",
            "000000000000000000000000000def45",
            None,
            "legacy_operation",
            1000,
            2000,
            "session-legacy",
            None,
            '{"gen_ai.conversation.id": "session-legacy"}',
        ),
    )

  exporter = SqliteSpanExporter(db_path=str(db_path))
  try:
    assert (
        exporter.export([
            _create_span(
                span_id=2,
                name="new_operation",
                kind=SpanKind.CLIENT,
                attributes={"gen_ai.conversation.id": "session-legacy"},
                start_time=3000,
            )
        ])
        == SpanExportResult.SUCCESS
    )
    restored = exporter.get_all_spans_for_session("session-legacy")
    assert [(span.name, span.kind) for span in restored] == [
        ("legacy_operation", SpanKind.INTERNAL),
        ("new_operation", SpanKind.CLIENT),
    ]
  finally:
    exporter.shutdown()


def test_database_from_newer_schema_version_is_not_downgraded(
    tmp_path: Path,
) -> None:
  db_path = tmp_path / "newer.db"
  SqliteSpanExporter(db_path=str(db_path)).shutdown()
  with sqlite3.connect(db_path) as conn:
    conn.execute("PRAGMA user_version = 99")

  SqliteSpanExporter(db_path=str(db_path)).shutdown()

  with sqlite3.connect(db_path) as conn:
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 99


def _column_names(conn: sqlite3.Connection, table: str) -> list[str]:
  return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_failing_migration_rolls_back_and_can_be_retried(
    tmp_path: Path,
) -> None:
  conn = sqlite3.connect(tmp_path / "migrations.db")
  try:
    create = ("CREATE TABLE t (a TEXT)",)
    _apply_migrations(conn, [create])

    with pytest.raises(sqlite3.OperationalError):
      _apply_migrations(
          conn,
          [
              create,
              ("ALTER TABLE t ADD COLUMN b TEXT", "SELECT * FROM missing"),
          ],
      )

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert _column_names(conn, "t") == ["a"]

    _apply_migrations(conn, [create, ("ALTER TABLE t ADD COLUMN b TEXT",)])

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert _column_names(conn, "t") == ["a", "b"]
  finally:
    conn.close()


def test_apply_migrations_skips_entries_already_applied(tmp_path: Path) -> None:
  conn = sqlite3.connect(tmp_path / "migrations.db")
  try:
    conn.execute("CREATE TABLE t (a TEXT)")
    conn.execute("PRAGMA user_version = 1")

    # Entry 0 would fail if re-run because the table already exists.
    _apply_migrations(
        conn,
        [("CREATE TABLE t (a TEXT)",), ("ALTER TABLE t ADD COLUMN b TEXT",)],
    )

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert _column_names(conn, "t") == ["a", "b"]
  finally:
    conn.close()


def test_current_database_opens_while_another_writer_holds_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  db_path = tmp_path / "locked.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))
  assert (
      exporter.export(
          [_create_span(attributes={"gen_ai.conversation.id": "session-lock"})]
      )
      == SpanExportResult.SUCCESS
  )
  exporter.shutdown()
  monkeypatch.setattr(
      "google.adk.telemetry.sqlite_span_exporter._DEFAULT_TIMEOUT_SECONDS",
      0.1,
  )

  writer = sqlite3.connect(db_path, isolation_level=None)
  try:
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("UPDATE spans SET name = 'uncommitted'")

    reopened = SqliteSpanExporter(db_path=str(db_path))
    try:
      restored = reopened.get_all_spans_for_session("session-lock")
      assert [span.name for span in restored] == ["test_span"]
    finally:
      reopened.shutdown()
  finally:
    writer.rollback()
    writer.close()


def test_spans_with_parent_context_exported_correctly(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  parent_span = _create_span(
      span_id=0xAAA,
      trace_id=0x123,
      name="parent",
      attributes={"gcp.vertex.agent.session_id": "session-001"},
  )

  child_span = _create_span(
      span_id=0xBBB,
      trace_id=0x123,
      parent_span_id=0xAAA,
      name="child",
      attributes={"gcp.vertex.agent.session_id": "session-001"},
  )

  exporter.export([parent_span, child_span])

  retrieved_spans = exporter.get_all_spans_for_session("session-001")

  assert len(retrieved_spans) == 2

  # Find child span in results
  child = next(s for s in retrieved_spans if s.name == "child")
  assert child.parent is not None
  assert child.parent.span_id == 0xAAA
  assert child.parent.trace_id == 0x123

  # Find parent span in results
  parent = next(s for s in retrieved_spans if s.name == "parent")
  assert parent.parent is None


def test_shutdown_closes_connection(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Create a span to ensure connection is open
  span = _create_span()
  exporter.export([span])

  # Verify connection exists
  assert exporter._conn is not None

  exporter.shutdown()

  # Verify connection is closed
  assert exporter._conn is None


def test_force_flush_returns_true(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  result = exporter.force_flush()

  assert result is True

  # Also test with timeout parameter
  result_with_timeout = exporter.force_flush(timeout_millis=5000)
  assert result_with_timeout is True


def test_export_handles_spans_with_none_attributes(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  span = _create_span(attributes=None)

  result = exporter.export([span])

  assert result == SpanExportResult.SUCCESS

  # Verify the span was stored correctly
  rows = exporter._query("SELECT attributes_json FROM spans", [])
  assert len(rows) == 1
  attributes_json = rows[0]["attributes_json"]
  assert json.loads(attributes_json) == {}


def test_duplicate_span_id_replaces_previous_row(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Export first version of span
  span1 = _create_span(
      span_id=0x999,
      name="first_version",
      attributes={"version": 1, "gcp.vertex.agent.session_id": "session-dup"},
  )
  exporter.export([span1])

  # Export second version with same span_id
  span2 = _create_span(
      span_id=0x999,
      name="second_version",
      attributes={"version": 2, "gcp.vertex.agent.session_id": "session-dup"},
  )
  exporter.export([span2])

  # Verify only one row exists with updated data
  retrieved_spans = exporter.get_all_spans_for_session("session-dup")
  assert len(retrieved_spans) == 1
  assert retrieved_spans[0].name == "second_version"
  assert retrieved_spans[0].attributes["version"] == 2


def test_non_serializable_attributes_use_fallback(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Create a non-serializable object
  class NonSerializable:
    pass

  attributes = {
      "gcp.vertex.agent.session_id": "session-nonser",
      "normal_attr": "value",
      "non_serializable": NonSerializable(),
  }

  span = _create_span(attributes=attributes)

  result = exporter.export([span])

  assert result == SpanExportResult.SUCCESS

  # Verify the span was stored and non-serializable attribute has fallback
  retrieved_spans = exporter.get_all_spans_for_session("session-nonser")
  assert len(retrieved_spans) == 1
  assert retrieved_spans[0].attributes["normal_attr"] == "value"
  assert (
      retrieved_spans[0].attributes["non_serializable"] == "<not serializable>"
  )


def test_export_multiple_spans_in_batch(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  spans = [
      _create_span(
          span_id=i,
          name=f"span_{i}",
          attributes={"gcp.vertex.agent.session_id": "batch-session"},
      )
      for i in range(10)
  ]

  result = exporter.export(spans)

  assert result == SpanExportResult.SUCCESS

  retrieved_spans = exporter.get_all_spans_for_session("batch-session")
  assert len(retrieved_spans) == 10
  names = {span.name for span in retrieved_spans}
  assert names == {f"span_{i}" for i in range(10)}


def test_export_with_alternative_session_id_attribute(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Test using gen_ai.conversation.id as fallback for session_id
  span = _create_span(
      attributes={"gen_ai.conversation.id": "conv-session-123"},
  )

  exporter.export([span])

  # Should be queryable by the conversation id
  result = exporter.get_all_spans_for_session("conv-session-123")

  assert len(result) == 1
  assert result[0].attributes["gen_ai.conversation.id"] == "conv-session-123"


def test_deserialize_handles_invalid_json(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Manually insert a row with invalid JSON
  conn = exporter._get_connection()
  conn.execute(
      "INSERT INTO spans (span_id, trace_id, name, attributes_json) VALUES (?,"
      " ?, ?, ?)",
      ("abc123", "def456", "test", "not valid json"),
  )
  conn.commit()

  # Try to retrieve the span - should not raise, but attributes should be empty
  rows = exporter._query("SELECT * FROM spans", [])
  span = exporter._row_to_readable_span(rows[0])

  assert span.name == "test"
  assert span.attributes == {}


def test_get_spans_ordered_by_start_time(tmp_path):
  db_path = tmp_path / "test.db"
  exporter = SqliteSpanExporter(db_path=str(db_path))

  # Create spans with different start times
  spans = [
      _create_span(
          span_id=0x300,
          start_time=3000,
          attributes={"gcp.vertex.agent.session_id": "session-order"},
      ),
      _create_span(
          span_id=0x100,
          start_time=1000,
          attributes={"gcp.vertex.agent.session_id": "session-order"},
      ),
      _create_span(
          span_id=0x200,
          start_time=2000,
          attributes={"gcp.vertex.agent.session_id": "session-order"},
      ),
  ]

  exporter.export(spans)

  result = exporter.get_all_spans_for_session("session-order")

  # Verify spans are ordered by start_time
  assert len(result) == 3
  assert result[0].context.span_id == 0x100
  assert result[1].context.span_id == 0x200
  assert result[2].context.span_id == 0x300
