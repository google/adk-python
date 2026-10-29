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

"""The ADK Web trace view: spans and their logs, elided content restored."""

from __future__ import annotations

import base64
import collections
from collections.abc import Mapping
import threading
from typing import TYPE_CHECKING

from opentelemetry.sdk._logs.export import LogRecordExporter
from opentelemetry.sdk._logs.export import LogRecordExportResult
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID
from opentelemetry.util import types as otel_types
from pydantic import BaseModel
from typing_extensions import override
from typing_extensions import TypeAliasType

from ._experimental_semconv import COMPLETION_DETAILS_EVENT_NAME
from ._experimental_semconv import set_operation_details_attributes_from_request
from ._experimental_semconv import set_operation_details_attributes_from_response
from ._finish_reason import is_reported_finish_reason
from ._instrumentation import inference_scope

if TYPE_CHECKING:
  from collections.abc import Iterable
  from collections.abc import Sequence

  from opentelemetry._logs import LogRecord
  from opentelemetry.sdk._logs import ReadableLogRecord
  from opentelemetry.sdk.trace import ReadableSpan

# Wraps OTel's recursive alias; pydantic resolves its self-reference by name.
AnyValue = TypeAliasType("AnyValue", otel_types.AnyValue)

_MAX_LOGS = 1000


class WebUiLogExporter(LogRecordExporter):
  """Keeps the latest completion-details records the trace view is served."""

  def __init__(self) -> None:
    # Each record pins its call's LlmRequest, so only the latest are kept.
    self._logs: collections.deque[ReadableLogRecord] = collections.deque(
        maxlen=_MAX_LOGS
    )
    self._lock = threading.Lock()

  def get_finished_logs(self) -> tuple[ReadableLogRecord, ...]:
    with self._lock:
      return tuple(self._logs)

  @override
  def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
    with self._lock:
      self._logs.extend(
          log
          for log in batch
          if log.log_record.event_name == COMPLETION_DETAILS_EVENT_NAME
      )
    return LogRecordExportResult.SUCCESS

  @override
  def shutdown(self) -> None:
    pass

  # Abstract in newer opentelemetry-sdk releases, absent from older ones.
  def force_flush(self, timeout_millis: int = 10000) -> bool:
    return True


class WebUiLog(BaseModel):
  """One log record recorded under a span."""

  event_name: str
  body: AnyValue
  attributes: dict[str, AnyValue]


class WebUiSpan(BaseModel):
  """One span, with the logs recorded under it."""

  name: str
  start_time: int | None
  end_time: int | None
  trace_id: int
  span_id: int
  parent_span_id: int | None
  attributes: dict[str, AnyValue]
  # ADK Web rejects a null here.
  logs: tuple[WebUiLog, ...] = ()


def ui_telemetry_from_otel(
    spans: Iterable[ReadableSpan],
    logs: Iterable[ReadableLogRecord],
) -> list[WebUiSpan]:
  """Maps OTel spans, and the logs recorded under them, to ADK Web's schema."""
  spans = list(spans)
  span_ids = {span.context.span_id for span in spans}

  logs_by_span_id: dict[int, list[WebUiLog]] = {}
  for log in logs:
    log_record = log.log_record
    # The exporter holds every session's records; rebuild only this one's.
    if log_record.span_id not in span_ids:
      continue
    logs_by_span_id.setdefault(log_record.span_id, []).append(
        WebUiLog(
            event_name=log_record.event_name or "",
            body=_safe_serialize(log_record.body),
            attributes={
                key: _safe_serialize(value)
                for key, value in _restored_attributes(log_record).items()
            },
        )
    )

  return [
      WebUiSpan(
          name=span.name,
          start_time=span.start_time,
          end_time=span.end_time,
          trace_id=span.context.trace_id,
          span_id=span.context.span_id,
          parent_span_id=span.parent.span_id if span.parent else None,
          attributes=dict(span.attributes or {}),
          logs=tuple(logs_by_span_id.get(span.context.span_id, ())),
      )
      for span in spans
  ]


def _restored_attributes(log_record: LogRecord) -> dict[str, AnyValue]:
  """Returns a copy of the record's attributes with the elided content put back.

  The record is shared with every other log processor, so it is not mutated.
  """
  attributes: dict[str, AnyValue] = dict(log_record.attributes or {})
  scope = inference_scope(log_record.context)
  if scope is None:
    return attributes
  content: dict[str, AnyValue] = {}
  set_operation_details_attributes_from_request(content, scope.llm_request)
  for llm_response in scope.llm_responses:
    set_operation_details_attributes_from_response(llm_response, content, {})
    # The event stops at the finish reason, as InferenceScope does.
    if is_reported_finish_reason(llm_response.finish_reason):
      break
  # Emitted only with content capture on, like the content itself.
  content[USER_ID] = scope.user_id
  return attributes | content


def _safe_serialize(obj: AnyValue) -> AnyValue:
  """Base64-encodes bytes, recursively.

  Standard alphabet: pydantic's ``ser_json_bytes`` is URL-safe, which the
  browser's ``atob`` rejects.
  """
  if isinstance(obj, bytes):
    return base64.b64encode(obj).decode("utf-8")
  if isinstance(obj, Mapping):
    return {key: _safe_serialize(value) for key, value in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_safe_serialize(item) for item in obj]
  return obj
