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

"""How OTel spans and logs are translated into what ADK Web is served."""

from __future__ import annotations

import base64
import json

from google.adk.telemetry._web_ui import ui_telemetry_from_otel
from google.adk.telemetry._web_ui import WebUiLog
from google.adk.telemetry._web_ui import WebUiLogExporter
from google.adk.telemetry._web_ui import WebUiSpan
from opentelemetry import trace
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import NonRecordingSpan
from opentelemetry.trace import SpanContext
from opentelemetry.util.types import Attributes

_TRACE_ID = 0xDEF45
_SPAN_ID = 0xABC1
_CHILD_SPAN_ID = 0xABC2
_OTHER_SESSION_SPAN_ID = 0xABC3
_COMPLETION_DETAILS = "gen_ai.client.inference.operation.details"


def _span_context(span_id: int) -> SpanContext:
  return SpanContext(trace_id=_TRACE_ID, span_id=span_id, is_remote=False)


def _span(span_id: int, parent_span_id: int | None) -> ReadableSpan:
  return ReadableSpan(
      name=f"span-{span_id}",
      context=_span_context(span_id),
      parent=(
          None if parent_span_id is None else _span_context(parent_span_id)
      ),
      attributes={"gen_ai.conversation.id": "s1"},
      start_time=1000,
      end_time=2000,
  )


def _log(
    span_id: int, event_name: str, body: object, attributes: Attributes
) -> ReadableLogRecord:
  return ReadableLogRecord(
      log_record=LogRecord(
          context=trace.set_span_in_context(
              NonRecordingSpan(_span_context(span_id))
          ),
          event_name=event_name,
          body=body,
          attributes=attributes,
      ),
      resource=Resource.create({}),
      instrumentation_scope=InstrumentationScope("test"),
  )


def test_each_span_is_served_with_its_own_logs() -> None:
  served = ui_telemetry_from_otel(
      [_span(_SPAN_ID, None), _span(_CHILD_SPAN_ID, _SPAN_ID)],
      [
          _log(_SPAN_ID, _COMPLETION_DETAILS, None, {"a": 1}),
          _log(_SPAN_ID, _COMPLETION_DETAILS, "second", None),
          _log(_OTHER_SESSION_SPAN_ID, _COMPLETION_DETAILS, "other", None),
      ],
  )

  assert served == [
      WebUiSpan(
          name=f"span-{_SPAN_ID}",
          start_time=1000,
          end_time=2000,
          trace_id=_TRACE_ID,
          span_id=_SPAN_ID,
          parent_span_id=None,
          attributes={"gen_ai.conversation.id": "s1"},
          logs=(
              WebUiLog(
                  event_name=_COMPLETION_DETAILS, body=None, attributes={"a": 1}
              ),
              WebUiLog(
                  event_name=_COMPLETION_DETAILS, body="second", attributes={}
              ),
          ),
      ),
      WebUiSpan(
          name=f"span-{_CHILD_SPAN_ID}",
          start_time=1000,
          end_time=2000,
          trace_id=_TRACE_ID,
          span_id=_CHILD_SPAN_ID,
          parent_span_id=_SPAN_ID,
          attributes={"gen_ai.conversation.id": "s1"},
          logs=(),
      ),
  ]
  # ADK Web rejects a null `logs`.
  assert json.loads(served[1].model_dump_json())["logs"] == []


def test_binary_content_is_served_as_standard_base64() -> None:
  image = b"\x89PNG\r\n\xff\xfe"
  (served,) = ui_telemetry_from_otel(
      [_span(_SPAN_ID, None)],
      [_log(_SPAN_ID, _COMPLETION_DETAILS, {"parts": [{"data": image}]}, None)],
  )

  encoded = base64.b64encode(image).decode()
  assert json.loads(served.model_dump_json())["logs"][0]["body"] == {
      "parts": [{"data": encoded}]
  }


def test_exporter_keeps_only_completion_details() -> None:
  exporter = WebUiLogExporter()
  kept = _log(_SPAN_ID, _COMPLETION_DETAILS, None, None)
  dropped = _log(_SPAN_ID, "gen_ai.user.message", None, None)

  exporter.export([kept, dropped])

  assert exporter.get_finished_logs() == (kept,)


def test_exporter_keeps_only_the_latest_1000() -> None:
  exporter = WebUiLogExporter()
  logs = [_log(_SPAN_ID, _COMPLETION_DETAILS, i, None) for i in range(1001)]

  exporter.export(logs)

  assert exporter.get_finished_logs() == tuple(logs[1:])
