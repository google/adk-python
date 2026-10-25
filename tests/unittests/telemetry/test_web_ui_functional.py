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

"""With capture off, the Web UI is served what capture on emits."""

from __future__ import annotations

import dataclasses
from typing import Literal
from typing import TYPE_CHECKING

from google.adk.telemetry._experimental_semconv import COMPLETION_DETAILS_EVENT_NAME
from google.adk.telemetry._web_ui import ui_telemetry_from_otel
from google.adk.telemetry._web_ui import WebUiLogExporter
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
import pytest

from .functional._digests import LogDigest
from .functional._digests import normalized_attributes
from .functional._digests import SpanDigest
from .functional._digests import TelemetryDigest
from .functional._divergences import DivergenceGroup
from .functional._divergences import DivergenceId
from .functional._divergences import divergences
from .functional._recording import _record
from .functional_test_cases import ALL_CASES
from .functional_test_cases import MCP_CASE
from .functional_test_cases import MCP_HTTP_CASE
from .functional_test_goldens import load_divergences

if TYPE_CHECKING:
  from .functional._divergences import InferenceInstrumentation
  from .functional._recording import FunctionalTestCase

_CASES = pytest.mark.parametrize(
    "case",
    [*ALL_CASES, MCP_CASE, MCP_HTTP_CASE],
    ids=lambda c: f"{c.scenario}-{c.test_id}",
)


def _with_capture(
    case: FunctionalTestCase, capture_content: str
) -> FunctionalTestCase:
  return dataclasses.replace(
      case,
      semconv_opt_in="gen_ai_latest_experimental",
      capture_content=capture_content,
  )


async def _run_and_capture_logs(
    case: FunctionalTestCase,
    exporter: InMemoryLogRecordExporter,
    instrumentation: InferenceInstrumentation,
    capture_content: Literal["NO_CONTENT", "SPAN_AND_EVENT"],
) -> list[LogDigest]:
  """The completion details `exporter` gets, via the Web UI for its own."""
  recording = await _record(
      _with_capture(case, capture_content),
      instrumentation,
      log_exporter=exporter,
  )
  if not isinstance(exporter, WebUiLogExporter):
    return [
        LogDigest.from_log(log)
        for log in recording.logs
        if log.log_record.event_name == COMPLETION_DETAILS_EVENT_NAME
    ]
  return [
      LogDigest(
          event_name=log.event_name,
          body=log.body,
          attributes=normalized_attributes(log.attributes),
      )
      for span in ui_telemetry_from_otel(recording.spans, recording.logs)
      for log in span.logs
  ]


def _unexplained(
    native: list[LogDigest], otel: list[LogDigest]
) -> list[DivergenceId]:
  """Where ``otel`` differs from ``native`` with nothing on file about why."""
  explained = DivergenceGroup.by_id(load_divergences())
  found = divergences(*(
      TelemetryDigest(
          root_span=SpanDigest(name="", attributes={}, logs=logs),
          metric_points={},
      )
      for logs in (native, otel)
  ))
  return [
      divergence_id
      for divergence_id in found
      if divergence_id not in explained
      or not explained[divergence_id].explained
  ]


@_CASES
@pytest.mark.asyncio
async def test_web_ui_is_served_what_capture_on_emits(
    case: FunctionalTestCase,
) -> None:
  native_web_ui_logs = await _run_and_capture_logs(
      case, WebUiLogExporter(), "native", "NO_CONTENT"
  )
  native_with_content_logs = await _run_and_capture_logs(
      case, InMemoryLogRecordExporter(), "native", "SPAN_AND_EVENT"
  )
  otel_web_ui_logs = await _run_and_capture_logs(
      case, WebUiLogExporter(), "otel", "NO_CONTENT"
  )
  otel_with_content_logs = await _run_and_capture_logs(
      case, InMemoryLogRecordExporter(), "otel", "SPAN_AND_EVENT"
  )

  assert native_web_ui_logs == native_with_content_logs
  # The instrumentor records its own metadata; where it differs is on file.
  assert not _unexplained(native_web_ui_logs, otel_web_ui_logs)
  assert not _unexplained(native_web_ui_logs, otel_with_content_logs)
