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

# pylint: disable=protected-access

import time
from unittest import mock

from google.adk.telemetry import _instrumentation
from opentelemetry import trace
import pytest


def test_get_elapsed_ms_span_none():
  """Tests fallback when span is None."""
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _instrumentation._get_elapsed_ms(None, start_time)
  assert elapsed == 2000.0  # (12 - 10) * 1000


def test_get_elapsed_ms_span_valid():
  """Tests duration calculation with valid span times."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000  # 1s in ns
  mock_span.end_time = 2000000000  # 2s in ns
  elapsed = _instrumentation._get_elapsed_ms(mock_span, time.monotonic())
  assert elapsed == 1000.0  # (2 - 1) * 1000 ms


def test_get_elapsed_ms_span_missing_start():
  """Tests fallback when start_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  del mock_span.start_time
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _instrumentation._get_elapsed_ms(mock_span, start_time)
  assert elapsed == 2000.0


def test_get_elapsed_ms_span_missing_end():
  """Tests fallback when end_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  del mock_span.end_time
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _instrumentation._get_elapsed_ms(mock_span, start_time)
  assert elapsed == 2000.0


def test_get_elapsed_ms_span_non_int_start():
  """Tests fallback when start_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000.0
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _instrumentation._get_elapsed_ms(mock_span, start_time)
  assert elapsed == 2000.0


def test_get_elapsed_ms_span_non_int_end():
  """Tests fallback when end_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  mock_span.end_time = 2000000000.0
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _instrumentation._get_elapsed_ms(mock_span, start_time)
  assert elapsed == 2000.0


@pytest.mark.asyncio
async def test_record_agent_invocation_tolerates_minimal_context():
  """Tolerates context-likes that lack user_content or session.

  Test doubles, partial migrations, and external embedders can pass an
  InvocationContext-like object without `user_content` or with a `session`
  that has no `events` attribute. The telemetry path must not raise
  AttributeError on the metrics call in those cases.
  """
  agent = mock.MagicMock()
  agent.name = "test_agent"
  # Bare object without `user_content` and without `session`.
  bare_ctx = object()

  with (
      mock.patch.object(
          _instrumentation, "_record_agent_metrics"
      ) as mock_record,
      mock.patch.object(_instrumentation, "tracing") as mock_tracing,
  ):
    mock_tracing.tracer.start_as_current_span.return_value.__enter__.return_value = mock.MagicMock(
        spec=trace.Span
    )
    async with _instrumentation.record_agent_invocation(bare_ctx, agent):
      pass

  mock_record.assert_called_once()
  call_args = mock_record.call_args
  # positional: (agent_name, elapsed_ms, user_content, events, caught_error)
  assert call_args.args[0] == "test_agent"
  assert call_args.args[2] is None  # user_content default
  assert call_args.args[3] == []  # events default
