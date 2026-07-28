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

"""Tests for trace context propagated from request headers."""

from __future__ import annotations

import fastapi
from google.adk.telemetry._agent_engine import get_propagated_context
from google.adk.telemetry._agent_engine import TopSpanProcessor
from opentelemetry import baggage
from opentelemetry import context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest

_AE_TRACEPARENT_HEADER = 'Google-Agent-Engine-Traceparent'
_TRACEPARENT_HEADER = 'traceparent'
_SUPPORT_ID_ATTRIBUTE = 'supportID'
_SUPPORT_ID_VALUE = 'support-id-value'
_TOP_SPAN = 'invocation'
_CHILD_SPAN = 'child'

_TRACE_ID_HEX = '4bf92f3577b34da6a3ce929d0e0e4736'
_REMOTE_SPAN_ID_HEX = '00f067aa0ba902b7'
_WELL_FORMED_TRACEPARENT = f'00-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}-01'

# Values the trace context propagator refuses, either because they do not
# match the wire format or because the ids they carry are not usable.
_REJECTED_TRACEPARENT_VALUES = [
    'x',
    '00-abc-zz-01',
    '',
    '00',
    '-',
    f'00-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}',
    f'00-{"0" * 32}-{_REMOTE_SPAN_ID_HEX}-01',
    f'ff-{_TRACE_ID_HEX}-{_REMOTE_SPAN_ID_HEX}-01',
]


def _request(**headers: str) -> fastapi.Request:
  """Builds a minimal request carrying the given headers."""
  return fastapi.Request({
      'type': 'http',
      'method': 'POST',
      'path': '/',
      'headers': [
          (name.lower().encode(), value.encode())
          for name, value in headers.items()
      ],
  })


def _record_spans(ctx: context.Context) -> dict[str, ReadableSpan]:
  """Traces a child span under a top span with ctx attached, keyed by name."""
  exporter = InMemorySpanExporter()
  provider = TracerProvider(shutdown_on_exit=False)
  provider.add_span_processor(TopSpanProcessor())
  provider.add_span_processor(SimpleSpanProcessor(exporter))
  tracer = provider.get_tracer(__name__)

  token = context.attach(ctx)
  try:
    with tracer.start_as_current_span(_TOP_SPAN):
      with tracer.start_as_current_span(_CHILD_SPAN):
        pass
  finally:
    context.detach(token)

  return {span.name: span for span in exporter.get_finished_spans()}


@pytest.mark.parametrize('header_value', _REJECTED_TRACEPARENT_VALUES)
def test_rejected_header_still_produces_child_spans(header_value):
  """A caller-supplied header must not be able to break span creation."""
  spans = _record_spans(
      get_propagated_context(_request(**{_AE_TRACEPARENT_HEADER: header_value}))
  )

  assert set(spans) == {_TOP_SPAN, _CHILD_SPAN}


@pytest.mark.parametrize('header_value', _REJECTED_TRACEPARENT_VALUES)
def test_rejected_header_is_not_stored_in_baggage(header_value):
  """Only a header the propagator accepted is worth carrying in baggage."""
  ctx = get_propagated_context(
      _request(**{_AE_TRACEPARENT_HEADER: header_value})
  )

  assert _TRACEPARENT_HEADER not in baggage.get_all(context=ctx)


@pytest.mark.parametrize('baggage_value', _REJECTED_TRACEPARENT_VALUES)
def test_rejected_value_in_baggage_still_produces_child_spans(baggage_value):
  """The processor runs on every span, so it cannot trust baggage contents."""
  spans = _record_spans(baggage.set_baggage(_TRACEPARENT_HEADER, baggage_value))

  assert set(spans) == {_TOP_SPAN, _CHILD_SPAN}


def test_well_formed_header_is_stored_in_baggage():
  """The top span check reads the accepted header back out of baggage."""
  ctx = get_propagated_context(
      _request(**{_AE_TRACEPARENT_HEADER: _WELL_FORMED_TRACEPARENT})
  )

  assert (
      baggage.get_all(context=ctx)[_TRACEPARENT_HEADER]
      == _WELL_FORMED_TRACEPARENT
  )


def test_well_formed_header_marks_first_span_as_top_span():
  """This is the propagation the rejected-header guards must not break."""
  spans = _record_spans(
      get_propagated_context(
          _request(**{
              _AE_TRACEPARENT_HEADER: _WELL_FORMED_TRACEPARENT,
              _TRACEPARENT_HEADER: _SUPPORT_ID_VALUE,
          })
      )
  )

  assert spans[_TOP_SPAN].parent.span_id == int(_REMOTE_SPAN_ID_HEX, 16)
  assert spans[_TOP_SPAN].attributes[_SUPPORT_ID_ATTRIBUTE] == _SUPPORT_ID_VALUE
  assert _SUPPORT_ID_ATTRIBUTE not in spans[_CHILD_SPAN].attributes


def test_first_span_is_parentless_when_header_is_rejected():
  """Rejecting the header leaves the first span parentless, still the top."""
  spans = _record_spans(
      get_propagated_context(
          _request(**{
              _AE_TRACEPARENT_HEADER: 'x',
              _TRACEPARENT_HEADER: _SUPPORT_ID_VALUE,
          })
      )
  )

  assert spans[_TOP_SPAN].parent is None
  assert spans[_TOP_SPAN].attributes[_SUPPORT_ID_ATTRIBUTE] == _SUPPORT_ID_VALUE
  assert _SUPPORT_ID_ATTRIBUTE not in spans[_CHILD_SPAN].attributes
