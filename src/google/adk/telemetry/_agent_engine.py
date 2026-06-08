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

from typing import Mapping
from typing import Optional

import fastapi
from opentelemetry import baggage
from opentelemetry import context
from opentelemetry.sdk import trace
from opentelemetry.trace.propagation import tracecontext

_GOOGLE_AE_TRACEPARENT_HEADER = "Google-Agent-Engine-Traceparent"
_TRACEPARENT_BAGGAGE_KEY = "traceparent"
_GOOGLE_TRACEPARENT_HEADER = "traceparent"
_GOOGLE_TRACEPARENT_BAGGAGE_KEY = "google_traceparent"
_GOOGLE_TRACEPARENT_SUPPORT_ATTRIBUTE_KEY = "supportID"


def get_propagated_context(request: fastapi.Request) -> context.Context:
  """Propagates context from the request headers."""
  ctx = context.get_current()

  if _GOOGLE_TRACEPARENT_HEADER in request.headers:
    original_traceparent = request.headers[_GOOGLE_TRACEPARENT_HEADER]
    ctx = baggage.set_baggage(
        _GOOGLE_TRACEPARENT_BAGGAGE_KEY,
        original_traceparent,
        context=ctx,
    )

  if _GOOGLE_AE_TRACEPARENT_HEADER in request.headers:
    carrier = {"traceparent": request.headers[_GOOGLE_AE_TRACEPARENT_HEADER]}
    ctx = baggage.set_baggage(
        _TRACEPARENT_BAGGAGE_KEY,
        request.headers[_GOOGLE_AE_TRACEPARENT_HEADER],
        context=ctx,
    )
    ctx = tracecontext.TraceContextTextMapPropagator().extract(
        carrier=carrier, context=ctx
    )

  return ctx


class TopSpanProcessor(trace.SpanProcessor):
  """Top span processor."""

  def on_start(
      self, span: trace.Span, parent_context: Optional[context.Context] = None
  ):
    """Adds support ID to the top span."""
    baggage_items = baggage.get_all(context=parent_context)
    if self._is_top_span(span, baggage_items) and (
        baggage_trace_header := baggage_items.get(
            _GOOGLE_TRACEPARENT_BAGGAGE_KEY
        )
    ):
      span.set_attribute(
          _GOOGLE_TRACEPARENT_SUPPORT_ATTRIBUTE_KEY, baggage_trace_header
      )

  def on_end(self, span: trace.ReadableSpan) -> None:
    pass

  def shutdown(self) -> None:
    pass

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True

  def _is_top_span(
      self, span: trace.Span, baggage_items: Mapping[str, object]
  ) -> bool:
    """Returns true if the span is a top span.

    Args:
      span: The span to check.
      baggage_items: The baggage items that carry the context.

    Top span (e.g. "Invocation" span) is defined as the first span generated in
    trace generation.
    Top span could have an empty parent or the parent could be the span
    provided by traceparent propagation.
    """
    if span.parent is None or span.parent.span_id == 0:
      return True
    if _TRACEPARENT_BAGGAGE_KEY in baggage_items:
      parent_id_hex = str(baggage_items[_TRACEPARENT_BAGGAGE_KEY]).split("-")[2]
      parent_id_int = int(parent_id_hex, 16)
      if span.parent.span_id == parent_id_int:
        return True
    return False
