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

import contextlib
import dataclasses
import time
from typing import Any
from typing import AsyncIterator
from typing import TYPE_CHECKING

from opentelemetry import trace
import opentelemetry.context as context_api

from . import _metrics
from . import tracing
from ..events import event as event_lib

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext


def _get_elapsed_ms(span: trace.Span | None, fallback_start: float) -> float:
  """Guarantees consistent time source for duration calculation.

  Note: This must be called with an ended span.

  Args:
    span (trace.Span | None): The ended span to extract duration from.
    fallback_start (float): Fallback start time in seconds (monotonic).

  Returns:
    float: Elapsed duration in milliseconds.
  """
  if span is None:
    return (time.monotonic() - fallback_start) * 1000

  start_ns = getattr(span, "start_time", None)
  end_ns = getattr(span, "end_time", None)

  if isinstance(start_ns, int) and isinstance(end_ns, int):
    return (end_ns - start_ns) / 1e6  # Convert ns to ms

  # Fallback if span times are missing
  return (time.monotonic() - fallback_start) * 1000


@dataclasses.dataclass
class TelemetryContext:
  """Stores all telemetry related state."""

  otel_context: context_api.Context
  function_response_event: event_lib.Event | None = None


@contextlib.asynccontextmanager
async def record_agent_invocation(
    ctx: InvocationContext, agent: BaseAgent
) -> AsyncIterator[TelemetryContext]:
  """Unified context manager for consolidated metrics and tracing."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  try:
    with tracing.tracer.start_as_current_span(
        f"invoke_agent {agent.name}"
    ) as s:
      span = s
      tracing.trace_agent_invocation(span, agent, ctx)
      _metrics.record_agent_request_size(agent.name, ctx.user_content)
      tel_ctx = TelemetryContext(otel_context=context_api.get_current())
      yield tel_ctx

  except Exception as e:
    caught_error = e
    raise
  finally:
    elapsed_ms = _get_elapsed_ms(span, start_time)
    _metrics.record_agent_invocation_duration(
        agent.name,
        elapsed_ms,
        ctx.user_content,
        ctx.session.events,
        caught_error,
    )
    _metrics.record_agent_response_size(agent.name, ctx.session.events)
    _metrics.record_agent_workflow_steps(agent.name, len(ctx.session.events))


@contextlib.asynccontextmanager
async def record_tool_execution(
    tool: BaseTool,
    agent: BaseAgent,
    invocation_context: InvocationContext,
    function_args: dict[str, Any],
) -> AsyncIterator[TelemetryContext]:
  """Unified context manager for consolidated tool execution telemetry."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  tel_ctx: TelemetryContext | None = None
  try:
    with tracing.tracer.start_as_current_span(f"execute_tool {tool.name}") as s:
      span = s
      tel_ctx = TelemetryContext(otel_context=context_api.get_current())
      try:
        yield tel_ctx
      except Exception as e:
        caught_error = e
        raise
      finally:
        response_event = (
            tel_ctx.function_response_event if caught_error is None else None
        )
        tracing.trace_tool_call(
            tool=tool,
            args=function_args,
            function_response_event=response_event,
            error=caught_error,
        )
  finally:
    elapsed_ms = _get_elapsed_ms(span, start_time)
    result_event = (
        tel_ctx.function_response_event if tel_ctx is not None else None
    )
    output_content = (
        result_event.content
        if isinstance(result_event, event_lib.Event)
        else None
    )
    _metrics.record_tool_execution_duration(
        tool_name=tool.name,
        agent_name=agent.name,
        elapsed_ms=elapsed_ms,
        input_content=invocation_context.user_content,
        output_content=output_content,
        error=caught_error,
    )
