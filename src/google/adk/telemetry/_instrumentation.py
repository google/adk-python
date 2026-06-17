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
import logging
import sys
import time
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
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool

logger = logging.getLogger("google_adk." + __name__)


def _get_elapsed_s(
    span: trace.Span | tracing.GenerateContentSpan | None,
    fallback_start: float,
) -> float:
  """Guarantees consistent time source for duration calculation.

  Note: This must be called with an ended span.

  Args:
    span (trace.Span | tracing.GenerateContentSpan | None): The ended span to
      extract duration from.
    fallback_start (float): Fallback start time in seconds (monotonic).

  Returns:
    float: Elapsed duration in seconds.
  """
  if span is None:
    return time.monotonic() - fallback_start

  span = span.span if hasattr(span, "span") else span
  start_ns = getattr(span, "start_time", None)
  end_ns = getattr(span, "end_time", None)

  if isinstance(start_ns, int) and isinstance(end_ns, int):
    return (end_ns - start_ns) / 1e9  # Convert ns to s

  # Fallback if span times are missing
  return time.monotonic() - fallback_start


@dataclasses.dataclass
class TelemetryContext:
  """Stores all telemetry related state."""

  otel_context: context_api.Context | None = None
  function_response_event: event_lib.Event | None = None
  error_type: str | None = None
  span: tracing.GenerateContentSpan | trace.Span | None = None
  _llm_responses: list[LlmResponse] = dataclasses.field(default_factory=list)

  @property
  def llm_responses(self) -> list[LlmResponse]:
    return self._llm_responses

  def record_llm_response(
      self, invocation_context: InvocationContext, response: LlmResponse
  ) -> None:
    self._llm_responses.append(response)
    tracing.trace_inference_result(invocation_context, self.span, response)


def _record_agent_metrics(
    agent_name: str,
    elapsed_s: float,
    user_content: object,
    events: object,
    caught_error: Exception | None,
) -> None:
  try:
    _metrics.record_agent_invocation_duration(
        agent_name,
        elapsed_s,
        caught_error,
    )
    _metrics.record_agent_request_size(agent_name, user_content)
    _metrics.record_agent_response_size(agent_name, events)
    _metrics.record_agent_workflow_steps(agent_name, events)
  except Exception:  # pylint: disable=broad-exception-caught
    logger.exception("Failed to record agent metrics for agent %s", agent_name)


@contextlib.asynccontextmanager
async def record_agent_invocation(
    ctx: InvocationContext, agent: BaseAgent
) -> AsyncIterator[TelemetryContext]:
  """Unified context manager for consolidated agent invocation telemetry."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  span_name = f"invoke_agent {agent.name}"
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
      span = s
      tracing.trace_agent_invocation(span, agent, ctx)
      tel_ctx = TelemetryContext(otel_context=context_api.get_current())
      yield tel_ctx
  except Exception as e:
    caught_error = e
    raise
  finally:
    _record_agent_metrics(
        agent.name,
        _get_elapsed_s(span, start_time),
        getattr(ctx, "user_content", None),
        getattr(getattr(ctx, "session", None), "events", []),
        caught_error,
    )


@contextlib.asynccontextmanager
async def record_tool_execution(
    tool: BaseTool,
    agent: BaseAgent,
    function_args: dict[str, object],
    invocation_context: InvocationContext | None = None,
) -> AsyncIterator[TelemetryContext]:
  """Unified context manager for consolidated tool execution telemetry."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  span_name = f"execute_tool {tool.name}"
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
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
            invocation_context=invocation_context,
            error_type=tel_ctx.error_type,
        )
  finally:
    try:
      _metrics.record_tool_execution_duration(
          tool_name=tool.name,
          agent_name=agent.name,
          elapsed_s=_get_elapsed_s(span, start_time),
          error=caught_error,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          "Failed to record tool execution duration for tool %s", tool.name
      )


@contextlib.asynccontextmanager
async def record_inference_telemetry(
    llm_request: LlmRequest,
    invocation_context: InvocationContext,
    model_response_event: event_lib.Event,
) -> AsyncIterator[TelemetryContext]:
  """Unified async context manager for consolidated inference metrics."""
  start_time = time.monotonic()
  tel_ctx: TelemetryContext = TelemetryContext()
  try:
    async with tracing.use_inference_span(
        llm_request,
        invocation_context,
        model_response_event,
    ) as gc_span:
      tel_ctx.span = gc_span
      yield tel_ctx
  finally:
    inference_error = sys.exc_info()[1]
    agent = invocation_context.agent
    elapsed_s = _get_elapsed_s(tel_ctx.span, start_time)
    try:
      if agent is not None and tracing._should_emit_native_telemetry(agent):
        _metrics.record_client_operation_duration(
            agent_name=agent.name,
            elapsed_s=elapsed_s,
            llm_request=llm_request,
            responses=tel_ctx.llm_responses,
            error=(
                inference_error
                if isinstance(inference_error, Exception)
                else None
            ),
        )
        _metrics.record_client_token_usage(
            agent_name=agent.name,
            llm_request=llm_request,
            responses=tel_ctx.llm_responses,
        )
    except Exception:  # pylint: disable=broad-exception-caught
      logger.exception(
          "Failed to record inference metrics for agent %s",
          agent.name if agent is not None else "<unknown>",
      )
