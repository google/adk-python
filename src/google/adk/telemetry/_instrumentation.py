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
from typing import Iterator
from typing import TYPE_CHECKING

from opentelemetry import trace
import opentelemetry.context as context_api

from . import _metrics
from . import tracing
from ._schema_version import resolve_schema_version
from ._schema_version import SCHEMA_VERSION_SEMCONV_ALIGNED

# pylint: disable=g-import-not-at-top
if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..events import event as event_lib
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..tools.base_tool import BaseTool
  from ..workflow._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)

_INVOKE_AGENT_TELEMETRY_KEY = context_api.create_key("invoke_agent_telemetry")


@contextlib.contextmanager
def record_invocation(
    entrypoint_node: BaseNode | None,
    conversation_id: str,
) -> Iterator[None]:
  """Top-level invocation span for a runner invocation.

  Schema v1 emits the legacy ``invocation`` span. Schema v2 replaces it with an
  entrypoint ``invoke_workflow {entrypoint}`` span (entrypoint = root agent or
  root node name), which omits the ``gen_ai.workflow.nested`` attribute, and a
  ``gen_ai.invoke_workflow.duration`` metric -- unless the entrypoint is itself
  a workflow, in which case its own node span is the entrypoint
  ``invoke_workflow`` span and we avoid double-emitting it here.

  Args:
    entrypoint_node: The runner's root agent/node.
    conversation_id: Session/conversation id (stamped on the v2 span).

  Yields:
    Nothing; the span (if any) is active for the duration of the block.
  """
  if resolve_schema_version() < SCHEMA_VERSION_SEMCONV_ALIGNED:
    with tracing.tracer.start_as_current_span("invocation"):
      yield
    return

  from . import node_tracing
  from ..workflow._workflow import Workflow

  if isinstance(entrypoint_node, Workflow):
    # The workflow's own node span is the entrypoint `invoke_workflow` span.
    yield
    return

  entrypoint_name = entrypoint_node.name if entrypoint_node else ""
  with node_tracing._use_invoke_workflow_span(entrypoint_name, conversation_id):
    yield


@dataclasses.dataclass
class TelemetryContext:
  """Stores all telemetry related state."""

  otel_context: context_api.Context | None = None
  function_response_event: event_lib.Event | None = None
  error_type: str | None = None
  span: tracing.GenerateContentSpan | trace.Span | None = None
  _llm_responses: list[LlmResponse] = dataclasses.field(default_factory=list)
  _inference_call_count: int = 0
  _tool_call_count: int = 0

  @property
  def inference_call_count(self) -> int:
    return self._inference_call_count

  def increment_inference_calls(self) -> None:
    self._inference_call_count += 1

  @property
  def tool_call_count(self) -> int:
    return self._tool_call_count

  def increment_tool_calls(self) -> None:
    self._tool_call_count += 1

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
    caught_error: Exception | None,
) -> None:
  try:
    _metrics.record_agent_invocation_duration(
        agent_name,
        elapsed_s,
        caught_error,
    )
  except Exception:  # pylint: disable=broad-exception-caught
    logger.exception("Failed to record agent metrics for agent %s", agent_name)


def _flush_invoke_agent_metrics(
    tel_ctx: TelemetryContext, agent_name: str
) -> None:
  """Flushes this span's accumulated inference/tool-call metrics."""
  _metrics.record_invoke_agent_inference_calls(
      agent_name, tel_ctx.inference_call_count
  )
  _metrics.record_invoke_agent_tool_calls(agent_name, tel_ctx.tool_call_count)


def _active_invoke_agent_tel_ctx() -> TelemetryContext | None:
  """Returns the TelemetryContext of the active invoke_agent span."""
  value = context_api.get_value(_INVOKE_AGENT_TELEMETRY_KEY)
  return value if isinstance(value, TelemetryContext) else None


def _accumulate_invoke_agent_tool_call() -> None:
  """Counts one tool call against the active invoke_agent span."""
  span_tel_ctx = _active_invoke_agent_tel_ctx()
  if span_tel_ctx is not None:
    span_tel_ctx.increment_tool_calls()


def _accumulate_invoke_agent_inference_call() -> None:
  """Counts one model call against the active invoke_agent span."""
  span_tel_ctx = _active_invoke_agent_tel_ctx()
  if span_tel_ctx is not None:
    span_tel_ctx.increment_inference_calls()


@contextlib.asynccontextmanager
async def record_agent_invocation(
    ctx: InvocationContext, agent: BaseAgent
) -> AsyncIterator[TelemetryContext]:
  """Unified context manager for consolidated agent invocation telemetry."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  span_name = f"invoke_agent {agent.name}"
  tel_ctx = TelemetryContext()
  token = context_api.attach(
      context_api.set_value(_INVOKE_AGENT_TELEMETRY_KEY, tel_ctx)
  )
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
      span = s
      tracing.trace_agent_invocation(span, agent, ctx)
      tel_ctx.otel_context = context_api.get_current()
      yield tel_ctx
  except Exception as e:
    caught_error = e
    raise
  finally:
    context_api.detach(token)
    _record_agent_metrics(
        agent.name,
        _metrics.get_elapsed_s(span, start_time),
        caught_error,
    )
    _flush_invoke_agent_metrics(tel_ctx, agent.name)


@contextlib.asynccontextmanager
async def record_tool_execution(
    tool: BaseTool,
    agent: BaseAgent,
    function_args: dict[str, object],
    invocation_context: InvocationContext,
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
    _accumulate_invoke_agent_tool_call()
    try:
      _metrics.record_tool_execution_duration(
          tool_name=tool.name,
          tool_type=tool.__class__.__name__,
          agent_name=agent.name,
          elapsed_s=_metrics.get_elapsed_s(span, start_time),
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
    _accumulate_invoke_agent_inference_call()
    agent = invocation_context.agent
    elapsed_s = _metrics.get_elapsed_s(tel_ctx.span, start_time)
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
