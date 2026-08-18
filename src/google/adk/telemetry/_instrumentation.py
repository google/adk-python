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
import enum
import logging
import sys
import time
from typing import AsyncIterator
from typing import Iterator
from typing import TYPE_CHECKING

from opentelemetry import trace
import opentelemetry.context as context_api

from . import _adk_attributes
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
  from ..skills.models import Skill
  from ..tools.base_tool import BaseTool
  from ..workflow._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)

_INVOKE_AGENT_TELEMETRY_KEY = context_api.create_key("invoke_agent_telemetry")
_TOOL_EXECUTION_TELEMETRY_KEY = context_api.create_key(
    "tool_execution_telemetry"
)


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


class SkillTelemetrySpanType(enum.Enum):
  """Skill telemetry type."""

  SKILL_LOAD = enum.auto()


@dataclasses.dataclass
class SkillTelemetry:
  """Skill related telemetry.

  Added to the enclosing tool execution via :func:`record_skill_telemetry`,
  which is what turns it into attributes on the skill related spans.

  Attributes:
    span_type: The type of skill telemetry being recorded.
    skill_name: The name of the skill.
    skill: The loaded skill, or None if the load did not produce one (unknown
      skill name, registry failure). Nothing is recorded in that case; the
      failure itself is already reported as the span's ``error.type``.
    cache_hit: Whether the skill came from the per-invocation fetch cache
      instead of the registry. Only meaningful for registry-sourced skills.
    additional_tools: The list of additional tools reported by the skill.
  """

  span_type: SkillTelemetrySpanType
  skill_name: str | None = None
  skill: Skill | None = None
  cache_hit: bool = False
  additional_tools: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TelemetryContext:
  """Stores all telemetry related state."""

  otel_context: context_api.Context | None = None
  function_response_event: event_lib.Event | None = None
  error_type: str | None = None
  span: tracing.GenerateContentSpan | trace.Span | None = None
  skill_telemetry: SkillTelemetry | None = None
  _llm_responses: list[LlmResponse] = dataclasses.field(default_factory=list)
  _inference_span_ended: bool = False
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
    # Anything after the span ended (a second complete response for the same
    # call) can no longer be recorded on it, but still counts for the metrics.
    if self._inference_span_ended:
      return

    tracing.trace_inference_result(invocation_context, self.span, response)
    # A non-partial response is the end of the inference: end the span before
    # the response is handed back to the caller, so what the caller does with
    # it (running the tool the model asked for) is not nested inside the
    # inference span. Partial chunks keep it open until the final one arrives.
    if not response.partial and isinstance(
        self.span, tracing.GenerateContentSpan
    ):
      self.span._close()  # pylint: disable=protected-access
      self._inference_span_ended = True


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


def _active_tool_execution_tel_ctx() -> TelemetryContext | None:
  """Returns the TelemetryContext of the active execute_tool span."""
  value = context_api.get_value(_TOOL_EXECUTION_TELEMETRY_KEY)
  return value if isinstance(value, TelemetryContext) else None


def record_skill_telemetry(
    telemetry_type: SkillTelemetrySpanType,
) -> SkillTelemetry:
  """Attaches skill telemetry to the enclosing tool execution.

  The attributes are written by :func:`record_tool_execution`, which owns the
  ``execute_tool`` span, once the tool call completes. Callers therefore never
  depend on a span being open: outside a tool execution this is a no-op rather
  than an attribute silently landing on whatever span happens to be current.

  A tool execution references a single skill, so a second call within the same
  tool execution replaces the first.

  Args:
    telemetry_type: The type of skill telemetry being recorded.

  Returns:
    skill_telemetry: Skill telemetry reference to record against the active tool
    call.
  """
  tel_ctx = _active_tool_execution_tel_ctx()
  if tel_ctx is None:
    logger.debug(
        "No tool execution is being recorded, skill telemetry will not be"
        " attached to current span."
    )
    return SkillTelemetry(span_type=telemetry_type)
  if tel_ctx.skill_telemetry is not None:
    logger.warning(
        "Tool execution already has attached skill telemetry, overwriting."
    )
  tel_ctx.skill_telemetry = SkillTelemetry(span_type=telemetry_type)
  return tel_ctx.skill_telemetry


def record_skill_cache_hit() -> None:
  """Records a skill cache hit against the enclosing tool execution."""
  tel_ctx = _active_tool_execution_tel_ctx()
  if tel_ctx is None:
    logger.debug(
        "Skipping skill cache hit: no tool execution is being recorded."
    )
    return
  if tel_ctx.skill_telemetry is None:
    logger.warning(
        "Tool execution has no attached skill telemetry, skipping cache hit."
    )
    return
  tel_ctx.skill_telemetry.cache_hit = True


def _trace_skill_load(
    span: trace.Span,
    skill_telemetry: SkillTelemetry | None,
    invocation_context: InvocationContext,
) -> None:
  """Stamps the skill load attributes onto the ``execute_tool`` span."""
  if skill_telemetry is None:
    return

  telemetry_config = tracing._telemetry_config_from_invocation_context(
      invocation_context
  )
  if not telemetry_config.should_emit_experimental_telemetry:
    return

  if skill_telemetry.skill_name is not None:
    span.set_attribute(
        _adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME, skill_telemetry.skill_name
    )

  skill = skill_telemetry.skill
  if skill is None:
    return

  span.set_attribute(
      _adk_attributes.ADK_EXPERIMENTAL_SKILL_DESCRIPTION, skill.description
  )

  if skill._uri is not None:
    span.set_attribute(
        _adk_attributes.ADK_EXPERIMENTAL_SKILL_SOURCE_URI, skill._uri
    )

    if skill._uri.startswith(("http", "https")):
      # Only meaningful if skill is from registry.
      span.set_attribute(
          _adk_attributes.ADK_EXPERIMENTAL_SKILL_CACHE_HIT,
          skill_telemetry.cache_hit,
      )

  span.set_attribute(
      _adk_attributes.ADK_EXPERIMENTAL_SKILL_ADDITIONAL_TOOLS,
      skill_telemetry.additional_tools,
  )


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
  detected_error_type: str | None = None
  span: trace.Span | None = None
  span_name = f"execute_tool {tool.name}"
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
      span = s
      tel_ctx = TelemetryContext(otel_context=context_api.get_current())
      # Published so the running tool can report telemetry back to this span
      # (see `record_skill_telemetry`) without reaching for the ambient span
      # itself.
      token = context_api.attach(
          context_api.set_value(_TOOL_EXECUTION_TELEMETRY_KEY, tel_ctx)
      )
      try:
        yield tel_ctx
      except Exception as e:
        caught_error = e
        raise
      finally:
        context_api.detach(token)
        detected_error_type = tel_ctx.error_type
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
        if (
            tel_ctx.skill_telemetry is not None
            and tel_ctx.skill_telemetry.span_type
            == SkillTelemetrySpanType.SKILL_LOAD
        ):
          _trace_skill_load(span, tel_ctx.skill_telemetry, invocation_context)
  finally:
    _accumulate_invoke_agent_tool_call()
    try:
      _metrics.record_tool_execution_duration(
          tool_name=tool.name,
          tool_type=tool.__class__.__name__,
          agent_name=agent.name,
          elapsed_s=_metrics.get_elapsed_s(span, start_time),
          error=caught_error,
          error_type=detected_error_type,
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
