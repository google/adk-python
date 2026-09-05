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
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from typing_extensions import assert_never

from . import _adk_attributes
from . import _hallucination
from . import _metrics
from . import _token_usage
from . import tracing
from ._finish_reason import is_reported_finish_reason
from ._schema_version import resolve_schema_version
from ._schema_version import SCHEMA_VERSION_SEMCONV_ALIGNED
from .context import TelemetryConfig

# pylint: disable=g-import-not-at-top
if TYPE_CHECKING:
  from opentelemetry.util.types import AttributeValue

  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..agents.run_config import RunConfig
  from ..events import event as event_lib
  from ..models.llm_request import LlmRequest
  from ..models.llm_response import LlmResponse
  from ..skills.models import Skill
  from ..tools.base_tool import BaseTool
  from ..workflow._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)

_AGENT_INVOCATION_SCOPE_KEY = context_api.create_key("agent_invocation_scope")
_TOOL_EXECUTION_TELEMETRY_KEY = context_api.create_key(
    "tool_execution_telemetry"
)
_WORKFLOW_SCOPE_KEY = context_api.create_key("adk_workflow_scope")


@dataclasses.dataclass(kw_only=True)
class _Accumulator:
  """Accumulates telemetry data for one active scope (e.g. a workflow).

  Inherited by every scope that accumulates, whether its totals cover only its
  own work or everything nested inside it too. Keyword-only so a subclass can
  still declare fields of its own without a default.
  """

  token_totals: _token_usage.TokenUsage | None = None
  """Token spend, None until a model call reports usage."""

  inference_call_count: int = 0
  """Model calls counted against this scope."""

  tool_call_count: int = 0
  """Tool calls counted against this scope, including the `transfer_to_agent`
  calls that route between agents."""

  skill_load_count: int = 0
  """Skill loads counted against this scope."""


@dataclasses.dataclass(kw_only=True)
class _AgentInvocationScope(_Accumulator):
  """State accumulated across one agent invocation."""

  agent_name: str
  """The agent these totals are exclusive to."""


@dataclasses.dataclass(kw_only=True)
class _WorkflowScope(_Accumulator):
  """State accumulated across one `invoke_workflow`, root or nested.

  Exactly one `invoke_workflow` per turn is unnested, so the root instance is
  the turn and the two need no separate types. Totals are inclusive of the
  workflows nested inside, so summing across datapoints double counts unless
  the query pins `gen_ai.workflow.nested`.
  """

  root_agent_name: str
  """The agent the turn entered at, not the app. A sticky `transfer_to_agent`
  moves it from one turn to the next."""

  telemetry_config: TelemetryConfig
  """The config the turn was claimed under, inherited by the workflows nested
  inside it, so one turn shares one config."""

  workflow_name: str | None
  """The name the datapoint is filed under. Set by the reporting span."""

  parent: _WorkflowScope | None
  """The enclosing scope, None on the root. Its presence is what
  `gen_ai.workflow.nested` reports."""

  def self_and_enclosing(self) -> Iterator[_WorkflowScope]:
    """Yields this scope, then each one enclosing it, outermost last.

    What a call spends belongs to every workflow it ran inside, so anything
    accumulated per call walks this chain rather than stopping here.
    """
    scope: _WorkflowScope | None = self
    while scope is not None:
      yield scope
      scope = scope.parent


@contextlib.contextmanager
def record_invocation(
    entrypoint_node: BaseNode | None,
    conversation_id: str,
    run_config: RunConfig | None = None,
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
    run_config: The run's config, whose telemetry settings carry the
      experimental opt-in the span's token scope is gated on.

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
  with node_tracing._use_invoke_workflow_span(
      entrypoint_name,
      conversation_id,
      telemetry_config=run_config.telemetry if run_config else None,
  ):
    yield


@dataclasses.dataclass
class _SkillTelemetryCommon:
  """Common attributes for skill related telemetry.

  Added to the enclosing tool execution via :func:`track_*` functions,
  which is what turns it into attributes on the skill related spans.

  Attributes:
    skill_name: The name of the skill, possibly hallucinated.
    skill: The loaded skill, or None if the load did not produce one (unknown
      skill name, registry failure). Nothing is recorded in that case; the
      failure itself is already reported as the span's ``error.type``.
  """

  skill_name: _hallucination.MaybeHallucinated[str]
  skill: Skill | None = dataclasses.field(default=None, init=False)


@dataclasses.dataclass
class SkillLoadTelemetry(_SkillTelemetryCommon):
  """Skill telemetry extension for skill load.

  Attributes:
    additional_tools: The list of additional tools reported by the skill.
  """

  @property
  def additional_tools(self) -> list[str] | None:
    if self.skill is None:
      return None
    return self.skill.frontmatter.metadata.get("adk_additional_tools", None)


@dataclasses.dataclass
class SkillResourceLoadTelemetry(_SkillTelemetryCommon):
  """Skill telemetry extension for skill resource load.

  See :class:`_SkillTelemetryCommon`, for more information.

  Attributes:
    resource_path: Path of resource being loaded from skill, possibly
      hallucinated.
  """

  resource_path: _hallucination.MaybeHallucinated[str]


@dataclasses.dataclass
class SkillScriptExecutionTelemetry(_SkillTelemetryCommon):
  """Skill telemetry extension for skill script execution.

  See :class:`_SkillTelemetryCommon`, for more information.

  Attributes:
    script_exit_code: The exit code of the skill script.
    script_path: The path of the skill script, possibly hallucinated.
  """

  script_path: _hallucination.MaybeHallucinated[str]
  script_exit_code: int | None = dataclasses.field(default=None, init=False)


SkillTelemetry = (
    SkillLoadTelemetry
    | SkillResourceLoadTelemetry
    | SkillScriptExecutionTelemetry
)


@dataclasses.dataclass
class ToolScope:
  """What one tool call reports back while its `execute_tool` span is open.

  Handed to its caller by yield and also published on the OTel context, so a
  tool running underneath can report back without reaching for the ambient
  span. Read back in the `finally` that closes the span.
  """

  function_response_event: event_lib.Event | None = None
  error_type: str | None = None
  skill_telemetry: SkillTelemetry | None = None


@dataclasses.dataclass
class InferenceScope:
  """What one model call collects while its inference span is open."""

  span: tracing.GenerateContentSpan | trace.Span | None = None
  _llm_responses: list[LlmResponse] = dataclasses.field(default_factory=list)
  _inference_span_ended: bool = False

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
    # The inference ends at the response that says why generation stopped. End
    # the span there, before the response is handed back to the caller, so what
    # the caller does with it (running the tool the model asked for) is not
    # nested inside the inference span.
    #
    # `response.partial` is deliberately not consulted. It is set within ADK
    # rather than by the model, and a `BaseLlm` implementation is free to leave
    # it out, which made chunk one look like the end of the turn. The finish
    # reason comes from the model and marks the last response reliably.
    if (
        is_reported_finish_reason(response.finish_reason)
        and isinstance(self.span, tracing.GenerateContentSpan)
        and (exit_stack := self.span._exit_stack) is not None  # pylint: disable=protected-access
    ):
      exit_stack.close()
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
    scope: _AgentInvocationScope, tel_cfg: TelemetryConfig
) -> None:
  """Flushes one agent invocation's accumulated metrics.

  Args:
    scope: The invocation's totals.
    tel_cfg: The config the invocation ran under, for the experimental gate.
  """
  # `token_totals` is only set under opt-in, so it carries the gate already.
  if scope.token_totals is not None:
    _metrics.record_invoke_agent_token_usage(
        scope.agent_name, scope.token_totals
    )
  _metrics.record_invoke_agent_inference_calls(
      scope.agent_name, scope.inference_call_count
  )
  _metrics.record_invoke_agent_tool_calls(
      scope.agent_name, scope.tool_call_count
  )
  if tel_cfg.should_emit_experimental_telemetry:
    _metrics.record_invoke_agent_skill_loads(
        scope.agent_name, scope.skill_load_count
    )


def _flush_workflow_metrics(scope: _WorkflowScope) -> None:
  """Flushes one workflow's metrics; called once, by the scope's owner."""
  if not scope.telemetry_config.should_emit_experimental_telemetry:
    return
  nested = scope.parent is not None
  # We always record call counts because a count of 0 is a valid, accurate
  # measurement. For tokens, nothing having reported usage isn't the same as
  # knowing the spend was exactly zero. So we skip tokens in that case.
  if scope.token_totals is not None:
    _metrics.record_invoke_workflow_token_usage(
        root_agent_name=scope.root_agent_name,
        workflow_name=scope.workflow_name,
        totals=scope.token_totals,
        nested=nested,
    )
  _metrics.record_invoke_workflow_inference_calls(
      root_agent_name=scope.root_agent_name,
      workflow_name=scope.workflow_name,
      count=scope.inference_call_count,
      nested=nested,
  )
  _metrics.record_invoke_workflow_tool_calls(
      root_agent_name=scope.root_agent_name,
      workflow_name=scope.workflow_name,
      count=scope.tool_call_count,
      nested=nested,
  )
  _metrics.record_invoke_workflow_skill_loads(
      root_agent_name=scope.root_agent_name,
      workflow_name=scope.workflow_name,
      count=scope.skill_load_count,
      nested=nested,
  )


def _agent_invocation_scope() -> _AgentInvocationScope | None:
  """Returns the agent invocation's scope."""
  value = context_api.get_value(_AGENT_INVOCATION_SCOPE_KEY)
  return value if isinstance(value, _AgentInvocationScope) else None


def _workflow_scope(
    otel_context: context_api.Context | None = None,
) -> _WorkflowScope | None:
  """Returns the innermost workflow scope, whose totals these are.

  Args:
    otel_context: Context to read from; defaults to the one in force.
  """
  value = context_api.get_value(_WORKFLOW_SCOPE_KEY, otel_context)
  return value if isinstance(value, _WorkflowScope) else None


def _accumulating_scopes(
    workflow_scope: _WorkflowScope | None,
) -> tuple[_Accumulator, ...]:
  """Returns every accumulating scope this call belongs to, outermost last.

  One call belongs to all of them at once: the per-agent totals are exclusive
  to that agent, every workflow total inclusive of what ran inside it. The
  workflow scopes form a chain, so a call three workflows deep belongs to all
  three and the turn is the last link.

  Every scope is returned even if the invocation has not finished, so that
  every token spent is recorded.

  Args:
    workflow_scope: The innermost workflow the call ran inside, read when the
      call started rather than here.
  """
  scopes: list[_Accumulator] = []
  agent_scope = _agent_invocation_scope()
  if agent_scope is not None:
    scopes.append(agent_scope)
  if workflow_scope is not None:
    scopes.extend(workflow_scope.self_and_enclosing())
  return tuple(scopes)


def _accumulate_tool_call(workflow_scope: _WorkflowScope | None) -> None:
  """Counts one tool call against every scope it belongs to.

  Args:
    workflow_scope: The workflow the call ran inside.
  """
  for scope in _accumulating_scopes(workflow_scope):
    scope.tool_call_count += 1


def _accumulate_inference_call(workflow_scope: _WorkflowScope | None) -> None:
  """Counts one model call against every scope it belongs to.

  Args:
    workflow_scope: The workflow the call ran inside.
  """
  for scope in _accumulating_scopes(workflow_scope):
    scope.inference_call_count += 1


def _accumulate_skill_load(workflow_scope: _WorkflowScope | None) -> None:
  """Counts one skill load against every scope it belongs to.

  Args:
    workflow_scope: The workflow the load ran inside.
  """
  for scope in _accumulating_scopes(workflow_scope):
    scope.skill_load_count += 1


def _active_tool_execution_tel_ctx() -> ToolScope | None:
  """Returns the scope of the active execute_tool span."""
  value = context_api.get_value(_TOOL_EXECUTION_TELEMETRY_KEY)
  return value if isinstance(value, ToolScope) else None


def track_skill_load(
    skill_name: _hallucination.MaybeHallucinated[str],
) -> SkillLoadTelemetry:
  """Creates a SkillLoadTelemetry for the given skill name, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillLoadTelemetry(skill_name)
  attach_skill_telemetry(skill_telemetry)
  return skill_telemetry


def track_skill_resource_load(
    skill_name: _hallucination.MaybeHallucinated[str],
    resource_path: _hallucination.MaybeHallucinated[str],
) -> SkillResourceLoadTelemetry:
  """Creates a SkillResourceLoadTelemetry for the given skill name and resource path, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillResourceLoadTelemetry(
      skill_name,
      resource_path,
  )
  attach_skill_telemetry(skill_telemetry)
  return skill_telemetry


def track_skill_script_execution(
    skill_name: _hallucination.MaybeHallucinated[str],
    script_path: _hallucination.MaybeHallucinated[str],
) -> SkillScriptExecutionTelemetry:
  """Creates a SkillScriptExecutionTelemetry for the given skill name and script path, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillScriptExecutionTelemetry(
      skill_name,
      script_path,
  )
  attach_skill_telemetry(skill_telemetry)
  return skill_telemetry


def attach_skill_telemetry(
    skill_telemetry: SkillTelemetry,
) -> None:
  """Attaches skill telemetry to the enclosing tool execution.

  The attributes are written by :func:`record_tool_execution`, which owns the
  ``execute_tool`` span, once the tool call completes. Callers therefore never
  depend on a span being open: outside a tool execution this is a no-op rather
  than an attribute silently landing on whatever span happens to be current.

  A tool execution references a single skill, so a second call within the same
  tool execution replaces the first.

  Args:
    skill_telemetry: Skill telemetry reference to attach to the active tool
      execution context.
  """
  tel_ctx = _active_tool_execution_tel_ctx()
  if tel_ctx is None:
    logger.debug(
        "No tool execution is being recorded, skill telemetry will not be"
        " attached to current span."
    )
    return
  if tel_ctx.skill_telemetry is not None:
    logger.warning(
        "Tool execution already has attached skill telemetry, overwriting."
    )
  tel_ctx.skill_telemetry = skill_telemetry


def _accumulate_tokens(
    usage: _token_usage.TokenUsage,
    workflow_scope: _WorkflowScope | None,
) -> None:
  """Adds one model call's token usage to every scope it belongs to.

  The turn included: several aggregations of one call, not double counting.
  Where an agent's total is exclusive to that agent, a workflow's is inclusive
  of everything that ran in it.

  Args:
    usage: What the call spent.
    workflow_scope: The workflow the call ran inside.
  """
  for scope in _accumulating_scopes(workflow_scope):
    if scope.token_totals is None:
      scope.token_totals = _token_usage.TokenUsage()
    scope.token_totals.add(usage)


@contextlib.asynccontextmanager
async def record_agent_invocation(
    ctx: InvocationContext, agent: BaseAgent
) -> AsyncIterator[_AgentInvocationScope]:
  """Unified context manager for consolidated agent invocation telemetry."""
  start_time = time.monotonic()
  caught_error: Exception | None = None
  span: trace.Span | None = None
  span_name = f"invoke_agent {agent.name}"
  scope = _AgentInvocationScope(agent_name=agent.name)
  token = context_api.attach(
      context_api.set_value(_AGENT_INVOCATION_SCOPE_KEY, scope)
  )
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
      span = s
      tracing.trace_agent_invocation(span, agent, ctx)
      yield scope
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
    tel_cfg = tracing._telemetry_config_from_invocation_context(ctx)
    _flush_invoke_agent_metrics(scope, tel_cfg)


@contextlib.asynccontextmanager
async def record_tool_execution(
    tool: BaseTool,
    agent: BaseAgent,
    function_args: dict[str, object],
    invocation_context: InvocationContext,
) -> AsyncIterator[ToolScope]:
  """Unified context manager for consolidated tool execution telemetry."""
  start_time = time.monotonic()
  workflow_scope = _workflow_scope()
  caught_error: Exception | None = None
  detected_error_type: str | None = None
  span: trace.Span | None = None
  span_name = f"execute_tool {tool.name}"
  try:
    with tracing.tracer.start_as_current_span(span_name) as s:
      span = s
      tel_ctx = ToolScope()
      # Published so the running tool can report telemetry back to this
      # span (see `record_skill_telemetry`) without reaching for the
      # ambient span itself.
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
        if tel_ctx.skill_telemetry is not None:
          _dispatch_skill_telemetry(
              span,
              tel_ctx.skill_telemetry,
              invocation_context,
              workflow_scope,
              error=caught_error,
              error_type=tel_ctx.error_type,
          )
  finally:
    _accumulate_tool_call(workflow_scope)
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
) -> AsyncIterator[InferenceScope]:
  """Unified async context manager for consolidated inference metrics."""
  start_time = time.monotonic()
  workflow_scope = _workflow_scope()
  tel_ctx = InferenceScope()
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
    _accumulate_inference_call(workflow_scope)
    # Tokens only: the metrics keyed on them are experimental, so a run without
    # the opt-in never builds the totals. The counts above accumulate either
    # way, and each flush gates what it emits.
    if tracing._telemetry_config_from_invocation_context(
        invocation_context
    ).should_emit_experimental_telemetry:
      usage = _token_usage.TokenUsage.from_llm_responses(tel_ctx.llm_responses)
      if usage is not None:
        _accumulate_tokens(usage, workflow_scope)
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


def _dispatch_skill_telemetry(
    span: trace.Span,
    skill_telemetry: SkillTelemetry,
    invocation_context: InvocationContext,
    workflow_scope: _WorkflowScope | None,
    error: Exception | None = None,
    error_type: str | None = None,
) -> None:
  """Selects and attaches the correct skill telemetry to the enclosing tool execution span.

  Args:
    span: The ``execute_tool`` span the skill work ran under.
    skill_telemetry: What the tool reported about the skill it touched.
    invocation_context: The invocation the tool call belongs to.
    workflow_scope: The workflow scope of the tool call.
    error: The exception the tool raised, if any.
    error_type: An error type the tool reported without raising. Ignored when
      `error` is also set.
  """
  telemetry_config = tracing._telemetry_config_from_invocation_context(
      invocation_context
  )
  if not telemetry_config.should_emit_experimental_telemetry:
    return

  error_type = (
      tracing.resolve_error_type(error) if error is not None else error_type
  )

  match skill_telemetry:
    case SkillLoadTelemetry():
      _accumulate_skill_load(workflow_scope)
      _trace_skill_load(span, skill_telemetry)
      if invocation_context.agent is None:
        return
      _metrics.record_skill_load(
          invocation_context.agent.name,
          skill_telemetry.skill_name,
          error_type,
      )
    case SkillResourceLoadTelemetry():
      _trace_skill_resource_load(span, skill_telemetry)
    case SkillScriptExecutionTelemetry():
      _trace_skill_script_execution(span, skill_telemetry)
      if invocation_context.agent is None:
        return
      _metrics.record_skill_script_execution(
          invocation_context.agent.name,
          skill_telemetry.skill_name,
          skill_telemetry.script_path,
          skill_telemetry.script_exit_code,
      )
    case _:
      assert_never(skill_telemetry)


def _trace_skill_load(
    span: trace.Span,
    skill_telemetry: SkillLoadTelemetry,
) -> None:
  """Stamps the skill load attributes onto the ``execute_tool`` span."""
  attributes: dict[str, AttributeValue] = {}
  attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME] = (
      skill_telemetry.skill_name.maybe_hallucinated_value
  )
  skill = skill_telemetry.skill

  if skill is not None:
    attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_DESCRIPTION] = (
        skill.description
    )

    if (uri := skill._uri) is not None:
      attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SOURCE_URI] = uri

    if (additional_tools := skill_telemetry.additional_tools) is not None:
      attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_ADDITIONAL_TOOLS] = (
          additional_tools
      )

  span.set_attributes(attributes)


def _trace_skill_resource_load(
    span: trace.Span,
    skill_telemetry: SkillResourceLoadTelemetry,
) -> None:
  """Stamps the skill resource loading information in the ``execute_tool load_skill_resource`` span."""
  attributes: dict[str, AttributeValue] = {}
  attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME] = (
      skill_telemetry.skill_name.maybe_hallucinated_value
  )
  if (skill := skill_telemetry.skill) is not None and (
      uri := skill._uri
  ) is not None:
    attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SOURCE_URI] = uri

  attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_RESOURCE_PATH] = (
      skill_telemetry.resource_path.maybe_hallucinated_value
  )

  span.set_attributes(attributes)


def _trace_skill_script_execution(
    span: trace.Span,
    skill_telemetry: SkillScriptExecutionTelemetry,
) -> None:
  """Stamps the skill script execution information in the ``execute_tool run_skill_script`` span."""
  attributes: dict[str, AttributeValue] = {}
  attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_NAME] = (
      skill_telemetry.skill_name.maybe_hallucinated_value
  )
  attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SCRIPT_PATH] = (
      skill_telemetry.script_path.maybe_hallucinated_value
  )

  if (script_exit_code := skill_telemetry.script_exit_code) is not None:
    attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SCRIPT_EXIT_CODE] = (
        script_exit_code
    )

    if script_exit_code != 0:
      span.set_status(
          trace.Status(trace.StatusCode.ERROR, "SKILL_SCRIPT_EXECUTION_ERROR")
      )
      span.set_attribute(ERROR_TYPE, "SKILL_SCRIPT_EXECUTION_ERROR")

  if (skill := skill_telemetry.skill) is not None and (
      uri := skill._uri
  ) is not None:
    attributes[_adk_attributes.ADK_EXPERIMENTAL_SKILL_SOURCE_URI] = uri

  span.set_attributes(attributes)
