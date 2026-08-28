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

_INVOKE_AGENT_TELEMETRY_KEY = context_api.create_key("invoke_agent_telemetry")
_TOOL_EXECUTION_TELEMETRY_KEY = context_api.create_key(
    "tool_execution_telemetry"
)
_WORKFLOW_SCOPE_KEY = context_api.create_key("adk_workflow_scope")


@dataclasses.dataclass
class _WorkflowScope:
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

  token_totals: _token_usage.InvocationTokenTotals | None = None
  """The accumulator, empty to begin with unlike the fields above."""

  inference_call_count: int = 0
  """Model calls made anywhere inside this workflow."""

  tool_call_count: int = 0
  """Tool calls made anywhere inside this workflow, including the
  `transfer_to_agent` calls that route between its agents."""

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
    skill_name: The name of the skill as the model wrote it, confirmed only once
      :func:`confirm_skill` has a loaded skill to back it. See
      :mod:`._hallucination`.
    skill: The loaded skill, or None if the load did not produce one (unknown
      skill name, registry failure). Nothing is recorded in that case; the
      failure itself is already reported as the span's ``error.type``.
  """

  skill_name: _hallucination.MaybeHallucinated[str]
  skill: Skill | None = dataclasses.field(default=None, init=False)

  def confirm_skill(self, skill: Skill) -> None:
    """Records the skill the name resolved to.

    Loading the skill is what proves the name real, so the two are set together
    rather than left for a caller to keep in step.

    Args:
      skill: The skill that ``skill_name`` named.
    """
    self.skill = skill
    self.skill_name = _hallucination.ConfirmedNotHallucinated(
        self.skill_name.maybe_hallucinated_value
    )


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
    resource_path: Path of resource being loaded from skill, confirmed only once
      :func:`confirm_resource_path` reports the resource was found.
  """

  resource_path: _hallucination.MaybeHallucinated[str]

  def confirm_resource_path(self) -> None:
    """Marks the path as one the skill turned out to hold a resource at."""
    self.resource_path = _hallucination.ConfirmedNotHallucinated(
        self.resource_path.maybe_hallucinated_value
    )


@dataclasses.dataclass
class SkillScriptExecutionTelemetry(_SkillTelemetryCommon):
  """Skill telemetry extension for skill script execution.

  See :class:`_SkillTelemetryCommon`, for more information.

  Attributes:
    script_exit_code: The exit code of the skill script.
    script_path: The path of the skill script, confirmed only once
      :func:`confirm_script_path` reports the script was found.
  """

  script_path: _hallucination.MaybeHallucinated[str]
  script_exit_code: int | None = dataclasses.field(default=None, init=False)

  def confirm_script_path(self) -> None:
    """Marks the path as one the skill turned out to hold a script at."""
    self.script_path = _hallucination.ConfirmedNotHallucinated(
        self.script_path.maybe_hallucinated_value
    )


SkillTelemetry = (
    SkillLoadTelemetry
    | SkillResourceLoadTelemetry
    | SkillScriptExecutionTelemetry
)


@dataclasses.dataclass
class TelemetryContext:
  """Stores all telemetry related state."""

  otel_context: context_api.Context | None = None
  function_response_event: event_lib.Event | None = None
  error_type: str | None = None
  span: tracing.GenerateContentSpan | trace.Span | None = None
  skill_telemetry: SkillTelemetry | None = None
  token_totals: _token_usage.InvocationTokenTotals | None = None
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
    if (
        not response.partial
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
    tel_ctx: TelemetryContext, agent_name: str
) -> None:
  """Flushes this span's accumulated inference/tool-call and token metrics."""
  if tel_ctx.token_totals is not None:
    _metrics.record_invoke_agent_token_usage(agent_name, tel_ctx.token_totals)
  _metrics.record_invoke_agent_inference_calls(
      agent_name, tel_ctx.inference_call_count
  )
  _metrics.record_invoke_agent_tool_calls(agent_name, tel_ctx.tool_call_count)


def _flush_workflow_metrics(scope: _WorkflowScope) -> None:
  """Flushes one workflow's metrics; called once, by the scope's owner."""
  if not scope.telemetry_config.should_emit_experimental_telemetry:
    return
  nested = scope.parent is not None
  # We always record call counts because a count of 0 is a valid, accurate
  # measurement. For tokens, `None` means usage just wasn't reported, which
  # isn't the same as knowing the spend was exactly zero. So we skip tokens if
  # they are `None`.
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


def _invoke_agent_tel_ctx() -> TelemetryContext | None:
  """Returns the invoke_agent span's TelemetryContext."""
  value = context_api.get_value(_INVOKE_AGENT_TELEMETRY_KEY)
  return value if isinstance(value, TelemetryContext) else None


def _workflow_scope(
    otel_context: context_api.Context | None = None,
) -> _WorkflowScope | None:
  """Returns the innermost workflow scope, whose totals these are.

  Args:
    otel_context: Context to read from; defaults to the one in force.
  """
  value = context_api.get_value(_WORKFLOW_SCOPE_KEY, otel_context)
  return value if isinstance(value, _WorkflowScope) else None


def _accumulate_invoke_agent_tool_call() -> None:
  """Counts one tool call against the invoke_agent span."""
  span_tel_ctx = _invoke_agent_tel_ctx()
  if span_tel_ctx is not None:
    span_tel_ctx.increment_tool_calls()


def _accumulate_invoke_agent_inference_call() -> None:
  """Counts one model call against the invoke_agent span."""
  span_tel_ctx = _invoke_agent_tel_ctx()
  if span_tel_ctx is not None:
    span_tel_ctx.increment_inference_calls()


def _accumulate_invoke_workflow_tool_call(scope: _WorkflowScope | None) -> None:
  """Counts one tool call against every workflow enclosing it.

  Args:
    scope: The workflow the call ran inside.
  """
  if scope is None:
    return
  for enclosing in scope.self_and_enclosing():
    enclosing.tool_call_count += 1


def _accumulate_invoke_workflow_inference_call(
    scope: _WorkflowScope | None,
) -> None:
  """Counts one model call against every workflow enclosing it.

  Args:
    scope: The workflow the call ran inside.
  """
  if scope is None:
    return
  for enclosing in scope.self_and_enclosing():
    enclosing.inference_call_count += 1


def _active_tool_execution_tel_ctx() -> TelemetryContext | None:
  """Returns the TelemetryContext of the active execute_tool span."""
  value = context_api.get_value(_TOOL_EXECUTION_TELEMETRY_KEY)
  return value if isinstance(value, TelemetryContext) else None


def track_skill_load(skill_name: str) -> SkillLoadTelemetry:
  """Creates a SkillLoadTelemetry for the given skill name, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillLoadTelemetry(
      skill_name=_hallucination.MaybeHallucinated(skill_name)
  )
  attach_skill_telemetry(skill_telemetry)
  return skill_telemetry


def track_skill_resource_load(
    skill_name: str, resource_path: str
) -> SkillResourceLoadTelemetry:
  """Creates a SkillResourceLoadTelemetry for the given skill name and resource path, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillResourceLoadTelemetry(
      skill_name=_hallucination.MaybeHallucinated(skill_name),
      resource_path=_hallucination.MaybeHallucinated(resource_path),
  )
  attach_skill_telemetry(skill_telemetry)
  return skill_telemetry


def track_skill_script_execution(
    skill_name: str, script_path: str
) -> SkillScriptExecutionTelemetry:
  """Creates a SkillScriptExecutionTelemetry for the given skill name and script path, and attaches it to the enclosing tool execution span."""
  skill_telemetry = SkillScriptExecutionTelemetry(
      skill_name=_hallucination.MaybeHallucinated(skill_name),
      script_path=_hallucination.MaybeHallucinated(script_path),
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


def _accumulate_invoke_agent_tokens(usage: _token_usage.TokenUsage) -> None:
  """Adds one model call's token usage to the active invoke_agent span."""
  span_tel_ctx = _invoke_agent_tel_ctx()
  if span_tel_ctx is None:
    return
  if span_tel_ctx.token_totals is None:
    span_tel_ctx.token_totals = _token_usage.InvocationTokenTotals()
  span_tel_ctx.token_totals.add(usage)


def _accumulate_invoke_workflow_tokens(
    usage: _token_usage.TokenUsage,
    scope: _WorkflowScope | None,
) -> None:
  """Adds one model call's token usage to every workflow enclosing the call.

  The turn included: several aggregations of one call, not double counting.
  Where an agent's total is exclusive to that agent, a workflow's is inclusive
  of everything that ran in it.

  Args:
    usage: What the call spent.
    scope: The workflow the call ran inside.
  """
  if scope is None:
    return
  for enclosing in scope.self_and_enclosing():
    if enclosing.token_totals is None:
      enclosing.token_totals = _token_usage.InvocationTokenTotals()
    enclosing.token_totals.add(usage)


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
  workflow_scope = _workflow_scope()
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
        if tel_ctx.skill_telemetry is not None:
          _dispatch_skill_telemetry(
              span, tel_ctx.skill_telemetry, invocation_context
          )
  finally:
    _accumulate_invoke_agent_tool_call()
    # Workflow-grain metrics are experimental, skip counting if not opted-in.
    if tracing._telemetry_config_from_invocation_context(
        invocation_context
    ).should_emit_experimental_telemetry:
      _accumulate_invoke_workflow_tool_call(workflow_scope)
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
  workflow_scope = _workflow_scope()
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
    # Skipped when not opted in: the token metrics keyed on it are experimental,
    # so a run without the opt-in does no bookkeeping and flushes nothing.
    if tracing._telemetry_config_from_invocation_context(
        invocation_context
    ).should_emit_experimental_telemetry:
      _accumulate_invoke_workflow_inference_call(workflow_scope)
      usage = _token_usage.TokenUsage.from_llm_responses(tel_ctx.llm_responses)
      if usage is not None:
        _accumulate_invoke_agent_tokens(usage)
        _accumulate_invoke_workflow_tokens(usage, workflow_scope)
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
) -> None:
  """Selects and attaches the correct skill telemetry to the enclosing tool execution span."""
  telemetry_config = tracing._telemetry_config_from_invocation_context(
      invocation_context
  )
  if not telemetry_config.should_emit_experimental_telemetry:
    return

  match skill_telemetry:
    case SkillLoadTelemetry():
      _trace_skill_load(span, skill_telemetry)
    case SkillResourceLoadTelemetry():
      _trace_skill_resource_load(span, skill_telemetry)
    case SkillScriptExecutionTelemetry():
      _trace_skill_script_execution(span, skill_telemetry)
      if invocation_context.agent is None:
        return
      try:
        _metrics.record_skill_script_execution(
            invocation_context.agent.name,
            skill_telemetry.skill_name,
            skill_telemetry.script_path,
            skill_telemetry.script_exit_code,
        )
      except Exception:  # pylint: disable=broad-exception-caught
        logger.exception(
            "Failed to record skill script execution metrics for agent %s",
            invocation_context.agent.name,
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
