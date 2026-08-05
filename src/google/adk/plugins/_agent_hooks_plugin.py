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

"""Governance plugin that makes ADK a host for the agent-hooks contract.

`agent-hooks <https://github.com/responsibleai/agent-hooks>`_ is a
framework-neutral *control* contract: a fixed set of agent lifecycle
interception points, the ``AgentContext`` a host supplies at each, and the
``Verdict`` an interceptor returns (``allow`` / ``deny`` / ``transform``). This
plugin turns ADK's :class:`~google.adk.plugins.base_plugin.BasePlugin` seam
into a conformant host: you register one or more interceptors (policy engines,
content filters, rate limiters, egress guards) once, and every governed ADK
lifecycle point delegates its decision to the agent-hooks emitter, which runs
the interceptors and records every decision as an auditable
``InterceptionRecord``.

Interception-point mapping (ADK callback -> agent-hooks point):
    ``before_run_callback``       -> ``agent_startup``
    ``on_user_message_callback``  -> ``input``
    ``before_model_callback``     -> ``pre_model_call``
    ``after_model_callback``      -> ``post_model_call``
    ``before_tool_callback``      -> ``pre_tool_call``
    ``after_tool_callback``       -> ``post_tool_call``
    ``on_event_callback`` (final) -> ``output``
    ``after_run_callback``        -> ``agent_shutdown``

Because ADK invokes the user-message seam before the run seam, ``input`` is
emitted before ``agent_startup`` in a turn; the agent-hooks ``sequence`` field
reflects that real ADK order.

Enforcement semantics (fail closed):
    - ``deny`` blocks the guarded action. A blocked model call returns a
      refusal ``LlmResponse``; a blocked tool returns an error result; a
      blocked run/input/output surfaces a refusal ``Content``.
    - ``transform`` rewrites the guarded value from the interceptor's
      ``$target`` transform: tool args (pre-tool), tool result (post-tool),
      user text (input), model text (post-model), or final output text.
    - Any engine-internal error, malformed verdict, or interceptor
      timeout is turned into a fail-closed *deny* by the emitter, and this
      plugin surfaces it as a blocked action — it never fails open.
    - ``pre_model_call`` supports ``allow`` / ``deny``; a ``transform`` there
      is treated as a fail-closed deny, because rebuilding a provider-native
      request from wire messages is not round-trip safe.

Trust model: agent-hooks is a *cooperative* control contract, **not** a
security boundary. Interceptors run in-process with full data access and the
interception points do not guarantee complete mediation. See the agent-hooks
``SECURITY.md`` before relying on it for isolation.

``agent-hooks`` is an **optional** dependency with a compiled native core;
install it with ``pip install "google-adk[agent-hooks]"``. Importing this
module never requires it — the import is deferred to
:class:`AgentHooksPlugin` construction, which raises an actionable
``ImportError`` when the package is missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import math
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..agents.callback_context import CallbackContext
from ..events.event import Event
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from collections.abc import Callable
  from collections.abc import Sequence

  from agent_hooks import AgentContext
  from agent_hooks import AgentContextBuilder
  from agent_hooks import CompositionConfig
  from agent_hooks import IdentityProvider
  from agent_hooks import InterceptionEmitter
  from agent_hooks import InterceptionRecord
  from agent_hooks import Interceptor

  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

#: ``framework`` identifier stamped on every emitted ``AgentContext``.
_FRAMEWORK = "google-adk"

#: Default identity provider name understood by the emitter (agent-hooks §10.2).
_DEFAULT_IDENTITY = "jcs-sha256"

#: Default bound on the emitter's in-memory record buffer per invocation.
_DEFAULT_MAX_RECORDS = 1000

_INSTALL_HINT = (
    "agent-hooks is not installed (or its native core failed to load). It is "
    "an optional dependency with a compiled core; install it with:\n"
    '    pip install "google-adk[agent-hooks]"\n'
    "See https://github.com/responsibleai/agent-hooks for details."
)


def _require_agent_hooks() -> Any:
  """Import and return the ``agent_hooks`` module, or fail actionably.

  Raises:
    ImportError: If the ``agent_hooks`` package (or its compiled core) cannot
      be imported.
  """
  try:
    return importlib.import_module("agent_hooks")
  except Exception as exc:  # ImportError or a native-core load failure.
    raise ImportError(_INSTALL_HINT) from exc


#: Maximum nesting depth normalized by :func:`_json_safe`; deeper structures
#: are truncated so an untrusted, deeply nested payload cannot raise
#: ``RecursionError`` before a decision is made.
_MAX_JSON_DEPTH = 100


def _json_safe(
    value: Any, _seen: frozenset[int] = frozenset(), _depth: int = 0
) -> Any:
  """Coerce ``value`` into a JSON-serializable form for an ``AgentContext``.

  Tool arguments, tool results, and model output are untrusted and may hold
  objects the agent-hooks wire format cannot carry (which would otherwise fail
  the emission closed). This normalizes them at the boundary so an interceptor
  always sees a stable, inspectable value. Reference cycles are broken with a
  ``"<cycle>"`` placeholder and nesting past ``_MAX_JSON_DEPTH`` is truncated to
  ``"<max-depth>"``, so a self-referential or deeply nested container cannot
  raise ``RecursionError`` before a decision is made.
  """
  if _depth > _MAX_JSON_DEPTH:
    return "<max-depth>"
  if value is None or isinstance(value, (bool, int, str)):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, dict):
    if id(value) in _seen:
      return "<cycle>"
    seen = _seen | {id(value)}
    return {str(k): _json_safe(v, seen, _depth + 1) for k, v in value.items()}
  if isinstance(value, (list, tuple, set, frozenset)):
    if id(value) in _seen:
      return "<cycle>"
    seen = _seen | {id(value)}
    return [_json_safe(v, seen, _depth + 1) for v in value]
  if isinstance(value, (bytes, bytearray)):
    return bytes(value).decode("utf-8", errors="replace")
  dump = getattr(value, "model_dump", None)
  if callable(dump):
    try:
      return _json_safe(dump(mode="json"), _seen, _depth + 1)
    except Exception:
      logger.debug("model_dump() failed while normalizing %s", type(value))
  return str(value)


def _content_text(content: Optional[types.Content]) -> str:
  """Join the text parts of a ``types.Content`` into a single string."""
  if content is None or not content.parts:
    return ""
  return "".join(part.text for part in content.parts if part.text)


def _text_content(text: str, *, role: str) -> types.Content:
  """Build a single-text-part ``types.Content`` with the given role."""
  return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def _target_text(target: Any) -> str:
  """Extract replacement text from a transformed ``input``/``output`` target.

  The ``input``/``output`` L1 envelopes wrap the value as ``{"content": ...}``;
  an interceptor may transform the whole envelope or replace ``$target`` with a
  bare value.
  """
  if isinstance(target, dict) and "content" in target:
    target = target["content"]
  return target if isinstance(target, str) else json.dumps(target, default=str)


def _request_messages(llm_request: LlmRequest) -> list[dict[str, Any]]:
  """Project an ``LlmRequest`` into inspectable ``{role, content}`` messages."""
  messages: list[dict[str, Any]] = []
  system = getattr(llm_request.config, "system_instruction", None)
  if isinstance(system, str) and system:
    messages.append({"role": "system", "content": system})
  for content in llm_request.contents:
    messages.append({
        "role": content.role or "user",
        "content": _content_text(content),
    })
  return messages


def _response_tool_calls(response: LlmResponse) -> list[dict[str, Any]]:
  """Surface the function calls a model response requested."""
  if response.content is None or not response.content.parts:
    return []
  calls: list[dict[str, Any]] = []
  for part in response.content.parts:
    call = part.function_call
    if call is not None:
      calls.append({
          "id": call.id or "",
          "name": call.name or "",
          "args": _json_safe(dict(call.args) if call.args else {}),
      })
  return calls


def _response_finish_reason(
    response: LlmResponse, *, has_tool_calls: bool
) -> str:
  """Best-effort finish reason for the ``post_model_call`` context."""
  reason = response.finish_reason
  if reason is not None:
    return getattr(reason, "name", str(reason)).lower()
  return "tool_calls" if has_tool_calls else "stop"


def _synth_call_id(name: str, args: dict[str, Any]) -> str:
  """Deterministic tool-call id when ADK supplies none.

  Derived from ``name`` and ``args``. The plugin stashes the pre-tool id so the
  matching post-tool record reuses it, because a pre-tool transform may rewrite
  the args in place before the post-tool point is reached.
  """
  digest = hashlib.sha256()
  digest.update(name.encode("utf-8"))
  digest.update(json.dumps(args, sort_keys=True, default=str).encode("utf-8"))
  return f"tc-{digest.hexdigest()[:16]}"


def _verdict_reason(record: Optional[InterceptionRecord]) -> str:
  """Human-readable reason for a blocked action (payload-free)."""
  if record is None:
    return "agent-hooks engine error (failing closed)"
  verdict = record.verdict
  reason = (verdict.reason or "").strip()
  message = (verdict.message or "").strip()
  if reason and message and reason != message:
    return f"{reason}: {message}"
  return reason or message or "blocked by agent-hooks policy"


class _InvocationState:
  """Per-invocation agent-hooks builder and emitter.

  One instance exists per ADK invocation id; it owns the monotonic
  ``sequence`` for that turn's records and is evicted when the invocation
  ends (or errors) so long-running processes do not leak state.
  """

  __slots__ = (
      "builder",
      "emitter",
      "startup_denied",
      "startup_record",
      "synth_call_ids",
  )

  def __init__(
      self, builder: AgentContextBuilder, emitter: InterceptionEmitter
  ) -> None:
    self.builder = builder
    self.emitter = emitter
    # Set when agent_startup denies, so the deny is re-enforced at the first
    # model call on runner paths that ignore before_run_callback's return.
    self.startup_denied: bool = False
    self.startup_record: Optional[InterceptionRecord] = None
    # Synthesized pre-tool call ids, queued per tool name so the matching
    # post-tool record reuses the same id after an in-place args transform.
    self.synth_call_ids: dict[str, list[str]] = {}


class AgentHooksPlugin(BasePlugin):
  """ADK plugin that enforces agent-hooks interceptors across the lifecycle.

  Register the plugin on the ``Runner`` with one or more interceptors; each
  governed ADK callback emits an ``AgentContext`` to the agent-hooks emitter
  and enforces the returned verdict, failing closed on any error.

  Example:
      >>> from google.adk.plugins import AgentHooksPlugin
      >>> from agent_hooks import AgentContext, Decision, Verdict
      >>>
      >>> class BlockDangerousTools:
      ...   def intercept(self, ctx: AgentContext) -> Verdict:
      ...     if (
      ...         ctx["interception_point"] == "pre_tool_call"
      ...         and ctx["tool_call"]["name"] == "delete_account"
      ...     ):
      ...       return Verdict.deny(reason="tool_denied")
      ...     return Verdict(decision=Decision.ALLOW)
      >>>
      >>> plugin = AgentHooksPlugin(interceptors=[BlockDangerousTools()])
      >>> # runner = InMemoryRunner(agent=root_agent, plugins=[plugin])
  """

  def __init__(
      self,
      interceptors: Sequence[Interceptor],
      *,
      name: str = "agent_hooks",
      mode: str = "enforce",
      timeout: Optional[float] = 5.0,
      composition: Optional[CompositionConfig] = None,
      identity_provider: Optional[str | IdentityProvider] = _DEFAULT_IDENTITY,
      record_sink: Optional[Callable[[InterceptionRecord], None]] = None,
      max_records: Optional[int] = _DEFAULT_MAX_RECORDS,
  ) -> None:
    """Initializes the plugin.

    Args:
      interceptors: The interceptors to run at every governed point, in
        registration order. Each is an object with an
        ``intercept(AgentContext) -> Verdict`` method (sync or async).
      name: Unique identifier for this plugin instance.
      mode: ``"enforce"`` (act on verdicts) or ``"evaluate_only"`` (record
        decisions without blocking or transforming).
      timeout: Per-interceptor timeout in seconds; ``None`` disables it.
      composition: agent-hooks composition profile; defaults to
        ``sequential/first_deny``.
      identity_provider: agent-hooks identity provider for audit records;
        ``"jcs-sha256"`` by default, or ``None`` for identity-unbound records.
      record_sink: Optional callback invoked with every ``InterceptionRecord``
        for audit persistence. A sink exception is swallowed by the emitter.
      max_records: Bound on the per-invocation in-memory record buffer.

    Raises:
      ImportError: If the optional ``agent-hooks`` package is not installed.
    """
    super().__init__(name)
    ah = _require_agent_hooks()
    self._ah = ah
    self._interceptors: list[Any] = list(interceptors)
    self._mode = ah.EnforcementMode(mode)
    self._enforcing = self._mode == ah.EnforcementMode("enforce")
    self._timeout = timeout
    self._composition = composition
    self._identity_provider = identity_provider
    self._record_sink = record_sink
    self._max_records = max_records
    self._states: dict[str, _InvocationState] = {}

  # --- state lifecycle -------------------------------------------------------

  def _new_state(
      self, *, invocation_id: str, session_id: str, agent_name: str
  ) -> _InvocationState:
    ah = self._ah
    builder = ah.AgentContextBuilder(
        agent_id=agent_name,
        framework=_FRAMEWORK,
        session_id=session_id,
        agent_name=agent_name,
    )
    emitter = ah.InterceptionEmitter(
        mode=self._mode,
        timeout=self._timeout,
        composition=self._composition,
        identity_provider=self._identity_provider,
    )
    for interceptor in self._interceptors:
      emitter.register(interceptor, type(interceptor).__name__)
    if self._record_sink is not None:
      emitter.set_record_sink(self._record_sink)
    if self._max_records is not None:
      emitter.set_max_records(self._max_records)
    state = _InvocationState(builder, emitter)
    self._states[invocation_id] = state
    return state

  def _state_for_invocation(
      self, invocation_context: InvocationContext
  ) -> _InvocationState:
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    state = self._states.get(invocation_context.invocation_id)
    if state is None:
      state = self._new_state(
          invocation_id=invocation_context.invocation_id,
          session_id=invocation_context.session.id,
          agent_name=agent_name,
      )
    return state

  def _state_for_context(
      self, context: CallbackContext | ToolContext
  ) -> _InvocationState:
    state = self._states.get(context.invocation_id)
    if state is None:
      state = self._new_state(
          invocation_id=context.invocation_id,
          session_id=context.session.id,
          agent_name=context.agent_name,
      )
    return state

  def _agent_envelope(self, agent_name: str) -> dict[str, Any]:
    return {"id": agent_name, "framework": _FRAMEWORK, "name": agent_name}

  async def _emit(
      self, state: _InvocationState, ctx: AgentContext, *, agent_name: str
  ) -> Optional[InterceptionRecord]:
    """Emit ``ctx`` and return the record, or ``None`` on engine failure.

    A ``None`` return signals the caller to fail closed. ``CancelledError``
    is propagated unchanged so task cancellation is honoured.
    """
    ctx["agent"] = self._agent_envelope(agent_name)
    try:
      return await state.emitter.emit_unchecked(ctx)
    except asyncio.CancelledError:
      raise
    except Exception:
      logger.exception(
          "agent-hooks emission failed at %s; failing closed",
          ctx.get("interception_point"),
      )
      return None

  @staticmethod
  def _blocked(record: Optional[InterceptionRecord]) -> bool:
    """Whether the record denies the guarded action (or is an engine error)."""
    return record is None or not record.proceeds

  def _is_transform(self, record: InterceptionRecord) -> bool:
    # Only enforce mode actually rewrites ``ctx["target"]``; evaluate_only
    # records the verdict without transforming, so applying it here is lossy.
    return self._enforcing and record.verdict.transform is not None

  def _pre_tool_call_id(
      self,
      state: _InvocationState,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> str:
    """Call id for the pre-tool record.

    When ADK supplies no ``function_call_id`` the id is synthesized and stashed
    so the matching post-tool record can reuse it even after a pre-tool
    transform rewrites the args in place.
    """
    function_call_id = tool_context.function_call_id
    if function_call_id:
      return function_call_id
    call_id = _synth_call_id(tool.name, tool_args)
    state.synth_call_ids.setdefault(tool.name, []).append(call_id)
    return call_id

  def _post_tool_call_id(
      self,
      state: _InvocationState,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> str:
    """Call id for the post-tool record, correlated with its pre-tool id."""
    function_call_id = tool_context.function_call_id
    if function_call_id:
      return function_call_id
    pending = state.synth_call_ids.get(tool.name)
    if pending:
      return pending.pop(0)
    return _synth_call_id(tool.name, tool_args)

  # --- lifecycle callbacks ---------------------------------------------------

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """agent_startup: deny halts the run with a refusal message."""
    state = self._state_for_invocation(invocation_context)
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    ctx = state.builder.agent_startup(tools_registered=self._tool_names(agent))
    record = await self._emit(state, ctx, agent_name=agent_name)
    if self._blocked(record):
      # Some runner paths ignore this return value; the flag makes the deny
      # stick by also blocking the first model call (see before_model_callback).
      state.startup_denied = True
      state.startup_record = record
      return _text_content(
          f"[blocked by agent-hooks: {_verdict_reason(record)}]", role="model"
      )
    return None

  @override
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """input: deny replaces the user message; transform rewrites it."""
    state = self._state_for_invocation(invocation_context)
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    ctx = state.builder.input(content=_content_text(user_message))
    record = await self._emit(state, ctx, agent_name=agent_name)
    if self._blocked(record):
      return _text_content(
          f"[input blocked by agent-hooks: {_verdict_reason(record)}]",
          role="user",
      )
    if record is not None and self._is_transform(record):
      return _text_content(_target_text(ctx.get("target")), role="user")
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """pre_model_call: deny (or transform, treated as deny) blocks the call."""
    state = self._state_for_context(callback_context)
    if state.startup_denied:
      # A denied agent_startup halts the run even on runner paths that ignore
      # before_run_callback's return value.
      return self._blocked_response(state.startup_record)
    ctx = state.builder.pre_model_call(
        model_id=llm_request.model or "unknown",
        messages=_request_messages(llm_request),
    )
    record = await self._emit(
        state, ctx, agent_name=callback_context.agent_name
    )
    if self._blocked(record):
      return self._blocked_response(record)
    if record is not None and self._is_transform(record):
      logger.warning(
          "agent-hooks pre_model_call transform is not applied to the "
          "provider request; failing closed"
      )
      return self._blocked_response(record)
    return None

  @override
  async def after_model_callback(
      self, *, callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    """post_model_call: deny blocks; transform rewrites the response text."""
    state = self._state_for_context(callback_context)
    tool_calls = _response_tool_calls(llm_response)
    ctx = state.builder.post_model_call(
        model_id=llm_response.model_version or "unknown",
        content=_content_text(llm_response.content),
        tool_calls=tool_calls,
        finish_reason=_response_finish_reason(
            llm_response, has_tool_calls=bool(tool_calls)
        ),
    )
    record = await self._emit(
        state, ctx, agent_name=callback_context.agent_name
    )
    if self._blocked(record):
      return self._blocked_response(record)
    if record is not None and self._is_transform(record):
      return llm_response.model_copy(
          update={
              "content": _text_content(
                  _target_text(ctx.get("target")), role="model"
              )
          }
      )
    return None

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict[str, Any]]:
    """pre_tool_call: deny blocks the tool; transform rewrites the args."""
    state = self._state_for_context(tool_context)
    call_id = self._pre_tool_call_id(state, tool, tool_args, tool_context)
    ctx = state.builder.pre_tool_call(
        call_id=call_id, name=tool.name, args=_json_safe(tool_args)
    )
    record = await self._emit(state, ctx, agent_name=tool_context.agent_name)
    if self._blocked(record):
      return self._blocked_tool_result(record)
    if record is not None and self._is_transform(record):
      new_args = ctx.get("target")
      if not isinstance(new_args, dict):
        logger.warning(
            "agent-hooks pre_tool_call transform did not yield an args "
            "object; failing closed"
        )
        return self._blocked_tool_result(record)
      # Mutating tool_args in place propagates to the actual tool call.
      tool_args.clear()
      tool_args.update(new_args)
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict[str, Any],
  ) -> Optional[dict[str, Any]]:
    """post_tool_call: deny blocks; transform replaces the tool result."""
    state = self._state_for_context(tool_context)
    call_id = self._post_tool_call_id(state, tool, tool_args, tool_context)
    ctx = state.builder.post_tool_call(
        call_id=call_id,
        name=tool.name,
        args=_json_safe(tool_args),
        value=_json_safe(result),
    )
    record = await self._emit(state, ctx, agent_name=tool_context.agent_name)
    if self._blocked(record):
      return self._blocked_tool_result(record)
    if record is not None and self._is_transform(record):
      new_value = ctx.get("target")
      return new_value if isinstance(new_value, dict) else {"result": new_value}
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """output: govern the final response event (deny/transform its content)."""
    if not event.is_final_response():
      return None
    state = self._state_for_invocation(invocation_context)
    agent = invocation_context.agent
    agent_name = event.author or (
        agent.name if agent is not None else "unknown"
    )
    ctx = state.builder.output(content=_content_text(event.content))
    record = await self._emit(state, ctx, agent_name=agent_name)
    if self._blocked(record):
      return event.model_copy(
          update={
              "content": _text_content(
                  f"[output blocked by agent-hooks: {_verdict_reason(record)}]",
                  role="model",
              )
          }
      )
    if record is not None and self._is_transform(record):
      return event.model_copy(
          update={
              "content": _text_content(
                  _target_text(ctx.get("target")), role="model"
              )
          }
      )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """agent_shutdown: emit for audit, then evict per-invocation state."""
    state = self._states.pop(invocation_context.invocation_id, None)
    if state is None:
      return None
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    try:
      ctx = state.builder.agent_shutdown(reason="completed")
      await self._emit(state, ctx, agent_name=agent_name)
    except Exception:
      logger.debug("agent_shutdown emission failed", exc_info=True)
    return None

  @override
  async def on_run_error_callback(
      self, *, invocation_context: InvocationContext, error: Exception
  ) -> None:
    """Evict per-invocation state on an error path (notification-only)."""
    state = self._states.pop(invocation_context.invocation_id, None)
    if state is None:
      return None
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    try:
      ctx = state.builder.agent_shutdown(
          reason="error", error=type(error).__name__
      )
      await self._emit(state, ctx, agent_name=agent_name)
    except Exception:
      logger.debug("agent_shutdown emission failed on run error", exc_info=True)
    return None

  @override
  async def close(self) -> None:
    """Drop any residual per-invocation state."""
    self._states.clear()

  # --- block-result builders -------------------------------------------------

  def _blocked_response(
      self, record: Optional[InterceptionRecord]
  ) -> LlmResponse:
    reason = _verdict_reason(record)
    return LlmResponse(
        content=_text_content(
            f"[blocked by agent-hooks: {reason}]", role="model"
        ),
        custom_metadata={"agent_hooks_blocked": True, "reason": reason},
    )

  def _blocked_tool_result(
      self, record: Optional[InterceptionRecord]
  ) -> dict[str, Any]:
    reason = _verdict_reason(record)
    return {
        "error": f"blocked by agent-hooks: {reason}",
        "agent_hooks_blocked": True,
        "reason": reason,
    }

  @staticmethod
  def _tool_names(agent: Any) -> list[str]:
    """Best-effort declared tool names for the ``agent_startup`` context."""
    names: list[str] = []
    tools = getattr(agent, "tools", None) or []
    for tool in tools:
      name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
      if isinstance(name, str) and name not in names:
        names.append(name)
    return names
