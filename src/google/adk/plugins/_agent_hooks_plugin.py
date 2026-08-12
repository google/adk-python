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
into an Agent Hooks-compatible host: you register one or more interceptors (policy engines,
content filters, rate limiters, egress guards) once, and every governed ADK
lifecycle point delegates its decision to the agent-hooks emitter, which runs
the interceptors and records every decision as an auditable
``InterceptionRecord``.

Interception-point mapping (ADK callback -> agent-hooks point):
  ``on_user_message_callback``  -> ``agent_startup`` (once), then ``input``
  ``before_run_callback``       -> startup fallback / terminal deny latch
    ``before_model_callback``     -> ``pre_model_call``
    ``after_model_callback``      -> ``post_model_call``
    ``on_model_error_callback``   -> ``post_model_call`` (error outcome)
    ``before_tool_callback``      -> ``pre_tool_call``
    ``after_tool_callback``       -> ``post_tool_call``
    ``on_tool_error_callback``    -> ``post_tool_call`` (error outcome)
    ``on_event_callback`` (final) -> ``output``
    ``on_run_complete_callback``  -> ``agent_shutdown`` (completed outcome)
    ``on_run_error_callback``     -> ``agent_shutdown`` (error outcome)
    ``on_run_cancelled_callback`` -> ``agent_shutdown`` (cancelled outcome)

A model or tool call that raises is routed by ADK to its ``on_*_error``
callback, so the paired ``post_*`` point is still emitted (the errored result is
discarded and the original error propagates), keeping every ``pre_*`` paired.

ADK invokes the user-message seam before the run seam, so the plugin lazily
emits ``agent_startup`` from that seam before emitting ``input``. The run seam
is an idempotent fallback for invocations without a user message and re-enforces
latched startup/input denies on runner paths that treat callback values as
replacements rather than terminal control signals.

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
      - Multimodal content is projected as structured JSON. Values that cannot
        be represented faithfully are denied rather than truncated or coerced.
      - A configured timeout requires async interceptors. Synchronous
        interceptors are accepted only with the explicit unbounded setting
        ``timeout=None``.
      - Custom audit-sink failures deny by default. Set
        ``audit_failure_mode="log"`` only when lossy audit delivery is an
        accepted deployment tradeoff.

Trust model: agent-hooks is a *cooperative* control contract, **not** a
security boundary. Interceptors run in-process with full data access and the
interception points do not guarantee complete mediation. See the agent-hooks
``SECURITY.md`` before relying on it for isolation.

Agent Hooks owns its governed ADK callbacks exclusively by default because
ADK short-circuits callback chains on the first replacement value. Set
``allow_unsafe_plugin_composition=True`` only after reviewing every overlapping
plugin and accepting that governance can be bypassed or invalidated.

``agent-hooks`` is an **optional** dependency with a compiled native core;
install it with ``pip install "google-adk[agent-hooks]"``. Importing this
module never requires it — the import is deferred to
:class:`AgentHooksPlugin` construction, which raises an actionable
``ImportError`` when the package is missing.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import hmac
import importlib
import inspect
import logging
import math
import secrets
from typing import Any
from typing import Literal
from typing import Optional
from typing import TYPE_CHECKING
import uuid

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

  from ..agents.invocation_context import _RunIdentityToken
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

#: ``framework`` identifier stamped on every emitted ``AgentContext``.
_FRAMEWORK = "google-adk"

#: Default identity provider name understood by the emitter (agent-hooks §10.2).
_DEFAULT_IDENTITY = "jcs-sha256"

#: Default bound on the emitter's in-memory record buffer per invocation.
_DEFAULT_MAX_RECORDS = 1000

#: Default bound on concurrently retained invocation state.
_DEFAULT_MAX_ACTIVE_INVOCATIONS = 1024

#: Default bound on tool-call identities retained within one invocation.
_DEFAULT_MAX_TRACKED_TOOL_CALLS = 1024

#: Default bound on concurrently dispatched model calls per invocation.
_DEFAULT_MAX_TRACKED_MODEL_CALLS = 128

#: Default bound on tool calls accepted from one model response.
_DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE = 64

#: Default bound on records retained for custom-sink retry per invocation.
_DEFAULT_MAX_RETAINED_AUDIT_RECORDS = 1024

_MODEL_BLOCK_REASON = "policy_denied"
_MODEL_BLOCK_TEXT = "[blocked by policy]"

_EXCLUSIVE_CALLBACKS = frozenset({
    "on_user_message_callback",
    "before_run_callback",
    "on_run_complete_callback",
    "before_model_callback",
    "after_model_callback",
    "on_model_error_callback",
    "before_tool_callback",
    "after_tool_callback",
    "on_tool_error_callback",
    "on_event_callback",
})

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


#: Maximum nesting depth accepted by :func:`_json_safe`.
_MAX_JSON_DEPTH = 100
_MAX_CONTEXT_NODES = 100_000
_MAX_CONTEXT_TEXT_BYTES = 5 * 1024 * 1024
_MAX_CONTEXT_BLOB_BYTES = 5 * 1024 * 1024
_MAX_CONTENT_PARTS = 1024
_MAX_MESSAGES = 4096


class _ContextProjectionError(ValueError):
  """Raised when an Agent Hooks context cannot preserve a source value."""


class _ProjectionBudget:
  """Finite work budget shared across one projected interception context."""

  __slots__ = ("blob_bytes", "nodes", "text_bytes")

  def __init__(self) -> None:
    self.nodes = 0
    self.text_bytes = 0
    self.blob_bytes = 0

  def consume_node(self) -> None:
    self.nodes += 1
    if self.nodes > _MAX_CONTEXT_NODES:
      raise _ContextProjectionError("maximum context node count exceeded")

  def consume_text(self, value: str) -> None:
    remaining = _MAX_CONTEXT_TEXT_BYTES - self.text_bytes
    if len(value) > remaining:
      raise _ContextProjectionError("maximum context text size exceeded")
    self.text_bytes += len(value.encode("utf-8", errors="strict"))
    if self.text_bytes > _MAX_CONTEXT_TEXT_BYTES:
      raise _ContextProjectionError("maximum context text size exceeded")

  def consume_blob(self, size: int) -> None:
    self.blob_bytes += size
    if self.blob_bytes > _MAX_CONTEXT_BLOB_BYTES:
      raise _ContextProjectionError("maximum context blob size exceeded")


def _json_safe(
    value: Any,
    _seen: frozenset[int] = frozenset(),
    _depth: int = 0,
    _budget: _ProjectionBudget | None = None,
) -> Any:
  """Return a faithful JSON value or reject the context fail closed."""
  budget = _budget or _ProjectionBudget()
  budget.consume_node()
  if _depth > _MAX_JSON_DEPTH:
    raise _ContextProjectionError("maximum context depth exceeded")
  if isinstance(value, str):
    budget.consume_text(value)
    return value
  if value is None or isinstance(value, (bool, int)):
    return value
  if isinstance(value, float):
    if not math.isfinite(value):
      raise _ContextProjectionError("non-finite number is not JSON")
    return value
  if isinstance(value, dict):
    if id(value) in _seen:
      raise _ContextProjectionError("reference cycle in context value")
    if not all(isinstance(key, str) for key in value):
      raise _ContextProjectionError("context mappings require string keys")
    seen = _seen | {id(value)}
    for key in value:
      budget.consume_text(key)
    return {
        key: _json_safe(item, seen, _depth + 1, budget)
        for key, item in value.items()
    }
  if isinstance(value, (list, tuple)):
    if id(value) in _seen:
      raise _ContextProjectionError("reference cycle in context value")
    seen = _seen | {id(value)}
    return [_json_safe(item, seen, _depth + 1, budget) for item in value]
  if isinstance(value, (bytes, bytearray, set, frozenset)):
    raise _ContextProjectionError(
        f"unsupported context value type: {type(value).__name__}"
    )
  dump = getattr(value, "model_dump", None)
  if callable(dump):
    try:
      inline_data = getattr(value, "inline_data", None)
      blob_data = getattr(inline_data, "data", None)
      if isinstance(blob_data, (bytes, bytearray)):
        budget.consume_blob(len(blob_data))
      return _json_safe(
          dump(mode="json", exclude_none=True),
          _seen,
          _depth + 1,
          budget,
      )
    except _ContextProjectionError:
      raise
    except Exception as exc:
      raise _ContextProjectionError(
          f"model serialization failed for {type(value).__name__}"
      ) from exc
  raise _ContextProjectionError(
      f"unsupported context value type: {type(value).__name__}"
  )


def _content_value(
    content: Optional[types.Content],
    budget: _ProjectionBudget | None = None,
) -> Any:
  """Project every content part, retaining a compact string for text-only."""
  if content is None or not content.parts:
    return ""
  if len(content.parts) > _MAX_CONTENT_PARTS:
    raise _ContextProjectionError("maximum content part count exceeded")
  projection_budget = budget or _ProjectionBudget()
  if content.role:
    projection_budget.consume_text(content.role)
  projected_parts = [
      _json_safe(part, _budget=projection_budget) for part in content.parts
  ]
  if all(
      isinstance(part, dict) and set(part) == {"text"}
      for part in projected_parts
  ):
    return "".join(part["text"] for part in projected_parts)
  return {"role": content.role, "parts": projected_parts}


def _text_content(text: str, *, role: str) -> types.Content:
  """Build a single-text-part ``types.Content`` with the given role."""
  return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def _target_content(target: Any, *, default_role: str) -> types.Content:
  """Parse a transformed input, response, or output target as ADK content."""
  target = _json_safe(target)
  role = default_role
  if isinstance(target, dict) and "content" in target:
    candidate_role = target.get("role")
    if isinstance(candidate_role, str):
      role = candidate_role
    target = target["content"]
  if isinstance(target, str):
    return _text_content(target, role=role)
  if isinstance(target, dict) and isinstance(target.get("parts"), list):
    content_data = dict(target)
    content_data.setdefault("role", role)
    try:
      return types.Content.model_validate(content_data)
    except Exception as exc:
      raise _ContextProjectionError(
          "transformed content does not match the ADK Content contract"
      ) from exc
  raise _ContextProjectionError(
      "transformed content must be text or a structured ADK Content value"
  )


def _request_messages(
    llm_request: LlmRequest, *, agent_name: str
) -> list[dict[str, Any]]:
  """Project provider-native ADK history into Agent Hooks message roles."""
  budget = _ProjectionBudget()
  messages: list[dict[str, Any]] = []

  def append_message(role: str, content: Any) -> None:
    if len(messages) >= _MAX_MESSAGES:
      raise _ContextProjectionError("maximum expanded message count exceeded")
    budget.consume_text(role)
    messages.append({"role": role, "content": content})

  if len(llm_request.contents) > _MAX_MESSAGES:
    raise _ContextProjectionError("maximum message count exceeded")
  system = getattr(llm_request.config, "system_instruction", None)
  if isinstance(system, str) and system:
    internal_prefix = f'You are an agent. Your internal name is "{agent_name}".'
    if system.startswith(internal_prefix):
      system = system[len(internal_prefix) :].strip()
    if system:
      budget.consume_text(system)
      append_message("system", system)
  elif isinstance(system, types.Content):
    append_message("system", _content_value(system, budget))
  elif system is not None:
    raise _ContextProjectionError(
        f"unsupported system instruction type: {type(system).__name__}"
    )
  for content in llm_request.contents:
    parts = content.parts or []
    if len(parts) > _MAX_CONTENT_PARTS:
      raise _ContextProjectionError("maximum content part count exceeded")
    ordinary_parts = []
    for part in parts:
      function_response = part.function_response
      if function_response is not None:
        response = _json_safe(function_response.response, _budget=budget)
        if (
            isinstance(response, dict)
            and response.get("agent_hooks_blocked") is True
        ):
          response = f"blocked: {response.get('reason') or 'policy_denied'}"
          budget.consume_text(response)
        elif isinstance(response, dict) and set(response) == {"result"}:
          response = response["result"]
        append_message("tool", response)
      else:
        ordinary_parts.append(part)
    if ordinary_parts:
      append_message(
          content.role or "user",
          _content_value(
              types.Content(role=content.role, parts=ordinary_parts), budget
          ),
      )
  return messages


def _response_tool_calls(
    response: LlmResponse, budget: _ProjectionBudget | None = None
) -> list[dict[str, Any]]:
  """Surface the function calls a model response requested."""
  if response.content is None or not response.content.parts:
    return []
  if len(response.content.parts) > _MAX_CONTENT_PARTS:
    raise _ContextProjectionError("maximum content part count exceeded")
  calls: list[dict[str, Any]] = []
  projection_budget = budget or _ProjectionBudget()
  for part in response.content.parts:
    call = part.function_call
    if call is not None:
      calls.append({
          "id": call.id or "",
          "name": call.name or "",
          "args": _json_safe(
              dict(call.args) if call.args else {},
              _budget=projection_budget,
          ),
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
  """Host-generated tool-call id when ADK supplies none."""
  del name, args
  return f"tc-{uuid.uuid4().hex}"


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
      "agent_name",
      "run_nonce",
      "run_identity_token",
      "resume_lineage_id",
      "owner_identity",
      "startup_lock",
      "emission_lock",
      "finalization_lock",
      "finalization_task",
      "terminal_reason",
      "terminal_error",
      "phase",
      "startup_emitted",
      "startup_denied",
      "startup_record",
      "input_denied",
      "input_record",
      "source_tool_call_ids",
      "open_model_calls",
      "open_tool_calls",
      "post_record_owners",
      "undelivered_records",
      "late_delivery_sequences",
      "delivered_record_sequences",
      "terminal_record_emitted",
  )

  def __init__(
      self,
      builder: AgentContextBuilder,
      emitter: InterceptionEmitter,
      agent_name: str,
      run_nonce: str,
      run_identity_token: _RunIdentityToken,
      resume_lineage_id: str,
      owner_identity: tuple[str, str, str, str],
  ) -> None:
    self.builder = builder
    self.emitter = emitter
    self.agent_name = agent_name
    self.run_nonce = run_nonce
    self.run_identity_token = run_identity_token
    self.resume_lineage_id = resume_lineage_id
    self.owner_identity = owner_identity
    self.startup_lock = asyncio.Lock()
    self.emission_lock = asyncio.Lock()
    self.finalization_lock = asyncio.Lock()
    self.finalization_task: asyncio.Task[None] | None = None
    self.terminal_reason: Literal["completed", "error", "cancelled"] | None = (
        None
    )
    self.terminal_error: str | None = None
    self.phase: Literal["active", "finalizing", "closed"] = "active"
    self.startup_emitted: bool = False
    # Set when agent_startup denies, so the deny is re-enforced at the first
    # model call on runner paths that ignore before_run_callback's return.
    self.startup_denied: bool = False
    self.startup_record: Optional[InterceptionRecord] = None
    # Set when input denies. ADK treats on_user_message_callback's non-None
    # return as a replacement message, not a halt, so the deny is latched and
    # re-enforced at the first model call (agent-hooks §6: at input the turn
    # MUST NOT begin).
    self.input_denied: bool = False
    self.input_record: Optional[InterceptionRecord] = None
    self.source_tool_call_ids: set[str] = set()
    self.open_model_calls: dict[int, tuple[object, str, str]] = {}
    # Call ids whose tool was dispatched (pre_tool_call emitted, not blocked)
    # and still owe exactly one post_tool_call (agent-hooks §3.1(5)).
    self.open_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
    self.post_record_owners: dict[int, tuple[str, object]] = {}
    self.undelivered_records: dict[int, InterceptionRecord] = {}
    self.late_delivery_sequences: set[int] = set()
    self.delivered_record_sequences: set[int] = set()
    self.terminal_record_emitted = False


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
      ...   async def intercept(self, ctx: AgentContext) -> Verdict:
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
      allow_unsafe_unbounded_records: bool = False,
      max_active_invocations: int = _DEFAULT_MAX_ACTIVE_INVOCATIONS,
      max_tracked_tool_calls: int = _DEFAULT_MAX_TRACKED_TOOL_CALLS,
      max_tracked_model_calls: int = _DEFAULT_MAX_TRACKED_MODEL_CALLS,
      max_tool_calls_per_response: int = _DEFAULT_MAX_TOOL_CALLS_PER_RESPONSE,
      allow_unsafe_plugin_composition: bool = False,
      audit_failure_mode: Literal["deny", "log"] = "deny",
      audit_timeout: float = 5.0,
      max_pending_audit_records: int = 32,
      audit_workers: int = 1,
      audit_shutdown_timeout: float = 5.0,
      max_retained_audit_records: int = _DEFAULT_MAX_RETAINED_AUDIT_RECORDS,
  ) -> None:
    """Initializes the plugin.

    Args:
      interceptors: The interceptors to run at every governed point, in
        registration order. Each is an object with an
        async ``intercept(AgentContext) -> Verdict`` method when ``timeout`` is
        configured. Synchronous interceptors require ``timeout=None`` because
        Python cannot safely preempt a blocking in-process call.
      name: Unique identifier for this plugin instance.
      mode: ``"enforce"`` (act on verdicts) or ``"evaluate_only"`` (record
        decisions without blocking or transforming).
      timeout: Per-interceptor timeout in seconds; ``None`` disables it.
      composition: agent-hooks composition profile; defaults to
        ``sequential/first_deny``.
      identity_provider: agent-hooks identity provider for audit records;
        ``"jcs-sha256"`` by default, or ``None`` for identity-unbound records.
      record_sink: Optional callback invoked with every ``InterceptionRecord``
        for audit persistence. Failures are logged without exception text and
        deny in enforce mode unless ``audit_failure_mode="log"`` is selected.
      max_records: Bound on the per-invocation in-memory record buffer.
      allow_unsafe_unbounded_records: Permit ``max_records=None``. Disabled by
        default because an unbounded audit buffer can exhaust process memory.
      max_active_invocations: Maximum invocation states retained concurrently.
        New invocations fail closed when capacity is exhausted.
      max_tracked_tool_calls: Maximum distinct tool-call identities tracked in
        one invocation. Calls beyond the limit fail closed.
      max_tracked_model_calls: Maximum model requests tracked concurrently in
        one invocation. Requests beyond the limit fail closed.
      max_tool_calls_per_response: Maximum tool calls accepted from one model
        response before ADK creates tool tasks.
      allow_unsafe_plugin_composition: Permit other plugins to implement the
        same governed callbacks. Disabled by default because ADK short-circuits
        callback chains, so overlapping plugins can bypass governance.
      audit_failure_mode: ``"deny"`` blocks the guarded action when a custom
        record sink fails; ``"log"`` reports the failure and preserves the
        policy verdict. Evaluate-only mode never enforces an audit failure.
      audit_timeout: Maximum seconds to admit and deliver one custom audit
        record before the guarded action fails closed.
      max_pending_audit_records: Maximum running and queued custom sink calls.
      audit_workers: Dedicated worker threads for custom sink delivery.
      audit_shutdown_timeout: Maximum seconds to drain custom sink calls.
      max_retained_audit_records: Maximum records retained for sink retry per
        invocation. One slot is reserved for ``agent_shutdown``.

    Raises:
      ImportError: If the optional ``agent-hooks`` package is not installed.
      ValueError: If a configured timeout cannot cover a synchronous
        interceptor or an option is invalid.
    """
    super().__init__(name)
    ah = _require_agent_hooks()
    self._ah = ah
    self._interceptors: list[Any] = list(interceptors)
    if timeout is not None:
      synchronous = [
          type(interceptor).__name__
          for interceptor in self._interceptors
          if not inspect.iscoroutinefunction(
              getattr(interceptor, "intercept", None)
          )
      ]
      if synchronous:
        names = ", ".join(synchronous)
        raise ValueError(
            "AgentHooksPlugin timeout cannot preempt synchronous "
            f"interceptors: {names}. Use async intercept methods or set "
            "timeout=None explicitly."
        )
    self._mode = ah.EnforcementMode(mode)
    self._enforcing = self._mode == ah.EnforcementMode("enforce")
    self._timeout = timeout
    self._composition = composition
    self._identity_provider = identity_provider
    self._record_sink = record_sink
    if max_records is None and not allow_unsafe_unbounded_records:
      raise ValueError(
          "max_records=None requires allow_unsafe_unbounded_records=True"
      )
    if max_records is not None and max_records < 1:
      raise ValueError("max_records must be at least 1")
    self._max_records = max_records
    if max_active_invocations < 1:
      raise ValueError("max_active_invocations must be at least 1")
    self._max_active_invocations = max_active_invocations
    if max_tracked_tool_calls < 1:
      raise ValueError("max_tracked_tool_calls must be at least 1")
    self._max_tracked_tool_calls = max_tracked_tool_calls
    if max_tracked_model_calls < 1:
      raise ValueError("max_tracked_model_calls must be at least 1")
    self._max_tracked_model_calls = max_tracked_model_calls
    if max_tool_calls_per_response < 1:
      raise ValueError("max_tool_calls_per_response must be at least 1")
    self._max_tool_calls_per_response = max_tool_calls_per_response
    self._allow_unsafe_plugin_composition = allow_unsafe_plugin_composition
    if audit_failure_mode not in ("deny", "log"):
      raise ValueError("audit_failure_mode must be 'deny' or 'log'")
    self._audit_failure_mode = audit_failure_mode
    self._lineage_key = secrets.token_bytes(32)
    if audit_timeout <= 0:
      raise ValueError("audit_timeout must be positive")
    if max_pending_audit_records < 1:
      raise ValueError("max_pending_audit_records must be at least 1")
    if audit_workers < 1 or audit_workers > max_pending_audit_records:
      raise ValueError(
          "audit_workers must be between 1 and max_pending_audit_records"
      )
    if audit_shutdown_timeout <= 0:
      raise ValueError("audit_shutdown_timeout must be positive")
    if max_retained_audit_records < 2:
      raise ValueError("max_retained_audit_records must be at least 2")
    self._audit_timeout = audit_timeout
    self._audit_shutdown_timeout = audit_shutdown_timeout
    self._max_retained_audit_records = max_retained_audit_records
    self._audit_slots = asyncio.Semaphore(max_pending_audit_records)
    self._audit_executor = (
        ThreadPoolExecutor(
            max_workers=audit_workers,
            thread_name_prefix="agent-hooks-audit",
        )
        if record_sink is not None
        else None
    )
    self._audit_futures: set[asyncio.Future[None]] = set()
    self._audit_attempts: dict[tuple[str, int], asyncio.Future[None]] = {}
    self._states: dict[str, _InvocationState] = {}
    self._closed = False
    self._close_task: asyncio.Task[None] | None = None

  @property
  def exclusive_callbacks(self) -> frozenset[str]:
    """Reserve governed callbacks unless unsafe composition is explicit."""
    if self._allow_unsafe_plugin_composition:
      return frozenset()
    return _EXCLUSIVE_CALLBACKS

  # --- state lifecycle -------------------------------------------------------

  def _new_state(
      self,
      *,
      run_nonce: str,
      run_identity_token: _RunIdentityToken,
      owner_identity: tuple[str, str, str, str],
      agent_name: str,
  ) -> _InvocationState:
    if self._closed:
      raise RuntimeError("AgentHooksPlugin is closed")
    if len(self._states) >= self._max_active_invocations:
      raise RuntimeError(
          "AgentHooksPlugin active invocation capacity exhausted"
      )
    ah = self._ah
    builder = ah.AgentContextBuilder(
        agent_id=agent_name,
        framework=_FRAMEWORK,
        session_id=run_nonce,
        agent_name=agent_name,
    )
    resume_lineage_id = self._resume_lineage_id(owner_identity)
    builder.with_l2(
        extensions={"adk": {"resume_lineage_id": resume_lineage_id}}
    )
    emitter = ah.InterceptionEmitter(
        mode=self._mode,
        timeout=self._timeout,
        composition=self._composition,
        identity_provider=self._identity_provider,
    )
    for interceptor in self._interceptors:
      emitter.register(interceptor, type(interceptor).__name__)
    if self._max_records is not None:
      emitter.set_max_records(self._max_records)
    state = _InvocationState(
        builder,
        emitter,
        agent_name,
        run_nonce,
        run_identity_token,
        resume_lineage_id,
        owner_identity,
    )
    self._states[run_nonce] = state
    return state

  def _resume_lineage_id(
      self, owner_identity: tuple[str, str, str, str]
  ) -> str:
    """Return a stable unforgeable lineage for retries of one public run ID."""
    payload = "\0".join(owner_identity).encode("utf-8")
    digest = hmac.new(self._lineage_key, payload, hashlib.sha256).hexdigest()
    return f"rl-{digest}"

  def _state_for_invocation(
      self, invocation_context: InvocationContext
  ) -> _InvocationState:
    if self._closed:
      raise RuntimeError("AgentHooksPlugin is closed")
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    owner_identity = (
        invocation_context.app_name,
        invocation_context.user_id,
        invocation_context.session.id,
        invocation_context.invocation_id,
    )
    run_identity_token = invocation_context._run_identity_token
    if run_identity_token.closed:
      raise RuntimeError("Agent Hooks invocation is closed")
    if run_identity_token.owner_identity != owner_identity:
      raise RuntimeError("Agent Hooks invocation owner identity changed")
    state = self._states.get(invocation_context.run_nonce)
    if state is None:
      state = self._new_state(
          run_nonce=invocation_context.run_nonce,
          run_identity_token=run_identity_token,
          owner_identity=owner_identity,
          agent_name=agent_name,
      )
    elif state.owner_identity != owner_identity:
      raise RuntimeError("Agent Hooks invocation owner identity changed")
    elif state.run_identity_token is not run_identity_token:
      raise RuntimeError("Agent Hooks run nonce collision")
    elif state.phase != "active":
      raise RuntimeError("Agent Hooks invocation is finalizing")
    return state

  def _state_for_context(
      self, context: CallbackContext | ToolContext
  ) -> _InvocationState:
    if self._closed:
      raise RuntimeError("AgentHooksPlugin is closed")
    owner_identity = (
        context.session.app_name,
        context.user_id,
        context.session.id,
        context.invocation_id,
    )
    run_identity_token = context._run_identity_token
    if run_identity_token.closed:
      raise RuntimeError("Agent Hooks invocation is closed")
    if run_identity_token.owner_identity != owner_identity:
      raise RuntimeError("Agent Hooks invocation owner identity changed")
    state = self._states.get(context.run_nonce)
    if state is None:
      state = self._new_state(
          run_nonce=context.run_nonce,
          run_identity_token=run_identity_token,
          owner_identity=owner_identity,
          agent_name=context.agent_name,
      )
    elif state.owner_identity != owner_identity:
      raise RuntimeError("Agent Hooks invocation owner identity changed")
    elif state.run_identity_token is not run_identity_token:
      raise RuntimeError("Agent Hooks run nonce collision")
    elif state.phase != "active":
      raise RuntimeError("Agent Hooks invocation is finalizing")
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
    async with state.emission_lock:
      if state.phase != "active":
        return None
      return await self._emit_unlocked(state, ctx, agent_name=agent_name)

  async def _emit_unlocked(
      self, state: _InvocationState, ctx: AgentContext, *, agent_name: str
  ) -> Optional[InterceptionRecord]:
    """Emit while the caller owns ``state.emission_lock``."""
    ctx["agent"] = self._agent_envelope(agent_name)
    try:
      is_shutdown = ctx.get("interception_point") == "agent_shutdown"
      retained_limit = self._max_retained_audit_records - (
          0 if is_shutdown else 1
      )
      if (
          self._record_sink is not None
          and len(state.undelivered_records) >= retained_limit
      ):
        logger.error(
            "agent-hooks audit retry capacity exhausted",
            extra={
                "agent_hooks_run_nonce": state.run_nonce,
                "agent_hooks_retained_records": len(state.undelivered_records),
            },
        )
        return None
      record = await state.emitter.emit_unchecked(ctx)
      state.undelivered_records[record.sequence] = record
      owner = state.post_record_owners.pop(record.sequence, None)
      if owner is not None:
        owner_kind, owner_key = owner
        if owner_kind == "model" and isinstance(owner_key, int):
          state.open_model_calls.pop(owner_key, None)
        elif owner_kind == "tool" and isinstance(owner_key, str):
          state.open_tool_calls.pop(owner_key, None)
      delivered = await self._deliver_record(state, record)
      if delivered:
        state.undelivered_records.pop(record.sequence, None)
      if (
          not delivered
          and self._enforcing
          and self._audit_failure_mode == "deny"
      ):
        return None
      return record
    except asyncio.CancelledError:
      raise
    except Exception:
      logger.exception(
          "agent-hooks emission failed at %s; failing closed",
          ctx.get("interception_point"),
      )
      return None

  @staticmethod
  def _emitted_record(
      state: _InvocationState,
      ctx: AgentContext,
      record: Optional[InterceptionRecord],
  ) -> Optional[InterceptionRecord]:
    """Return the SDK record even when required sink delivery failed."""
    if record is not None:
      return record
    sequence = ctx.get("sequence")
    if isinstance(sequence, int):
      return state.undelivered_records.get(sequence)
    return None

  async def _deliver_record(
      self, state: _InvocationState, record: InterceptionRecord
  ) -> bool:
    """Log and durably deliver one record behind bounded admission."""
    logger.info(
        "agent-hooks decision",
        extra={
            "agent_hooks_interception_point": record.interception_point.value,
            "agent_hooks_decision": record.verdict.decision.value,
            "agent_hooks_reason": record.verdict.reason,
            "agent_hooks_session_id": record.session_id,
            "agent_hooks_sequence": record.sequence,
            "agent_hooks_adk_invocation_id": state.owner_identity[3],
            "agent_hooks_adk_session_id": state.owner_identity[2],
            "agent_hooks_run_nonce": state.run_nonce,
            "agent_hooks_resume_lineage_id": state.resume_lineage_id,
            "agent_hooks_input_identity": record.input_identity,
            "agent_hooks_enforced_identity": record.enforced_identity,
        },
    )
    if self._record_sink is None:
      return True
    if record.sequence in state.delivered_record_sequences:
      state.delivered_record_sequences.discard(record.sequence)
      state.late_delivery_sequences.discard(record.sequence)
      return True
    executor = self._audit_executor
    if executor is None:
      return False
    attempt_key = (state.run_nonce, record.sequence)
    future = self._audit_attempts.get(attempt_key)
    if future is None:
      try:
        await asyncio.wait_for(
            self._audit_slots.acquire(), timeout=self._audit_timeout
        )
      except asyncio.TimeoutError:
        self._log_audit_failure(state, record, "AdmissionTimeout")
        return False

      loop = asyncio.get_running_loop()
      future = loop.run_in_executor(executor, self._record_sink, record)
      self._audit_attempts[attempt_key] = future
      self._audit_futures.add(future)

      def release_slot(completed: asyncio.Future[None]) -> None:
        self._audit_futures.discard(completed)
        self._audit_slots.release()
        self._audit_attempts.pop(attempt_key, None)
        if (
            record.sequence in state.late_delivery_sequences
            and not completed.cancelled()
            and completed.exception() is None
        ):
          state.delivered_record_sequences.add(record.sequence)

      future.add_done_callback(release_slot)
    try:
      await asyncio.wait_for(
          asyncio.shield(future), timeout=self._audit_timeout
      )
      state.late_delivery_sequences.discard(record.sequence)
      state.delivered_record_sequences.discard(record.sequence)
      return True
    except asyncio.TimeoutError:
      if future.done():
        try:
          future.result()
        except Exception as error:
          self._log_audit_failure(state, record, type(error).__name__)
          return False
        state.late_delivery_sequences.discard(record.sequence)
        state.delivered_record_sequences.discard(record.sequence)
        return True
      state.late_delivery_sequences.add(record.sequence)
      self._log_audit_failure(state, record, "TimeoutError")
      return False
    except asyncio.CancelledError:
      if future.done():
        try:
          future.result()
        except Exception as error:
          state.late_delivery_sequences.discard(record.sequence)
          state.delivered_record_sequences.discard(record.sequence)
          self._log_audit_failure(state, record, type(error).__name__)
        else:
          state.delivered_record_sequences.add(record.sequence)
      else:
        state.late_delivery_sequences.add(record.sequence)
      raise
    except Exception as error:
      state.late_delivery_sequences.discard(record.sequence)
      state.delivered_record_sequences.discard(record.sequence)
      self._log_audit_failure(state, record, type(error).__name__)
      return False

  @staticmethod
  def _log_audit_failure(
      state: _InvocationState,
      record: InterceptionRecord,
      error_type: str,
  ) -> None:
    logger.error(
        "agent-hooks audit sink failed",
        extra={
            "agent_hooks_interception_point": record.interception_point.value,
            "agent_hooks_session_id": record.session_id,
            "agent_hooks_sequence": record.sequence,
            "agent_hooks_adk_invocation_id": state.owner_identity[3],
            "agent_hooks_adk_session_id": state.owner_identity[2],
            "agent_hooks_run_nonce": state.run_nonce,
            "agent_hooks_resume_lineage_id": state.resume_lineage_id,
            "agent_hooks_audit_error_type": error_type,
        },
    )

  @staticmethod
  def _blocked(record: Optional[InterceptionRecord]) -> bool:
    """Whether the record denies the guarded action (or is an engine error)."""
    return record is None or not record.proceeds

  @staticmethod
  def _session_terminally_denied(state: _InvocationState) -> bool:
    """Whether agent_startup or input denied this turn (agent-hooks §6/§6.1a).

    ADK ignores before_run_callback's return on some runner paths and treats
    on_user_message_callback's return as a replacement message, so a deny at
    either point is latched and re-enforced at every later guarded point.
    """
    return state.startup_denied or state.input_denied

  @staticmethod
  def _terminal_deny_record(
      state: _InvocationState,
  ) -> Optional[InterceptionRecord]:
    """The record for the latched agent_startup/input deny, if any."""
    return state.startup_record if state.startup_denied else state.input_record

  def _is_transform(self, record: InterceptionRecord) -> bool:
    # Only enforce mode actually rewrites ``ctx["target"]``; evaluate_only
    # records the verdict without transforming, so applying it here is lossy.
    return self._enforcing and record.verdict.transform is not None

  @staticmethod
  def _log_projection_failure(
      interception_point: str, error: _ContextProjectionError
  ) -> None:
    logger.warning(
        "agent-hooks context projection failed at %s (%s); failing closed",
        interception_point,
        type(error).__name__,
    )

  async def _emit_invalid_context(
      self,
      state: _InvocationState,
      ctx: AgentContext,
      *,
      agent_name: str,
      error: _ContextProjectionError,
  ) -> Optional[InterceptionRecord]:
    """Emit an unprojectable context so the SDK records context_invalid."""
    self._log_projection_failure(ctx["interception_point"], error)
    invalid_ctx = self._invalid_context(state, ctx, agent_name=agent_name)
    return await self._emit(state, invalid_ctx, agent_name=agent_name)

  def _invalid_context(
      self,
      state: _InvocationState,
      ctx: AgentContext,
      *,
      agent_name: str,
  ) -> AgentContext:
    """Build a serializable envelope that fails point-specific validation."""
    return {
        "spec": ctx.get("spec", "agent-hooks/0.1"),
        "interception_point": ctx["interception_point"],
        "timestamp": ctx.get("timestamp"),
        "sequence": ctx.get("sequence"),
        "agent": self._agent_envelope(agent_name),
        "session": {"id": state.run_nonce},
        "target": None,
    }

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
    source_call_id = tool_context.function_call_id
    violation: str | None = None
    if source_call_id in state.source_tool_call_ids:
      violation = "duplicate tool call id"
      call_id = _synth_call_id(tool.name, tool_args)
    elif (
        source_call_id
        and len(state.source_tool_call_ids) >= self._max_tracked_tool_calls
    ):
      violation = "tool call identity capacity exhausted"
      call_id = _synth_call_id(tool.name, tool_args)
    elif source_call_id:
      call_id = source_call_id
      state.source_tool_call_ids.add(source_call_id)
    else:
      call_id = _synth_call_id(tool.name, tool_args)
    setattr(tool_context, "_agent_hooks_call_id", call_id)
    setattr(tool_context, "_agent_hooks_call_id_violation", violation)
    return call_id

  def _post_tool_call_id(
      self,
      state: _InvocationState,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> str:
    """Call id for the post-tool record, correlated with its pre-tool id."""
    call_id = getattr(tool_context, "_agent_hooks_call_id", None)
    if isinstance(call_id, str):
      return call_id
    del state
    return _synth_call_id(tool.name, tool_args)

  async def _ensure_startup(
      self, invocation_context: InvocationContext
  ) -> _InvocationState:
    """Emit agent_startup once, before any input for this invocation."""
    state = self._state_for_invocation(invocation_context)
    async with state.startup_lock:
      if state.startup_emitted:
        return state
      agent = invocation_context.agent
      agent_name = agent.name if agent is not None else "unknown"
      ctx = state.builder.agent_startup(
          tools_registered=self._tool_names(agent)
      )
      record = await self._emit(state, ctx, agent_name=agent_name)
      state.startup_record = record
      state.startup_denied = self._blocked(record)
      state.startup_emitted = True
    return state

  # --- lifecycle callbacks ---------------------------------------------------

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """agent_startup: deny halts the run with a refusal message."""
    state = await self._ensure_startup(invocation_context)
    if self._session_terminally_denied(state):
      return _text_content(_MODEL_BLOCK_TEXT, role="model")
    return None

  @override
  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """input: deny replaces the user message; transform rewrites it."""
    state = await self._ensure_startup(invocation_context)
    agent = invocation_context.agent
    agent_name = agent.name if agent is not None else "unknown"
    if state.startup_denied:
      return _text_content(_MODEL_BLOCK_TEXT, role="user")
    try:
      content = _content_value(user_message)
    except _ContextProjectionError as error:
      ctx = state.builder.input(content=user_message)
      record = await self._emit_invalid_context(
          state, ctx, agent_name=agent_name, error=error
      )
      state.input_denied = True
      state.input_record = record
      return _text_content(_MODEL_BLOCK_TEXT, role="user")
    ctx = state.builder.input(content=content)
    record = await self._emit(state, ctx, agent_name=agent_name)
    if self._blocked(record):
      # ADK treats this return as a replacement message and still runs the
      # turn, so latch the deny and re-enforce it at the first model call
      # (agent-hooks §6: at input the turn MUST NOT begin).
      state.input_denied = True
      state.input_record = record
      return _text_content(_MODEL_BLOCK_TEXT, role="user")
    if record is not None and self._is_transform(record):
      try:
        return _target_content(ctx.get("target"), default_role="user")
      except _ContextProjectionError as error:
        self._log_projection_failure("input", error)
        state.input_denied = True
        state.input_record = None
        return _text_content(_MODEL_BLOCK_TEXT, role="user")
    return None

  @override
  async def before_model_callback(
      self, *, callback_context: CallbackContext, llm_request: LlmRequest
  ) -> Optional[LlmResponse]:
    """pre_model_call: deny (or transform, treated as deny) blocks the call."""
    state = self._state_for_context(callback_context)
    if self._session_terminally_denied(state):
      # A denied agent_startup or input halts the run even on runner paths that
      # ignore before_run_callback's return or treat on_user_message_callback's
      # return as a replacement message rather than a halt.
      return self._blocked_response(self._terminal_deny_record(state))
    try:
      messages = _request_messages(
          llm_request, agent_name=callback_context.agent_name
      )
    except _ContextProjectionError as error:
      ctx = state.builder.pre_model_call(
          model_id=llm_request.model or "unknown",
          messages=[{"role": "unprojectable", "content": llm_request}],
      )
      record = await self._emit_invalid_context(
          state,
          ctx,
          agent_name=callback_context.agent_name,
          error=error,
      )
      return self._blocked_response(record)
    request_key = id(callback_context.actions)
    request_id = f"mc-{uuid.uuid4().hex}"
    existing_request = state.open_model_calls.get(request_key)
    capacity_error: str | None = None
    if (
        existing_request is not None
        and existing_request[0] is callback_context.actions
    ):
      capacity_error = "duplicate model request identity"
    elif len(state.open_model_calls) >= self._max_tracked_model_calls:
      capacity_error = "model call tracking capacity exhausted"
    ctx = state.builder.pre_model_call(
        model_id=llm_request.model or "unknown",
        messages=messages,
        request_id=request_id,
    )
    if capacity_error is not None:
      self._log_projection_failure(
          "pre_model_call", _ContextProjectionError(capacity_error)
      )
      invalid_ctx = self._invalid_context(
          state, ctx, agent_name=callback_context.agent_name
      )
      record = await self._emit(
          state, invalid_ctx, agent_name=callback_context.agent_name
      )
      return self._blocked_response(record)
    state.open_model_calls[request_key] = (
        callback_context.actions,
        request_id,
        llm_request.model or "unknown",
    )
    try:
      record = await self._emit(
          state, ctx, agent_name=callback_context.agent_name
      )
    except BaseException:
      state.open_model_calls.pop(request_key, None)
      raise
    if self._blocked(record):
      state.open_model_calls.pop(request_key, None)
      return self._blocked_response(record)
    if record is not None and self._is_transform(record):
      state.open_model_calls.pop(request_key, None)
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
    request_key = id(callback_context.actions)
    open_request = state.open_model_calls.get(request_key)
    if open_request is None or open_request[0] is not callback_context.actions:
      logger.warning("suppressing unpaired agent-hooks post_model_call")
      return None
    _, request_id, requested_model_id = open_request
    response_parts = (
        list(llm_response.content.parts or [])
        if llm_response.content is not None
        else []
    )
    if len(response_parts) > _MAX_CONTENT_PARTS:
      ctx = state.builder.post_model_call(
          model_id=llm_response.model_version or requested_model_id,
          content="",
          tool_calls=[],
          finish_reason="error",
          request_id=request_id,
      )
      state.post_record_owners[ctx["sequence"]] = ("model", request_key)
      record = await self._emit_invalid_context(
          state,
          ctx,
          agent_name=callback_context.agent_name,
          error=_ContextProjectionError("maximum content part count exceeded"),
      )
      if self._emitted_record(state, ctx, record) is not None:
        state.open_model_calls.pop(request_key, None)
      return self._blocked_response(record)
    raw_tool_call_count = sum(
        1 for part in response_parts if part.function_call is not None
    )
    if raw_tool_call_count > self._max_tool_calls_per_response:
      ctx = state.builder.post_model_call(
          model_id=llm_response.model_version or requested_model_id,
          content="",
          tool_calls=[],
          finish_reason="error",
          request_id=request_id,
      )
      state.post_record_owners[ctx["sequence"]] = ("model", request_key)
      record = await self._emit_invalid_context(
          state,
          ctx,
          agent_name=callback_context.agent_name,
          error=_ContextProjectionError(
              "model response tool-call capacity exhausted"
          ),
      )
      if self._emitted_record(state, ctx, record) is not None:
        state.open_model_calls.pop(request_key, None)
      return self._blocked_response(record)
    try:
      budget = _ProjectionBudget()
      tool_calls = _response_tool_calls(llm_response, budget)
      content = _content_value(llm_response.content, budget)
    except _ContextProjectionError as error:
      ctx = state.builder.post_model_call(
          model_id=llm_response.model_version or requested_model_id,
          content=llm_response,
          tool_calls=[],
          finish_reason="error",
          request_id=request_id,
      )
      state.post_record_owners[ctx["sequence"]] = ("model", request_key)
      record = await self._emit_invalid_context(
          state,
          ctx,
          agent_name=callback_context.agent_name,
          error=error,
      )
      if self._emitted_record(state, ctx, record) is not None:
        state.open_model_calls.pop(request_key, None)
      return self._blocked_response(record)
    ctx = state.builder.post_model_call(
        model_id=llm_response.model_version or requested_model_id,
        content=content,
        tool_calls=tool_calls,
        request_id=request_id,
        finish_reason=_response_finish_reason(
            llm_response, has_tool_calls=bool(tool_calls)
        ),
    )
    state.post_record_owners[ctx["sequence"]] = ("model", request_key)
    record = await self._emit(
        state, ctx, agent_name=callback_context.agent_name
    )
    if self._emitted_record(state, ctx, record) is not None:
      state.open_model_calls.pop(request_key, None)
    if self._blocked(record):
      return self._blocked_response(record)
    if record is not None and self._is_transform(record):
      try:
        content = _target_content(ctx.get("target"), default_role="model")
      except _ContextProjectionError as error:
        self._log_projection_failure("post_model_call", error)
        return self._blocked_response(None)
      return llm_response.model_copy(update={"content": content})
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
    tracked_call_count = len(state.open_tool_calls)
    call_id_violation = getattr(
        tool_context, "_agent_hooks_call_id_violation", None
    )
    if call_id_violation or tracked_call_count >= self._max_tracked_tool_calls:
      error = _ContextProjectionError(
          call_id_violation or "tool call tracking capacity exhausted"
      )
      ctx = state.builder.pre_tool_call(
          call_id=call_id, name=tool.name, args={}
      )
      record = await self._emit_invalid_context(
          state, ctx, agent_name=tool_context.agent_name, error=error
      )
      return self._blocked_tool_result(record)
    try:
      projected_args = _json_safe(tool_args)
    except _ContextProjectionError as error:
      ctx = state.builder.pre_tool_call(
          call_id=call_id, name=tool.name, args=tool_args
      )
      record = await self._emit_invalid_context(
          state, ctx, agent_name=tool_context.agent_name, error=error
      )
      return self._blocked_tool_result(record)
    ctx = state.builder.pre_tool_call(
        call_id=call_id, name=tool.name, args=projected_args
    )
    state.open_tool_calls[call_id] = (tool.name, projected_args)
    try:
      record = await self._emit(state, ctx, agent_name=tool_context.agent_name)
    except BaseException:
      state.open_tool_calls.pop(call_id, None)
      raise
    if self._blocked(record):
      state.open_tool_calls.pop(call_id, None)
      return self._blocked_tool_result(record)
    if record is not None and self._is_transform(record):
      new_args = ctx.get("target")
      if not isinstance(new_args, dict):
        logger.warning(
            "agent-hooks pre_tool_call transform did not yield an args "
            "object; failing closed"
        )
        state.open_tool_calls.pop(call_id, None)
        return self._blocked_tool_result(record)
      # Mutating tool_args in place propagates to the actual tool call.
      tool_args.clear()
      tool_args.update(new_args)
    # The tool will run; it now owes exactly one post_tool_call (§3.1(5)).
    state.open_tool_calls[call_id] = (tool.name, _json_safe(tool_args))
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
    if call_id not in state.open_tool_calls:
      logger.warning("suppressing unpaired agent-hooks post_tool_call")
      return None
    try:
      budget = _ProjectionBudget()
      projected_args = _json_safe(tool_args, _budget=budget)
      projected_result = _json_safe(result, _budget=budget)
    except _ContextProjectionError as error:
      ctx = state.builder.post_tool_call(
          call_id=call_id,
          name=tool.name,
          args=tool_args,
          value=result,
      )
      state.post_record_owners[ctx["sequence"]] = ("tool", call_id)
      record = await self._emit_invalid_context(
          state, ctx, agent_name=tool_context.agent_name, error=error
      )
      if self._emitted_record(state, ctx, record) is not None:
        state.open_tool_calls.pop(call_id, None)
      return self._blocked_tool_result(record)
    ctx = state.builder.post_tool_call(
        call_id=call_id,
        name=tool.name,
        args=projected_args,
        value=projected_result,
    )
    state.post_record_owners[ctx["sequence"]] = ("tool", call_id)
    record = await self._emit(state, ctx, agent_name=tool_context.agent_name)
    if self._emitted_record(state, ctx, record) is not None:
      state.open_tool_calls.pop(call_id, None)
    if self._blocked(record):
      return self._blocked_tool_result(record)
    if record is not None and self._is_transform(record):
      new_value = ctx.get("target")
      return new_value if isinstance(new_value, dict) else {"result": new_value}
    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict[str, Any]]:
    """post_tool_call for a tool that raised (agent-hooks §3.1(5)).

    ADK routes a tool exception here instead of after_tool_callback, so a
    dispatched tool that errors would otherwise leave its pre_tool_call
    unpaired. The paired record is emitted for audit and the original error is
    left to propagate (the errored result is discarded per §6.1); the plugin
    never turns a tool failure into a silent success.
    """
    state = self._state_for_context(tool_context)
    call_id = self._post_tool_call_id(state, tool, tool_args, tool_context)
    if call_id not in state.open_tool_calls:
      # No matching pre_tool_call (e.g. an unresolved tool name that never
      # reached before_tool_callback); a post here would be unpaired.
      return None
    try:
      projected_args = _json_safe(tool_args)
    except _ContextProjectionError as projection_error:
      ctx = state.builder.post_tool_call(
          call_id=call_id,
          name=tool.name,
          args=tool_args,
          value={"error": type(error).__name__},
          is_error=True,
      )
      state.post_record_owners[ctx["sequence"]] = ("tool", call_id)
      record = await self._emit_invalid_context(
          state,
          ctx,
          agent_name=tool_context.agent_name,
          error=projection_error,
      )
      if self._emitted_record(state, ctx, record) is not None:
        state.open_tool_calls.pop(call_id, None)
      return None
    ctx = state.builder.post_tool_call(
        call_id=call_id,
        name=tool.name,
        args=projected_args,
        value={"error": type(error).__name__},
        is_error=True,
    )
    state.post_record_owners[ctx["sequence"]] = ("tool", call_id)
    record = await self._emit(state, ctx, agent_name=tool_context.agent_name)
    if self._emitted_record(state, ctx, record) is not None:
      state.open_tool_calls.pop(call_id, None)
    return None

  @override
  async def on_model_error_callback(
      self,
      *,
      callback_context: CallbackContext,
      llm_request: LlmRequest,
      error: Exception,
  ) -> Optional[LlmResponse]:
    """post_model_call for a model request that raised (agent-hooks §3.1(4)).

    before_model_callback short-circuits a blocked call, so this fires only for
    a dispatched request that errored; it emits the paired post_model_call for
    audit and lets ADK propagate the error (the errored response is discarded
    per §6.1).
    """
    state = self._state_for_context(callback_context)
    request_key = id(callback_context.actions)
    open_request = state.open_model_calls.get(request_key)
    if open_request is None or open_request[0] is not callback_context.actions:
      return None
    _, request_id, requested_model_id = open_request
    ctx = state.builder.post_model_call(
        model_id=llm_request.model or requested_model_id,
        content="",
        tool_calls=[],
        finish_reason="error",
        request_id=request_id,
    )
    state.post_record_owners[ctx["sequence"]] = ("model", request_key)
    record = await self._emit(
        state, ctx, agent_name=callback_context.agent_name
    )
    if self._emitted_record(state, ctx, record) is not None:
      state.open_model_calls.pop(request_key, None)
    return None

  @override
  async def on_event_callback(
      self, *, invocation_context: InvocationContext, event: Event
  ) -> Optional[Event]:
    """output: govern the final response event (deny/transform its content)."""
    if not event.is_final_response():
      return None
    state = self._state_for_invocation(invocation_context)
    if self._session_terminally_denied(state):
      # A denied agent_startup/input bars the output point too (§6.1a/§6); the
      # blocked response already produced upstream still reaches the caller.
      return None
    agent = invocation_context.agent
    agent_name = event.author or (
        agent.name if agent is not None else "unknown"
    )
    try:
      content = _content_value(event.content)
    except _ContextProjectionError as error:
      ctx = state.builder.output(content=event.content)
      await self._emit_invalid_context(
          state, ctx, agent_name=agent_name, error=error
      )
      return event.model_copy(
          update={"content": _text_content(_MODEL_BLOCK_TEXT, role="model")}
      )
    ctx = state.builder.output(content=content)
    record = await self._emit(state, ctx, agent_name=agent_name)
    if self._blocked(record):
      return event.model_copy(
          update={"content": _text_content(_MODEL_BLOCK_TEXT, role="model")}
      )
    if record is not None and self._is_transform(record):
      try:
        content = _target_content(ctx.get("target"), default_role="model")
      except _ContextProjectionError as error:
        self._log_projection_failure("output", error)
        return event.model_copy(
            update={"content": _text_content(_MODEL_BLOCK_TEXT, role="model")}
        )
      return event.model_copy(update={"content": content})
    return None

  @override
  async def on_run_complete_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Emit completed agent_shutdown after all ordinary cleanup succeeds."""
    state = self._states.get(invocation_context.run_nonce)
    reason = (
        "error"
        if state is not None
        and (
            self._session_terminally_denied(state)
            or bool(state.open_model_calls)
            or bool(state.open_tool_calls)
        )
        else "completed"
    )
    await self._finalize_invocation(invocation_context.run_nonce, reason)

  @override
  async def on_run_error_callback(
      self, *, invocation_context: InvocationContext, error: Exception
  ) -> None:
    """Evict per-invocation state on an error path (notification-only)."""
    await self._finalize_invocation(
        invocation_context.run_nonce,
        "error",
        error=type(error).__name__,
    )

  @override
  async def on_run_cancelled_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Close and evict a cancelled invocation without consuming cancellation."""
    await self._finalize_invocation(invocation_context.run_nonce, "cancelled")

  async def _finalize_invocation(
      self, run_nonce: str, reason: str, *, error: str | None = None
  ) -> None:
    """Await one owned terminal drain; error/cancellation outrank completion."""
    state = self._states.get(run_nonce)
    if state is None:
      return
    async with state.finalization_lock:
      self._raise_terminal_priority(state, reason, error=error)
      finalization_failed = (
          state.finalization_task is not None
          and state.finalization_task.done()
          and state.finalization_task.exception() is not None
      )
      if state.finalization_task is None or finalization_failed:
        state.phase = "finalizing"
        state.finalization_task = asyncio.create_task(
            self._drain_finalization(state, run_nonce)
        )
      finalization_task = state.finalization_task
    try:
      await asyncio.shield(finalization_task)
    except asyncio.CancelledError:
      await asyncio.shield(finalization_task)
      raise

  @staticmethod
  def _raise_terminal_priority(
      state: _InvocationState, reason: str, *, error: str | None = None
  ) -> None:
    priorities = {"completed": 0, "error": 1, "cancelled": 1}
    current = state.terminal_reason
    if current is None or priorities[reason] > priorities[current]:
      state.terminal_reason = reason  # type: ignore[assignment]
    if error is not None:
      state.terminal_error = error

  async def _drain_finalization(
      self, state: _InvocationState, run_nonce: str
  ) -> None:
    """Pair open actions, emit terminal state, and release all retention."""
    completed = False
    try:
      async with state.emission_lock:
        await self._redeliver_undelivered_records(state)
        terminal_reason = state.terminal_reason or "error"
        action_reason = (
            "cancelled" if terminal_reason == "cancelled" else "error"
        )
        error_name = (
            "CancelledError"
            if terminal_reason == "cancelled"
            else state.terminal_error or "RunError"
        )
        while state.open_model_calls:
          request_key, (_, request_id, model_id) = next(
              iter(state.open_model_calls.items())
          )
          ctx = state.builder.post_model_call(
              model_id=model_id,
              content="",
              tool_calls=[],
              finish_reason=action_reason,
              request_id=request_id,
          )
          record = await self._emit_unlocked(
              state, ctx, agent_name=state.agent_name
          )
          if self._emitted_record(state, ctx, record) is None:
            raise RuntimeError("Agent Hooks model post record was not emitted")
          state.open_model_calls.pop(request_key, None)
        for call_id, (tool_name, tool_args) in list(
            state.open_tool_calls.items()
        ):
          ctx = state.builder.post_tool_call(
              call_id=call_id,
              name=tool_name,
              args=tool_args,
              value={"error": error_name},
              is_error=True,
          )
          record = await self._emit_unlocked(
              state, ctx, agent_name=state.agent_name
          )
          if self._emitted_record(state, ctx, record) is None:
            raise RuntimeError("Agent Hooks tool post record was not emitted")
          state.open_tool_calls.pop(call_id, None)
        if not state.terminal_record_emitted:
          terminal_reason = state.terminal_reason or "error"
          summary = (
              {"error": state.terminal_error}
              if state.terminal_error is not None
              else {}
          )
          ctx = state.builder.agent_shutdown(reason=terminal_reason, **summary)
          record = await self._emit_unlocked(
              state, ctx, agent_name=state.agent_name
          )
          if self._emitted_record(state, ctx, record) is None:
            raise RuntimeError("Agent Hooks shutdown record was not emitted")
          state.terminal_record_emitted = True
        await self._redeliver_undelivered_records(state)
        state.late_delivery_sequences.clear()
        state.delivered_record_sequences.clear()
        completed = True
    except Exception:
      logger.exception("agent_shutdown emission failed for run %s", run_nonce)
      raise
    finally:
      state.phase = "closed"
      state.run_identity_token.close()
      if completed and self._states.get(run_nonce) is state:
        self._states.pop(run_nonce, None)

  async def _redeliver_undelivered_records(
      self, state: _InvocationState
  ) -> None:
    """Retry required audit records before terminal state is released."""
    for sequence, record in list(state.undelivered_records.items()):
      if not await self._deliver_record(state, record):
        raise RuntimeError(
            f"Agent Hooks audit record {sequence} was not delivered"
        )
      state.undelivered_records.pop(sequence, None)

  @override
  async def close(self) -> None:
    """Finalize any residual invocation state as cancelled."""
    self._closed = True
    close_failed = (
        self._close_task is not None
        and self._close_task.done()
        and (
            self._close_task.cancelled()
            or self._close_task.exception() is not None
        )
    )
    if self._close_task is None or close_failed:
      self._close_task = asyncio.create_task(self._close_states())
    close_task = self._close_task
    try:
      await asyncio.shield(close_task)
    except asyncio.CancelledError:
      await asyncio.shield(close_task)
      raise
    except Exception:
      if self._close_task is close_task:
        self._close_task = None
      raise

  async def _close_states(self) -> None:
    """Finalize the bounded state snapshot owned by :meth:`close`."""
    results = await asyncio.gather(
        *(
            self._finalize_invocation(run_nonce, "cancelled")
            for run_nonce in list(self._states)
        ),
        return_exceptions=True,
    )
    failures = [
        result for result in results if isinstance(result, BaseException)
    ]
    if failures:
      raise RuntimeError(
          f"{len(failures)} Agent Hooks invocation(s) failed to finalize"
      ) from failures[0]
    await self._shutdown_audit_executor()

  async def _shutdown_audit_executor(self) -> None:
    """Drain bounded audit jobs or surface the retained running work."""
    executor = self._audit_executor
    if executor is None:
      return
    pending = set(self._audit_futures)
    if pending:
      _, pending = await asyncio.wait(
          pending, timeout=self._audit_shutdown_timeout
      )
    if pending:
      raise RuntimeError(
          f"{len(pending)} Agent Hooks audit record(s) did not drain"
      )
    executor.shutdown(wait=True, cancel_futures=True)
    self._audit_executor = None
    self._audit_attempts.clear()

  # --- block-result builders -------------------------------------------------

  def _blocked_response(
      self, record: Optional[InterceptionRecord]
  ) -> LlmResponse:
    del record
    return LlmResponse(
        content=_text_content(_MODEL_BLOCK_TEXT, role="model"),
        custom_metadata={
            "agent_hooks_blocked": True,
            "reason": _MODEL_BLOCK_REASON,
        },
    )

  def _blocked_tool_result(
      self, record: Optional[InterceptionRecord]
  ) -> dict[str, Any]:
    del record
    return {
        "error": "blocked by policy",
        "agent_hooks_blocked": True,
        "reason": _MODEL_BLOCK_REASON,
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
