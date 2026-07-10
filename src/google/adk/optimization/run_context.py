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

"""Run-scoped usage, budget, and cancellation controls for agent optimizers.

Experimental. All names in this module are subject to change until the
callback/error semantics have at least one downstream implementation.

An :class:`OptimizationRunContext` is an optional, one-shot, caller-owned
object attached to a single ``AgentOptimizer.optimize(...)`` run. It records
every *logical* optimizer-owned model invocation (control-plane metadata only,
never prompt or response content), enforces configured call/token budgets, and
carries a cooperative cancellation signal that instrumented optimizers observe
between logical model invocations.

The ledger records provider-reported usage truthfully: missing token counters
stay ``None`` and are classified via :class:`UsageCoverage`, never coerced to
zero. The final :class:`OptimizationRunSnapshot` is readable from the context
on success *and* failure, so a governance caller can persist the attempt in a
``finally`` block without parsing logs.
"""

from __future__ import annotations

import enum
import threading
import time
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class UsageCoverage(str, enum.Enum):
  """How completely the provider reported token usage for one logical call."""

  VERIFIED = "verified"
  """The provider supplied an authoritative total token count."""

  PARTIAL = "partial"
  """At least one token counter was supplied without an authoritative total."""

  UNREPORTED = "unreported"
  """No token counter was supplied."""


class ModelCallStage(str, enum.Enum):
  """Which optimizer stage issued the logical model invocation."""

  CANDIDATE_GENERATION = "candidate_generation"
  REFLECTION = "reflection"


class ModelCallState(str, enum.Enum):
  """Terminal state of one logical model invocation."""

  COMPLETED = "completed"
  PROVIDER_ERROR = "provider_error"
  CANCELLED = "cancelled"
  BUDGET_EXCEEDED = "budget_exceeded"


class ModelCallEvent(BaseModel):
  """Control-plane record of one logical model invocation.

  Prompt and response content are deliberately not part of this event;
  content capture remains an explicit evaluator/sampler concern.
  """

  sequence: int
  stage: ModelCallStage
  requested_model: Optional[str] = None
  returned_model_version: Optional[str] = None
  start_time: float
  end_time: Optional[float] = None
  state: Optional[ModelCallState] = None
  prompt_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  reasoning_tokens: Optional[int] = None
  cached_tokens: Optional[int] = None
  tool_use_tokens: Optional[int] = None
  total_tokens: Optional[int] = None
  usage_coverage: Optional[UsageCoverage] = None
  error_message: Optional[str] = None


class OptimizationRunSnapshot(BaseModel):
  """Immutable view of the run ledger at a point in time."""

  events: list[ModelCallEvent] = Field(default_factory=list)
  started_calls: int = 0
  completed_calls: int = 0
  cumulative_total_tokens: int = 0
  """Sum of provider-reported authoritative totals (verified calls only)."""

  usage_coverage: UsageCoverage = UsageCoverage.UNREPORTED
  """VERIFIED only if every completed call was verified; PARTIAL if any call
  reported any counter; UNREPORTED otherwise."""

  cancel_requested: bool = False
  cancel_reason: Optional[str] = None
  terminal_control_state: Optional[str] = None
  """Run-level control outcome (e.g. ``call_budget_rejected``,
  ``budget_exceeded``, ``cancelled``); ``None`` while running or on a
  normal completion."""


class OptimizationBudgets(BaseModel):
  """Configured ceilings for optimizer-owned logical model invocations."""

  max_model_calls: Optional[int] = Field(
      default=None,
      description=(
          "Maximum number of logical optimizer-owned model invocations that"
          " may start. Checked before each call; a preflight rejection does"
          " not create a model-call event."
      ),
  )
  max_total_tokens: Optional[int] = Field(
      default=None,
      description=(
          "Hard ceiling on cumulative provider-reported total tokens."
          " Checked after each terminal response is committed; an over-budget"
          " final call is committed first, then the run terminates."
      ),
  )
  on_budget_exceeded: Literal["raise", "return_partial"] = Field(
      default="raise",
      description=(
          "Terminal behavior on budget exhaustion. Both modes stop scheduling"
          " immediately after the final in-flight call settles. 'raise' raises"
          " OptimizationBudgetExceeded; 'return_partial' lets the optimizer"
          " return its best-so-far result marked terminal_status="
          "'budget_exceeded' (never an unmarked success)."
      ),
  )


class OptimizationRunContextError(Exception):
  """Base class for run-context errors carrying the final snapshot."""

  def __init__(self, message: str, snapshot: OptimizationRunSnapshot):
    super().__init__(message)
    self.snapshot = snapshot


class OptimizationBudgetExceeded(OptimizationRunContextError):
  """A configured call or token budget was exhausted."""


class OptimizationCancelledError(OptimizationRunContextError):
  """The run stopped because the context's cancellation was requested."""


class UnsupportedOptimizationContextError(Exception):
  """The optimizer does not support run-context instrumentation."""


class ContextAlreadyAttachedError(Exception):
  """A one-shot context was attached to more than one optimization run."""


class _CallHandle:
  """Opaque handle for one in-flight logical model invocation."""

  def __init__(self, event: ModelCallEvent):
    self._event = event
    self._closed = False


class OptimizationRunContext:
  """One-shot, concurrency-safe ledger and control channel for one run.

  Instrumented optimizers call :meth:`begin_model_call` immediately before a
  logical ``BaseLlm.generate_content_async()`` invocation and
  :meth:`end_model_call` with the terminal response, in-band error, or raised
  provider error. Governance callers own the context, may call
  :meth:`request_cancel` from any thread, and read :meth:`snapshot` at any
  time, including after a failure.
  """

  def __init__(self, budgets: Optional[OptimizationBudgets] = None):
    self._budgets = budgets or OptimizationBudgets()
    self._lock = threading.Lock()
    self._events: list[ModelCallEvent] = []
    self._started_calls = 0
    self._completed_calls = 0
    self._cumulative_total_tokens = 0
    self._cancel_requested = False
    self._cancel_reason: Optional[str] = None
    self._terminal_control_state: Optional[str] = None
    self._attached_owner: Optional[object] = None

  @property
  def budgets(self) -> OptimizationBudgets:
    return self._budgets

  def attach(self, owner: object) -> None:
    """Binds the context to one optimization run; reuse is rejected."""
    with self._lock:
      if self._attached_owner is not None:
        raise ContextAlreadyAttachedError(
            "OptimizationRunContext is one-shot: it is already attached to an"
            " optimization run and cannot be reused or shared."
        )
      self._attached_owner = owner

  def request_cancel(self, reason: str = "requested") -> None:
    """Thread-safe, idempotent cooperative cancellation signal."""
    with self._lock:
      if not self._cancel_requested:
        self._cancel_requested = True
        self._cancel_reason = reason

  @property
  def cancel_requested(self) -> bool:
    with self._lock:
      return self._cancel_requested

  def raise_if_cancelled(self) -> None:
    """Raises ``OptimizationCancelledError`` if cancellation was requested."""
    with self._lock:
      if not self._cancel_requested:
        return
      self._terminal_control_state = "cancelled"
      snapshot = self._snapshot_locked()
      reason = self._cancel_reason
    raise OptimizationCancelledError(
        f"Optimization cancelled: {reason}", snapshot
    )

  def begin_model_call(
      self,
      stage: ModelCallStage,
      requested_model: Optional[str] = None,
  ) -> _CallHandle:
    """Reserves and records the start of one logical model invocation.

    Checks cancellation and the logical-call budget *before* the invocation
    starts. A preflight rejection does not create a model-call event and does
    not increment the started-call count; it records the run-level control
    state and raises.
    """
    with self._lock:
      if self._cancel_requested:
        self._terminal_control_state = "cancelled"
        snapshot = self._snapshot_locked()
        reason = self._cancel_reason
        raise OptimizationCancelledError(
            f"Optimization cancelled: {reason}", snapshot
        )
      max_calls = self._budgets.max_model_calls
      if max_calls is not None and self._started_calls >= max_calls:
        self._terminal_control_state = "call_budget_rejected"
        snapshot = self._snapshot_locked()
        raise OptimizationBudgetExceeded(
            f"Logical model-call budget exhausted ({max_calls}).", snapshot
        )
      self._started_calls += 1
      event = ModelCallEvent(
          sequence=self._started_calls,
          stage=stage,
          requested_model=requested_model,
          start_time=time.monotonic(),
      )
      self._events.append(event)
      return _CallHandle(event)

  def end_model_call(
      self,
      handle: _CallHandle,
      *,
      usage_metadata: Any = None,
      returned_model_version: Optional[str] = None,
      error_message: Optional[str] = None,
  ) -> None:
    """Commits the terminal outcome of one logical model invocation.

    Token usage is committed before budget enforcement: if the new cumulative
    total crosses the hard ceiling, the completed call is persisted first and
    ``OptimizationBudgetExceeded`` is raised immediately afterwards, so an
    over-budget final call can never produce an unmarked success. A terminal
    provider error closes the call as ``provider_error``, preserves any
    reported usage, and re-raising the primary error remains the caller's
    responsibility.
    """
    if handle._closed:
      return
    handle._closed = True
    event = handle._event
    with self._lock:
      event.end_time = time.monotonic()
      event.returned_model_version = returned_model_version
      _apply_usage(event, usage_metadata)
      if error_message is not None:
        event.state = ModelCallState.PROVIDER_ERROR
        event.error_message = error_message
      else:
        event.state = ModelCallState.COMPLETED
      self._completed_calls += 1
      if event.total_tokens is not None:
        self._cumulative_total_tokens += event.total_tokens
      max_tokens = self._budgets.max_total_tokens
      over_budget = (
          max_tokens is not None
          and self._cumulative_total_tokens > max_tokens
      )
      if over_budget:
        self._terminal_control_state = "budget_exceeded"
        snapshot = self._snapshot_locked()
    if over_budget:
      raise OptimizationBudgetExceeded(
          f"Token budget exhausted ({max_tokens}).", snapshot
      )

  def snapshot(self) -> OptimizationRunSnapshot:
    with self._lock:
      return self._snapshot_locked()

  def _snapshot_locked(self) -> OptimizationRunSnapshot:
    completed = [
        e for e in self._events if e.state is not None
    ]
    if completed and all(
        e.usage_coverage == UsageCoverage.VERIFIED for e in completed
    ):
      run_coverage = UsageCoverage.VERIFIED
    elif any(
        e.usage_coverage in (UsageCoverage.VERIFIED, UsageCoverage.PARTIAL)
        for e in completed
    ):
      run_coverage = UsageCoverage.PARTIAL
    else:
      run_coverage = UsageCoverage.UNREPORTED
    return OptimizationRunSnapshot(
        events=[e.model_copy(deep=True) for e in self._events],
        started_calls=self._started_calls,
        completed_calls=self._completed_calls,
        cumulative_total_tokens=self._cumulative_total_tokens,
        usage_coverage=run_coverage,
        cancel_requested=self._cancel_requested,
        cancel_reason=self._cancel_reason,
        terminal_control_state=self._terminal_control_state,
    )


def _apply_usage(event: ModelCallEvent, usage_metadata: Any) -> None:
  """Copies provider-reported token counters onto the event, truthfully.

  Missing counters stay ``None``. Coverage: ``verified`` iff the provider
  supplied ``total_token_count``; ``partial`` if any other counter was
  supplied; ``unreported`` otherwise.
  """
  if usage_metadata is None:
    event.usage_coverage = UsageCoverage.UNREPORTED
    return

  def _get(name: str) -> Optional[int]:
    value = getattr(usage_metadata, name, None)
    return int(value) if isinstance(value, (int, float)) else None

  event.prompt_tokens = _get("prompt_token_count")
  event.output_tokens = _get("candidates_token_count")
  event.reasoning_tokens = _get("thoughts_token_count")
  event.cached_tokens = _get("cached_content_token_count")
  event.tool_use_tokens = _get("tool_use_prompt_token_count")
  event.total_tokens = _get("total_token_count")
  if event.total_tokens is not None:
    event.usage_coverage = UsageCoverage.VERIFIED
  elif any(
      v is not None
      for v in (
          event.prompt_tokens,
          event.output_tokens,
          event.reasoning_tokens,
          event.cached_tokens,
          event.tool_use_tokens,
      )
  ):
    event.usage_coverage = UsageCoverage.PARTIAL
  else:
    event.usage_coverage = UsageCoverage.UNREPORTED


class OptimizerCapabilities(BaseModel):
  """What run-context instrumentation an optimizer actually supports.

  Capabilities describe ADK instrumentation, not provider metadata guarantees
  or transport-attempt visibility. Conservative defaults (all ``False``) are
  correct for optimizers that predate or do not implement the run-context
  seam, so a governance wrapper can reject an incompatible optimizer at
  preflight instead of discovering opacity after spend occurs.
  """

  model_calls_observable: bool = False
  """Optimizer-owned logical model invocations are recorded on the context."""

  call_limits_enforceable: bool = False
  """Configured logical-invocation limits stop the next call from starting."""

  cooperative_cancellation: bool = False
  """``request_cancel`` is observed at the documented boundaries."""

  sampler_usage_included: bool = False
  """Always ``False`` for this context: candidate execution and evaluator
  inference belong to the supplied sampler/Runner, not the optimizer."""
