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
never prompt or response content), enforces configured call/token budgets,
carries a cooperative cancellation signal, and owns an explicit terminal
state machine: every governed run finalizes exactly once as ``completed``,
``budget_exceeded``, ``cancelled``, or ``failed`` (first terminal transition
wins).

The ledger records provider-reported usage truthfully: missing token counters
stay ``None`` and are classified via usage coverage, never coerced to zero.
Snapshots are immutable values; the final snapshot is readable from the
context on success *and* failure, so a governance caller can persist the
attempt in a ``finally`` block without parsing logs.
"""

from __future__ import annotations

import enum
import math
import re
import threading
import time
from typing import Any
from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

# Initially defined optimizer stages. The stage field is an extensible string
# so third-party optimizers can record their own stages without an ADK
# release; these constants cover the built-in optimizers.
STAGE_CANDIDATE_GENERATION = "candidate_generation"
STAGE_REFLECTION = "reflection"


class UsageCoverage(str, enum.Enum):
  """How completely the provider reported token usage for one logical call."""

  VERIFIED = "verified"
  """The provider supplied an authoritative total token count."""

  PARTIAL = "partial"
  """At least one token counter was supplied without an authoritative total."""

  UNREPORTED = "unreported"
  """No token counter was supplied."""


class ModelCallState(str, enum.Enum):
  """Terminal state of one logical model invocation (call status).

  Budget exhaustion is a *run* status, not a call status: the triggering
  successful call remains ``completed`` while the run becomes
  ``budget_exceeded``.
  """

  COMPLETED = "completed"
  PROVIDER_ERROR = "provider_error"
  CANCELLED = "cancelled"
  ABORTED = "aborted"
  """The invocation was admitted but aborted locally (request construction,
  scheduling, or other optimizer-side failure) before or during execution --
  distinct from a provider failure and from cancellation."""


class RunStatus(str, enum.Enum):
  """Terminal status of the optimization run (run status)."""

  COMPLETED = "completed"
  BUDGET_EXCEEDED = "budget_exceeded"
  CANCELLED = "cancelled"
  FAILED = "failed"


class TokenBudgetStatus(str, enum.Enum):
  """Compliance of actual token usage with the configured reported-token limit.

  ``indeterminate`` whenever any completed call's usage coverage is partial or
  unreported: missing totals are never interpreted as proof that actual usage
  stayed below the ceiling. Downstream policy may fail closed on that state.
  """

  WITHIN_LIMIT = "within_limit"
  EXCEEDED = "exceeded"
  INDETERMINATE = "indeterminate"


class ModelCallEvent(BaseModel):
  """Immutable control-plane record of one logical model invocation.

  Prompt and response content are deliberately not part of this event;
  content capture remains an explicit evaluator/sampler concern. Error
  metadata is structured and sanitized: a short provider error code and the
  exception type name, never raw exception or payload text.
  """

  model_config = ConfigDict(frozen=True)

  sequence: int
  stage: str
  requested_model: Optional[str] = None
  returned_model_version: Optional[str] = None
  start_time: float
  """Wall-clock (epoch seconds): durable and comparable across workers."""

  end_time: Optional[float] = None
  state: Optional[ModelCallState] = None
  prompt_tokens: Optional[int] = None
  output_tokens: Optional[int] = None
  reasoning_tokens: Optional[int] = None
  cached_tokens: Optional[int] = None
  tool_use_tokens: Optional[int] = None
  total_tokens: Optional[int] = None
  usage_coverage: Optional[UsageCoverage] = None
  error_code: Optional[str] = None
  error_type: Optional[str] = None


class OptimizationBudgets(BaseModel):
  """Configured ceilings for optimizer-owned logical model invocations.

  Immutable by construction, so limits cannot change mid-run.
  """

  model_config = ConfigDict(frozen=True)

  max_model_calls: Optional[int] = Field(
      default=None,
      ge=0,
      description=(
          "Maximum number of logical optimizer-owned model invocations that"
          " may start. Checked before each call via atomic slot admission; a"
          " rejected reservation is not a call event."
      ),
  )
  max_provider_reported_tokens: Optional[int] = Field(
      default=None,
      ge=0,
      description=(
          "Ceiling on cumulative provider-reported authoritative total"
          " tokens. Checked after each terminal response is committed; the"
          " over-budget final call is committed first, then the run"
          " terminates. Deliberately not a hard bound on billing or physical"
          " tokens: when any call's usage is partial or unreported,"
          " actual-token compliance is indeterminate."
      ),
  )
  on_budget_exceeded: Literal["raise", "return_partial"] = Field(
      default="raise",
      description=(
          "Terminal behavior on budget exhaustion. Both modes stop scheduling"
          " immediately after the final in-flight call settles. 'raise'"
          " raises OptimizationBudgetExceeded; 'return_partial' lets the"
          " optimizer return its best-so-far result while the caller-owned"
          " snapshot (run_status='budget_exceeded') remains the authoritative"
          " record."
      ),
  )


class OptimizationRunSnapshot(BaseModel):
  """Immutable view of the run ledger at a point in time."""

  model_config = ConfigDict(frozen=True)

  events: tuple[ModelCallEvent, ...] = ()
  budgets: OptimizationBudgets = Field(default_factory=OptimizationBudgets)
  started_calls: int = 0
  completed_calls: int = 0
  """Count of settled (closed) calls, whatever their terminal call status --
  completed, provider_error, or cancelled."""

  cumulative_total_tokens: int = 0
  """Sum of provider-reported authoritative totals (verified calls only)."""

  usage_coverage: UsageCoverage = UsageCoverage.UNREPORTED
  """VERIFIED only if every closed call was verified; PARTIAL if any call
  reported any counter; UNREPORTED otherwise."""

  token_budget_status: Optional[TokenBudgetStatus] = None
  """Only set when a reported-token limit is configured."""

  cancel_requested: bool = False
  cancel_reason: Optional[str] = None
  run_status: Optional[RunStatus] = None
  """Terminal run status; ``None`` only while the run is still in progress."""

  terminal_sequence: Optional[int] = None
  """Sequence of the logical invocation that triggered the terminal state,
  when one did."""

  terminal_error_code: Optional[str] = None
  terminal_error_type: Optional[str] = None


class OptimizationRunContextError(Exception):
  """Base class for run-context errors carrying the final snapshot."""

  def __init__(self, message: str, snapshot: OptimizationRunSnapshot):
    super().__init__(message)
    self.snapshot = snapshot


class OptimizationBudgetExceeded(OptimizationRunContextError):
  """A configured call or token budget was exhausted."""


class OptimizationCancelledError(OptimizationRunContextError):
  """The run stopped because the context's cancellation was requested."""


class OptimizationProviderError(OptimizationRunContextError):
  """A governed run terminated on a provider-call failure.

  Usage reported before the failure is preserved on the snapshot. Provider
  failure takes precedence over a simultaneous token overshoot: both the
  committed usage and the token-compliance evidence are preserved, but the
  run terminates ``failed`` and this error is raised.
  """


class OptimizationFailedError(OptimizationRunContextError):
  """A governed run terminated on a non-provider failure.

  Sampler, adapter, validation, and optimizer-internal failures recorded via
  ``finalize_failed`` surface as this generic type;
  ``OptimizationProviderError`` is reserved for failures of governed
  provider calls.
  """


class OptimizationRunFinalizedError(Exception):
  """A terminal context was used where an in-progress context is required."""


class UnsupportedOptimizationContextError(Exception):
  """The optimizer does not support run-context instrumentation."""


class ContextAlreadyAttachedError(Exception):
  """A one-shot context was attached to more than one optimization run."""


class _CallRecord:
  """Mutable internal record for one logical model invocation."""

  __slots__ = (
      "sequence",
      "stage",
      "requested_model",
      "returned_model_version",
      "start_time",
      "end_time",
      "state",
      "usage",
      "error_code",
      "error_type",
      "closed",
  )

  def __init__(
      self, sequence: int, stage: str, requested_model: Optional[str]
  ) -> None:
    self.sequence: int = sequence
    self.stage: str = stage
    self.requested_model: Optional[str] = requested_model
    self.returned_model_version: Optional[str] = None
    self.start_time: float = time.time()
    self.end_time: Optional[float] = None
    self.state: Optional[ModelCallState] = None
    self.usage: dict[str, Optional[int]] = {}
    self.error_code: Optional[str] = None
    self.error_type: Optional[str] = None
    self.closed: bool = False


class _CallHandle:
  """Opaque handle for one in-flight logical model invocation.

  Bound to exactly one context; committing it into a different context is a
  typed misuse error.
  """

  def __init__(
      self, context: "OptimizationRunContext", record: _CallRecord
  ) -> None:
    self._context: "OptimizationRunContext" = context
    self._record: _CallRecord = record


_UNATTACHED = object()


class OptimizationRunContext:
  """One-shot, concurrency-safe ledger and control channel for one run.

  Instrumented optimizers call :meth:`begin_model_call` immediately before a
  logical ``BaseLlm.generate_content_async()`` invocation and
  :meth:`end_model_call` with the terminal outcome. Every governed run must
  finalize exactly once (first terminal transition wins): optimizers call
  :meth:`finalize_success` on a normal return; budget/cancel/provider
  terminals are committed by the corresponding ledger operations or by
  :meth:`finalize_cancelled` / :meth:`finalize_failed` on exceptional exits.
  Governance callers own the context, may call :meth:`request_cancel` from
  any thread, and read :meth:`snapshot` at any time, including after failure.
  """

  def __init__(self, budgets: Optional[OptimizationBudgets] = None) -> None:
    # OptimizationBudgets is frozen; keep a private reference and never
    # expose a mutable path to it.
    self._budgets = budgets or OptimizationBudgets()
    self._lock = threading.Lock()
    self._records: list[_CallRecord] = []
    self._started_calls = 0
    self._settled_calls = 0
    self._cumulative_total_tokens = 0
    self._cancel_requested = False
    self._cancel_reason: Optional[str] = None
    self._run_status: Optional[RunStatus] = None
    self._terminal_sequence: Optional[int] = None
    self._terminal_error_code: Optional[str] = None
    self._terminal_error_type: Optional[str] = None
    self._terminal_from_provider_call = False
    self._attached_owner: object = _UNATTACHED

  @property
  def budgets(self) -> OptimizationBudgets:
    """The configured (immutable) budgets."""
    return self._budgets

  def attach(self, owner: object) -> None:
    """Binds the context to one optimization run; reuse is rejected."""
    with self._lock:
      if self._attached_owner is not _UNATTACHED:
        raise ContextAlreadyAttachedError(
            "OptimizationRunContext is one-shot: it is already attached to an"
            " optimization run and cannot be reused or shared."
        )
      self._attached_owner = owner

  # --- cancellation ---------------------------------------------------------

  def request_cancel(self, reason: str = "requested") -> None:
    """Thread-safe, idempotent cooperative cancellation signal.

    Requesting cancellation does not itself finalize the run; the terminal
    ``cancelled`` transition is committed when the signal is observed at a
    documented boundary (or via :meth:`finalize_cancelled`), and never
    overwrites an earlier terminal state.
    """
    with self._lock:
      if not self._cancel_requested:
        self._cancel_requested = True
        self._cancel_reason = reason

  @property
  def cancel_requested(self) -> bool:
    with self._lock:
      return self._cancel_requested

  def raise_if_cancelled(self) -> None:
    """Raises ``OptimizationCancelledError`` if cancellation was requested.

    First terminal wins: if the run already terminated for another reason,
    this re-raises that terminal outcome instead of overwriting it.
    """
    with self._lock:
      if self._run_status is not None:
        error = self._terminal_error_locked()
      elif self._cancel_requested:
        self._transition_locked(RunStatus.CANCELLED)
        error = OptimizationCancelledError(
            f"Optimization cancelled: {self._cancel_reason}",
            self._snapshot_locked(),
        )
      else:
        return
    raise error

  # --- terminal transitions -------------------------------------------------

  def finalize_success(self) -> None:
    """Commits ``completed`` -- and enforces that success is actually possible.

    This is an invariant boundary, not a status setter:

    - idempotent only when the run is already ``completed``; any other
      existing terminal re-raises its typed outcome so a caller cannot
      swallow a committed budget/provider/cancel terminal into success;
    - a pending cancellation commits ``cancelled`` and raises;
    - an open (unsettled) call makes success impossible: the run commits
      ``failed`` (``OPEN_MODEL_CALLS``) and raises
      ``OptimizationFailedError``. Already-admitted calls may still settle
      later and will observe that terminal.
    """
    with self._lock:
      if self._run_status is not None:
        if self._run_status == RunStatus.COMPLETED:
          return
        error = self._terminal_error_locked()
      elif self._cancel_requested:
        self._transition_locked(RunStatus.CANCELLED)
        error = OptimizationCancelledError(
            f"Optimization cancelled: {self._cancel_reason}",
            self._snapshot_locked(),
        )
      elif any(not r.closed for r in self._records):
        self._transition_locked(
            RunStatus.FAILED,
            error_code="OPEN_MODEL_CALLS",
            error_type="OptimizationRunFinalizedError",
        )
        error = OptimizationFailedError(
            "Cannot finalize success with unsettled model calls.",
            self._snapshot_locked(),
        )
      else:
        self._transition_locked(RunStatus.COMPLETED)
        return
    raise error

  def finalize_cancelled(self, reason: str = "cancelled") -> None:
    """Commits the ``cancelled`` terminal (e.g. on native task cancellation).

    Closes any open call event as ``cancelled`` first, so the final snapshot
    carries no open invocations. First terminal wins.
    """
    with self._lock:
      if not self._cancel_requested:
        self._cancel_requested = True
        self._cancel_reason = reason
      for record in self._records:
        if not record.closed:
          record.closed = True
          record.end_time = time.time()
          record.state = ModelCallState.CANCELLED
          self._settled_calls += 1
      self._transition_locked(RunStatus.CANCELLED)

  def finalize_failed(
      self,
      *,
      error_code: Optional[str] = None,
      error_type: Optional[str] = None,
  ) -> None:
    """Commits the ``failed`` terminal for a non-provider-call failure path.

    First terminal wins. Error metadata must be sanitized identifiers (a
    short code and an exception type name), never raw payload text.
    """
    with self._lock:
      self._transition_locked(
          RunStatus.FAILED,
          error_code=_sanitize_error_meta(error_code),
          error_type=_sanitize_error_meta(error_type),
      )

  def _transition_locked(
      self,
      status: RunStatus,
      *,
      sequence: Optional[int] = None,
      error_code: Optional[str] = None,
      error_type: Optional[str] = None,
      provider: bool = False,
  ) -> bool:
    """First-terminal-wins transition; returns True if this call won."""
    if self._run_status is not None:
      return False
    self._run_status = status
    self._terminal_sequence = sequence
    self._terminal_error_code = error_code
    self._terminal_error_type = error_type
    self._terminal_from_provider_call = provider
    return True

  def _terminal_error_locked(self) -> Exception:
    """The typed error corresponding to an existing terminal state."""
    snapshot = self._snapshot_locked()
    if self._run_status == RunStatus.BUDGET_EXCEEDED:
      return OptimizationBudgetExceeded(
          "Optimization budget exhausted.", snapshot
      )
    if self._run_status == RunStatus.CANCELLED:
      return OptimizationCancelledError(
          f"Optimization cancelled: {self._cancel_reason}", snapshot
      )
    if self._run_status == RunStatus.FAILED:
      if self._terminal_from_provider_call:
        return OptimizationProviderError(
            f"Provider failure: {self._terminal_error_code}", snapshot
        )
      return OptimizationFailedError(
          f"Optimization failed: {self._terminal_error_code}", snapshot
      )
    return OptimizationRunFinalizedError(
        f"OptimizationRunContext already finalized as {self._run_status.value}."
    )

  # --- the ledger -----------------------------------------------------------

  def begin_model_call(
      self,
      stage: str,
      requested_model: Optional[str] = None,
  ) -> _CallHandle:
    """Atomically admits and records the start of one logical invocation.

    Under one lock acquisition: rejects a terminal context, observes
    cancellation, checks the logical-call budget, reserves the slot, assigns
    the sequence ID, and creates the ledger entry. A preflight rejection is
    not a call event and does not increment the started-call count.
    """
    with self._lock:
      if self._run_status is not None:
        error = self._terminal_error_locked()
      elif self._cancel_requested:
        self._transition_locked(RunStatus.CANCELLED)
        error = OptimizationCancelledError(
            f"Optimization cancelled: {self._cancel_reason}",
            self._snapshot_locked(),
        )
      else:
        max_calls = self._budgets.max_model_calls
        if max_calls is not None and self._started_calls >= max_calls:
          self._transition_locked(RunStatus.BUDGET_EXCEEDED)
          error = OptimizationBudgetExceeded(
              f"Logical model-call budget exhausted ({max_calls}).",
              self._snapshot_locked(),
          )
        else:
          self._started_calls += 1
          record = _CallRecord(self._started_calls, stage, requested_model)
          self._records.append(record)
          return _CallHandle(self, record)
    raise error

  def end_model_call(
      self,
      handle: _CallHandle,
      *,
      usage_metadata: Any = None,
      returned_model_version: Optional[str] = None,
      error_code: Optional[str] = None,
      error_type: Optional[str] = None,
      cancelled: bool = False,
  ) -> None:
    """Commits the terminal outcome of one logical model invocation.

    All usage and error metadata is normalized and validated *before* the
    record is mutated, then the event, counters, and any terminal transition
    commit atomically under the context lock -- no exception path can expose
    a half-closed event. Ownership and close are validated under the same
    lock: a handle from another context is a typed misuse error, and a
    concurrent double-close commits exactly once.

    Provider failure takes precedence over token overshoot: usage is
    committed either way, but a call ending in a provider error transitions
    the run to ``failed`` and raises ``OptimizationProviderError``. A clean
    over-budget final call is committed with call status ``completed`` while
    the run transitions to ``budget_exceeded`` and
    ``OptimizationBudgetExceeded`` is raised. ``cancelled=True`` settles the
    call as ``cancelled`` (preserving usage evidence) without raising.

    If the run was already terminated by another call, this call still
    settles for truthful accounting, and the existing terminal outcome is
    then raised so no caller can continue scheduling work past a committed
    terminal.
    """
    # Normalize/validate everything BEFORE touching the record.
    usage = _extract_usage(usage_metadata)
    clean_code = _sanitize_error_meta(error_code)
    clean_type = _sanitize_error_meta(error_type)
    is_provider_error = error_code is not None or error_type is not None
    if is_provider_error and clean_code is None:
      clean_code = "PROVIDER_ERROR"

    record = handle._record
    with self._lock:
      if handle._context is not self:
        raise OptimizationRunFinalizedError(
            "Call handle belongs to a different OptimizationRunContext."
        )
      if record.closed:
        return
      was_terminal = self._run_status is not None
      record.closed = True
      record.end_time = time.time()
      record.returned_model_version = _sanitize_optional_str(
          returned_model_version, _MODEL_VERSION_MAX_LEN
      )
      record.usage = usage
      self._settled_calls += 1
      total = usage.get("total_tokens")
      if total is not None:
        self._cumulative_total_tokens += total

      error: Optional[Exception] = None
      if cancelled:
        record.state = ModelCallState.CANCELLED
      elif is_provider_error:
        record.state = ModelCallState.PROVIDER_ERROR
        record.error_code = clean_code
        record.error_type = clean_type
        # Provider failure is primary even when the same terminal response
        # also crossed the token ceiling; both usage and token-compliance
        # evidence are preserved on the snapshot.
        if self._transition_locked(
            RunStatus.FAILED,
            sequence=record.sequence,
            error_code=clean_code,
            error_type=clean_type,
            provider=True,
        ):
          error = OptimizationProviderError(
              f"Provider failure on call {record.sequence}: {clean_code}",
              self._snapshot_locked(),
          )
      else:
        record.state = ModelCallState.COMPLETED
        max_tokens = self._budgets.max_provider_reported_tokens
        if (
            max_tokens is not None
            and self._cumulative_total_tokens > max_tokens
        ):
          if self._transition_locked(
              RunStatus.BUDGET_EXCEEDED, sequence=record.sequence
          ):
            error = OptimizationBudgetExceeded(
                f"Token budget exhausted ({max_tokens}).",
                self._snapshot_locked(),
            )
      if error is None and was_terminal and not cancelled:
        # Another call already terminated the run; the settling caller must
        # observe that terminal instead of continuing.
        error = self._terminal_error_locked()
    if error is not None:
      raise error

  def abort_model_call(
      self,
      handle: _CallHandle,
      *,
      error_code: Optional[str] = None,
      error_type: Optional[str] = None,
  ) -> None:
    """Settles an admitted call that failed locally (non-provider).

    Closes the record as ``aborted``, transitions the run to ``failed`` with
    sanitized metadata, and raises ``OptimizationFailedError`` -- truthfully
    distinct from provider failures and from cancellation. Idempotent on an
    already-settled handle; foreign handles are a typed misuse error.
    """
    clean_code = _sanitize_error_meta(error_code) or "LOCAL_CALL_ABORT"
    clean_type = _sanitize_error_meta(error_type)
    record = handle._record
    with self._lock:
      if handle._context is not self:
        raise OptimizationRunFinalizedError(
            "Call handle belongs to a different OptimizationRunContext."
        )
      error: Optional[Exception] = None
      if not record.closed:
        record.closed = True
        record.end_time = time.time()
        record.state = ModelCallState.ABORTED
        record.error_code = clean_code
        record.error_type = clean_type
        self._settled_calls += 1
        if self._transition_locked(
            RunStatus.FAILED,
            sequence=record.sequence,
            error_code=clean_code,
            error_type=clean_type,
        ):
          error = OptimizationFailedError(
              f"Local failure on call {record.sequence}: {clean_code}",
              self._snapshot_locked(),
          )
      if error is None and self._run_status is not None:
        error = self._terminal_error_locked()
    if error is not None:
      raise error

  # --- snapshots --------------------------------------------------------------

  def snapshot(self) -> OptimizationRunSnapshot:
    with self._lock:
      return self._snapshot_locked()

  def _snapshot_locked(self) -> OptimizationRunSnapshot:
    events = tuple(
        ModelCallEvent(
            sequence=r.sequence,
            stage=r.stage,
            requested_model=r.requested_model,
            returned_model_version=r.returned_model_version,
            start_time=r.start_time,
            end_time=r.end_time,
            state=r.state,
            prompt_tokens=r.usage.get("prompt_tokens"),
            output_tokens=r.usage.get("output_tokens"),
            reasoning_tokens=r.usage.get("reasoning_tokens"),
            cached_tokens=r.usage.get("cached_tokens"),
            tool_use_tokens=r.usage.get("tool_use_tokens"),
            total_tokens=r.usage.get("total_tokens"),
            usage_coverage=_coverage_of(r) if r.closed else None,
            error_code=r.error_code,
            error_type=r.error_type,
        )
        for r in self._records
    )
    closed = [r for r in self._records if r.closed]
    coverages = [_coverage_of(r) for r in closed]
    if coverages and all(c == UsageCoverage.VERIFIED for c in coverages):
      run_coverage = UsageCoverage.VERIFIED
    elif any(
        c in (UsageCoverage.VERIFIED, UsageCoverage.PARTIAL) for c in coverages
    ):
      run_coverage = UsageCoverage.PARTIAL
    else:
      run_coverage = UsageCoverage.UNREPORTED
    token_status = None
    if self._budgets.max_provider_reported_tokens is not None:
      if (
          self._cumulative_total_tokens
          > self._budgets.max_provider_reported_tokens
      ):
        token_status = TokenBudgetStatus.EXCEEDED
      elif not closed or run_coverage == UsageCoverage.VERIFIED:
        token_status = TokenBudgetStatus.WITHIN_LIMIT
      else:
        # Missing totals are never proof of compliance.
        token_status = TokenBudgetStatus.INDETERMINATE
    return OptimizationRunSnapshot(
        events=events,
        budgets=self._budgets,
        started_calls=self._started_calls,
        completed_calls=self._settled_calls,
        cumulative_total_tokens=self._cumulative_total_tokens,
        usage_coverage=run_coverage,
        token_budget_status=token_status,
        cancel_requested=self._cancel_requested,
        cancel_reason=self._cancel_reason,
        run_status=self._run_status,
        terminal_sequence=self._terminal_sequence,
        terminal_error_code=self._terminal_error_code,
        terminal_error_type=self._terminal_error_type,
    )


_ERROR_META_MAX_LEN = 128
_ERROR_META_SAFE = re.compile(r"[^A-Za-z0-9_.:-]")


_MODEL_VERSION_MAX_LEN = 256


def _sanitize_optional_str(value: Any, max_len: int) -> Optional[str]:
  """Bounded optional-string normalization for ledger fields."""
  if value is None:
    return None
  text = str(value).strip()
  return text[:max_len] if text else None


def _sanitize_error_meta(value: Any) -> Optional[str]:
  """Normalizes error metadata to a bounded, single-token identifier.

  Accepts any input (providers report numeric codes, enums, or strings) and
  never lets raw multiline/path-like text into the ledger.
  """
  if value is None:
    return None
  text = str(value).strip()
  if not text:
    return None
  text = _ERROR_META_SAFE.sub("_", text)
  return text[:_ERROR_META_MAX_LEN]


def _extract_usage(usage_metadata: Any) -> dict[str, Optional[int]]:
  """Copies provider-reported token counters, truthfully (no zero-coercion).

  Counters must be finite, non-negative numbers; anything else (NaN, inf,
  negative, non-numeric) is treated as not reported rather than corrupting
  the ledger.
  """
  if usage_metadata is None:
    return {}

  def _get(name: str) -> Optional[int]:
    value = getattr(usage_metadata, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      return None
    if not math.isfinite(value) or value < 0:
      return None
    if isinstance(value, float) and not value.is_integer():
      # Token counters are integral evidence; silent truncation is not
      # truthful normalization -- fractional values are "not reported".
      return None
    return int(value)

  return {
      "prompt_tokens": _get("prompt_token_count"),
      "output_tokens": _get("candidates_token_count"),
      "reasoning_tokens": _get("thoughts_token_count"),
      "cached_tokens": _get("cached_content_token_count"),
      "tool_use_tokens": _get("tool_use_prompt_token_count"),
      "total_tokens": _get("total_token_count"),
  }


def _coverage_of(record: _CallRecord) -> UsageCoverage:
  if record.usage.get("total_tokens") is not None:
    return UsageCoverage.VERIFIED
  if any(v is not None for v in record.usage.values()):
    return UsageCoverage.PARTIAL
  return UsageCoverage.UNREPORTED


class OptimizerCapabilities(BaseModel):
  """What run-context instrumentation an optimizer actually supports.

  Capabilities describe ADK instrumentation, not provider metadata guarantees
  or transport-attempt visibility. Conservative defaults (all ``False``) are
  correct for optimizers that predate or do not implement the run-context
  seam, so a governance wrapper can reject an incompatible optimizer at
  preflight instead of discovering opacity after spend occurs.
  """

  model_config = ConfigDict(frozen=True)

  accepts_run_context: bool = False
  """``optimize`` accepts a run context at all. When ``False``, callers must
  omit the keyword entirely (protects pre-existing third-party overrides)."""

  model_calls_observable: bool = False
  """Optimizer-owned logical model invocations are recorded on the context."""

  logical_call_limits_enforceable: bool = False
  """Configured logical-invocation limits stop the next call from starting
  (hard: atomic slot admission)."""

  reported_token_limits_enforceable: bool = False
  """Reported-token ceilings terminate the run reactively after the
  triggering call commits. Not a bound on unreported usage."""

  cooperative_cancellation: bool = False
  """``request_cancel`` is observed at the documented boundaries."""

  sampler_usage_included: bool = False
  """Always ``False`` for this context: candidate execution and evaluator
  inference belong to the supplied sampler/Runner, not the optimizer."""
