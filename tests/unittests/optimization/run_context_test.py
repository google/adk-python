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

import threading
from types import SimpleNamespace

from google.adk.optimization import ContextAlreadyAttachedError
from google.adk.optimization import ModelCallState
from google.adk.optimization import OptimizationBudgetExceeded
from google.adk.optimization import OptimizationBudgets
from google.adk.optimization import OptimizationCancelledError
from google.adk.optimization import OptimizationProviderError
from google.adk.optimization import OptimizationRunContext
from google.adk.optimization import OptimizationRunFinalizedError
from google.adk.optimization import OptimizerCapabilities
from google.adk.optimization import RunStatus
from google.adk.optimization import STAGE_CANDIDATE_GENERATION
from google.adk.optimization import STAGE_REFLECTION
from google.adk.optimization import TokenBudgetStatus
from google.adk.optimization import UsageCoverage
import pydantic
import pytest


def _usage(**kwargs) -> SimpleNamespace:
  return SimpleNamespace(**kwargs)


class TestOneShotAttachment:

  def test_attach_twice_rejected(self):
    ctx = OptimizationRunContext()
    ctx.attach(owner=object())
    with pytest.raises(ContextAlreadyAttachedError):
      ctx.attach(owner=object())

  def test_attach_none_owner_still_one_shot(self):
    # None must not double as the unattached sentinel.
    ctx = OptimizationRunContext()
    ctx.attach(owner=None)
    with pytest.raises(ContextAlreadyAttachedError):
      ctx.attach(owner=None)

  def test_two_runs_have_isolated_contexts(self):
    a, b = OptimizationRunContext(), OptimizationRunContext()
    a.attach(owner="run-a")
    b.attach(owner="run-b")
    h = a.begin_model_call(STAGE_REFLECTION)
    a.end_model_call(h, usage_metadata=_usage(total_token_count=7))
    assert a.snapshot().completed_calls == 1
    assert b.snapshot().completed_calls == 0


class TestLedgerIntegrity:

  def test_foreign_handle_rejected(self):
    a, b = OptimizationRunContext(), OptimizationRunContext()
    handle_from_a = a.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationRunFinalizedError):
      b.end_model_call(handle_from_a, usage_metadata=None)
    # B's ledger is untouched by the attempt.
    assert b.snapshot().completed_calls == 0
    assert b.snapshot().started_calls == 0

  def test_concurrent_double_close_commits_once(self):
    ctx = OptimizationRunContext()
    handle = ctx.begin_model_call(STAGE_REFLECTION)
    errors: list[Exception] = []

    def close():
      try:
        ctx.end_model_call(handle, usage_metadata=_usage(total_token_count=10))
      except Exception as e:  # pylint: disable=broad-except
        errors.append(e)

    threads = [threading.Thread(target=close) for _ in range(8)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    assert not errors
    snap = ctx.snapshot()
    assert snap.completed_calls == 1
    assert snap.cumulative_total_tokens == 10

  def test_budgets_are_immutable(self):
    budgets = OptimizationBudgets(max_model_calls=3)
    with pytest.raises(pydantic.ValidationError):
      budgets.max_model_calls = 100
    ctx = OptimizationRunContext(budgets)
    with pytest.raises(pydantic.ValidationError):
      ctx.budgets.max_model_calls = 100

  def test_negative_budgets_rejected(self):
    with pytest.raises(pydantic.ValidationError):
      OptimizationBudgets(max_model_calls=-1)
    with pytest.raises(pydantic.ValidationError):
      OptimizationBudgets(max_provider_reported_tokens=-5)

  def test_snapshot_and_events_are_immutable(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(total_token_count=1))
    snap = ctx.snapshot()
    with pytest.raises(pydantic.ValidationError):
      snap.started_calls = 99
    with pytest.raises(pydantic.ValidationError):
      snap.events[0].total_tokens = 99
    assert isinstance(snap.events, tuple)

  def test_extensible_stage_strings(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call("my_custom_stage")
    ctx.end_model_call(h, usage_metadata=None)
    assert ctx.snapshot().events[0].stage == "my_custom_stage"
    assert STAGE_CANDIDATE_GENERATION == "candidate_generation"


class TestTerminalStateMachine:

  def test_success_finalizer(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(total_token_count=1))
    ctx.finalize_success()
    assert ctx.snapshot().run_status == RunStatus.COMPLETED

  def test_first_terminal_wins_late_cancel_does_not_overwrite(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    ctx.request_cancel("late")
    with pytest.raises(OptimizationBudgetExceeded):
      # Re-raises the EARLIER terminal outcome, not cancellation.
      ctx.raise_if_cancelled()
    assert ctx.snapshot().run_status == RunStatus.BUDGET_EXCEEDED

  def test_no_call_admitted_after_terminal(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.begin_model_call(STAGE_REFLECTION)
    assert ctx.snapshot().started_calls == 1

  def test_no_call_admitted_after_completed(self):
    ctx = OptimizationRunContext()
    ctx.finalize_success()
    with pytest.raises(OptimizationRunFinalizedError):
      ctx.begin_model_call(STAGE_REFLECTION)

  def test_finalize_cancelled_closes_open_events(self):
    ctx = OptimizationRunContext()
    ctx.begin_model_call(STAGE_REFLECTION)  # left open (native cancel)
    ctx.finalize_cancelled("task_cancelled")
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.CANCELLED
    assert snap.events[0].state == ModelCallState.CANCELLED
    assert snap.events[0].end_time is not None

  def test_snapshot_carries_limits_and_terminal_metadata(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    snap = ctx.snapshot()
    assert snap.budgets.max_provider_reported_tokens == 10
    assert snap.terminal_sequence == 1


class TestCallBudget:

  def test_exactly_n_calls_may_start(self):
    ctx = OptimizationRunContext(OptimizationBudgets(max_model_calls=2))
    for _ in range(2):
      h = ctx.begin_model_call(STAGE_CANDIDATE_GENERATION)
      ctx.end_model_call(h, usage_metadata=None)
    with pytest.raises(OptimizationBudgetExceeded) as exc:
      ctx.begin_model_call(STAGE_CANDIDATE_GENERATION)
    snap = exc.value.snapshot
    # The rejected reservation is not a call event and did not start a call.
    assert snap.started_calls == 2
    assert len(snap.events) == 2
    assert snap.run_status == RunStatus.BUDGET_EXCEEDED


class TestTokenBudget:

  def test_overshoot_commits_then_raises(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=100)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded) as exc:
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=150))
    snap = exc.value.snapshot
    # The over-budget final call is committed before the raise; the CALL
    # status stays completed while the RUN status becomes budget_exceeded.
    assert snap.completed_calls == 1
    assert snap.events[0].state == ModelCallState.COMPLETED
    assert snap.cumulative_total_tokens == 150
    assert snap.run_status == RunStatus.BUDGET_EXCEEDED
    assert snap.token_budget_status == TokenBudgetStatus.EXCEEDED

  def test_unreported_usage_is_indeterminate_not_compliant(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=None)  # no raise
    snap = ctx.snapshot()
    assert snap.cumulative_total_tokens == 0
    # Missing totals are never proof of compliance.
    assert snap.token_budget_status == TokenBudgetStatus.INDETERMINATE

  def test_all_verified_under_limit_is_within_limit(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=100)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(total_token_count=40))
    assert ctx.snapshot().token_budget_status == TokenBudgetStatus.WITHIN_LIMIT


class TestUsageClassification:

  def test_verified_partial_unreported(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(
        h, usage_metadata=_usage(prompt_token_count=5, total_token_count=9)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(prompt_token_count=5))
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=None)
    events = ctx.snapshot().events
    assert events[0].usage_coverage == UsageCoverage.VERIFIED
    assert events[1].usage_coverage == UsageCoverage.PARTIAL
    assert events[1].total_tokens is None  # never coerced to zero
    assert events[2].usage_coverage == UsageCoverage.UNREPORTED
    # Run-level coverage degrades to partial, not verified.
    assert ctx.snapshot().usage_coverage == UsageCoverage.PARTIAL

  def test_run_coverage_verified_only_when_all_verified(self):
    ctx = OptimizationRunContext()
    for _ in range(2):
      h = ctx.begin_model_call(STAGE_CANDIDATE_GENERATION)
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=3))
    assert ctx.snapshot().usage_coverage == UsageCoverage.VERIFIED


class TestProviderFailure:

  def test_provider_error_is_a_governed_terminal(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationProviderError) as exc:
      ctx.end_model_call(
          h,
          usage_metadata=_usage(total_token_count=42),
          error_code="RESOURCE_EXHAUSTED",
          error_type="ClientError",
      )
    snap = exc.value.snapshot
    event = snap.events[0]
    assert event.state == ModelCallState.PROVIDER_ERROR
    assert event.total_tokens == 42  # usage preserved
    assert event.error_code == "RESOURCE_EXHAUSTED"
    assert event.error_type == "ClientError"
    assert snap.run_status == RunStatus.FAILED
    assert snap.terminal_error_code == "RESOURCE_EXHAUSTED"

  def test_provider_error_precedence_over_token_overshoot(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationProviderError) as exc:
      ctx.end_model_call(
          h,
          usage_metadata=_usage(total_token_count=50),
          error_code="INTERNAL",
      )
    snap = exc.value.snapshot
    # Provider failure is primary; usage and compliance evidence preserved.
    assert snap.run_status == RunStatus.FAILED
    assert snap.cumulative_total_tokens == 50
    assert snap.token_budget_status == TokenBudgetStatus.EXCEEDED

  def test_finalize_failed_for_non_call_failures(self):
    ctx = OptimizationRunContext()
    ctx.finalize_failed(error_code="ADAPTER_CRASH", error_type="ValueError")
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.FAILED
    assert snap.terminal_error_type == "ValueError"


class TestCancellation:

  def test_cancel_is_idempotent_and_first_reason_wins(self):
    ctx = OptimizationRunContext()
    ctx.request_cancel("deadline")
    ctx.request_cancel("other")
    with pytest.raises(OptimizationCancelledError) as exc:
      ctx.begin_model_call(STAGE_REFLECTION)
    assert "deadline" in str(exc.value)
    assert exc.value.snapshot.cancel_reason == "deadline"
    assert exc.value.snapshot.run_status == RunStatus.CANCELLED

  def test_raise_if_cancelled_noop_when_not_cancelled(self):
    OptimizationRunContext().raise_if_cancelled()

  def test_cancel_from_another_thread_observed(self):
    ctx = OptimizationRunContext()
    t = threading.Thread(target=ctx.request_cancel, args=("watchdog",))
    t.start()
    t.join()
    with pytest.raises(OptimizationCancelledError):
      ctx.raise_if_cancelled()

  def test_snapshot_readable_after_failure(self):
    ctx = OptimizationRunContext(OptimizationBudgets(max_model_calls=0))
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.begin_model_call(STAGE_CANDIDATE_GENERATION)
    # A governance caller can persist the attempt from a finally block.
    assert ctx.snapshot().run_status == RunStatus.BUDGET_EXCEEDED


class TestCapabilitiesDefaults:

  def test_conservative_defaults(self):
    caps = OptimizerCapabilities()
    assert not caps.accepts_run_context
    assert not caps.model_calls_observable
    assert not caps.logical_call_limits_enforceable
    assert not caps.reported_token_limits_enforceable
    assert not caps.cooperative_cancellation
    assert not caps.sampler_usage_included

  def test_base_optimizer_reports_conservative_capabilities(self):
    from google.adk.optimization.agent_optimizer import AgentOptimizer

    class _Impl(AgentOptimizer):

      async def optimize(self, initial_agent, sampler, *, run_context=None):
        raise NotImplementedError()

    assert _Impl().capabilities == OptimizerCapabilities()


class TestResultShapeUnchanged:

  def test_optimizer_result_has_no_run_context_fields(self):
    # The caller-owned snapshot is authoritative for run status; the public
    # OptimizerResult schema stays untouched for observable equivalence.
    from google.adk.optimization.data_types import OptimizerResult

    assert "terminal_status" not in OptimizerResult.model_fields


# The contracts-PR behavior (built-ins reject a supplied context) is
# superseded in this stacked enforcement PR: SimplePromptOptimizer and
# GEPARootAgentPromptOptimizer now support run contexts (see
# run_context_enforcement_test.py); GEPARootAgentOptimizer still rejects,
# covered there as well.


class TestSuccessInvariantBoundary:
  """Round-3: finalize_success is an invariant boundary, not a status setter."""

  def test_success_with_open_call_commits_failed(self):
    from google.adk.optimization import OptimizationFailedError

    ctx = OptimizationRunContext()
    ctx.begin_model_call(STAGE_REFLECTION)  # left open
    with pytest.raises(OptimizationFailedError):
      ctx.finalize_success()
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.FAILED
    assert snap.terminal_error_code == "OPEN_MODEL_CALLS"

  def test_success_after_budget_terminal_reraises(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.finalize_success()  # cannot swallow the committed terminal
    assert ctx.snapshot().run_status == RunStatus.BUDGET_EXCEEDED

  def test_success_after_provider_terminal_reraises(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationProviderError):
      ctx.end_model_call(h, error_code="INTERNAL")
    with pytest.raises(OptimizationProviderError):
      ctx.finalize_success()

  def test_success_after_cancel_terminal_reraises(self):
    ctx = OptimizationRunContext()
    ctx.finalize_cancelled("x")
    with pytest.raises(OptimizationCancelledError):
      ctx.finalize_success()

  def test_success_idempotent_only_when_completed(self):
    ctx = OptimizationRunContext()
    ctx.finalize_success()
    ctx.finalize_success()  # idempotent
    assert ctx.snapshot().run_status == RunStatus.COMPLETED


class TestLedgerFieldNormalization:
  """Round-3: every event field normalized before mutation."""

  def test_malformed_model_version_cannot_corrupt_snapshots(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(
        h,
        usage_metadata=_usage(total_token_count=3),
        returned_model_version=123,  # non-string from a custom BaseLlm
    )
    snap = ctx.snapshot()  # must not raise
    assert snap.events[0].returned_model_version == "123"

  def test_fractional_usage_is_unreported_not_truncated(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(total_token_count=1.9))
    event = ctx.snapshot().events[0]
    assert event.total_tokens is None
    assert event.usage_coverage == UsageCoverage.UNREPORTED

  def test_integral_float_zero_and_large_int_accepted(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(
        h,
        usage_metadata=_usage(
            total_token_count=2.0,
            prompt_token_count=0,
            candidates_token_count=10**12,
        ),
    )
    event = ctx.snapshot().events[0]
    assert event.total_tokens == 2
    assert event.prompt_tokens == 0
    assert event.output_tokens == 10**12


class TestLocalCallAbort:

  def test_abort_settles_as_failed_not_provider(self):
    from google.adk.optimization import OptimizationFailedError

    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationFailedError):
      ctx.abort_model_call(
          h, error_code="SCHEDULING_FAILURE", error_type="RuntimeError"
      )
    snap = ctx.snapshot()
    assert snap.events[0].state == ModelCallState.ABORTED
    assert snap.run_status == RunStatus.FAILED
    assert snap.completed_calls == 1  # settled


class TestConcurrencyRaces:
  """Round-3: the RFC-required concurrent attachment and admission races."""

  def test_concurrent_attach_exactly_one_wins(self):
    ctx = OptimizationRunContext()
    results: list[bool] = []

    def try_attach():
      try:
        ctx.attach(owner=object())
        results.append(True)
      except ContextAlreadyAttachedError:
        results.append(False)

    threads = [threading.Thread(target=try_attach) for _ in range(16)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    assert results.count(True) == 1

  def test_concurrent_admission_admits_exactly_k(self):
    k = 3
    ctx = OptimizationRunContext(OptimizationBudgets(max_model_calls=k))
    admitted: list[int] = []
    rejected: list[Exception] = []

    def try_admit():
      try:
        h = ctx.begin_model_call(STAGE_REFLECTION)
        admitted.append(h._record.sequence)
      except OptimizationBudgetExceeded as e:
        rejected.append(e)

    threads = [threading.Thread(target=try_admit) for _ in range(16)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    assert len(admitted) == k
    assert sorted(admitted) == [1, 2, 3]  # unique sequence ids
    assert len(rejected) == 16 - k
    assert all(
        e.snapshot.run_status == RunStatus.BUDGET_EXCEEDED for e in rejected
    )

  def test_success_racing_last_settlement_never_completes_with_open_call(
      self,
  ):
    from google.adk.optimization import OptimizationFailedError

    for _ in range(20):
      ctx = OptimizationRunContext()
      h = ctx.begin_model_call(STAGE_REFLECTION)
      barrier = threading.Barrier(2)
      outcomes: dict[str, object] = {}

      def settle(handle=h, context=ctx, sync=barrier, sink=outcomes):
        sync.wait()
        try:
          context.end_model_call(handle, usage_metadata=None)
          sink["settle"] = "ok"
        except OptimizationFailedError:
          # Success won the race, committed FAILED/OPEN_MODEL_CALLS, and
          # this settling caller observes it after committing its event.
          sink["settle"] = "observed_terminal"

      def finish(context=ctx, sync=barrier, sink=outcomes):
        sync.wait()
        try:
          context.finalize_success()
          sink["success"] = "completed"
        except OptimizationFailedError:
          sink["success"] = "failed_open"

      t1 = threading.Thread(target=settle)
      t2 = threading.Thread(target=finish)
      t1.start()
      t2.start()
      t1.join()
      t2.join()
      snap = ctx.snapshot()
      # Exactly two legal resolutions -- and never COMPLETED with open calls.
      assert snap.completed_calls == snap.started_calls == 1
      if outcomes["success"] == "completed":
        assert outcomes["settle"] == "ok"
        assert snap.run_status == RunStatus.COMPLETED
        assert snap.events[0].state is not None
      else:
        assert outcomes["success"] == "failed_open"
        assert outcomes["settle"] == "observed_terminal"
        assert snap.run_status == RunStatus.FAILED
        assert snap.terminal_error_code == "OPEN_MODEL_CALLS"


class TestRoundFourFindings:

  def test_throwing_usage_accessor_degrades_to_unreported(self):
    class ExplodingUsage:

      @property
      def total_token_count(self):
        raise RuntimeError("bad provider metadata")

      prompt_token_count = 5

    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=ExplodingUsage())
    snap = ctx.snapshot()
    # The call settles atomically; the throwing counter is unreported and
    # the well-behaved counter survives.
    assert snap.completed_calls == 1
    event = snap.events[0]
    assert event.state == ModelCallState.COMPLETED
    assert event.total_tokens is None
    assert event.prompt_tokens == 5

  def test_hostile_str_cannot_defeat_settlement(self):
    class HostileCode:

      def __str__(self):
        raise RuntimeError("no string for you")

    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationProviderError):
      ctx.end_model_call(h, error_code=HostileCode())
    snap = ctx.snapshot()
    assert snap.completed_calls == 1
    assert snap.events[0].error_code == "UNSTRINGABLE"

  def test_duplicate_terminal_settlement_reraises_terminal(self):
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=10)
    )
    h = ctx.begin_model_call(STAGE_REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    # A duplicate close on a terminal run must not return normally.
    with pytest.raises(OptimizationBudgetExceeded):
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=50))
    assert ctx.snapshot().completed_calls == 1  # committed exactly once

  def test_duplicate_close_on_nonterminal_run_is_noop(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(STAGE_REFLECTION)
    ctx.end_model_call(h, usage_metadata=None)
    ctx.end_model_call(h, usage_metadata=None)  # no raise, no double count
    assert ctx.snapshot().completed_calls == 1

  def test_admission_metadata_normalized(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(123, requested_model=456)  # hostile callers
    ctx.end_model_call(h, usage_metadata=None)
    snap = ctx.snapshot()  # must not raise
    assert snap.events[0].stage == "123"
    assert snap.events[0].requested_model == "456"

  def test_empty_stage_rejected(self):
    ctx = OptimizationRunContext()
    with pytest.raises(ValueError):
      ctx.begin_model_call("   ")
    assert ctx.snapshot().started_calls == 0

  def test_cancel_reason_normalized(self):
    ctx = OptimizationRunContext()
    ctx.request_cancel(123)
    snap = ctx.snapshot()  # must not raise
    assert snap.cancel_reason == "123"
