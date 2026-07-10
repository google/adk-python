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

from google.adk.optimization.run_context import ContextAlreadyAttachedError
from google.adk.optimization.run_context import ModelCallStage
from google.adk.optimization.run_context import ModelCallState
from google.adk.optimization.run_context import OptimizationBudgetExceeded
from google.adk.optimization.run_context import OptimizationBudgets
from google.adk.optimization.run_context import OptimizationCancelledError
from google.adk.optimization.run_context import OptimizationRunContext
from google.adk.optimization.run_context import OptimizerCapabilities
from google.adk.optimization.run_context import UsageCoverage
import pytest


def _usage(**kwargs) -> SimpleNamespace:
  return SimpleNamespace(**kwargs)


class TestOneShotAttachment:

  def test_attach_twice_rejected(self):
    ctx = OptimizationRunContext()
    ctx.attach(owner=object())
    with pytest.raises(ContextAlreadyAttachedError):
      ctx.attach(owner=object())

  def test_two_runs_have_isolated_contexts(self):
    a, b = OptimizationRunContext(), OptimizationRunContext()
    a.attach(owner="run-a")
    b.attach(owner="run-b")
    h = a.begin_model_call(ModelCallStage.REFLECTION)
    a.end_model_call(h, usage_metadata=_usage(total_token_count=7))
    assert a.snapshot().completed_calls == 1
    assert b.snapshot().completed_calls == 0


class TestCallBudget:

  def test_exactly_n_calls_may_start(self):
    ctx = OptimizationRunContext(OptimizationBudgets(max_model_calls=2))
    for _ in range(2):
      h = ctx.begin_model_call(ModelCallStage.CANDIDATE_GENERATION)
      ctx.end_model_call(h, usage_metadata=None)
    with pytest.raises(OptimizationBudgetExceeded) as exc:
      ctx.begin_model_call(ModelCallStage.CANDIDATE_GENERATION)
    snap = exc.value.snapshot
    # The rejected reservation is not a call event and did not start a call.
    assert snap.started_calls == 2
    assert len(snap.events) == 2
    assert snap.terminal_control_state == "call_budget_rejected"


class TestTokenBudget:

  def test_overshoot_commits_then_raises(self):
    ctx = OptimizationRunContext(OptimizationBudgets(max_total_tokens=100))
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
    with pytest.raises(OptimizationBudgetExceeded) as exc:
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=150))
    snap = exc.value.snapshot
    # The over-budget final call is committed before the raise.
    assert snap.completed_calls == 1
    assert snap.events[0].state == ModelCallState.COMPLETED
    assert snap.cumulative_total_tokens == 150
    assert snap.terminal_control_state == "budget_exceeded"

  def test_unreported_usage_does_not_consume_token_budget(self):
    ctx = OptimizationRunContext(OptimizationBudgets(max_total_tokens=10))
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
    ctx.end_model_call(h, usage_metadata=None)  # no raise
    assert ctx.snapshot().cumulative_total_tokens == 0


class TestUsageClassification:

  def test_verified_partial_unreported(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
    ctx.end_model_call(
        h, usage_metadata=_usage(prompt_token_count=5, total_token_count=9)
    )
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
    ctx.end_model_call(h, usage_metadata=_usage(prompt_token_count=5))
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
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
      h = ctx.begin_model_call(ModelCallStage.CANDIDATE_GENERATION)
      ctx.end_model_call(h, usage_metadata=_usage(total_token_count=3))
    assert ctx.snapshot().usage_coverage == UsageCoverage.VERIFIED


class TestProviderError:

  def test_error_preserves_usage_so_far(self):
    ctx = OptimizationRunContext()
    h = ctx.begin_model_call(ModelCallStage.REFLECTION)
    ctx.end_model_call(
        h,
        usage_metadata=_usage(total_token_count=42),
        error_message="RESOURCE_EXHAUSTED",
    )
    event = ctx.snapshot().events[0]
    assert event.state == ModelCallState.PROVIDER_ERROR
    assert event.total_tokens == 42
    assert event.error_message == "RESOURCE_EXHAUSTED"


class TestCancellation:

  def test_cancel_is_idempotent_and_first_reason_wins(self):
    ctx = OptimizationRunContext()
    ctx.request_cancel("deadline")
    ctx.request_cancel("other")
    with pytest.raises(OptimizationCancelledError) as exc:
      ctx.begin_model_call(ModelCallStage.REFLECTION)
    assert "deadline" in str(exc.value)
    assert exc.value.snapshot.cancel_reason == "deadline"
    assert exc.value.snapshot.terminal_control_state == "cancelled"

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
      ctx.begin_model_call(ModelCallStage.CANDIDATE_GENERATION)
    # A governance caller can persist the attempt from a finally block.
    assert ctx.snapshot().terminal_control_state == "call_budget_rejected"


class TestCapabilitiesDefaults:

  def test_conservative_defaults(self):
    caps = OptimizerCapabilities()
    assert not caps.model_calls_observable
    assert not caps.call_limits_enforceable
    assert not caps.cooperative_cancellation
    assert not caps.sampler_usage_included

  def test_base_optimizer_reports_conservative_capabilities(self):
    from google.adk.optimization.agent_optimizer import AgentOptimizer

    class _Impl(AgentOptimizer):

      async def optimize(self, initial_agent, sampler, *, run_context=None):
        raise NotImplementedError()

    assert _Impl().capabilities == OptimizerCapabilities()


class TestResultTerminalStatus:

  def test_default_none(self):
    from google.adk.optimization.data_types import AgentWithScores
    from google.adk.optimization.data_types import OptimizerResult

    result = OptimizerResult(optimized_agents=[])
    assert result.terminal_status is None
    result = OptimizerResult(
        optimized_agents=[], terminal_status="budget_exceeded"
    )
    assert result.terminal_status == "budget_exceeded"
