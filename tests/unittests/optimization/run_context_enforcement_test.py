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

"""Enforcement tests: run-context instrumentation in the P0a optimizers."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.optimization import ContextAlreadyAttachedError
from google.adk.optimization import OptimizationBudgetExceeded
from google.adk.optimization import OptimizationBudgets
from google.adk.optimization import OptimizationCancelledError
from google.adk.optimization import OptimizationProviderError
from google.adk.optimization import OptimizationRunContext
from google.adk.optimization import RunStatus
from google.adk.optimization import STAGE_CANDIDATE_GENERATION
from google.adk.optimization import UsageCoverage
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.sampler import Sampler
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizer
from google.adk.optimization.simple_prompt_optimizer import SimplePromptOptimizerConfig
from google.genai import types as genai_types
import pytest


def _mock_sampler() -> mock.MagicMock:
  sampler = mock.MagicMock(spec=Sampler)
  sampler.get_train_example_ids.return_value = ["e1", "e2"]
  sampler.get_validation_example_ids.return_value = ["v1"]

  async def mock_sample_and_score(
      agent, example_set="validation", batch=None, capture_full_eval_data=False
  ):
    ids = batch or (["v1"] if example_set == "validation" else ["e1", "e2"])
    return UnstructuredSamplingResult(scores={i: 0.5 for i in ids})

  sampler.sample_and_score.side_effect = mock_sample_and_score
  return sampler


@pytest.fixture
def mock_sampler() -> mock.MagicMock:
  return _mock_sampler()


def _mock_llm(usage=None, error_code=None, raise_exc=None):
  mock_llm = mock.MagicMock()

  async def mock_generate_content_async(*args, **kwargs):
    if raise_exc is not None:
      raise raise_exc
    yield SimpleNamespace(
        content=genai_types.Content(
            parts=[genai_types.Part(text="improved prompt")], role="model"
        ),
        usage_metadata=usage,
        model_version="test-model-001",
        error_code=error_code,
        partial=False,
    )

  mock_llm.generate_content_async.side_effect = mock_generate_content_async
  return mock.MagicMock(return_value=mock_llm)


def _optimizer(mock_llm_class, iterations=2):
  with mock.patch(
      "google.adk.optimization.simple_prompt_optimizer.LLMRegistry.resolve",
      return_value=mock_llm_class,
  ):
    return SimplePromptOptimizer(
        SimplePromptOptimizerConfig(num_iterations=iterations, batch_size=1)
    )


def _agent() -> Agent:
  return Agent(name="test_agent", model="gemini-2.5-flash", instruction="v0")


class TestSimplePromptOptimizerEnforcement:

  def test_capabilities_opt_in(self):
    caps = _optimizer(_mock_llm()).capabilities
    assert caps.accepts_run_context
    assert caps.model_calls_observable
    assert caps.logical_call_limits_enforceable
    assert caps.reported_token_limits_enforceable
    assert caps.cooperative_cancellation
    assert not caps.sampler_usage_included

  @pytest.mark.asyncio
  async def test_normal_success_finalizes_completed(self, mock_sampler):
    usage = SimpleNamespace(total_token_count=15)
    optimizer = _optimizer(_mock_llm(usage), iterations=3)
    ctx = OptimizationRunContext()
    result = await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    # One candidate-generation call per iteration; nothing more.
    assert snap.started_calls == 3
    assert snap.completed_calls == 3
    assert all(e.stage == STAGE_CANDIDATE_GENERATION for e in snap.events)
    assert snap.cumulative_total_tokens == 45
    assert snap.usage_coverage == UsageCoverage.VERIFIED
    # Normal success is finalized COMPLETED -- the snapshot is authoritative.
    assert snap.run_status == RunStatus.COMPLETED
    assert result.optimized_agents[0].overall_score is not None

  @pytest.mark.asyncio
  async def test_missing_usage_is_unreported_not_zero(self, mock_sampler):
    optimizer = _optimizer(_mock_llm(usage=None), iterations=1)
    ctx = OptimizationRunContext()
    await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    event = ctx.snapshot().events[0]
    assert event.total_tokens is None
    assert event.usage_coverage == UsageCoverage.UNREPORTED

  @pytest.mark.asyncio
  async def test_call_budget_raise_mode(self, mock_sampler):
    optimizer = _optimizer(_mock_llm(), iterations=5)
    ctx = OptimizationRunContext(OptimizationBudgets(max_model_calls=2))
    with pytest.raises(OptimizationBudgetExceeded):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    # Exactly N calls started; the rejected third reservation is not an event.
    assert ctx.snapshot().started_calls == 2
    assert len(ctx.snapshot().events) == 2

  @pytest.mark.asyncio
  async def test_token_budget_return_partial(self, mock_sampler):
    usage = SimpleNamespace(total_token_count=100)
    optimizer = _optimizer(_mock_llm(usage), iterations=5)
    ctx = OptimizationRunContext(
        OptimizationBudgets(
            max_provider_reported_tokens=150,
            on_budget_exceeded="return_partial",
        )
    )
    sampler_calls_before = mock_sampler.sample_and_score.await_count
    result = await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.BUDGET_EXCEEDED
    assert snap.cumulative_total_tokens == 200
    # Validation never ran, so overall_score is None; the result schema is
    # unchanged and the snapshot is the authoritative record.
    assert result.optimized_agents[0].overall_score is None
    # Baseline + iteration-1 candidate scoring only: no scoring after stop.
    assert mock_sampler.sample_and_score.await_count - sampler_calls_before == 2

  @pytest.mark.asyncio
  async def test_in_band_provider_error_terminates_governed_run(
      self, mock_sampler
  ):
    usage = SimpleNamespace(total_token_count=30)
    optimizer = _optimizer(
        _mock_llm(usage, error_code="RESOURCE_EXHAUSTED"), iterations=2
    )
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    event = snap.events[0]
    assert event.total_tokens == 30  # usage preserved
    assert event.error_code == "RESOURCE_EXHAUSTED"
    assert snap.run_status == RunStatus.FAILED

  @pytest.mark.asyncio
  async def test_raised_provider_error_finalizes_failed(self, mock_sampler):
    optimizer = _optimizer(
        _mock_llm(raise_exc=RuntimeError("boom")), iterations=2
    )
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.FAILED
    assert snap.events[0].error_type == "RuntimeError"
    # Sanitized: the raw exception text is not in the ledger.
    assert snap.events[0].error_code == "PROVIDER_EXCEPTION"

  @pytest.mark.asyncio
  async def test_pre_cancelled_run_does_zero_work(self, mock_sampler):
    optimizer = _optimizer(_mock_llm(), iterations=5)
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    # Zero model calls AND zero sampler calls.
    assert ctx.snapshot().started_calls == 0
    assert mock_sampler.sample_and_score.await_count == 0

  @pytest.mark.asyncio
  async def test_cancel_during_model_work_prevents_next_sampler_call(
      self, mock_sampler
  ):
    ctx = OptimizationRunContext()
    mock_llm = mock.MagicMock()

    async def gen_and_cancel(*args, **kwargs):
      ctx.request_cancel("mid-model")
      yield SimpleNamespace(
          content=genai_types.Content(
              parts=[genai_types.Part(text="p")], role="model"
          ),
          usage_metadata=None,
          model_version=None,
          error_code=None,
      )

    mock_llm.generate_content_async.side_effect = gen_and_cancel
    optimizer = _optimizer(mock.MagicMock(return_value=mock_llm), iterations=3)
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    # Only the baseline scoring ran; the post-generation boundary stopped
    # candidate scoring.
    assert mock_sampler.sample_and_score.await_count == 1
    assert ctx.snapshot().run_status == RunStatus.CANCELLED

  @pytest.mark.asyncio
  async def test_native_cancellation_finalizes_and_reraises(self, mock_sampler):
    ctx = OptimizationRunContext()
    mock_llm = mock.MagicMock()

    async def gen_cancelled(*args, **kwargs):
      raise asyncio.CancelledError()
      yield  # pylint: disable=unreachable

    mock_llm.generate_content_async.side_effect = gen_cancelled
    optimizer = _optimizer(mock.MagicMock(return_value=mock_llm), iterations=2)
    with pytest.raises(asyncio.CancelledError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    # The open call event is closed and the run finalized CANCELLED.
    assert snap.run_status == RunStatus.CANCELLED
    assert all(e.end_time is not None for e in snap.events)

  @pytest.mark.asyncio
  async def test_no_context_path_unchanged(self, mock_sampler):
    optimizer = _optimizer(_mock_llm(), iterations=2)
    result = await optimizer.optimize(_agent(), mock_sampler)
    assert result.optimized_agents

  @pytest.mark.asyncio
  async def test_context_reuse_rejected_across_runs(self, mock_sampler):
    optimizer = _optimizer(_mock_llm(), iterations=1)
    ctx = OptimizationRunContext()
    await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    with pytest.raises(ContextAlreadyAttachedError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)


class TestRootSkillOptimizerRejectsContext:

  @pytest.mark.asyncio
  async def test_unsupported_context_rejected_before_work(self):
    pytest.importorskip("gepa")
    from google.adk.optimization import UnsupportedOptimizationContextError
    from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizer
    from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizerConfig

    optimizer = GEPARootAgentOptimizer(GEPARootAgentOptimizerConfig())
    sampler = mock.MagicMock(spec=Sampler)
    with pytest.raises(UnsupportedOptimizationContextError):
      await optimizer.optimize(
          _agent(), sampler, run_context=OptimizationRunContext()
      )
    # Rejected before any sampler or model work.
    sampler.get_train_example_ids.assert_not_called()
    sampler.sample_and_score.assert_not_called()

  def test_conservative_capabilities(self):
    pytest.importorskip("gepa")
    from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizer
    from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizerConfig

    caps = GEPARootAgentOptimizer(GEPARootAgentOptimizerConfig()).capabilities
    assert not caps.accepts_run_context
    assert not caps.model_calls_observable


def _gepa_stub(mocker, evaluate_counter):
  """A gepa stub with faithful engine semantics: per-iteration exceptions are
  swallowed ("no proposal") and stop callbacks are checked at loop
  boundaries."""

  class MockEvaluationBatch:

    def __init__(self, outputs, scores, trajectories):
      self.outputs, self.scores, self.trajectories = (
          outputs,
          scores,
          trajectories,
      )

  class MockGEPAAdapter:

    def __class_getitem__(cls, item):
      return cls

  def fake_optimize(
      *,
      seed_candidate,
      trainset,
      valset,
      adapter,
      max_metric_calls,
      reflection_lm,
      reflection_minibatch_size,
      run_dir,
      stop_callbacks=None,
      **_kwargs,
  ):
    def should_stop():
      return any(cb(None) for cb in stop_callbacks or [])

    for _ in range(5):
      if should_stop():
        break
      try:
        reflection_lm("reflect")
      except Exception:  # engine.py:588 -- swallowed, loop continues
        continue
      if should_stop():
        break
      evaluate_counter["count"] += 1
      adapter.evaluate(list(trainset), dict(seed_candidate), False)
    return SimpleNamespace(
        candidates=[dict(seed_candidate)],
        val_aggregate_scores=[0.0],
        to_dict=lambda: {},
    )

  gepa_module = mocker.MagicMock()
  gepa_module.optimize = fake_optimize
  gepa_module.core.adapter.EvaluationBatch = MockEvaluationBatch
  gepa_module.core.adapter.GEPAAdapter = MockGEPAAdapter
  mocker.patch.dict(
      sys.modules,
      {
          "gepa": gepa_module,
          "gepa.core": gepa_module.core,
          "gepa.core.adapter": gepa_module.core.adapter,
      },
  )
  return gepa_module


def _gepa_optimizer(llm_class):
  from google.adk.optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizer
  from google.adk.optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizerConfig

  with mock.patch(
      "google.adk.optimization.gepa_root_agent_prompt_optimizer.LLMRegistry.resolve",
      return_value=llm_class,
  ):
    return GEPARootAgentPromptOptimizer(GEPARootAgentPromptOptimizerConfig())


class TestGepaSentinelBridge:

  @pytest.mark.asyncio
  async def test_no_sampler_call_after_overshoot_commits(self, mocker):
    counter = {"count": 0}
    _gepa_stub(mocker, counter)
    usage = SimpleNamespace(total_token_count=500)
    optimizer = _gepa_optimizer(_mock_llm(usage))
    sampler = _mock_sampler()
    ctx = OptimizationRunContext(
        OptimizationBudgets(
            max_provider_reported_tokens=100,
            on_budget_exceeded="return_partial",
        )
    )
    result = await optimizer.optimize(_agent(), sampler, run_context=ctx)
    # The first reflection commits 500 tokens (> 100): the sentinel is
    # swallowed by the engine, the stopper ends the loop at the next
    # boundary, and NO sampler evaluation is ever scheduled.
    assert counter["count"] == 0
    assert sampler.sample_and_score.call_count == 0
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.BUDGET_EXCEEDED
    assert snap.completed_calls == 1  # committed; call status completed
    assert result.optimized_agents  # best-so-far partial, schema unchanged

  @pytest.mark.asyncio
  async def test_raise_mode_maps_post_check_to_typed_error(self, mocker):
    _gepa_stub(mocker, {"count": 0})
    usage = SimpleNamespace(total_token_count=500)
    optimizer = _gepa_optimizer(_mock_llm(usage))
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=100)
    )
    with pytest.raises(OptimizationBudgetExceeded):
      await optimizer.optimize(_agent(), _mock_sampler(), run_context=ctx)

  @pytest.mark.asyncio
  async def test_in_band_provider_error_cannot_become_success(self, mocker):
    _gepa_stub(mocker, {"count": 0})
    usage = SimpleNamespace(total_token_count=10)
    optimizer = _gepa_optimizer(
        _mock_llm(usage, error_code="RESOURCE_EXHAUSTED")
    )
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), _mock_sampler(), run_context=ctx)
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.FAILED
    assert snap.terminal_error_code == "RESOURCE_EXHAUSTED"
    assert snap.events[0].total_tokens == 10  # usage preserved

  @pytest.mark.asyncio
  async def test_raised_provider_error_cannot_become_success(self, mocker):
    _gepa_stub(mocker, {"count": 0})
    optimizer = _gepa_optimizer(_mock_llm(raise_exc=RuntimeError("boom")))
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), _mock_sampler(), run_context=ctx)
    assert ctx.snapshot().run_status == RunStatus.FAILED

  @pytest.mark.asyncio
  async def test_normal_gepa_success_finalizes_completed(self, mocker):
    counter = {"count": 0}
    _gepa_stub(mocker, counter)
    usage = SimpleNamespace(total_token_count=5)
    optimizer = _gepa_optimizer(_mock_llm(usage))
    ctx = OptimizationRunContext()
    result = await optimizer.optimize(
        _agent(), _mock_sampler(), run_context=ctx
    )
    assert result.optimized_agents
    assert ctx.snapshot().run_status == RunStatus.COMPLETED
    assert counter["count"] == 5  # all iterations ran

  @pytest.mark.asyncio
  async def test_pre_cancelled_gepa_run_does_zero_work(self, mocker):
    counter = {"count": 0}
    _gepa_stub(mocker, counter)
    optimizer = _gepa_optimizer(_mock_llm())
    sampler = _mock_sampler()
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    assert ctx.snapshot().started_calls == 0
    assert sampler.sample_and_score.call_count == 0


class TestRealGepaIntegration:
  """Budget/cancel/provider paths against the real installed gepa engine.

  These prove the sentinel/stopper/post-check bridge against actual GEPA
  loop semantics, not the stub. A min-plus-latest gepa version matrix is a
  CI concern tracked in the RFC.
  """

  @pytest.mark.asyncio
  async def test_real_gepa_token_overshoot_raise_mode(self):
    pytest.importorskip("gepa")
    usage = SimpleNamespace(total_token_count=10_000)
    optimizer = _gepa_optimizer(_mock_llm(usage))
    sampler = _mock_sampler()
    ctx = OptimizationRunContext(
        OptimizationBudgets(max_provider_reported_tokens=100)
    )
    with pytest.raises(OptimizationBudgetExceeded):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.BUDGET_EXCEEDED
    # The triggering reflection is committed with call status completed.
    reflections = [e for e in snap.events if e.stage == "reflection"]
    assert reflections and reflections[-1].state is not None

  @pytest.mark.asyncio
  async def test_real_gepa_pre_cancelled_stops_promptly(self):
    pytest.importorskip("gepa")
    optimizer = _gepa_optimizer(_mock_llm())
    sampler = _mock_sampler()
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    assert ctx.snapshot().started_calls == 0


class TestRoundTwoEnforcementFindings:
  """Regression tests for the round-2 adversarial findings."""

  @pytest.mark.asyncio
  async def test_no_context_gepa_stream_keeps_legacy_semantics(self, mocker):
    # An error-bearing chunk followed by a valid final chunk: the legacy
    # (no-context) path must keep iterating and reflect the FINAL text.
    reflected = {}

    def fake_optimize(*, reflection_lm, stop_callbacks=None, **_kwargs):
      reflected["text"] = reflection_lm("reflect")
      return SimpleNamespace(
          candidates=[{"agent_prompt": "p1"}],
          val_aggregate_scores=[0.0],
          to_dict=lambda: {},
      )

    gepa_module = mocker.MagicMock()
    gepa_module.optimize = fake_optimize

    class MockEvaluationBatch:

      def __init__(self, outputs, scores, trajectories):
        pass

    class MockGEPAAdapter:

      def __class_getitem__(cls, item):
        return cls

    gepa_module.core.adapter.EvaluationBatch = MockEvaluationBatch
    gepa_module.core.adapter.GEPAAdapter = MockGEPAAdapter
    mocker.patch.dict(
        sys.modules,
        {
            "gepa": gepa_module,
            "gepa.core": gepa_module.core,
            "gepa.core.adapter": gepa_module.core.adapter,
        },
    )

    mock_llm = mock.MagicMock()

    async def two_chunk_stream(*args, **kwargs):
      yield SimpleNamespace(
          content=genai_types.Content(parts=[], role="model"),
          usage_metadata=None,
          model_version=None,
          error_code="TRANSIENT",
      )
      yield SimpleNamespace(
          content=genai_types.Content(
              parts=[genai_types.Part(text="legacy final")], role="model"
          ),
          usage_metadata=None,
          model_version=None,
          error_code=None,
      )

    mock_llm.generate_content_async.side_effect = two_chunk_stream
    optimizer = _gepa_optimizer(mock.MagicMock(return_value=mock_llm))
    await optimizer.optimize(_agent(), _mock_sampler())  # no run_context
    assert reflected["text"] == "legacy final"

  @pytest.mark.asyncio
  async def test_gepa_sampler_setup_failure_finalizes_failed(self, mocker):
    _gepa_stub(mocker, {"count": 0})
    optimizer = _gepa_optimizer(_mock_llm())
    sampler = _mock_sampler()
    sampler.get_train_example_ids.side_effect = ValueError("setup crash")
    ctx = OptimizationRunContext()
    with pytest.raises(ValueError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    # The attached lifecycle terminalizes even before the executor boundary.
    assert ctx.snapshot().run_status == RunStatus.FAILED
    assert ctx.snapshot().terminal_error_type == "ValueError"

  @pytest.mark.asyncio
  async def test_simple_native_cancel_preserves_usage_evidence(
      self, mock_sampler
  ):
    from google.adk.optimization import ModelCallState

    ctx = OptimizationRunContext()
    mock_llm = mock.MagicMock()

    async def gen_usage_then_cancel(*args, **kwargs):
      yield SimpleNamespace(
          content=genai_types.Content(
              parts=[genai_types.Part(text="p")], role="model"
          ),
          usage_metadata=SimpleNamespace(total_token_count=7),
          model_version=None,
          error_code=None,
      )
      raise asyncio.CancelledError()

    mock_llm.generate_content_async.side_effect = gen_usage_then_cancel
    optimizer = _optimizer(mock.MagicMock(return_value=mock_llm), iterations=2)
    with pytest.raises(asyncio.CancelledError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    snap = ctx.snapshot()
    # The seven reported tokens survive as evidence on the cancelled call.
    assert snap.events[0].state == ModelCallState.CANCELLED
    assert snap.events[0].total_tokens == 7
    assert snap.completed_calls == 1  # settled
    assert snap.run_status == RunStatus.CANCELLED

  @pytest.mark.asyncio
  async def test_cancel_during_final_validation_never_completes(self):
    ctx = OptimizationRunContext()
    sampler = mock.MagicMock(spec=Sampler)
    sampler.get_train_example_ids.return_value = ["e1"]
    sampler.get_validation_example_ids.return_value = ["v1"]

    async def sample_and_score(
        agent,
        example_set="validation",
        batch=None,
        capture_full_eval_data=False,
    ):
      if example_set == "validation":
        # Cancellation lands while final validation is in flight.
        ctx.request_cancel("during-validation")
      ids = batch or (["v1"] if example_set == "validation" else ["e1"])
      return UnstructuredSamplingResult(scores={i: 0.5 for i in ids})

    sampler.sample_and_score.side_effect = sample_and_score
    optimizer = _optimizer(_mock_llm(), iterations=1)
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    assert ctx.snapshot().run_status == RunStatus.CANCELLED

  @pytest.mark.asyncio
  async def test_numeric_error_code_normalized_through_raised_path(
      self, mock_sampler
  ):
    exc = RuntimeError("boom")
    exc.error_code = 429  # numeric provider code
    optimizer = _optimizer(_mock_llm(raise_exc=exc), iterations=1)
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), mock_sampler, run_context=ctx)
    # Normalized to a bounded string at both the adapter and the context.
    assert ctx.snapshot().events[0].error_code == "429"


class TestRoundThreeEnforcementFindings:
  """Regression tests for the round-3 adversarial findings."""

  @pytest.mark.asyncio
  async def test_reflection_request_failure_leaves_clean_ledger(self, mocker):
    # Request construction happens BEFORE slot admission: a failure is a
    # GEPA-native no-proposal with a clean ledger, never an open call, and
    # finalize_success (second line of defense) can commit truthfully.
    counter = {"count": 0}
    _gepa_stub(mocker, counter)
    mocker.patch(
        "google.adk.optimization.gepa_root_agent_prompt_optimizer.LlmRequest",
        side_effect=RuntimeError("bad request config"),
    )
    optimizer = _gepa_optimizer(_mock_llm())
    ctx = OptimizationRunContext()
    result = await optimizer.optimize(
        _agent(), _mock_sampler(), run_context=ctx
    )
    snap = ctx.snapshot()
    assert snap.started_calls == 0  # no admitted call was left open
    assert snap.run_status == RunStatus.COMPLETED
    assert result.optimized_agents

  @pytest.mark.asyncio
  async def test_reflection_scheduling_failure_becomes_failed_not_success(
      self, mocker
  ):
    from google.adk.optimization import OptimizationFailedError

    counter = {"count": 0}
    _gepa_stub(mocker, counter)
    optimizer = _gepa_optimizer(_mock_llm())
    mocker.patch(
        "google.adk.optimization.gepa_root_agent_prompt_optimizer.asyncio"
        ".run_coroutine_threadsafe",
        side_effect=RuntimeError("loop closed"),
    )
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationFailedError):
      await optimizer.optimize(_agent(), _mock_sampler(), run_context=ctx)
    snap = ctx.snapshot()
    # The admitted call is settled (aborted), never open; the failure is
    # generic, not a provider failure, and can never become success.
    assert snap.run_status == RunStatus.FAILED
    assert not snap.terminal_from_provider_call
    assert snap.completed_calls == snap.started_calls == 1
    assert counter["count"] == 0  # no sampler evaluation after the abort

  @pytest.mark.asyncio
  async def test_pre_cancel_wins_before_simple_discovery(self):
    optimizer = _optimizer(_mock_llm(), iterations=2)
    sampler = mock.MagicMock(spec=Sampler)
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    # Every sampler method remains untouched.
    sampler.get_train_example_ids.assert_not_called()
    sampler.get_validation_example_ids.assert_not_called()
    sampler.sample_and_score.assert_not_called()

  @pytest.mark.asyncio
  async def test_pre_cancel_wins_before_gepa_discovery(self, mocker):
    _gepa_stub(mocker, {"count": 0})
    optimizer = _gepa_optimizer(_mock_llm())
    sampler = mock.MagicMock(spec=Sampler)
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    sampler.get_train_example_ids.assert_not_called()
    sampler.get_validation_example_ids.assert_not_called()

  @pytest.mark.asyncio
  async def test_pre_cancel_wins_over_discovery_exception(self):
    # Cancellation pending + discovery that would raise: cancellation wins.
    optimizer = _optimizer(_mock_llm(), iterations=2)
    sampler = mock.MagicMock(spec=Sampler)
    sampler.get_train_example_ids.side_effect = ValueError("discovery crash")
    ctx = OptimizationRunContext()
    ctx.request_cancel("pre")
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    assert ctx.snapshot().run_status == RunStatus.CANCELLED


class TestRealGepaRoundThree:
  """Round-3 durable real-GEPA coverage additions."""

  @pytest.mark.asyncio
  async def test_real_gepa_in_band_provider_error_is_typed(self):
    pytest.importorskip("gepa")
    usage = SimpleNamespace(total_token_count=10)
    optimizer = _gepa_optimizer(
        _mock_llm(usage, error_code="RESOURCE_EXHAUSTED")
    )
    ctx = OptimizationRunContext()
    with pytest.raises(OptimizationProviderError):
      await optimizer.optimize(_agent(), _mock_sampler(), run_context=ctx)
    snap = ctx.snapshot()
    assert snap.run_status == RunStatus.FAILED
    assert snap.terminal_from_provider_call

  @pytest.mark.asyncio
  async def test_real_gepa_post_sampler_cancellation_is_typed(self):
    pytest.importorskip("gepa")
    optimizer = _gepa_optimizer(_mock_llm())
    ctx = OptimizationRunContext()
    sampler = mock.MagicMock(spec=Sampler)
    sampler.get_train_example_ids.return_value = ["e1", "e2"]
    sampler.get_validation_example_ids.return_value = ["v1"]

    async def sample_then_cancel(
        agent,
        example_set="validation",
        batch=None,
        capture_full_eval_data=False,
    ):
      ids = batch or (["v1"] if example_set == "validation" else ["e1", "e2"])
      ctx.request_cancel("post-sampler")
      return UnstructuredSamplingResult(scores={i: 0.5 for i in ids})

    sampler.sample_and_score.side_effect = sample_then_cancel
    with pytest.raises(OptimizationCancelledError):
      await optimizer.optimize(_agent(), sampler, run_context=ctx)
    assert ctx.snapshot().run_status == RunStatus.CANCELLED

  @pytest.mark.asyncio
  async def test_real_gepa_reflection_request_failure_completes_cleanly(
      self, mocker
  ):
    pytest.importorskip("gepa")
    mocker.patch(
        "google.adk.optimization.gepa_root_agent_prompt_optimizer.LlmRequest",
        side_effect=RuntimeError("bad request config"),
    )
    optimizer = _gepa_optimizer(_mock_llm())
    ctx = OptimizationRunContext()
    result = await optimizer.optimize(
        _agent(), _mock_sampler(), run_context=ctx
    )
    snap = ctx.snapshot()
    assert snap.started_calls == 0
    assert snap.run_status == RunStatus.COMPLETED
    assert result.optimized_agents
