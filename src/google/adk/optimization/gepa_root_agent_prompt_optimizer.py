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

import asyncio
import contextvars
import logging
from typing import Any
from typing import Optional

from google.genai import types as genai_types
from pydantic import BaseModel
from pydantic import Field

from ..agents.llm_agent import Agent
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..models.registry import LLMRegistry
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from ._run_context import OptimizationBudgetExceeded
from ._run_context import OptimizationFailedError
from ._run_context import OptimizationProviderError
from ._run_context import OptimizationRunContext
from ._run_context import OptimizationRunContextError
from ._run_context import OptimizerCapabilities
from ._run_context import RunStatus
from ._run_context import STAGE_REFLECTION
from .agent_optimizer import AgentOptimizer
from .data_types import AgentWithScores
from .data_types import OptimizerResult
from .data_types import UnstructuredSamplingResult
from .sampler import Sampler

_logger = logging.getLogger("google_adk." + __name__)

_AGENT_PROMPT_NAME = "agent_prompt"


class _ReflectionCapture:
  """Typed carrier for reflection stream metadata, visible to error paths."""

  __slots__ = ("usage", "model_version", "error_code")

  def __init__(self) -> None:
    self.usage: Any = None
    self.model_version: Optional[str] = None
    self.error_code: Optional[str] = None


class _GovernedRunAbort(Exception):
  """Private iteration-abort sentinel for governed GEPA runs.

  GEPA's reflective-mutation proposer converts reflection/proposal
  exceptions into "no proposal" (``reflective_mutation.py``), independent of
  the engine's ``raise_on_exception`` flag -- so a typed budget, provider, or
  cancellation error raised from the reflection callable would be swallowed
  and the run could still return success. (Other iteration failures default
  to ``raise_on_exception=True`` in the admitted GEPA releases; governed runs
  pin that explicitly.) The governed terminal state is
  already committed on the run context when this sentinel is raised; the stop
  callback observes it at the next loop boundary and the post-check after
  ``gepa.optimize`` returns maps it to the typed outcome.
  """


class GEPARootAgentPromptOptimizerConfig(BaseModel):
  """Contains configuration options required by the GEPARootAgentPromptOptimizer."""

  optimizer_model: str = Field(
      default="gemini-2.5-flash",
      description=(
          "The model used to analyze the eval results and optimize the agent."
      ),
  )

  model_configuration: genai_types.GenerateContentConfig = Field(
      default_factory=lambda: genai_types.GenerateContentConfig(
          thinking_config=genai_types.ThinkingConfig(
              include_thoughts=True,
              thinking_budget=10240,
          )
      ),
      description="The configuration for the optimizer model.",
  )

  max_metric_calls: int = Field(
      default=100,
      description="The maximum number of metric calls (evaluations) to make.",
  )

  reflection_minibatch_size: int = Field(
      default=3,
      description="The number of examples to use for reflection.",
  )

  run_dir: Optional[str] = Field(
      default=None,
      description=(
          "The directory to save the intermediate/final optimization results."
      ),
  )


class GEPARootAgentPromptOptimizerResult(OptimizerResult[AgentWithScores]):
  """The final result of the GEPARootAgentPromptOptimizer."""

  gepa_result: Optional[dict[str, Any]] = Field(
      default=None,
      description="The raw result dictionary from the GEPA optimizer.",
  )


def _create_agent_gepa_adapter_class():
  """Creates the _AgentGEPAAdapter class dynamically to avoid top-level gepa imports."""
  from gepa.core.adapter import EvaluationBatch
  from gepa.core.adapter import GEPAAdapter

  class _AgentGEPAAdapter(GEPAAdapter[str, dict[str, Any], dict[str, Any]]):
    """A GEPA adapter for ADK agents."""

    def __init__(
        self,
        initial_agent: Agent,
        sampler: Sampler[UnstructuredSamplingResult],
        main_loop: asyncio.AbstractEventLoop,
        run_context: Optional[OptimizationRunContext] = None,
    ):
      self._initial_agent = initial_agent
      self._sampler = sampler
      self._main_loop = main_loop
      self._run_context = run_context

      self._train_example_ids = set(sampler.get_train_example_ids())
      self._validation_example_ids = set(sampler.get_validation_example_ids())

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
      prompt = candidate[_AGENT_PROMPT_NAME]
      _logger.info(
          "Evaluating agent on batch:\n%s\nwith prompt:\n%s", batch, prompt
      )
      # Clone the agent and update the instruction
      new_agent = self._initial_agent.clone(update={"instruction": prompt})

      if set(batch) <= self._train_example_ids:
        example_set = "train"
      elif set(batch) <= self._validation_example_ids:
        example_set = "validation"
      else:
        raise ValueError(f"Invalid batch composition: {batch}")

      if self._run_context is not None:
        # Boundary: before a sampler evaluation is scheduled onto the main
        # loop. Raising here is converted to the private sentinel below so
        # GEPA cannot swallow it into a later success.
        try:
          self._run_context.raise_if_cancelled()
        except OptimizationRunContextError as e:
          raise _GovernedRunAbort(str(e)) from e

      # Run the evaluation in the main loop
      future = asyncio.run_coroutine_threadsafe(
          self._sampler.sample_and_score(
              new_agent,
              example_set=example_set,
              batch=batch,
              capture_full_eval_data=capture_traces,
          ),
          self._main_loop,
      )
      result: UnstructuredSamplingResult = future.result()

      if self._run_context is not None:
        # Boundary: immediately after the sampler evaluation settles.
        try:
          self._run_context.raise_if_cancelled()
        except OptimizationRunContextError as e:
          raise _GovernedRunAbort(str(e)) from e

      scores = []
      outputs = []
      trajectories = []

      for example_id in batch:
        score = result.scores[example_id]
        scores.append(score)

        eval_data = result.data.get(example_id, {}) if result.data else {}
        outputs.append(eval_data)
        trajectories.append(eval_data)

      return EvaluationBatch(
          outputs=outputs, scores=scores, trajectories=trajectories
      )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
      dataset: list[dict[str, Any]] = []
      trace_instances: list[tuple[float, dict[str, Any]]] = list(
          zip(
              eval_batch.scores,
              eval_batch.trajectories,
              strict=True,
          )
      )
      for trace_instance in trace_instances:
        score, eval_data = trace_instance

        dataset.append({
            _AGENT_PROMPT_NAME: candidate[_AGENT_PROMPT_NAME],
            "score": score,
            "eval_data": eval_data,
        })

      # same data for all components (should be only one)
      result = {comp: dataset for comp in components_to_update}

      return result

  return _AgentGEPAAdapter


@experimental
class GEPARootAgentPromptOptimizer(
    AgentOptimizer[UnstructuredSamplingResult, AgentWithScores]
):
  """An optimizer that improves the root agent prompt using the GEPA framework."""

  def __init__(
      self,
      config: GEPARootAgentPromptOptimizerConfig,
  ):
    self._config = config
    llm_registry = LLMRegistry()
    self._llm_class = llm_registry.resolve(self._config.optimizer_model)

  @property
  def capabilities(self) -> OptimizerCapabilities:
    return OptimizerCapabilities(
        accepts_run_context=True,
        model_calls_observable=True,
        logical_call_limits_enforceable=True,
        reported_token_limits_enforceable=True,
        cooperative_cancellation=True,
        sampler_usage_included=False,
    )

  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[UnstructuredSamplingResult],
      *,
      run_context: Optional[OptimizationRunContext] = None,
  ) -> GEPARootAgentPromptOptimizerResult:
    """Runs the GEPARootAgentPromptOptimizer.

    Args:
      initial_agent: The initial agent whose prompt is to be optimized. Only the
        root agent prompt will be optimized.
      sampler: The interface used to get training and validation example UIDs,
        request agent evaluations, and get useful data for optimizing the agent.

    Returns:
      The final result of the optimization process, containing the optimized
      agent instance, its scores on the validation examples, and other metrics.
    """
    if run_context is not None:
      run_context.attach(owner=self)

    try:
      if run_context is not None:
        # A pending cancellation wins over everything -- observed before
        # imports, model construction, adapter construction, and sampler
        # discovery.
        run_context.raise_if_cancelled()

      if initial_agent.sub_agents:
        _logger.warning(
            "The GEPARootAgentPromptOptimizer will not optimize prompts for"
            " sub-agents."
        )

      _logger.info("Setting up the GEPA optimizer...")

      try:
        import gepa  # lazy import as gepa is not in core ADK package

        _AgentGEPAAdapter = _create_agent_gepa_adapter_class()
      except ImportError as e:
        raise ImportError(MISSING_EVAL_DEPENDENCIES_MESSAGE) from e

      loop = asyncio.get_running_loop()

      adapter = _AgentGEPAAdapter(
          initial_agent=initial_agent,
          sampler=sampler,
          main_loop=loop,
          run_context=run_context,
      )

      llm = self._llm_class(model=self._config.optimizer_model)

      def reflection_lm(prompt: str) -> str:
        # Build and validate the request BEFORE reserving the model-call slot: a
        # construction failure is a failed proposal (GEPA-native no-proposal
        # path) with a clean ledger -- no admitted call is left open.
        llm_request = LlmRequest(
            model=self._config.optimizer_model,
            config=self._config.model_configuration,
            contents=[
                genai_types.Content(
                    parts=[genai_types.Part(text=prompt)],
                    role="user",
                )
            ],
        )

        capture = _ReflectionCapture()

        async def _generate() -> str:
          response_text = ""
          async with Aclosing(llm.generate_content_async(llm_request)) as agen:
            async for llm_response in agen:
              llm_response: LlmResponse
              if getattr(llm_response, "usage_metadata", None) is not None:
                capture.usage = llm_response.usage_metadata
              if getattr(llm_response, "model_version", None):
                capture.model_version = llm_response.model_version
              error_code = getattr(llm_response, "error_code", None)
              if error_code and run_context is not None:
                # Governed-only: terminate on the in-band provider error. With
                # no context, legacy stream semantics continue iterating.
                capture.error_code = str(error_code)
                return response_text
              generated_content: genai_types.Content = llm_response.content
              if not generated_content.parts:
                continue
              response_text = "".join(
                  part.text
                  for part in generated_content.parts
                  if part.text and not part.thought
              )
          return response_text

        if run_context is None:
          # Absent-context parity: the legacy scheduling/return path exactly.
          future = asyncio.run_coroutine_threadsafe(_generate(), loop)
          return future.result()

        # One logical optimizer-owned reflection invocation. Atomic admission
        # observes cancellation and the call budget; from admission onward,
        # every operation lives inside one settlement boundary so no exception
        # can leave the call open.
        try:
          handle = run_context.begin_model_call(
              STAGE_REFLECTION,
              requested_model=self._config.optimizer_model,
          )
        except OptimizationRunContextError as e:
          raise _GovernedRunAbort(str(e)) from e

        try:
          future = asyncio.run_coroutine_threadsafe(_generate(), loop)
        except Exception as e:
          # Local scheduling failure: settle as an aborted call -> run FAILED
          # (OptimizationFailedError), truthfully distinct from a provider
          # failure and from cancellation.
          try:
            run_context.abort_model_call(
                handle,
                error_code="SCHEDULING_FAILURE",
                error_type=type(e).__name__,
            )
          except OptimizationRunContextError as ctx_err:
            raise _GovernedRunAbort(str(ctx_err)) from ctx_err
          raise _GovernedRunAbort(str(e)) from e

        try:
          response_text = future.result()
        except Exception as e:
          try:
            run_context.end_model_call(
                handle,
                usage_metadata=capture.usage,
                error_code=str(
                    getattr(e, "error_code", None) or "PROVIDER_EXCEPTION"
                ),
                error_type=type(e).__name__,
            )
          except OptimizationRunContextError as ctx_err:
            raise _GovernedRunAbort(str(ctx_err)) from ctx_err
          raise _GovernedRunAbort(str(e)) from e

        try:
          if capture.error_code:
            # In-band provider error: commit sanitized failure atomically
            # (run status FAILED, provider-error-first precedence).
            run_context.end_model_call(
                handle,
                usage_metadata=capture.usage,
                error_code=capture.error_code,
                error_type="LlmResponseError",
            )
          else:
            # Commit-then-terminate: an over-budget final reflection is
            # persisted first (call status completed; run status
            # budget_exceeded).
            run_context.end_model_call(
                handle,
                usage_metadata=capture.usage,
                returned_model_version=capture.model_version,
            )
        except OptimizationRunContextError as e:
          raise _GovernedRunAbort(str(e)) from e
        return response_text

      train_ids = sampler.get_train_example_ids()
      val_ids = sampler.get_validation_example_ids()

      if set(train_ids).intersection(val_ids):
        _logger.warning(
            "The training and validation example UIDs overlap. This WILL cause"
            " aliasing issues unless each common UID refers to the same example"
            " in both sets."
        )

      def run_gepa():
        if run_context is None:
          # Legacy call shape preserved exactly for absent-context parity.
          return gepa.optimize(
              seed_candidate={_AGENT_PROMPT_NAME: initial_agent.instruction},
              trainset=train_ids,
              valset=val_ids,
              adapter=adapter,
              max_metric_calls=self._config.max_metric_calls,
              reflection_lm=reflection_lm,
              reflection_minibatch_size=self._config.reflection_minibatch_size,
              run_dir=self._config.run_dir,
          )

        # Governed: the stopper is observed at every loop boundary and must
        # reflect ANY terminal governed state (budget, provider failure,
        # cancellation), because GEPA's reflective-mutation proposer converts an
        # aborted reflection into "no proposal" and keeps looping.
        # raise_on_exception=True is pinned so generic evaluator/sampler failures
        # stay fail-closed even if the upstream default changes under the
        # unbounded gepa>=0.1 range. GEPA types stay private to ADK.
        def _governed_stop(*_args: Any, **_kwargs: Any) -> bool:
          return (
              run_context.cancel_requested
              or run_context.snapshot().run_status is not None
          )

        return gepa.optimize(
            seed_candidate={_AGENT_PROMPT_NAME: initial_agent.instruction},
            trainset=train_ids,
            valset=val_ids,
            adapter=adapter,
            max_metric_calls=self._config.max_metric_calls,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=self._config.reflection_minibatch_size,
            run_dir=self._config.run_dir,
            stop_callbacks=[_governed_stop],
            raise_on_exception=True,
        )

      _logger.info("Running the GEPA optimizer...")

      ctx = contextvars.copy_context()
      if run_context is None:
        # Absent-context parity: the legacy direct await is preserved exactly,
        # including immediate task-cancellation propagation semantics.
        gepa_results = await loop.run_in_executor(
            None, lambda: ctx.run(run_gepa)
        )
      else:
        executor_future = loop.run_in_executor(None, lambda: ctx.run(run_gepa))
        try:
          gepa_results = await asyncio.shield(executor_future)
        except asyncio.CancelledError:
          # Governed native cancellation: request the cooperative stop, drain
          # the executor worker once the active operation reaches a boundary,
          # finalize the snapshot, then re-raise promptly. The caller's
          # watchdog remains the bounded hard backstop.
          run_context.request_cancel("task_cancelled")
          try:
            await asyncio.shield(executor_future)
          except BaseException:  # pylint: disable=broad-except
            pass
          run_context.finalize_cancelled("task_cancelled")
          raise
        except _GovernedRunAbort:
          # The sentinel escaped gepa.optimize (e.g. raised during the seed
          # evaluation, outside GEPA's swallowing loop). The terminal state is
          # already committed; map it to the typed outcome -- the raw sentinel
          # never reaches the caller.
          snap = run_context.snapshot()
          if snap.run_status == RunStatus.FAILED:
            if snap.terminal_from_provider_call:
              raise OptimizationProviderError(
                  f"Provider failure: {snap.terminal_error_code}", snap
              ) from None
            raise OptimizationFailedError(
                f"Optimization failed: {snap.terminal_error_code}", snap
            ) from None
          if snap.run_status == RunStatus.BUDGET_EXCEEDED:
            if run_context.budgets.on_budget_exceeded == "return_partial":
              _logger.warning(
                  "Optimization budget exhausted; returning the seed as the"
                  " partial result."
              )
              return GEPARootAgentPromptOptimizerResult(
                  optimized_agents=[
                      AgentWithScores(optimized_agent=initial_agent)
                  ],
              )
            raise OptimizationBudgetExceeded(
                "Optimization budget exhausted.", snap
            ) from None
          run_context.raise_if_cancelled()
          raise  # pragma: no cover -- unknown sentinel state
        except Exception as e:
          run_context.finalize_failed(
              error_code=getattr(e, "error_code", None) or "OPTIMIZER_FAILURE",
              error_type=type(e).__name__,
          )
          raise

        # Post-check: GEPA returns normally when the stopper ends the loop, so
        # a committed governed terminal state must map to its typed outcome
        # here -- never a false success.
        snap = run_context.snapshot()
        if snap.run_status == RunStatus.FAILED:
          if snap.terminal_from_provider_call:
            raise OptimizationProviderError(
                f"Provider failure: {snap.terminal_error_code}", snap
            )
          raise OptimizationFailedError(
              f"Optimization failed: {snap.terminal_error_code}", snap
          )
        if snap.cancel_requested and snap.run_status in (
            None,
            RunStatus.CANCELLED,
        ):
          run_context.raise_if_cancelled()
        if snap.run_status == RunStatus.BUDGET_EXCEEDED:
          if run_context.budgets.on_budget_exceeded == "return_partial":
            # GEPA stopped at a loop boundary, so its best-so-far candidates
            # are available. The snapshot (run_status=budget_exceeded) is the
            # authoritative record; the result schema stays unchanged and a
            # downstream gate treats the attempt as never promotion-eligible.
            _logger.warning(
                "Optimization budget exhausted; returning partial result."
            )
          else:
            raise OptimizationBudgetExceeded(
                "Optimization budget exhausted.", snap
            )

          _logger.info("GEPA optimization finished. Preparing final results...")

      optimized_prompts = [
          candidate[_AGENT_PROMPT_NAME] for candidate in gepa_results.candidates
      ]
      scores = gepa_results.val_aggregate_scores

      optimized_agents = [
          AgentWithScores(
              optimized_agent=initial_agent.clone(
                  update={"instruction": optimized_prompt},
              ),
              overall_score=score,
          )
          for optimized_prompt, score in zip(optimized_prompts, scores)
      ]

      if run_context is not None and run_context.snapshot().run_status is None:
        # Only a genuinely non-terminal run may finalize as success; a
        # governed budget partial keeps its authoritative BUDGET_EXCEEDED
        # terminal on the snapshot.
        run_context.finalize_success()
      return GEPARootAgentPromptOptimizerResult(
          optimized_agents=optimized_agents,
          gepa_result=gepa_results.to_dict(),
      )
    except asyncio.CancelledError:
      if run_context is not None:
        run_context.finalize_cancelled("task_cancelled")
      raise
    except OptimizationRunContextError:
      # Typed run-context outcomes already committed their terminal state.
      raise
    except Exception as e:
      if run_context is not None:
        # Any failure after successful attachment -- including sampler ID
        # setup before the executor boundary -- finalizes the authoritative
        # snapshot (first terminal wins).
        run_context.finalize_failed(
            error_code=getattr(e, "error_code", None) or "OPTIMIZER_FAILURE",
            error_type=type(e).__name__,
        )
      raise
