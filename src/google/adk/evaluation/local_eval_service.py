# Copyright 2025 Google LLC
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
import inspect
import logging
from typing import AsyncGenerator
from typing import Callable
from typing import Optional
import uuid

from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..errors.not_found_error import NotFoundError
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..utils.feature_decorator import experimental
from .base_eval_service import BaseEvalService
from .base_eval_service import EvaluateConfig
from .base_eval_service import EvaluateRequest
from .base_eval_service import InferenceRequest
from .base_eval_service import InferenceResult
from .base_eval_service import InferenceStatus
from .eval_case import Invocation
from .eval_metrics import EvalMetric
from .eval_metrics import EvalMetricResult
from .eval_metrics import EvalMetricResultDetails
from .eval_metrics import EvalMetricResultPerInvocation
from .eval_result import EvalCaseResult
from .eval_set import EvalCase
from .eval_set_results_manager import EvalSetResultsManager
from .eval_sets_manager import EvalSetsManager
from .evaluation_generator import EvaluationGenerator
from .evaluator import EvalStatus
from .evaluator import EvaluationResult
from .metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from .metric_evaluator_registry import MetricEvaluatorRegistry

logger = logging.getLogger('google_adk.' + __name__)

EVAL_SESSION_ID_PREFIX = '___eval___session___'


def _get_session_id() -> str:
  return f'{EVAL_SESSION_ID_PREFIX}{str(uuid.uuid4())}'


@experimental
class LocalEvalService(BaseEvalService):
  """An implementation of BaseEvalService, that runs the evals locally."""

  def __init__(
      self,
      root_agent: BaseAgent,
      eval_sets_manager: EvalSetsManager,
      metric_evaluator_registry: Optional[MetricEvaluatorRegistry] = None,
      session_service: Optional[BaseSessionService] = None,
      artifact_service: Optional[BaseArtifactService] = None,
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
      session_id_supplier: Callable[[], str] = _get_session_id,
  ):
    self._root_agent = root_agent
    self._eval_sets_manager = eval_sets_manager
    metric_evaluator_registry = (
        metric_evaluator_registry or DEFAULT_METRIC_EVALUATOR_REGISTRY
    )
    session_service = session_service or InMemorySessionService()
    artifact_service = artifact_service or InMemoryArtifactService()
    self._metric_evaluator_registry = metric_evaluator_registry
    self._session_service = session_service
    self._artifact_service = artifact_service
    self._eval_set_results_manager = eval_set_results_manager
    self._session_id_supplier = session_id_supplier

  @override
  async def perform_inference(
      self,
      inference_request: InferenceRequest,
  ) -> AsyncGenerator[InferenceResult, None]:
    """Returns InferenceResult obtained from the Agent as and when they are available.

    Args:
      inference_request: The request for generating inferences.
    """
    # Get the eval set from the storage.
    eval_set = self._eval_sets_manager.get_eval_set(
        app_name=inference_request.app_name,
        eval_set_id=inference_request.eval_set_id,
    )

    if not eval_set:
      raise NotFoundError(
          f'Eval set with id {inference_request.eval_set_id} not found for app'
          f' {inference_request.app_name}'
      )

    # Select eval cases for which we need to run inferencing. If the inference
    # request specified eval cases, then we use only those.
    eval_cases = eval_set.eval_cases
    if inference_request.eval_case_ids:
      eval_cases = [
          eval_case
          for eval_case in eval_cases
          if eval_case.eval_id in inference_request.eval_case_ids
      ]

    semaphore = asyncio.Semaphore(
        value=inference_request.inference_config.parallelism
    )

    async def run_inference(eval_case):
      async with semaphore:
        return await self._perform_inference_sigle_eval_item(
            app_name=inference_request.app_name,
            eval_set_id=inference_request.eval_set_id,
            eval_case=eval_case,
            root_agent=self._root_agent,
        )

    inference_results = [run_inference(eval_case) for eval_case in eval_cases]
    for inference_result in asyncio.as_completed(inference_results):
      yield await inference_result

  @override
  async def evaluate(
      self,
      evaluate_request: EvaluateRequest,
  ) -> AsyncGenerator[EvalCaseResult, None]:
    """Returns EvalCaseResult for each item as and when they are available.

    Args:
      evaluate_request: The request to perform metric evaluations on the
        inferences.
    """
    semaphore = asyncio.Semaphore(
        value=evaluate_request.evaluate_config.parallelism
    )

    async def run_evaluation(inference_result):
      async with semaphore:
        return await self._evaluate_single_inference_result(
            inference_result=inference_result,
            evaluate_config=evaluate_request.evaluate_config,
        )

    evaluation_tasks = [
        run_evaluation(inference_result)
        for inference_result in evaluate_request.inference_results
    ]

    for evaluation_task in asyncio.as_completed(evaluation_tasks):
      inference_result, eval_case_result = await evaluation_task

      if self._eval_set_results_manager:
        self._eval_set_results_manager.save_eval_set_result(
            app_name=inference_result.app_name,
            eval_set_id=inference_result.eval_set_id,
            eval_case_results=[eval_case_result],
        )

      yield eval_case_result

  async def _evaluate_single_inference_result(
      self, inference_result: InferenceResult, evaluate_config: EvaluateConfig
  ) -> tuple[InferenceResult, EvalCaseResult]:
    """Returns EvalCaseResult for the given inference result.

    A single inference result can have multiple invocations. For each
    invocaiton, this method evaluates the metrics present in evaluate config.

    The EvalCaseResult contains scores for each metric per invocation and the
    overall score.
    """
    eval_case = self._eval_sets_manager.get_eval_case(
        app_name=inference_result.app_name,
        eval_set_id=inference_result.eval_set_id,
        eval_case_id=inference_result.eval_case_id,
    )

    if eval_case is None:
      raise NotFoundError(
          f'Eval case with id {inference_result.eval_case_id} not found for'
          f' app {inference_result.app_name} and eval set'
          f' {inference_result.eval_set_id}.'
      )

    # Metric results for each invocation
    eval_metric_result_per_invocation = []

    # We also keep track of the overall score for a metric, derived from all
    # invocation. For example, if we were keeping track the metric that compares
    # how well is the final resposne as compared to a golden answer, then each
    # invocation will have the value of this metric. We will also have an
    # overall score using aggregation strategy across all invocations. This
    # would be the score for the eval case.
    overall_eval_metric_results = []

    if len(inference_result.inferences) != len(eval_case.conversation):
      raise ValueError(
          'Inferences should match conversations in eval case. Found'
          f'{len(inference_result.inferences)} inferences '
          f'{len(eval_case.conversation)} conversations in eval cases.'
      )

    # Pre-creating the EvalMetricResults entries for each invocation.
    for actual, expected in zip(
        inference_result.inferences, eval_case.conversation
    ):
      eval_metric_result_per_invocation.append(
          EvalMetricResultPerInvocation(
              actual_invocation=actual,
              expected_invocation=expected,
              # We will fill this as we evaluate each metric per invocation.
              eval_metric_results=[],
          )
      )

    for eval_metric in evaluate_config.eval_metrics:
      # Perform evaluation of the metric.
      evaluation_result = await self._evaluate_metric(
          eval_metric=eval_metric,
          actual_invocations=inference_result.inferences,
          expected_invocations=eval_case.conversation,
      )

      # Track overall scrore across all invocations.
      eval_metric_result_details = EvalMetricResultDetails(
          rubric_scores=evaluation_result.overall_rubric_scores
      )
      overall_eval_metric_results.append(
          EvalMetricResult(
              score=evaluation_result.overall_score,
              eval_status=evaluation_result.overall_eval_status,
              details=eval_metric_result_details,
              **eval_metric.model_dump(),
          )
      )

      if len(evaluation_result.per_invocation_results) != len(
          eval_metric_result_per_invocation
      ):
        raise ValueError(
            'Eval metric should return results for each invocation. Found '
            f'{len(evaluation_result.per_invocation_results)} results for '
            f'{len(eval_metric_result_per_invocation)} invocations.'
        )

      # Track score across individual invocations.
      for invocation_result, invocation in zip(
          evaluation_result.per_invocation_results,
          eval_metric_result_per_invocation,
      ):
        eval_metric_result_details = EvalMetricResultDetails(
            rubric_scores=invocation_result.rubric_scores
        )
        invocation.eval_metric_results.append(
            EvalMetricResult(
                score=invocation_result.score,
                eval_status=invocation_result.eval_status,
                details=eval_metric_result_details,
                **eval_metric.model_dump(),
            )
        )

    final_eval_status = self._generate_final_eval_status(
        overall_eval_metric_results
    )
    user_id = (
        eval_case.session_input.user_id
        if eval_case.session_input and eval_case.session_input.user_id
        else 'test_user_id'
    )

    eval_case_result = EvalCaseResult(
        eval_set_file=inference_result.eval_set_id,
        eval_set_id=inference_result.eval_set_id,
        eval_id=inference_result.eval_case_id,
        final_eval_status=final_eval_status,
        overall_eval_metric_results=overall_eval_metric_results,
        eval_metric_result_per_invocation=eval_metric_result_per_invocation,
        session_id=inference_result.session_id,
        session_details=await self._session_service.get_session(
            app_name=inference_result.app_name,
            user_id=user_id,
            session_id=inference_result.session_id,
        ),
        user_id=user_id,
    )

    return (inference_result, eval_case_result)

  async def _evaluate_metric(
      self,
      eval_metric: EvalMetric,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ) -> EvaluationResult:
    """Returns EvaluationResult obtained from evaluating a metric using an Evaluator."""

    # Get the metric evaluator from the registry.
    metric_evaluator = self._metric_evaluator_registry.get_evaluator(
        eval_metric=eval_metric
    )

    if inspect.iscoroutinefunction(metric_evaluator.evaluate_invocations):
      # Some evaluators could be async, for example those that use llm as a
      # judge, so we need to make sure that we wait on them.
      return await metric_evaluator.evaluate_invocations(
          actual_invocations=actual_invocations,
          expected_invocations=expected_invocations,
      )
    else:
      # Metrics that perform computation synchronously, mostly these don't
      # perform any i/o. An example of this would calculation of rouge_1 score.
      return metric_evaluator.evaluate_invocations(
          actual_invocations=actual_invocations,
          expected_invocations=expected_invocations,
      )

  def _generate_final_eval_status(
      self, overall_eval_metric_results: list[EvalMetricResult]
  ) -> EvalStatus:
    final_eval_status = EvalStatus.NOT_EVALUATED
    # Go over the all the eval statuses and mark the final eval status as
    # passed if all of them pass, otherwise mark the final eval status to
    # failed.
    for overall_eval_metric_result in overall_eval_metric_results:
      overall_eval_status = overall_eval_metric_result.eval_status
      if overall_eval_status == EvalStatus.PASSED:
        final_eval_status = EvalStatus.PASSED
      elif overall_eval_status == EvalStatus.NOT_EVALUATED:
        continue
      elif overall_eval_status == EvalStatus.FAILED:
        final_eval_status = EvalStatus.FAILED
        break
      else:
        raise ValueError(f'Unknown eval status: {overall_eval_status}.')

    return final_eval_status

  async def _perform_inference_sigle_eval_item(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case: EvalCase,
      root_agent: BaseAgent,
  ) -> InferenceResult:
    initial_session = eval_case.session_input
    session_id = self._session_id_supplier()
    inference_result = InferenceResult(
        app_name=app_name,
        eval_set_id=eval_set_id,
        eval_case_id=eval_case.eval_id,
        session_id=session_id,
    )

    try:
      inferences = (
          await EvaluationGenerator._generate_inferences_from_root_agent(
              invocations=eval_case.conversation,
              root_agent=root_agent,
              initial_session=initial_session,
              session_id=session_id,
              session_service=self._session_service,
              artifact_service=self._artifact_service,
          )
      )

      inference_result.inferences = inferences
      inference_result.status = InferenceStatus.SUCCESS

      return inference_result
    except Exception as e:
      # We intentionally catch the Exception as we don't failures to affect
      # other inferences.
      logger.error(
          'Inference failed for eval case `%s` with error %s',
          eval_case.eval_id,
          e,
      )
      inference_result.status = InferenceStatus.FAILURE
      inference_result.error_message = str(e)
      return inference_result
