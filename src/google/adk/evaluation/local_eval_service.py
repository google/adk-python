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
import logging
from typing import AsyncGenerator
from typing import Callable
from typing import Optional
import uuid

from typing_extensions import override

from ..agents import BaseAgent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..errors.not_found_error import NotFoundError
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..utils.feature_decorator import working_in_progress
from .base_eval_service import BaseEvalService
from .base_eval_service import EvaluateRequest
from .base_eval_service import InferenceRequest
from .base_eval_service import InferenceResult
from .base_eval_service import InferenceStatus
from .eval_result import EvalCaseResult
from .eval_set import EvalCase
from .eval_set_results_manager import EvalSetResultsManager
from .eval_sets_manager import EvalSetsManager
from .evaluation_generator import EvaluationGenerator
from .metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from .metric_evaluator_registry import MetricEvaluatorRegistry

logger = logging.getLogger('google_adk.' + __name__)

EVAL_SESSION_ID_PREFIX = '___eval___session___'


def _get_session_id() -> str:
  return f'{EVAL_SESSION_ID_PREFIX}{str(uuid.uuid4())}'


@working_in_progress("Incomplete feature, don't use yet")
class LocalEvalService(BaseEvalService):
  """An implementation of BaseEvalService, that runs the evals locally."""

  def __init__(
      self,
      root_agent: BaseAgent,
      eval_sets_manager: EvalSetsManager,
      metric_evaluator_registry: MetricEvaluatorRegistry = DEFAULT_METRIC_EVALUATOR_REGISTRY,
      session_service: BaseSessionService = InMemorySessionService(),
      artifact_service: BaseArtifactService = InMemoryArtifactService(),
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
      session_id_supplier: Callable[[], str] = _get_session_id,
  ):
    self._root_agent = root_agent
    self._eval_sets_manager = eval_sets_manager
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

    root_agent = self._root_agent.clone()

    semaphore = asyncio.Semaphore(
        value=inference_request.inference_config.parallelism
    )

    async def run_inference(eval_case):
      async with semaphore:
        return await self._perform_inference_sigle_eval_item(
            app_name=inference_request.app_name,
            eval_set_id=inference_request.eval_set_id,
            eval_case=eval_case,
            root_agent=root_agent,
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
    raise NotImplementedError()

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
