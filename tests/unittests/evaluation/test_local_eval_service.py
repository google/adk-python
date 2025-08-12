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

from unittest import mock

from google.adk.agents.llm_agent import LlmAgent
from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_metrics import EvalMetric
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import Interval
from google.adk.evaluation.eval_metrics import MetricInfo
from google.adk.evaluation.eval_metrics import MetricValueInfo
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_set import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.evaluator import EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult
from google.adk.evaluation.evaluator import Evaluator
from google.adk.evaluation.evaluator import PerInvocationResult
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from google.adk.models.registry import LLMRegistry
from google.genai import types as genai_types
import pytest


@pytest.fixture
def mock_eval_sets_manager():
  return mock.create_autospec(EvalSetsManager)


@pytest.fixture
def dummy_agent():
  llm = LLMRegistry.new_llm("gemini-pro")
  return LlmAgent(name="test_agent", model=llm)


@pytest.fixture
def mock_eval_set_results_manager():
  return mock.create_autospec(EvalSetResultsManager)


@pytest.fixture
def eval_service(
    dummy_agent, mock_eval_sets_manager, mock_eval_set_results_manager
):
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeEvaluator.get_metric_info(), evaluator=FakeEvaluator
  )
  return LocalEvalService(
      root_agent=dummy_agent,
      eval_sets_manager=mock_eval_sets_manager,
      eval_set_results_manager=mock_eval_set_results_manager,
  )


class FakeEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name="fake_metric",
        description="Fake metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: list[Invocation],
  ):
    per_invocation_results = []
    for actual, expected in zip(actual_invocations, expected_invocations):
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              expected_invocation=expected,
              score=0.9,
              eval_status=EvalStatus.PASSED,
          )
      )
    return EvaluationResult(
        overall_score=0.9,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=per_invocation_results,
    )


@pytest.mark.asyncio
async def test_perform_inference_success(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
          EvalCase(eval_id="case2", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mock.MagicMock()
  eval_service._perform_inference_sigle_eval_item = mock.AsyncMock(
      return_value=mock_inference_result
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      inference_config=InferenceConfig(parallelism=2),
  )

  results = []
  async for result in eval_service.perform_inference(inference_request):
    results.append(result)

  assert len(results) == 2
  assert results[0] == mock_inference_result
  assert results[1] == mock_inference_result
  mock_eval_sets_manager.get_eval_set.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set"
  )
  assert eval_service._perform_inference_sigle_eval_item.call_count == 2


@pytest.mark.asyncio
async def test_perform_inference_with_case_ids(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
          EvalCase(eval_id="case2", conversation=[], session_input=None),
          EvalCase(eval_id="case3", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mock.MagicMock()
  eval_service._perform_inference_sigle_eval_item = mock.AsyncMock(
      return_value=mock_inference_result
  )

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_ids=["case1", "case3"],
      inference_config=InferenceConfig(parallelism=1),
  )

  results = []
  async for result in eval_service.perform_inference(inference_request):
    results.append(result)

  assert len(results) == 2
  eval_service._perform_inference_sigle_eval_item.assert_any_call(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[0],
      root_agent=dummy_agent,
  )
  eval_service._perform_inference_sigle_eval_item.assert_any_call(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[2],
      root_agent=dummy_agent,
  )


@pytest.mark.asyncio
async def test_perform_inference_eval_set_not_found(
    eval_service,
    mock_eval_sets_manager,
):
  mock_eval_sets_manager.get_eval_set.return_value = None

  inference_request = InferenceRequest(
      app_name="test_app",
      eval_set_id="not_found_set",
      inference_config=InferenceConfig(parallelism=1),
  )

  with pytest.raises(NotFoundError):
    async for _ in eval_service.perform_inference(inference_request):
      pass


@pytest.mark.asyncio
async def test_evaluate_success(
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager
):
  inference_results = [
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case1",
          inferences=[],
          session_id="session1",
      ),
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case2",
          inferences=[],
          session_id="session2",
      ),
  ]
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=[eval_metric], parallelism=2),
  )

  mock_eval_case = mock.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = []
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  results = []
  async for result in eval_service.evaluate(evaluate_request):
    results.append(result)

  assert len(results) == 2
  assert isinstance(results[0], EvalCaseResult)
  assert isinstance(results[1], EvalCaseResult)
  assert mock_eval_sets_manager.get_eval_case.call_count == 2
  assert mock_eval_set_results_manager.save_eval_set_result.call_count == 2


@pytest.mark.asyncio
async def test_evaluate_eval_case_not_found(
    eval_service,
    mock_eval_sets_manager,
):
  inference_results = [
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case1",
          inferences=[],
          session_id="session1",
      ),
  ]
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=[eval_metric], parallelism=1),
  )

  mock_eval_sets_manager.get_eval_case.return_value = None

  with pytest.raises(NotFoundError):
    async for _ in eval_service.evaluate(evaluate_request):
      pass

  mock_eval_sets_manager.get_eval_case.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_single_inference_result(
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager
):
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_result = InferenceResult(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case_id="case1",
      inferences=[
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
          invocation.model_copy(deep=True),
      ],
      session_id="session1",
  )
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mock.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
  ]
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )

  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.session_id == "session1"
  assert len(result.overall_eval_metric_results) == 1
  assert result.overall_eval_metric_results[0].metric_name == "fake_metric"
  assert result.overall_eval_metric_results[0].score == 0.9
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3
  for i in range(3):
    invocation_result = result.eval_metric_result_per_invocation[i]
    assert invocation_result.actual_invocation == inference_result.inferences[i]
    assert (
        invocation_result.expected_invocation == mock_eval_case.conversation[i]
    )
    assert len(invocation_result.eval_metric_results) == 1
    metric_result = invocation_result.eval_metric_results[0]
    assert metric_result.metric_name == "fake_metric"
    assert metric_result.score == 0.9
    assert metric_result.eval_status == EvalStatus.PASSED


def test_generate_final_eval_status_doesn_t_throw_on(eval_service):
  # How to fix if this test case fails?
  # This test case has failed mainly because a new EvalStatus got added. You
  # mostly need to update _generate_final_eval_status method to handle the new
  # eval case.

  # We go over all the possible values of EvalStatus one by one and expect
  # the _generate_final_eval_status to handle it without throwing an exeception.
  for status in EvalStatus:
    eval_metric_result = EvalMetricResult(
        metric_name="metric1", threshold=0.5, eval_status=status
    )
    eval_service._generate_final_eval_status([eval_metric_result])
