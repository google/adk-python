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
import sys
from typing import Optional

from google.adk.agents.llm_agent import LlmAgent
from google.adk.errors.not_found_error import NotFoundError
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.base_eval_service import InferenceStatus
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
def mock_eval_sets_manager(mocker):
  return mocker.create_autospec(EvalSetsManager)


@pytest.fixture
def dummy_agent():
  llm = LLMRegistry.new_llm("gemini-pro")
  return LlmAgent(name="test_agent", model=llm)


@pytest.fixture
def mock_eval_set_results_manager(mocker):
  return mocker.create_autospec(EvalSetResultsManager)


@pytest.fixture
def eval_service(
    dummy_agent, mock_eval_sets_manager, mock_eval_set_results_manager
):
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeEvaluator.get_metric_info(), evaluator=FakeEvaluator
  )
  DEFAULT_METRIC_EVALUATOR_REGISTRY.register_evaluator(
      metric_info=FakeSingleSidedEvaluator.get_metric_info(),
      evaluator=FakeSingleSidedEvaluator,
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
      expected_invocations: Optional[list[Invocation]],
  ):
    if expected_invocations is None:
      raise ValueError("expected_invocations is required for this metric.")
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


class FakeSingleSidedEvaluator(Evaluator):

  def __init__(self, eval_metric: EvalMetric):
    self._eval_metric = eval_metric

  @staticmethod
  def get_metric_info() -> MetricInfo:
    return MetricInfo(
        metric_name="fake_single_sided_metric",
        description="Fake single sided metric description",
        metric_value_info=MetricValueInfo(
            interval=Interval(min_value=0.0, max_value=1.0)
        ),
    )

  def evaluate_invocations(
      self,
      actual_invocations: list[Invocation],
      expected_invocations: Optional[list[Invocation]],
  ):
    per_invocation_results = []
    for actual in actual_invocations:
      per_invocation_results.append(
          PerInvocationResult(
              actual_invocation=actual,
              score=0.995,
              eval_status=EvalStatus.PASSED,
          )
      )
    return EvaluationResult(
        overall_score=0.95,
        overall_eval_status=EvalStatus.PASSED,
        per_invocation_results=per_invocation_results,
    )


@pytest.mark.asyncio
async def test_perform_inference_success(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
):
  eval_set = EvalSet(
      eval_set_id="test_eval_set",
      eval_cases=[
          EvalCase(eval_id="case1", conversation=[], session_input=None),
          EvalCase(eval_id="case2", conversation=[], session_input=None),
      ],
  )
  mock_eval_sets_manager.get_eval_set.return_value = eval_set

  mock_inference_result = mocker.MagicMock()
  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
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
  assert eval_service._perform_inference_single_eval_item.call_count == 2


@pytest.mark.asyncio
async def test_perform_inference_with_case_ids(
    eval_service,
    dummy_agent,
    mock_eval_sets_manager,
    mocker,
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

  mock_inference_result = mocker.MagicMock()
  eval_service._perform_inference_single_eval_item = mocker.AsyncMock(
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
  eval_service._perform_inference_single_eval_item.assert_any_call(
      app_name="test_app",
      eval_set_id="test_eval_set",
      eval_case=eval_set.eval_cases[0],
      root_agent=dummy_agent,
  )
  eval_service._perform_inference_single_eval_item.assert_any_call(
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
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager, mocker
):
  invocation = Invocation(
      user_content=genai_types.Content(
          parts=[genai_types.Part(text="test user content.")]
      ),
      final_response=genai_types.Content(
          parts=[genai_types.Part(text="test final response.")]
      ),
  )
  inference_results = [
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case1",
          inferences=[invocation.model_copy(deep=True)],
          session_id="session1",
      ),
      InferenceResult(
          app_name="test_app",
          eval_set_id="test_eval_set",
          eval_case_id="case2",
          inferences=[invocation.model_copy(deep=True)],
          session_id="session2",
      ),
  ]
  eval_metric = EvalMetric(metric_name="fake_metric", threshold=0.5)
  evaluate_request = EvaluateRequest(
      inference_results=inference_results,
      evaluate_config=EvaluateConfig(eval_metrics=[eval_metric], parallelism=2),
  )

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [invocation.model_copy(deep=True)]
  mock_eval_case.conversation_scenario = None
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
    eval_service, mock_eval_sets_manager, mock_eval_set_results_manager, mocker
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

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = [
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
      invocation.model_copy(deep=True),
  ]
  mock_eval_case.conversation_scenario = None
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


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_for_conversation_scenario(
    eval_service, mock_eval_sets_manager, mocker
):
  """To be removed once evaluation is implemented for conversation scenarios."""
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
  eval_metric = EvalMetric(
      metric_name="fake_single_sided_metric", threshold=0.5
  )
  evaluate_config = EvaluateConfig(eval_metrics=[eval_metric], parallelism=1)

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.conversation = None
  mock_eval_case.conversation_scenario = mocker.MagicMock()
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )
  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.final_eval_status == EvalStatus.PASSED
  assert len(result.overall_eval_metric_results) == 1
  assert (
      result.overall_eval_metric_results[0].metric_name
      == "fake_single_sided_metric"
  )
  assert result.overall_eval_metric_results[0].score == 0.95
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3
  for i in range(3):
    invocation_result = result.eval_metric_result_per_invocation[i]
    assert invocation_result.actual_invocation == inference_result.inferences[i]
    assert invocation_result.expected_invocation == None
    assert len(invocation_result.eval_metric_results) == 1
    metric_result = invocation_result.eval_metric_results[0]
    assert metric_result.metric_name == "fake_single_sided_metric"
    assert metric_result.score == 0.995
    assert metric_result.eval_status == EvalStatus.PASSED


@pytest.mark.asyncio
async def test_evaluate_single_inference_result_for_conversation_scenario_with_unsupported_metric(
    eval_service, mock_eval_sets_manager, mocker
):
  """To be removed once evaluation is implemented for conversation scenarios."""
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

  mock_eval_case = mocker.MagicMock(spec=EvalCase)
  mock_eval_case.eval_id = "case1"
  mock_eval_case.conversation = None
  mock_eval_case.conversation_scenario = mocker.MagicMock()
  mock_eval_case.session_input = None
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  _, result = await eval_service._evaluate_single_inference_result(
      inference_result=inference_result, evaluate_config=evaluate_config
  )
  assert isinstance(result, EvalCaseResult)
  assert result.eval_id == "case1"
  assert result.final_eval_status == EvalStatus.NOT_EVALUATED
  assert len(result.overall_eval_metric_results) == 1
  assert result.overall_eval_metric_results[0].metric_name == "fake_metric"
  assert result.overall_eval_metric_results[0].score is None
  mock_eval_sets_manager.get_eval_case.assert_called_once_with(
      app_name="test_app", eval_set_id="test_eval_set", eval_case_id="case1"
  )

  assert len(result.eval_metric_result_per_invocation) == 3


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


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="MCP tool requires Python 3.10+"
)
async def test_mcp_stdio_agent_no_runtime_error(mocker):
  """Test that LocalEvalService can handle MCP stdio agents without RuntimeError.

  This is a regression test for GitHub issue #2196:
  "RuntimeError: Attempted to exit cancel scope in a different task than it was
  entered in"

  The fix ensures that Runner.close() is called to properly cleanup MCP
  connections.
  """
  import tempfile

  from google.adk.evaluation.local_eval_service import LocalEvalService
  from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
  from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
  from mcp import StdioServerParameters

  # Mock LLM responses to avoid real API calls
  from tests.unittests.testing_utils import MockModel

  mock_responses = [
      genai_types.Content(
          parts=[genai_types.Part(text="Mocked response from test agent")]
      )
  ]
  mock_model = MockModel.create(responses=mock_responses)

  # Create a test agent with MCP stdio toolset and mocked model
  test_dir = tempfile.mkdtemp()
  try:
    agent = LlmAgent(
        model=mock_model,
        name="test_mcp_agent",
        instruction="Test agent for MCP stdio regression test.",
        tools=[
            MCPToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command="npx",
                        args=[
                            "-y",
                            "@modelcontextprotocol/server-filesystem",
                            test_dir,
                        ],
                    ),
                    timeout=5,
                ),
                tool_filter=["read_file", "list_directory"],
            )
        ],
    )

    # Create a mock eval sets manager that returns an eval case
    mock_eval_sets_manager = mocker.create_autospec(EvalSetsManager)
    test_eval_case = EvalCase(
        eval_id="test_mcp_case",
        conversation=[
            Invocation(
                user_content=genai_types.Content(
                    parts=[genai_types.Part(text="List directory contents")]
                ),
            )
        ],
    )
    mock_eval_sets_manager.get_eval_case.return_value = test_eval_case
    eval_set = EvalSet(
        eval_set_id="test_set",
        eval_cases=[test_eval_case],
    )
    mock_eval_sets_manager.get_eval_set.return_value = eval_set

    # Create LocalEvalService with MCP agent
    eval_service = LocalEvalService(
        root_agent=agent,
        eval_sets_manager=mock_eval_sets_manager,
    )

    # Create inference request to actually trigger the code path with the fix
    inference_request = InferenceRequest(
        app_name="test_app",
        eval_set_id="test_set",
        inference_config=InferenceConfig(parallelism=1),
    )

    # The main test: actually call perform_inference which will trigger
    # _generate_inferences_from_root_agent where the fix is located

    # Note: In Python 3.10 and 3.11, there may be asyncio.CancelledError during cleanup
    # due to anyio cancel scope context violations when MCP toolsets are cleaned up
    # via asyncio.wait_for() in different task contexts. Python 3.12+ enhanced task
    # context management (Task.get_context(), improved context propagation) resolves this.

    try:
      results = []
      async for result in eval_service.perform_inference(inference_request):
        results.append(result)
        # We should get at least one result since we mocked the LLM
        break

      # Test passes if we get here without the cancel scope RuntimeError
      # With mocked model, we should get successful inference results
      assert len(results) >= 1

    except RuntimeError as e:
      # If we get a RuntimeError about cancel scope, the fix isn't working
      if "cancel scope" in str(e) and "different task" in str(e):
        pytest.fail(f"MCP stdio RuntimeError regression detected: {e}")
      else:
        # Other RuntimeErrors might be acceptable
        pass
    except asyncio.CancelledError as e:
      # In Python 3.10 and 3.11, anyio cancel scope context violations may manifest as CancelledError
      # when MCP RequestResponder.__exit__() is called in a different task than __enter__()
      if (
          hasattr(e, "args")
          and len(e.args) > 0
          and "cancel scope" in str(e.args[0])
      ):
        pytest.fail(f"MCP stdio cancel scope error regression detected: {e}")
      else:
        # Re-raise other CancelledErrors
        raise
    except Exception as e:
      # Check if this is the specific cancel scope error we're testing for
      if "cancel scope" in str(e) and "different task" in str(e):
        pytest.fail(f"MCP stdio RuntimeError regression detected: {e}")
      # Other exceptions are acceptable for this test

    # The main goal is to ensure the test completes without the specific
    # RuntimeError about cancel scopes. If we reach here, the fix is working.

  finally:
    # Cleanup
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)
