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
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.eval_set import EvalCase
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.evaluation.local_eval_service import LocalEvalService
from google.adk.models.registry import LLMRegistry
import pytest


@pytest.fixture
def mock_eval_sets_manager():
  return mock.create_autospec(EvalSetsManager)


@pytest.fixture
def dummy_agent():
  llm = LLMRegistry.new_llm("gemini-pro")
  return LlmAgent(name="test_agent", model=llm)


@pytest.fixture
def eval_service(dummy_agent, mock_eval_sets_manager):
  return LocalEvalService(
      root_agent=dummy_agent,
      eval_sets_manager=mock_eval_sets_manager,
  )


@pytest.mark.asyncio
async def test_perform_inference_success(
    eval_service, dummy_agent, mock_eval_sets_manager
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
    eval_service, dummy_agent, mock_eval_sets_manager
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
    eval_service, mock_eval_sets_manager
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
