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

from google.adk.agents.llm_agent import Agent
from google.adk.evaluation.base_eval_service import EvaluateConfig
from google.adk.evaluation.base_eval_service import EvaluateRequest
from google.adk.evaluation.base_eval_service import InferenceConfig
from google.adk.evaluation.base_eval_service import InferenceRequest
from google.adk.evaluation.base_eval_service import InferenceResult
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import InvocationEvent
from google.adk.evaluation.eval_case import InvocationEvents
from google.adk.evaluation.eval_config import EvalConfig
from google.adk.evaluation.eval_config import EvalMetric
from google.adk.evaluation.eval_metrics import EvalMetricResult
from google.adk.evaluation.eval_metrics import EvalMetricResultPerInvocation
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_sets_manager import EvalSetsManager
from google.adk.optimization.local_eval_sampler import _log_eval_summary
from google.adk.optimization.local_eval_sampler import extract_single_invocation_info
from google.adk.optimization.local_eval_sampler import extract_tool_call_data
from google.adk.optimization.local_eval_sampler import LocalEvalSampler
from google.adk.optimization.local_eval_sampler import LocalEvalSamplerConfig
from google.genai import types
import pytest


def test_log_eval_summary(mocker):
  statuses = (
      [EvalStatus.PASSED] * 3
      + [EvalStatus.FAILED] * 2
      + [EvalStatus.NOT_EVALUATED]
  )
  expected_log = "Evaluation summary: 3 PASSED, 2 FAILED, 1 OTHER"

  eval_results = [
      mocker.MagicMock(spec=EvalCaseResult, final_eval_status=status)
      for status in statuses
  ]
  mock_logger = mocker.patch(
      "google.adk.optimization.local_eval_sampler.logger"
  )

  _log_eval_summary(eval_results)

  mock_logger.info.assert_called_once_with(expected_log)


def test_extract_tool_call_data():
  # omitting IntermediateData tests as it is no longer used
  # case 1: empty invocation events
  assert not extract_tool_call_data(InvocationEvents())
  # case 2: multi call invocation events
  multi_call_invocation_events = InvocationEvents(
      invocation_events=[
          InvocationEvent(
              author="agent",
              content=types.Content(
                  parts=[
                      types.Part(
                          function_call=types.FunctionCall(
                              id="call_1",
                              name="tool_1",
                              args={"a": 1},
                          )
                      ),
                      types.Part(
                          function_call=types.FunctionCall(
                              id="call_2",
                              name="tool_2",
                              args={"b": 2},
                          )
                      ),
                      types.Part(
                          function_response=types.FunctionResponse(
                              id="call_1",
                              name="tool_1",
                              response={"result_1": "done"},
                          )
                      ),
                      types.Part(
                          function_response=types.FunctionResponse(
                              id="call_2",
                              name="tool_2",
                              response={"result_2": "done"},
                          )
                      ),
                  ]
              ),
          )
      ]
  )
  expected_entries = [
      {
          "name": "tool_1",
          "args": {"a": 1},
          "response": {"result_1": "done"},
      },
      {
          "name": "tool_2",
          "args": {"b": 2},
          "response": {"result_2": "done"},
      },
  ]
  result = extract_tool_call_data(multi_call_invocation_events)
  # order is not guaranteed
  for expected_entry in expected_entries:
    assert expected_entry in result
  assert len(result) == len(expected_entries)


def test_extract_single_invocation_info():
  invocation = Invocation(
      user_content=types.Content(
          parts=[
              types.Part(text="user thought", thought=True),
              types.Part(text="Hello agent!"),
          ]
      ),
      final_response=types.Content(
          parts=[
              types.Part(text="agent thought", thought=True),
              types.Part(text="Hello user!"),
          ]
      ),
  )

  result = extract_single_invocation_info(invocation)

  assert result == {
      "user_prompt": "Hello agent!",
      "agent_response": "Hello user!",
  }


@pytest.mark.parametrize(
    "config_kwargs, expected_attrs",
    [
        (
            {"train_eval_set": "train_set"},
            {
                "_train_eval_set": "train_set",
                "_train_eval_case_ids": ["train_set_1", "train_set_2"],
                "_validation_eval_set": "train_set",
                "_validation_eval_case_ids": ["train_set_1", "train_set_2"],
            },
        ),
        (
            {"train_eval_set": "train_set", "train_eval_case_ids": ["t1"]},
            {
                "_train_eval_case_ids": ["t1"],
                "_validation_eval_case_ids": ["t1"],
            },
        ),
        (
            {"train_eval_set": "train_set", "validation_eval_set": "val_set"},
            {
                "_validation_eval_set": "val_set",
                "_validation_eval_case_ids": ["val_set_1", "val_set_2"],
            },
        ),
        (
            {"train_eval_set": "train_set", "validation_eval_case_ids": ["v1"]},
            {
                "_validation_eval_case_ids": ["v1"],
            },
        ),
        (
            {
                "train_eval_set": "train_set",
                "train_eval_case_ids": ["t1"],
                "validation_eval_set": "val_set",
                "validation_eval_case_ids": ["v1"],
            },
            {
                "_train_eval_case_ids": ["t1"],
                "_validation_eval_set": "val_set",
                "_validation_eval_case_ids": ["v1"],
            },
        ),
    ],
)
def test_local_eval_service_interface_init(
    mocker, config_kwargs, expected_attrs
):
  mock_eval_sets_manager = mocker.MagicMock(spec=EvalSetsManager)

  def mock_get_eval_case_ids(self, eval_set_id):
    return [f"{eval_set_id}_1", f"{eval_set_id}_2"]

  mocker.patch.object(
      LocalEvalSampler,
      "_get_eval_case_ids",
      autospec=True,
      side_effect=mock_get_eval_case_ids,
  )

  config = LocalEvalSamplerConfig(
      eval_config=EvalConfig(), app_name="test_app", **config_kwargs
  )
  interface = LocalEvalSampler(config, mock_eval_sets_manager)

  for attr, expected_value in expected_attrs.items():
    assert getattr(interface, attr) == expected_value


@pytest.mark.asyncio
async def test_evaluate_agent(mocker):
  # Mocking LocalEvalService and its methods
  mock_eval_service_cls = mocker.patch(
      "google.adk.optimization.local_eval_sampler.LocalEvalService"
  )
  mock_eval_service = mock_eval_service_cls.return_value

  # mocking inference
  mock_inference_result = mocker.MagicMock(spec=InferenceResult)

  async def mock_perform_inference(*args, **kwargs):
    yield mock_inference_result

  mock_eval_service.perform_inference.side_effect = mock_perform_inference

  # mocking evaluate
  mock_eval_case_result = mocker.MagicMock(spec=EvalCaseResult)

  async def mock_evaluate(*args, **kwargs):
    yield mock_eval_case_result

  mock_eval_service.evaluate.side_effect = mock_evaluate

  # mocking get_eval_metrics_from_config
  mock_metrics = [EvalMetric(metric_name="test_metric")]
  mocker.patch(
      "google.adk.optimization.local_eval_sampler.get_eval_metrics_from_config",
      return_value=mock_metrics,
  )

  mocker.patch("google.adk.evaluation.base_eval_service.EvaluateConfig")

  # Initialize Interface
  config = LocalEvalSamplerConfig(
      eval_config=EvalConfig(),
      app_name="test_app",
      train_eval_set="train_set",
      train_eval_case_ids=["t1"],
  )
  interface = LocalEvalSampler(config, mocker.MagicMock(spec=EvalSetsManager))

  # Call _evaluate_agent
  results = await interface._evaluate_agent(
      mocker.MagicMock(spec=Agent), "train_set", ["t1"]
  )

  # Assertions
  mock_eval_service.perform_inference.assert_called_once_with(
      inference_request=InferenceRequest(
          app_name="test_app",
          eval_set_id="train_set",
          eval_case_ids=["t1"],
          inference_config=InferenceConfig(),
      )
  )
  mock_eval_service.evaluate.assert_called_once_with(
      evaluate_request=EvaluateRequest(
          inference_results=[mock_inference_result],
          evaluate_config=EvaluateConfig(eval_metrics=mock_metrics),
      )
  )
  assert results == [mock_eval_case_result]


@pytest.mark.asyncio
async def test_extract_eval_data(mocker):
  # Mock components
  mock_eval_sets_manager = mocker.MagicMock(spec=EvalSetsManager)
  mock_eval_case = mocker.MagicMock()
  mock_eval_case.conversation_scenario = "test_scenario"
  mock_eval_sets_manager.get_eval_case.return_value = mock_eval_case

  # Mock per invocation result
  mock_actual_invocation = mocker.MagicMock(spec=Invocation)
  mock_expected_invocation = mocker.MagicMock(spec=Invocation)
  mock_metric_result = mocker.MagicMock(spec=EvalMetricResult)
  mock_metric_result.metric_name = "test_metric"
  mock_metric_result.score = 0.854  # should be rounded to 0.85
  mock_metric_result.eval_status = EvalStatus.PASSED

  mock_per_inv_result = mocker.MagicMock(spec=EvalMetricResultPerInvocation)
  mock_per_inv_result.actual_invocation = mock_actual_invocation
  mock_per_inv_result.expected_invocation = mock_expected_invocation
  mock_per_inv_result.eval_metric_results = [mock_metric_result]

  mock_eval_result = mocker.MagicMock(spec=EvalCaseResult)
  mock_eval_result.eval_id = "t1"
  mock_eval_result.eval_metric_result_per_invocation = [mock_per_inv_result]

  # Mock extract_single_invocation_info
  mocker.patch(
      "google.adk.optimization.local_eval_sampler.extract_single_invocation_info",
      side_effect=[{"info": "actual"}, {"info": "expected"}],
  )

  # Initialize Interface
  config = LocalEvalSamplerConfig(
      eval_config=EvalConfig(),
      app_name="test_app",
      train_eval_set="train_set",
      train_eval_case_ids=["t1"],
  )
  interface = LocalEvalSampler(config, mock_eval_sets_manager)

  # Call _extract_eval_data
  eval_data = interface._extract_eval_data("train_set", [mock_eval_result])

  # Assertions
  assert "t1" in eval_data
  assert eval_data["t1"]["conversation_scenario"] == "test_scenario"
  assert len(eval_data["t1"]["invocations"]) == 1
  inv = eval_data["t1"]["invocations"][0]
  assert inv["actual_invocation"] == {"info": "actual"}
  assert inv["expected_invocation"] == {"info": "expected"}
  assert inv["eval_metric_results"] == [
      {"metric_name": "test_metric", "score": 0.85, "eval_status": "PASSED"}
  ]


@pytest.mark.asyncio
async def test_sample_and_score(mocker):
  # Mock results
  mock_eval_result_1 = mocker.MagicMock(spec=EvalCaseResult)
  mock_eval_result_1.eval_id = "t1"
  mock_eval_result_1.final_eval_status = EvalStatus.PASSED

  mock_eval_result_2 = mocker.MagicMock(spec=EvalCaseResult)
  mock_eval_result_2.eval_id = "t2"
  mock_eval_result_2.final_eval_status = EvalStatus.FAILED

  eval_results = [mock_eval_result_1, mock_eval_result_2]

  # Initialize Interface
  config = LocalEvalSamplerConfig(
      eval_config=EvalConfig(),
      app_name="test_app",
      train_eval_set="train_set",
      train_eval_case_ids=["t1", "t2"],
  )
  interface = LocalEvalSampler(config, mocker.MagicMock(spec=EvalSetsManager))

  # Patch internal methods
  mocker.patch.object(interface, "_evaluate_agent", return_value=eval_results)
  mock_log_summary = mocker.patch(
      "google.adk.optimization.local_eval_sampler._log_eval_summary"
  )
  mock_extract_data = mocker.patch.object(
      interface, "_extract_eval_data", return_value={"t1": {}, "t2": {}}
  )

  # Call sample_and_score
  result = await interface.sample_and_score(
      mocker.MagicMock(spec=Agent),
      example_set="train",
      capture_full_eval_data=True,
  )

  # Assertions
  assert result.scores == {"t1": 1.0, "t2": 0.0}
  assert result.data == {"t1": {}, "t2": {}}
  mock_log_summary.assert_called_once_with(eval_results)
  mock_extract_data.assert_called_once_with("train_set", eval_results)
