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
import sys

from google.adk.agents.llm_agent import Agent
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.gepa_root_agent_prompt_optimizer import _create_agent_gepa_adapter_class
from google.adk.optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizer
from google.adk.optimization.gepa_root_agent_prompt_optimizer import GEPARootAgentPromptOptimizerConfig
from google.adk.optimization.sampler import Sampler
import pytest


class MockEvaluationBatch:

  def __init__(self, outputs, scores, trajectories):
    self.outputs = outputs
    self.scores = scores
    self.trajectories = trajectories


class MockGEPAAdapter:
  """Mock that supports generic type hints."""

  def __class_getitem__(cls, item):
    return cls


@pytest.fixture(name="mock_gepa")
def fixture_mock_gepa(mocker):
  # mock gepa before it gets imported by the optimizer module
  mock_gepa_module = mocker.MagicMock()
  mock_gepa_adapter = mocker.MagicMock()

  mock_gepa_adapter.EvaluationBatch = MockEvaluationBatch
  mock_gepa_adapter.GEPAAdapter = MockGEPAAdapter

  mock_gepa_module.core = mocker.MagicMock()
  mock_gepa_module.core.adapter = mock_gepa_adapter

  mocker.patch.dict(
      sys.modules,
      {
          "gepa": mock_gepa_module,
          "gepa.core": mock_gepa_module.core,
          "gepa.core.adapter": mock_gepa_adapter,
      },
  )
  return mock_gepa_module


@pytest.fixture
def mock_sampler(mocker):
  sampler = mocker.MagicMock(spec=Sampler)
  sampler.get_train_example_ids.return_value = ["train1", "train2"]
  sampler.get_validation_example_ids.return_value = ["val1", "val2"]
  return sampler


@pytest.fixture
def mock_agent(mocker):
  agent = mocker.MagicMock(spec=Agent)
  agent.instruction = "Initial instruction"
  agent.sub_agents = {}
  agent.mode = None
  agent.clone.return_value = agent
  return agent


def test_adapter_init(mock_gepa, mock_sampler, mock_agent):
  del mock_gepa  # only needed to mock gepa in background
  loop = asyncio.new_event_loop()
  _AdapterClass = _create_agent_gepa_adapter_class()
  adapter = _AdapterClass(mock_agent, mock_sampler, loop)
  assert adapter._initial_agent == mock_agent
  assert adapter._sampler == mock_sampler
  assert adapter._main_loop == loop
  assert adapter._train_example_ids == {"train1", "train2"}
  assert adapter._validation_example_ids == {"val1", "val2"}
  loop.close()


def test_adapter_evaluate_train(mocker, mock_gepa, mock_sampler, mock_agent):
  del mock_gepa  # only needed to mock gepa in background
  loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
  _AdapterClass = _create_agent_gepa_adapter_class()
  adapter = _AdapterClass(mock_agent, mock_sampler, loop)

  candidate = {"agent_prompt": "New prompt"}
  batch = ["train1"]

  # mock the future returned by run_coroutine_threadsafe
  mock_future = mocker.MagicMock()
  expected_result = UnstructuredSamplingResult(
      scores={"train1": 0.8},
      data={"train1": {"output": "result"}},
  )
  mock_future.result.return_value = expected_result

  mock_rct = mocker.patch(
      "asyncio.run_coroutine_threadsafe", return_value=mock_future
  )
  eval_batch = adapter.evaluate(batch, candidate, capture_traces=True)

  mock_rct.assert_called_once()
  mock_sampler.sample_and_score.assert_called_once_with(
      mocker.ANY,
      example_set="train",
      batch=batch,
      capture_full_eval_data=True,
  )

  mock_agent.clone.assert_called_once_with(update={"instruction": "New prompt"})

  assert isinstance(eval_batch, MockEvaluationBatch)
  assert eval_batch.scores == [0.8]
  assert eval_batch.outputs == [{"output": "result"}]
  assert eval_batch.trajectories == [{"output": "result"}]


def test_adapter_evaluate_validation(
    mocker, mock_gepa, mock_sampler, mock_agent
):
  del mock_gepa  # only needed to mock gepa in background
  loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
  _AdapterClass = _create_agent_gepa_adapter_class()
  adapter = _AdapterClass(mock_agent, mock_sampler, loop)

  candidate = {"agent_prompt": "New prompt"}
  batch = ["val1"]

  mock_future = mocker.MagicMock()
  expected_result = UnstructuredSamplingResult(scores={"val1": 0.5}, data={})
  mock_future.result.return_value = expected_result

  mocker.patch("asyncio.run_coroutine_threadsafe", return_value=mock_future)
  adapter.evaluate(batch, candidate)

  mock_sampler.sample_and_score.assert_called_once_with(
      mocker.ANY,
      example_set="validation",
      batch=batch,
      capture_full_eval_data=False,
  )


def test_adapter_make_reflective_dataset(
    mocker, mock_gepa, mock_sampler, mock_agent
):
  del mock_gepa  # only needed to mock gepa in background
  loop = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
  _AdapterClass = _create_agent_gepa_adapter_class()
  adapter = _AdapterClass(mock_agent, mock_sampler, loop)

  candidate = {"agent_prompt": "Prompt"}
  eval_batch = MockEvaluationBatch(
      outputs=[{"o": 1}, {"o": 2}],
      scores=[0.9, 0.1],
      trajectories=[{"t": 1}, {"t": 2}],
  )
  components = ["component1"]

  dataset = adapter.make_reflective_dataset(candidate, eval_batch, components)

  assert "component1" in dataset
  assert len(dataset["component1"]) == 2
  assert dataset["component1"][0] == {
      "agent_prompt": "Prompt",
      "score": 0.9,
      "eval_data": {"t": 1},
  }
  assert dataset["component1"][1] == {
      "agent_prompt": "Prompt",
      "score": 0.1,
      "eval_data": {"t": 2},
  }


@pytest.mark.asyncio
async def test_optimize(mocker, mock_gepa, mock_sampler, mock_agent):
  config = GEPARootAgentPromptOptimizerConfig()
  optimizer = GEPARootAgentPromptOptimizer(config)

  # mock LLM
  mock_llm_class = mocker.MagicMock()
  mock_llm = mocker.MagicMock()
  mock_llm_class.return_value = mock_llm
  optimizer._llm_class = mock_llm_class

  # mock gepa.optimize return value
  mock_gepa_result = mocker.MagicMock()
  mock_gepa_result.candidates = [{"agent_prompt": "Optimized instruction"}]
  mock_gepa_result.val_aggregate_scores = [0.95]
  mock_gepa_result.to_dict.return_value = {"full": "result"}
  mock_gepa.optimize.return_value = mock_gepa_result

  result = await optimizer.optimize(mock_agent, mock_sampler)

  mock_gepa.optimize.assert_called_once()
  call_kwargs = mock_gepa.optimize.call_args[1]

  assert call_kwargs["seed_candidate"] == {
      "agent_prompt": "Initial instruction"
  }
  assert call_kwargs["trainset"] == ["train1", "train2"]
  assert call_kwargs["valset"] == ["val1", "val2"]

  assert len(result.optimized_agents) == 1
  assert result.optimized_agents[0].overall_score == 0.95
  mock_agent.clone.assert_called_with(
      update={"instruction": "Optimized instruction"}
  )
  assert result.gepa_result == {"full": "result"}


@pytest.mark.asyncio
async def test_optimize_logs_warning_on_overlapping_ids(
    mocker, mock_gepa, mock_sampler, mock_agent
):
  # Setup overlapping IDs
  mock_sampler.get_train_example_ids.return_value = ["id1", "id2"]
  mock_sampler.get_validation_example_ids.return_value = ["id2", "id3"]

  config = GEPARootAgentPromptOptimizerConfig()
  optimizer = GEPARootAgentPromptOptimizer(config)

  # Mock LLM class
  mock_llm_class = mocker.MagicMock()
  optimizer._llm_class = mock_llm_class

  # Mock gepa.optimize return value
  mock_gepa_result = mocker.MagicMock()
  mock_gepa_result.candidates = []
  mock_gepa_result.val_aggregate_scores = []
  mock_gepa_result.to_dict.return_value = {}
  mock_gepa.optimize.return_value = mock_gepa_result

  mock_logger = mocker.patch(
      "google.adk.optimization.gepa_root_agent_prompt_optimizer._logger"
  )

  # Run optimization
  await optimizer.optimize(mock_agent, mock_sampler)

  # Verify warning
  mock_logger.warning.assert_called_with(
      "The training and validation example UIDs overlap. This WILL cause"
      " aliasing issues unless each common UID refers to the same example"
      " in both sets."
  )
