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
from collections.abc import Callable
import sys
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.optimization import gepa_root_agent_optimizer
from google.adk.optimization.data_types import UnstructuredSamplingResult
from google.adk.optimization.gepa_root_agent_optimizer import _create_agent_from_candidate
from google.adk.optimization.gepa_root_agent_optimizer import _create_agent_gepa_adapter_class
from google.adk.optimization.gepa_root_agent_optimizer import _update_skill_toolset
from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizer
from google.adk.optimization.gepa_root_agent_optimizer import GEPARootAgentOptimizerConfig
from google.adk.optimization.sampler import Sampler
from google.adk.skills import models
from google.adk.tools.skill_toolset import SkillToolset
import pytest

# Spec structures used to autospec the dynamically mocked third-party `gepa`
# package. Since gepa is a lazy-loaded dependency in the runtime code, it may
# not be available in our standard hermetic test environment at import time.
# These placeholders allow us to build strict type/interface checks using
# `create_autospec` without requiring the gepa dependency.


class MockEvaluationBatchSpec:

  def __init__(self, outputs, scores, trajectories):
    self.outputs = outputs
    self.scores = scores
    self.trajectories = trajectories


class MockGEPAAdapterSpec:
  """Mock that supports generic type hints."""

  def __class_getitem__(cls, item):
    return cls


class MockAdapterModuleSpec:
  EvaluationBatch = MockEvaluationBatchSpec
  GEPAAdapter = MockGEPAAdapterSpec


class MockInstructionProposalSignatureSpec:

  @staticmethod
  def prompt_renderer(input_dict):
    pass

  @staticmethod
  def output_extractor(lm_out):
    pass


class MockInstructionProposalSpec:
  InstructionProposalSignature = MockInstructionProposalSignatureSpec


class MockStrategiesSpec:
  instruction_proposal = MockInstructionProposalSpec


class MockCoreSpec:
  adapter_module = MockAdapterModuleSpec


class MockGEPAModuleSpec:
  core = MockCoreSpec
  strategies = MockStrategiesSpec

  @staticmethod
  def optimize(*args, **kwargs):
    pass


class MockGEPAResultSpec:
  candidates: list[dict[str, str]] = []
  val_aggregate_scores: list[float] = []

  def to_dict(self) -> dict[str, Any]:
    return {}


class MockSamplerSpec:

  def get_train_example_ids(self) -> list[str]:
    return []

  def get_validation_example_ids(self) -> list[str]:
    return []

  def sample_and_score(self, *args, **kwargs):
    pass


@pytest.fixture(name="mock_gepa")
def fixture_mock_gepa(mocker):
  # mock gepa before it gets imported by the optimizer module
  mock_gepa_module = mocker.create_autospec(MockGEPAModuleSpec)
  mock_gepa_adapter_module = mocker.create_autospec(MockAdapterModuleSpec)

  mock_gepa_adapter_module.EvaluationBatch = MockEvaluationBatchSpec
  mock_gepa_adapter_module.GEPAAdapter = MockGEPAAdapterSpec

  mock_gepa_module.core = mocker.create_autospec(MockCoreSpec)
  mock_gepa_module.core.adapter = mock_gepa_adapter_module

  mock_gepa_module.strategies = mocker.create_autospec(MockStrategiesSpec)
  mock_ip = mocker.create_autospec(MockInstructionProposalSpec)
  mock_gepa_module.strategies.instruction_proposal = mock_ip
  mock_ip.InstructionProposalSignature = mocker.create_autospec(
      MockInstructionProposalSignatureSpec
  )

  mocker.patch.dict(
      sys.modules,
      {
          "gepa": mock_gepa_module,
          "gepa.core": mock_gepa_module.core,
          "gepa.core.adapter": mock_gepa_adapter_module,
          "gepa.strategies": mock_gepa_module.strategies,
          "gepa.strategies.instruction_proposal": (
              mock_gepa_module.strategies.instruction_proposal
          ),
      },
  )
  return mock_gepa_module


@pytest.fixture
def mock_sampler(mocker):
  sampler = mocker.create_autospec(MockSamplerSpec)
  sampler.get_train_example_ids.return_value = ["train1", "train2"]
  sampler.get_validation_example_ids.return_value = ["val1", "val2"]
  return sampler


@pytest.fixture
def mock_agent(mocker):
  agent = mocker.create_autospec(Agent, instance=True)
  agent.instruction = "Initial instruction"
  agent.sub_agents = {}
  agent.clone.return_value = agent
  agent.tools = []
  return agent


@pytest.fixture
def mock_adapter(mocker, mock_gepa, mock_agent, mock_sampler):
  del mock_gepa  # only needed to mock gepa in background
  loop = mocker.create_autospec(asyncio.AbstractEventLoop, instance=True)
  mock_reflection_lm = mocker.create_autospec(Callable)
  _AdapterClass = _create_agent_gepa_adapter_class()
  return _AdapterClass(mock_agent, mock_sampler, loop, mock_reflection_lm)


def test_create_agent_from_candidate(mock_agent):
  mock_agent.tools = []
  candidate = {"agent_prompt": "New prompt"}
  new_agent = _create_agent_from_candidate(mock_agent, candidate)

  mock_agent.clone.assert_called_once_with(update={"instruction": "New prompt"})
  assert new_agent == mock_agent


def test_update_skill_toolset(mocker):
  mock_skill = mocker.create_autospec(models.Skill, instance=True)
  mock_skill.name = "my_skill"
  mock_skill.instructions = "Old skill inst"
  mock_skill_copy = mocker.create_autospec(models.Skill, instance=True)
  mock_skill.model_copy.return_value = mock_skill_copy

  mock_skill_toolset = mocker.create_autospec(SkillToolset, instance=True)
  type(mock_skill_toolset).skills = mocker.PropertyMock(
      return_value=[mock_skill]
  )
  mock_new_toolset = mocker.create_autospec(SkillToolset, instance=True)
  mock_skill_toolset.clone_with_updated_skills.return_value = mock_new_toolset

  candidate = {
      "skill_instructions:my_skill": "New skill inst",
  }

  result = _update_skill_toolset(mock_skill_toolset, candidate)

  mock_skill.model_copy.assert_called_once_with(
      update={"instructions": "New skill inst"}
  )
  mock_skill_toolset.clone_with_updated_skills.assert_called_once_with(
      [mock_skill_copy]
  )
  assert result is mock_new_toolset


def test_create_agent_from_candidate_with_skills(mocker, mock_agent):
  mock_skill_toolset = mocker.create_autospec(SkillToolset, instance=True)
  mock_new_toolset = mocker.create_autospec(SkillToolset, instance=True)

  mock_update = mocker.patch.object(
      gepa_root_agent_optimizer,
      "_update_skill_toolset",
      return_value=mock_new_toolset,
      autospec=True,
  )

  mock_agent.tools = [mock_skill_toolset]

  candidate = {
      "agent_prompt": "New prompt",
      "skill_instructions:my_skill": "New skill inst",
  }

  new_agent = _create_agent_from_candidate(mock_agent, candidate)

  mock_agent.clone.assert_called_once_with(update={"instruction": "New prompt"})
  mock_update.assert_called_once_with(mock_skill_toolset, candidate)

  assert len(new_agent.tools) == 1
  assert new_agent.tools[0] is mock_new_toolset


def test_adapter_init(mocker, mock_gepa, mock_sampler, mock_agent):
  del mock_gepa  # only needed to mock gepa in background
  loop = asyncio.new_event_loop()
  _AdapterClass = _create_agent_gepa_adapter_class()
  mock_reflection_lm = mocker.create_autospec(Callable)
  adapter = _AdapterClass(mock_agent, mock_sampler, loop, mock_reflection_lm)
  assert adapter._initial_agent == mock_agent
  assert adapter._sampler == mock_sampler
  assert adapter._main_loop == loop
  assert adapter._reflection_lm == mock_reflection_lm
  assert adapter._train_example_ids == {"train1", "train2"}
  assert adapter._validation_example_ids == {"val1", "val2"}
  loop.close()


def test_adapter_evaluate_train(mocker, mock_adapter, mock_sampler, mock_agent):
  candidate = {"agent_prompt": "New prompt"}
  batch = ["train1"]

  # mock the future returned by run_coroutine_threadsafe
  mock_future = mocker.create_autospec(asyncio.Future, instance=True)
  expected_result = UnstructuredSamplingResult(
      scores={"train1": 0.8},
      data={"train1": {"output": "result"}},
  )
  mock_future.result.return_value = expected_result

  mock_rct = mocker.patch.object(
      asyncio,
      "run_coroutine_threadsafe",
      return_value=mock_future,
      autospec=True,
  )
  eval_batch = mock_adapter.evaluate(batch, candidate, capture_traces=True)

  mock_rct.assert_called_once()
  mock_sampler.sample_and_score.assert_called_once_with(
      mocker.ANY,
      example_set="train",
      batch=batch,
      capture_full_eval_data=True,
  )

  mock_agent.clone.assert_called_once_with(update={"instruction": "New prompt"})

  assert isinstance(eval_batch, MockEvaluationBatchSpec)
  assert eval_batch.scores == [0.8]
  assert eval_batch.outputs == [{"output": "result"}]
  assert eval_batch.trajectories == [{"output": "result"}]


def test_adapter_evaluate_validation(mocker, mock_adapter, mock_sampler):
  candidate = {"agent_prompt": "New prompt"}
  batch = ["val1"]

  mock_future = mocker.create_autospec(asyncio.Future, instance=True)
  expected_result = UnstructuredSamplingResult(scores={"val1": 0.5}, data={})
  mock_future.result.return_value = expected_result

  mocker.patch.object(
      asyncio,
      "run_coroutine_threadsafe",
      return_value=mock_future,
      autospec=True,
  )
  mock_adapter.evaluate(batch, candidate)

  mock_sampler.sample_and_score.assert_called_once_with(
      mocker.ANY,
      example_set="validation",
      batch=batch,
      capture_full_eval_data=False,
  )


def test_adapter_make_reflective_dataset(mock_adapter):
  candidate = {"agent_prompt": "Prompt"}
  eval_batch = MockEvaluationBatchSpec(
      outputs=[{"o": 1}, {"o": 2}],
      scores=[0.9, 0.1],
      trajectories=[{"t": "uses my_skill"}, {"t": "does not use skill"}],
  )
  components = ["agent_prompt", "skill_instructions:my_skill"]

  dataset = mock_adapter.make_reflective_dataset(
      candidate, eval_batch, components
  )

  assert dataset == {
      "agent_prompt": [
          {
              "score": 0.9,
              "eval_data": {"t": "uses my_skill"},
          },
          {
              "score": 0.1,
              "eval_data": {"t": "does not use skill"},
          },
      ],
      "skill_instructions:my_skill": [
          {
              "score": 0.9,
              "eval_data": {"t": "uses my_skill"},
          },
      ],
  }


def test_adapter_propose_new_texts(mock_gepa, mock_adapter):
  mock_adapter._reflection_lm.return_value = "lm output"

  candidate = {
      "agent_prompt": "Old prompt",
      "skill_instructions:my_skill": "Old skill inst",
  }
  reflective_dataset = {
      "agent_prompt": [{"score": 1.0, "eval_data": {}}],
      "skill_instructions:my_skill": [{"score": 0.9, "eval_data": {}}],
  }
  components = ["agent_prompt", "skill_instructions:my_skill"]

  mock_ips = (
      mock_gepa.strategies.instruction_proposal.InstructionProposalSignature
  )
  mock_ips.prompt_renderer.return_value = "rendered prompt"
  mock_ips.output_extractor.side_effect = [
      {"new_instruction": "New prompt"},
      {"new_instruction": "New skill inst"},
  ]

  new_texts = mock_adapter.propose_new_texts(
      candidate, reflective_dataset, components
  )

  assert mock_ips.prompt_renderer.call_count == 2
  assert mock_adapter._reflection_lm.call_count == 2
  assert mock_ips.output_extractor.call_count == 2
  assert new_texts == {
      "agent_prompt": "New prompt",
      "skill_instructions:my_skill": "New skill inst",
  }


async def test_optimize(mocker, mock_gepa, mock_sampler, mock_agent):
  config = GEPARootAgentOptimizerConfig()
  optimizer = GEPARootAgentOptimizer(config)

  # mock LLM
  mock_llm_class = mocker.create_autospec(Callable)
  mock_llm = mocker.create_autospec(Callable)
  mock_llm_class.return_value = mock_llm
  optimizer._llm_class = mock_llm_class

  # mock gepa.optimize return value
  mock_gepa_result = mocker.create_autospec(MockGEPAResultSpec, instance=True)
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


async def test_optimize_logs_warning_on_overlapping_ids(
    mocker, mock_gepa, mock_sampler, mock_agent
):
  # Setup overlapping IDs
  mock_sampler.get_train_example_ids.return_value = ["id1", "id2"]
  mock_sampler.get_validation_example_ids.return_value = ["id2", "id3"]

  config = GEPARootAgentOptimizerConfig()
  optimizer = GEPARootAgentOptimizer(config)

  # Mock LLM class
  mock_llm_class = mocker.create_autospec(Callable)
  optimizer._llm_class = mock_llm_class

  # Mock gepa.optimize return value
  mock_gepa_result = mocker.create_autospec(MockGEPAResultSpec, instance=True)
  mock_gepa_result.candidates = []
  mock_gepa_result.val_aggregate_scores = []
  mock_gepa_result.to_dict.return_value = {}
  mock_gepa.optimize.return_value = mock_gepa_result

  mock_logger = mocker.patch.object(
      gepa_root_agent_optimizer, "logger", autospec=True
  )

  # Run optimization
  await optimizer.optimize(mock_agent, mock_sampler)

  # Verify warning
  mock_logger.warning.assert_called_with(
      "The training and validation example UIDs overlap. This WILL cause"
      " aliasing issues unless each common UID refers to the same example"
      " in both sets."
  )
