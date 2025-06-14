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

from abc import ABC
from abc import abstractmethod
from typing import AsyncGenerator
from typing import Optional
from typing import Union

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..agents import Agent
from ..artifacts.base_artifact_service import BaseArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..models.base_llm import BaseLlm
from ..sessions.base_session_service import BaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from .eval_case import Invocation
from .eval_metrioc import EvalMetric
from .eval_result import EvalCaseResult
from .eval_set_results_manager import EvalSetResultsManager
from .eval_sets_manager import EvalSetsManager
from .metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY
from .metric_evaluator_registry import MetricEvaluatorRegistry


class EvalConfig(BaseModel):
  """Contains configurations need to run an Eval."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  eval_description: Optional[str] = Field(
      default=None,
      description='Optional description for this eval run.',
  )

  model_override: Optional[Union[str, BaseLlm]] = Field(
      default=None,
      description="""The model override for the agent and its sub-agents, if
specified.

This field is provided for convinience, where the root and all the sub-agent
use the same underlying model. So, instead of specifying the model overrides
for each agent in the system using `per_agent_model_override` field, use this
field.

If the value is not specified, the Eval will use whatever model value was set on
the Agent.""",
  )

  per_agent_model_override: Optional[dict[str, Union[str, BaseLlm]]] = Field(
      default=None,
      description="""A  mapping from Agent, identified by the name, to the 
models that they should be using.

While it is not common, but it is quite possible that in a multi-agent system 
there are different models used by each sub-agent. For example, some agents in a
system could use a thinking model, while others in the same multi-agent system 
may use non-thinking model.

If the root agent and all of its sub-agent use the same underlying model,
please consider using `model_override` field.""",
  )

  per_agent_instructions_override: Optional[dict[str, str]] = Field(
      default=None,
      description="""Override the instructions on a per agent basis.""",
  )

  eval_metrics: list[EvalMetric] = Field(
      description="""The list of metrics to be used in Eval.""",
  )

  labels: Optional[dict[str, str]] = Field(
      default=None,
      description="""Labels with user-defined metadata to break down billed
charges.""",
  )


class ItemToEval(BaseModel):
  """Represents a single item that needs to be evaluated."""

  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  eval_set_id: str = Field(description="""Id of the eval set.""")

  eval_case_id: str = Field(
      description="""Id of the eval case that needs to be evaluated.""",
  )


class InferenceResult(BaseModel):
  model_config = ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )

  item_to_eval: ItemToEval = Field(
      description="""The item that was evaluated."""
  )

  inferences: list[Invocation] = Field(
      description="""Inferences obtained from the Agent for the eval item."""
  )


class AgentCreator(ABC):
  """Creates an Agent for the purposes of Eval."""

  @abstractmethod
  def get_agent(
      self,
      default_model_override: Optional[Union[str, BaseLlm]] = None,
      per_agent_model_override: Optional[dict[str, Union[str, BaseLlm]]] = None,
      per_agent_instructions_override: Optional[dict[str, str]] = None,
  ) -> Agent:
    """Returns an instance of an Agent to be used for Eval purposes."""


class BaseEvalService(ABC):
  """A service that brings together all dependencies that one would need to run Evals for an ADK agent."""

  def __init__(
      self,
      agent_creator: AgentCreator,
      eval_sets_manager: EvalSetsManager,
      metric_evaluator_registry: MetricEvaluatorRegistry = DEFAULT_METRIC_EVALUATOR_REGISTRY,
      session_service: BaseSessionService = InMemorySessionService(),
      artifact_service: BaseArtifactService = InMemoryArtifactService(),
      eval_set_results_manager: Optional[EvalSetResultsManager] = None,
  ):
    self._agent_creator = agent_creator
    self._eval_sets_manager = eval_sets_manager
    self.metric_evaluator_registry = metric_evaluator_registry
    self._session_service = session_service
    self._artifact_service = artifact_service
    self._eval_set_results_manager = eval_set_results_manager

  @abstractmethod
  async def perform_inference(
      self,
      items_to_eval: list[ItemToEval],
      eval_config: EvalConfig,
  ) -> AsyncGenerator[InferenceResult, None]:
    """Returns InferenceResult obtained from the Agent as and when they are available.

    Args:
      items_to_eval: The list of items that need to be evaluated.
      eval_config: The evaluation config for this run.
    """

  @abstractmethod
  async def evaluate(
      self,
      inference_results: list[InferenceResult],
      eval_config: EvalConfig,
  ) -> AsyncGenerator[EvalCaseResult, None]:
    """Returns EvalCaseResult for each item as and when they are available.

    Args:
      inference_results: The list of items that need to be evaluated with
        inference results.
      eval_config: The evaluation config for this run.
    """
