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

from abc import ABC
from abc import abstractmethod
from typing import Generic

from ..agents.llm_agent import Agent
from .data_types import AgentWithScores
from .data_types import OptimizerResult
from .data_types import SamplingResult
from .sampler import Sampler


class AgentOptimizer(ABC, Generic[SamplingResult, AgentWithScores]):
  """Base class for agent optimizers."""

  @abstractmethod
  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[SamplingResult],
  ) -> OptimizerResult[AgentWithScores]:
    """Runs the optimizer.

    Args:
      initial_agent: The initial agent to be optimized.
      sampler: The interface used to get training and validation example UIDs,
        request agent evaluations, and get useful data for optimizing the agent.

    Returns:
      The final result of the optimization process, containing the optimized
      agent instances along with their corresponding scores on the validation
      examples and any optimization metadata.
    """
    ...
