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
from typing import Optional

from ..agents.llm_agent import Agent
from ._run_context import OptimizationRunContext
from ._run_context import OptimizerCapabilities
from .data_types import AgentWithScoresT
from .data_types import OptimizerResult
from .data_types import SamplingResultT
from .sampler import Sampler


class AgentOptimizer(ABC, Generic[SamplingResultT, AgentWithScoresT]):
  """Base class for agent optimizers."""

  @property
  def capabilities(self) -> OptimizerCapabilities:
    """Run-context instrumentation this optimizer supports (experimental).

    Conservative defaults (all unsupported) are correct for optimizers that
    do not implement the run-context seam. Callers should check capabilities
    and call ``optimize`` without a ``run_context`` when unsupported.
    """
    return OptimizerCapabilities()

  @abstractmethod
  async def optimize(
      self,
      initial_agent: Agent,
      sampler: Sampler[SamplingResultT],
      *,
      run_context: Optional[OptimizationRunContext] = None,
  ) -> OptimizerResult[AgentWithScoresT]:
    """Runs the optimizer.

    Args:
      initial_agent: The initial agent to be optimized.
      sampler: The interface used to get training and validation example UIDs,
        request agent evaluations, and get useful data for optimizing the agent.
      run_context: Optional one-shot run-scoped usage/budget/cancellation
        controls (experimental). Optimizers whose ``capabilities`` report no
        support must raise ``UnsupportedOptimizationContextError`` when a
        context is supplied, before any sampler or model work.

    Returns:
      The final result of the optimization process, containing the optimized
      agent instances along with their corresponding scores on the validation
      examples and any optimization metadata.
    """
    ...
