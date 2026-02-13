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
from typing import Literal
from typing import Optional

from ..agents.llm_agent import Agent
from .data_types import SamplingResult


class Sampler(ABC, Generic[SamplingResult]):
  """Base class for agent optimizers to sample and score candidate agents.

  The developer must implement this interface for their evaluation service to
  work with the optimizer. The optimizer will call the sample_and_score method
  to get evaluation results for the candidate agent on the batch of examples.
  """

  @abstractmethod
  def get_train_example_ids(self) -> list[str]:
    """Returns the UIDs of examples to use for training the agent."""
    ...

  @abstractmethod
  def get_validation_example_ids(self) -> list[str]:
    """Returns the UIDs of examples to use for validating the optimized agent."""
    ...

  @abstractmethod
  async def sample_and_score(
      self,
      candidate: Agent,
      example_set: Literal["train", "validation"] = "validation",
      batch: Optional[list[str]] = None,
      capture_full_eval_data: bool = False,
  ) -> SamplingResult:
    """Evaluates the candidate agent on the batch of examples.

    Args:
      candidate: The candidate agent to be evaluated.
      example_set: The set of examples to evaluate the candidate agent on.
        Possible values are "train" and "validation".
      batch: List of UIDs of examples to evaluate the candidate agent on. If not
        provided, all examples from the chosen set will be used.
      capture_full_eval_data: If false, it is enough to only calculate the
        scores for each example. If true, this method should also capture all
        other data required for optimizing the agent (e.g., outputs,
        trajectories, and tool calls).

    Returns:
      The evaluation results, containing the scores for each example and (if
      requested) other data required for optimization.
    """
    ...
