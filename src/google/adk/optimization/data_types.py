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

from typing import Any
from typing import Generic
from typing import Optional
from typing import TypeVar

from google.adk.agents.llm_agent import Agent
from pydantic import BaseModel
from pydantic import Field


class BaseSamplingResult(BaseModel):
  """Base class for evaluation results of the candidate agent on the batch of examples."""

  scores: dict[str, float] = Field(
      required=True,
      description=(
          "A map from example UID to the agent's overall score on that example."
          " (higher is better)."
      ),
  )


# SamplingResult: the per-component evaluation results for a batch of examples.
# Should at least include per-example scores and may also contain other data
# required for optimizing the agent (e.g., outputs, trajectories, and metrics).
SamplingResult = TypeVar("SamplingResult", bound=BaseSamplingResult)


class BaseAgentWithScores(BaseModel):
  """An optimized agent with its scores.

  Optimizers may use the overall_score field and can return custom metrics by
  sub-classing this class.
  """

  optimized_agent: Agent = Field(
      required=True,
      description="The optimized agent.",
  )

  overall_score: Optional[float] = Field(
      default=None,
      description="The overall score of the optimized agent.",
  )


AgentWithScores = TypeVar("AgentWithScores", bound=BaseAgentWithScores)


class OptimizerResult(BaseModel, Generic[AgentWithScores]):
  """Base class for optimizer final results."""

  optimized_agents: list[AgentWithScores] = Field(
      required=True,
      description=(
          "A list of optimized agents which cannot be considered strictly"
          " better than one another (see"
          " https://en.wikipedia.org/wiki/Pareto_front), along with scores."
      ),
  )


class UnstructuredSamplingResult(BaseSamplingResult):
  """Evaluation result providing per-example unstructured evaluation data."""

  data: Optional[dict[str, dict[str, Any]]] = Field(
      default=None,
      description=(
          "A map from example UID to JSON-serializable evaluation data useful"
          " for agent optimization. Recommended contents include inputs,"
          " trajectories, and metrics. Must be provided if requested by the"
          " optimizer."
      ),
  )
