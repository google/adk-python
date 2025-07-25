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

from typing import Any
from typing import Union

from pydantic import Discriminator
from pydantic import RootModel

from ..utils.feature_decorator import working_in_progress
from .base_agent import BaseAgentConfig
from .llm_agent_config import LlmAgentConfig
from .loop_agent_config import LoopAgentConfig
from .parallel_agent import ParallelAgentConfig
from .sequential_agent import SequentialAgentConfig

# A discriminated union of all possible agent configurations.
ConfigsUnion = Union[
    LlmAgentConfig,
    LoopAgentConfig,
    ParallelAgentConfig,
    SequentialAgentConfig,
    BaseAgentConfig,
]


def agent_config_discriminator(v: Any):
  if isinstance(v, dict):
    agent_class = v.get("agent_class", "LlmAgent")
    if agent_class in [
        "LlmAgent",
        "LoopAgent",
        "ParallelAgent",
        "SequentialAgent",
    ]:
      return agent_class
    else:
      return "BaseAgent"

  raise ValueError(f"Invalid agent config: {v}")


# Use a RootModel to represent the agent directly at the top level.
# The `discriminator` is applied to the union within the RootModel.
@working_in_progress("AgentConfig is not ready for use.")
class AgentConfig(RootModel[ConfigsUnion]):
  """The config for the YAML schema to create an agent."""

  class Config:
    # Pydantic v2 requires this for discriminated unions on RootModel
    # This tells the model to look at the 'agent_class' field of the input
    # data to decide which model from the `ConfigsUnion` to use.
    discriminator = Discriminator(agent_config_discriminator)
