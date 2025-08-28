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

from typing import Annotated
from typing import Any
from typing import get_args
from typing import Union

from pydantic import Discriminator
from pydantic import RootModel
from pydantic import Tag

from ..utils.feature_decorator import experimental
from .base_agent_config import BaseAgentConfig
from .llm_agent_config import LlmAgentConfig
from .loop_agent_config import LoopAgentConfig
from .parallel_agent_config import ParallelAgentConfig
from .sequential_agent_config import SequentialAgentConfig

_ADK_AGENT_CLASSES: set[str] = {
    "LlmAgent",
    "LoopAgent",
    "ParallelAgent",
    "SequentialAgent",
}


def agent_config_discriminator(v: Any) -> str:
  """Discriminator function that returns the tag name for Pydantic."""
  if isinstance(v, dict):
    agent_class: str = v.get("agent_class", "LlmAgent")

    # Look up the agent_class in our dynamically built mapping
    if agent_class in _ADK_AGENT_CLASSES:
      return agent_class

    # For non ADK agent classes, use BaseAgent to handle it.
    return "BaseAgent"

  raise ValueError(f"Invalid agent config: {v}")


# A discriminated union of all possible agent configurations.
ConfigsUnion = Annotated[
    Union[
        Annotated[LlmAgentConfig, Tag("LlmAgent")],
        Annotated[LoopAgentConfig, Tag("LoopAgent")],
        Annotated[ParallelAgentConfig, Tag("ParallelAgent")],
        Annotated[SequentialAgentConfig, Tag("SequentialAgent")],
        Annotated[BaseAgentConfig, Tag("BaseAgent")],
    ],
    Discriminator(agent_config_discriminator),
]


# Use a RootModel to represent the agent directly at the top level.
# The `discriminator` is applied to the union within the RootModel.
@experimental
class AgentConfig(RootModel[ConfigsUnion]):
  """The config for the YAML schema to create an agent."""
