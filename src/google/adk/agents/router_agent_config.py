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

"""Config definition for RouterAgent."""

from __future__ import annotations

from typing import Dict

from pydantic import ConfigDict
from pydantic import Field

from ..agents.base_agent_config import BaseAgentConfig
from ..features import experimental
from ..features import FeatureName


@experimental(FeatureName.AGENT_CONFIG)
class RouterAgentConfig(BaseAgentConfig):
  """The config for the YAML schema of a RouterAgent."""

  model_config = ConfigDict(
      extra="forbid",
  )

  agent_class: str = Field(
      default="RouterAgent",
      description=(
          "The value is used to uniquely identify the RouterAgent class."
      ),
  )

  classifier_agent_name: str = Field(
      description=(
          "The name of the sub-agent that acts as the routing classifier."
      ),
  )

  routing_key: str = Field(
      default="route",
      description=(
          "The JSON key to extract the chosen route from the classifier's"
          " output."
      ),
  )

  routes: Dict[str, str] = Field(
      default_factory=dict,
      description=(
          "A dictionary mapping the extracted route string to the target"
          " sub-agent name."
      ),
  )

  default_route: str = Field(
      default="",
      description=(
          "The fallback sub-agent name if the route key doesn't match any"
          " mapping."
      ),
  )
