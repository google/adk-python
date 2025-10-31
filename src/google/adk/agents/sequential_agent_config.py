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

"""Config definition for SequentialAgent."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic import Field

from ..agents.base_agent import experimental
from ..agents.base_agent_config import BaseAgentConfig


@experimental
class SequentialAgentConfig(BaseAgentConfig):
  """The config for the YAML schema of a SequentialAgent."""

  model_config = ConfigDict(
      extra="forbid",
  )

  agent_class: str = Field(
      default="SequentialAgent",
      description=(
          "The value is used to uniquely identify the SequentialAgent class."
      ),
  )
