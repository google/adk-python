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

from typing import List
from typing import Literal
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from ..utils.feature_decorator import experimental
from .common_configs import AgentRefConfig
from .common_configs import CodeConfig

TBaseAgentConfig = TypeVar('TBaseAgentConfig', bound='BaseAgentConfig')


@experimental
class BaseAgentConfig(BaseModel):
  """The config for the YAML schema of a BaseAgent.

  Do not use this class directly. It's the base class for all agent configs.
  """

  model_config = ConfigDict(
      extra='allow',
  )

  agent_class: Union[Literal['BaseAgent'], str] = Field(
      default='BaseAgent',
      description=(
          'Required. The class of the agent. The value is used to differentiate'
          ' among different agent classes.'
      ),
  )

  name: str = Field(description='Required. The name of the agent.')

  description: str = Field(
      default='', description='Optional. The description of the agent.'
  )

  sub_agents: Optional[List[AgentRefConfig]] = Field(
      default=None, description='Optional. The sub-agents of the agent.'
  )

  before_agent_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description="""\
Optional. The before_agent_callbacks of the agent.

Example:

  ```
  before_agent_callbacks:
    - name: my_library.security_callbacks.before_agent_callback
  ```""",
  )

  after_agent_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description='Optional. The after_agent_callbacks of the agent.',
  )
