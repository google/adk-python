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

import inspect
from typing import Any
from typing import AsyncGenerator
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from google.genai import types
from opentelemetry import trace
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import override
from typing_extensions import TypeAlias

from ..events.event import Event
from ..utils.feature_decorator import working_in_progress
from .callback_context import CallbackContext
from .common_configs import AgentRefConfig
from .common_configs import CodeConfig

if TYPE_CHECKING:
  from .invocation_context import InvocationContext


TBaseAgentConfig = TypeVar('TBaseAgentConfig', bound='BaseAgentConfig')


@working_in_progress('BaseAgentConfig is not ready for use.')
class BaseAgentConfig(BaseModel):
  """The config for the YAML schema of a BaseAgent.

  Do not use this class directly. It's the base class for all agent configs.
  """

  model_config = ConfigDict(
      extra='allow',
  )

  agent_class: Union[Literal['BaseAgent'], str] = 'BaseAgent'
  """Required. The class of the agent. The value is used to differentiate
  among different agent classes."""

  name: str
  """Required. The name of the agent."""

  description: str = ''
  """Optional. The description of the agent."""

  sub_agents: Optional[List[AgentRefConfig]] = None
  """Optional. The sub-agents of the agent."""

  before_agent_callbacks: Optional[List[CodeConfig]] = None
  """Optional. The before_agent_callbacks of the agent.

  Example:

    ```
    before_agent_callbacks:
      - name: my_library.security_callbacks.before_agent_callback
    ```
  """

  after_agent_callbacks: Optional[List[CodeConfig]] = None
  """Optional. The after_agent_callbacks of the agent."""

  def to_agent_config(
      self, custom_agent_config_cls: Type[TBaseAgentConfig]
  ) -> TBaseAgentConfig:
    """Converts this config to the concrete agent config type.

    NOTE: this is for ADK framework use only.
    """
    return custom_agent_config_cls.model_validate(self.model_dump())
