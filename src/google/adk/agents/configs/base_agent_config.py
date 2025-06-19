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
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from ...utils.feature_decorator import working_in_progress
from .common_config import CodeConfig
from .common_config import SubAgentConfig


@working_in_progress("agent config is under development")
class BaseAgentConfig(BaseModel):
  """The config for a BaseAgent."""

  model_config = ConfigDict(extra="forbid")

  agent_class: Optional[CodeConfig] = None
  """The class to for this agent. [when unset, default is LlmAgent]

  Examples

    ```
    agent_class:
      builtin_name: LlmAgent
    ```

    ```
    agent_class:
      code:
        py_name: my_library.custom_agents.ToyAgent
        java_name: com.acme.agents.ToyAgent
    ```
  """

  name: str
  """BaseAgent.name. Required."""

  description: str = ""
  """BaseAgent.description. Optional."""

  before_agent_callbacks: Optional[List[CodeConfig]] = None
  """BaseAgent.before_agent_callbacks. Optional.

  Examples:

    Below is a sample yaml snippet of two callbacks of both python and Java:

    ```
    before_agent_callbacks:
      # No.1 callback defined via fully qualified name in both Java and Python
      - py_name: my_library.security_callbacks.before_agent_callback
        java_name: com.acme.security.Callbacks.beforeAgentCallback
      # No.2 callback defined via inline code only in Python
      - py_name: my_callback
        py_inline: |
          def my_callback(callback_context):
            ...
    ```
  """

  after_agent_callbacks: Optional[List[CodeConfig]] = None
  """BaseAgent.after_agent_callbacks. Optional."""

  sub_agents: Optional[List[SubAgentConfig]] = None
  """The sub_agents of this agent.

  Examples:

    Below is a sample with two sub-agents in yaml.
    - The first agent is defined via config.
    - The second agent is implemented Python and Java respectively.

    ```
    sub_agents:
      - config: search_agent.yaml
      - code_agent:
          py_name: my_library.my_toy_agent
          java_name: com.acme.agents.ToyAgent.toyAgent
    ```
  """
