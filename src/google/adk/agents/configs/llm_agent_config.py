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

from pydantic import BaseModel
from pydantic import ConfigDict

from ...utils.feature_decorator import working_in_progress
from .base_agent_config import BaseAgentConfig
from .common_config import CodeConfig
from .common_config import SubAgentConfig


@working_in_progress('agent config is under development')
class AgentToolConfig(BaseModel):
  """Config for an agent tool in LlmAgent."""

  model_config = ConfigDict(extra='forbid')

  agent: SubAgentConfig
  skip_summarization: bool = False


@working_in_progress('agent config is under development')
class ToolConfig(BaseModel):
  """Config for a tool in LlmAgent."""

  model_config = ConfigDict(extra='forbid')

  code_tool: Optional[CodeConfig] = None
  """A tool defined via code.

  Examples:

    Below is a sample using ADK builtin tools:

    ```
    tools:
      - code_tool:
          builtin_name: google_search
      - code_tool:
          builtin_name: load_memory
    ```

    Below is a sample using inline code to define tools in Python:

    ```
    tools:
      - code_tool:
          py_inline:
            name: my_tool
            code: |
              def my_tool(tool_context):
                ...
    ```

    Below is a sample using custom tools in Python and Java respectively:

    ```
    tools:
      - code_tool:
          py_name: my_library.my_tools.my_tool
          java_name: com.acme.tools.MyAgent.myTool
    ```
  """

  agent_tool: Optional[AgentToolConfig] = None
  """The config-based agent tool.

  NOTE: agent tool can also be defined via code_tool. This field is only for
  using a config-based agent as tool.

  Examples:

    ```
    tools:
      - agent_tool:
          config: search_agent.yaml
          skip_summarization: true
    ```
  """


@working_in_progress('agent config is under development')
class LlmAgentConfig(BaseAgentConfig):
  """Config for LlmAgent."""

  model: Optional[str] = None
  """LlmAgent.model. Optional.

  When not set, using the same model as the parent model.
  """

  instruction: str
  """LlmAgent.instruction. Required."""

  tools: Optional[List[ToolConfig]] = None
  """LlmAgent.tools. Optional."""

  input_schema: Optional[CodeConfig] = None
  """LlmAgent.input_schema. Optional."""
  output_schema: Optional[CodeConfig] = None
  """LlmAgent.output_schema. Optional."""

  before_model_callbacks: Optional[List[CodeConfig]] = None
  after_model_callbacks: Optional[List[CodeConfig]] = None
  before_tool_callbacks: Optional[List[CodeConfig]] = None
  after_tool_callbacks: Optional[List[CodeConfig]] = None

  disallow_transfer_to_parent: bool = False
  """LlmAgent.disallow_transfer_to_parent. Optional."""

  disallow_transfer_to_peers: bool = False
  """LlmAgent.disallow_transfer_to_peers. Optional."""

  include_contents: Literal['default', 'none'] = 'default'
  """LlmAgent.include_contents. Optional."""

  output_key: Optional[str] = None
  """LlmAgent.output_key. Optional."""
