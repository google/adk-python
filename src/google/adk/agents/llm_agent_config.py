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

import logging
from typing import List
from typing import Literal
from typing import Optional

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field

from ..tools.tool_configs import ToolConfig
from .base_agent_config import BaseAgentConfig
from .common_configs import CodeConfig

logger = logging.getLogger('google_adk.' + __name__)


class LlmAgentConfig(BaseAgentConfig):
  """The config for the YAML schema of a LlmAgent."""

  model_config = ConfigDict(
      extra='forbid',
      # Allow arbitrary types to support types.ContentUnion for static_instruction.
      # ContentUnion includes PIL.Image.Image which doesn't have Pydantic schema
      # support, but we validate it at runtime using google.genai._transformers.t_content()
      arbitrary_types_allowed=True,
  )

  agent_class: str = Field(
      default='LlmAgent',
      description=(
          'The value is used to uniquely identify the LlmAgent class. If it is'
          ' empty, it is by default an LlmAgent.'
      ),
  )

  model: Optional[str] = Field(
      default=None,
      description=(
          'Optional. LlmAgent.model. If not set, the model will be inherited'
          ' from the ancestor.'
      ),
  )

  instruction: str = Field(
      description=(
          'Required. LlmAgent.instruction. Dynamic instructions with'
          ' placeholder support. Behavior: if static_instruction is None, goes'
          ' to system_instruction; if static_instruction is set, goes to user'
          ' content after static content.'
      )
  )

  static_instruction: Optional[types.ContentUnion] = Field(
      default=None,
      description=(
          'Optional. LlmAgent.static_instruction. Static content sent literally'
          ' at position 0 without placeholder processing. When set, changes'
          ' instruction behavior to go to user content instead of'
          ' system_instruction. Supports context caching. Accepts'
          ' types.ContentUnion (str, types.Content, types.Part,'
          ' PIL.Image.Image, types.File, or list[PartUnion]).'
      ),
  )

  disallow_transfer_to_parent: Optional[bool] = Field(
      default=None,
      description='Optional. LlmAgent.disallow_transfer_to_parent.',
  )

  disallow_transfer_to_peers: Optional[bool] = Field(
      default=None, description='Optional. LlmAgent.disallow_transfer_to_peers.'
  )

  input_schema: Optional[CodeConfig] = Field(
      default=None, description='Optional. LlmAgent.input_schema.'
  )

  output_schema: Optional[CodeConfig] = Field(
      default=None, description='Optional. LlmAgent.output_schema.'
  )

  output_key: Optional[str] = Field(
      default=None, description='Optional. LlmAgent.output_key.'
  )

  include_contents: Literal['default', 'none'] = Field(
      default='default', description='Optional. LlmAgent.include_contents.'
  )

  tools: Optional[list[ToolConfig]] = Field(
      default=None,
      description="""\
Optional. LlmAgent.tools.

Examples:

  For ADK built-in tools in `google.adk.tools` package, they can be referenced
  directly with the name:

    ```
    tools:
      - name: google_search
      - name: load_memory
    ```

  For user-defined tools, they can be referenced with fully qualified name:

    ```
    tools:
      - name: my_library.my_tools.my_tool
    ```

  For tools that needs to be created via functions:

    ```
    tools:
      - name: my_library.my_tools.create_tool
        args:
          - name: param1
            value: value1
          - name: param2
            value: value2
    ```

  For more advanced tools, instead of specifying arguments in config, it's
  recommended to define them in Python files and reference them. E.g.,

    ```
    # tools.py
    my_mcp_toolset = McpToolset(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "@notionhq/notion-mcp-server"],
            env={"OPENAPI_MCP_HEADERS": NOTION_HEADERS},
        )
    )
    ```

  Then, reference the toolset in config:

  ```
  tools:
    - name: tools.my_mcp_toolset
  ```""",
  )

  before_model_callbacks: Optional[List[CodeConfig]] = Field(
      default=None,
      description="""\
Optional. LlmAgent.before_model_callbacks.

Example:

  ```
  before_model_callbacks:
    - name: my_library.callbacks.before_model_callback
  ```""",
  )

  after_model_callbacks: Optional[List[CodeConfig]] = Field(
      default=None, description='Optional. LlmAgent.after_model_callbacks.'
  )

  before_tool_callbacks: Optional[List[CodeConfig]] = Field(
      default=None, description='Optional. LlmAgent.before_tool_callbacks.'
  )

  after_tool_callbacks: Optional[List[CodeConfig]] = Field(
      default=None, description='Optional. LlmAgent.after_tool_callbacks.'
  )

  generate_content_config: Optional[types.GenerateContentConfig] = Field(
      default=None, description='Optional. LlmAgent.generate_content_config.'
  )
