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

from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from ..utils.feature_decorator import experimental


@experimental
class BaseToolConfig(BaseModel):
  """The base class for all tool configs."""

  model_config = ConfigDict(extra="forbid")
  """Forbid extra fields."""


@experimental
class ToolArgsConfig(BaseModel):
  """Config to host free key-value pairs for the args in ToolConfig."""

  model_config = ConfigDict(extra="allow")


@experimental
class ToolConfig(BaseModel):
  """The configuration for a tool.

  The config supports these types of tools:
  1. ADK built-in tools
  2. User-defined tool instances
  3. User-defined tool classes
  4. User-defined functions that generate tool instances
  5. User-defined function tools

  For examples:

    1. For ADK built-in tool instances or classes in `google.adk.tools` package,
    they can be referenced directly with the `name` and optionally with
    `args`.

    ```
    tools:
      - name: google_search
      - name: AgentTool
        args:
          agent: ./another_agent.yaml
          skip_summarization: true
    ```

    2. For user-defined tool instances, the `name` is the fully qualified path
    to the tool instance.

    ```
    tools:
      - name: my_package.my_module.my_tool
    ```

    3. For user-defined tool classes (custom tools), the `name` is the fully
    qualified path to the tool class and `args` is the arguments for the tool.

    ```
    tools:
      - name: my_package.my_module.my_tool_class
        args:
          my_tool_arg1: value1
          my_tool_arg2: value2
    ```

    4. For user-defined functions that generate tool instances, the `name` is
    the fully qualified path to the function and `args` is passed to the
    function as arguments.

    ```
    tools:
      - name: my_package.my_module.my_tool_function
        args:
          my_function_arg1: value1
          my_function_arg2: value2
    ```

    The function must have the following signature:
    ```
    def my_function(args: ToolArgsConfig) -> BaseTool:
      ...
    ```

    5. For user-defined function tools, the `name` is the fully qualified path
    to the function.

    ```
    tools:
      - name: my_package.my_module.my_function_tool
    ```

    If the above use cases don't suffice, users can define a custom tool config
    by extending BaseToolConfig and override from_config() in the custom tool.
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  """The name of the tool.

  For ADK built-in tools, `name` is the name of the tool, e.g. `google_search`
  or `AgentTool`.

  For user-defined tools, the name is the fully qualified path to the tool, e.g.
  `my_package.my_module.my_tool`.
  """

  args: Optional[ToolArgsConfig] = None
  """The args for the tool."""
