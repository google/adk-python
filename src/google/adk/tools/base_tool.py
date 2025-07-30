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

from abc import ABC
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar

from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict

from ..utils.variant_utils import get_google_llm_variant
from ..utils.variant_utils import GoogleLLMVariant
from .tool_context import ToolContext

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest

SelfTool = TypeVar("SelfTool", bound="BaseTool")


class BaseTool(ABC):
  """The base class for all tools."""

  name: str
  """The name of the tool."""
  description: str
  """The description of the tool."""

  is_long_running: bool = False
  """Whether the tool is a long running operation, which typically returns a
  resource id first and finishes the operation later."""

  def __init__(self, *, name, description, is_long_running: bool = False):
    self.name = name
    self.description = description
    self.is_long_running = is_long_running

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    """Gets the OpenAPI specification of this tool in the form of a FunctionDeclaration.

    NOTE:
      - Required if subclass uses the default implementation of
        `process_llm_request` to add function declaration to LLM request.
      - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
        Gemini.

    Returns:
      The FunctionDeclaration of this tool, or None if it doesn't need to be
      added to LlmRequest.config.
    """
    return None

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Runs the tool with the given arguments and context.

    NOTE:
      - Required if this tool needs to run at the client side.
      - Otherwise, can be skipped, e.g. for a built-in GoogleSearch tool for
        Gemini.

    Args:
      args: The LLM-filled arguments.
      tool_context: The context of the tool.

    Returns:
      The result of running the tool.
    """
    raise NotImplementedError(f"{type(self)} is not implemented")

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request for this tool.

    Use cases:
    - Most common use case is adding this tool to the LLM request.
    - Some tools may just preprocess the LLM request before it's sent out.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request, mutable this method.
    """
    if (function_declaration := self._get_declaration()) is None:
      return

    llm_request.tools_dict[self.name] = self
    if tool_with_function_declarations := _find_tool_with_function_declarations(
        llm_request
    ):
      if tool_with_function_declarations.function_declarations is None:
        tool_with_function_declarations.function_declarations = []
      tool_with_function_declarations.function_declarations.append(
          function_declaration
      )
    else:
      llm_request.config = (
          types.GenerateContentConfig()
          if not llm_request.config
          else llm_request.config
      )
      llm_request.config.tools = (
          [] if not llm_request.config.tools else llm_request.config.tools
      )
      llm_request.config.tools.append(
          types.Tool(function_declarations=[function_declaration])
      )

  @property
  def _api_variant(self) -> GoogleLLMVariant:
    return get_google_llm_variant()

  @classmethod
  def from_config(
      cls: Type[SelfTool], config: ToolArgsConfig, config_abs_path: str
  ) -> SelfTool:
    """Creates a tool instance from a config.

    Subclasses should override and implement this method to do custom
    initialization from a config.

    Args:
      config: The config for the tool.
      config_abs_path: The absolute path to the config file that contains the
        tool config.

    Returns:
      The tool instance.
    """
    raise NotImplementedError(f"from_config for {cls} not implemented.")


def _find_tool_with_function_declarations(
    llm_request: LlmRequest,
) -> Optional[types.Tool]:
  # TODO: add individual tool with declaration and merge in google_llm.py
  if not llm_request.config or not llm_request.config.tools:
    return None

  return next(
      (
          tool
          for tool in llm_request.config.tools
          if isinstance(tool, types.Tool) and tool.function_declarations
      ),
      None,
  )


class ToolArgsConfig(BaseModel):
  """The configuration for tool arguments.

  This config allows arbitrary key-value pairs as tool arguments.
  """

  model_config = ConfigDict(extra="allow")


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
    `config`.

    ```
    tools:
      - name: google_search
      - name: AgentTool
        config:
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
    qualified path to the tool class and `config` is the arguments for the tool.

    ```
    tools:
      - name: my_package.my_module.my_tool_class
        config:
          my_tool_arg1: value1
          my_tool_arg2: value2
    ```

    4. For user-defined functions that generate tool instances, the `name` is the
    fully qualified path to the function and `config` is passed to the function
    as arguments.

    ```
    tools:
      - name: my_package.my_module.my_tool_function
        config:
          my_function_arg1: value1
          my_function_arg2: value2
    ```

    The function must have the following signature:
    ```
    def my_function(config: ToolArgsConfig) -> BaseTool:
      ...
    ```

    5. For user-defined function tools, the `name` is the fully qualified path
    to the function.

    ```
    tools:
      - name: my_package.my_module.my_function_tool
    ```
  """

  model_config = ConfigDict(extra="forbid")

  name: str
  """The name of the tool.

  For ADK built-in tools, the name is the name of the tool, e.g. `google_search`
  or `AgentTool`.

  For user-defined tools, the name is the fully qualified path to the tool, e.g.
  `my_package.my_module.my_tool`.
  """

  args: Optional[ToolArgsConfig] = None
  """The args for the tool."""


class BaseToolConfig(BaseModel):
  """The base configurations for all the tools."""

  model_config = ConfigDict(extra="forbid")
  """Forbid extra fields."""
