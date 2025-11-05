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
from typing import Union

from google.genai import types
from langchain_core.tools import BaseTool as LangchainBaseTool
from langchain_core.tools import Tool
from langchain_core.tools.structured import StructuredTool
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig


class LangchainTool(FunctionTool):
  """Adapter class that wraps a Langchain tool for use with ADK.

  This adapter converts Langchain tools into a format compatible with Google's
  generative AI function calling interface. It preserves the tool's name,
  description, and functionality while adapting its schema.

  The original tool's name and description can be overridden if needed.

  Args:
      tool: A Langchain tool to wrap (BaseTool or a tool with a .run method)
      name: Optional override for the tool's name
      description: Optional override for the tool's description

  Examples::

      from langchain.tools import DuckDuckGoSearchTool
      from google.genai.tools import LangchainTool

      search_tool = DuckDuckGoSearchTool()
      wrapped_tool = LangchainTool(search_tool)
  """

  _langchain_tool: Union[LangchainBaseTool, object]
  """The wrapped langchain tool."""

  def __init__(
      self,
      tool: Union[LangchainBaseTool, object],
      name: Optional[str] = None,
      description: Optional[str] = None,
  ):
    if not hasattr(tool, 'run') and not hasattr(tool, '_run'):
      raise ValueError(
          "Tool must be a Langchain tool, have a 'run' or '_run' method."
      )

    # Determine which function to use
    if isinstance(tool, StructuredTool):
      func = tool.func
      # For async tools, func might be None but coroutine exists
      if func is None and hasattr(tool, 'coroutine') and tool.coroutine:
        func = tool.coroutine
    elif hasattr(tool, '_run') or hasattr(tool, 'run'):
      func = tool._run if hasattr(tool, '_run') else tool.run
    else:
      raise ValueError(
          "This is not supported. Tool must be a Langchain tool, have a 'run'"
          " or '_run' method. The tool is: ",
          type(tool),
      )

    super().__init__(func)
    # run_manager is a special parameter for langchain tool
    self._ignore_params.append('run_manager')
    self._langchain_tool = tool

    # Set name: priority is 1) explicitly provided name, 2) tool's name, 3) default
    if name is not None:
      self.name = name
    elif hasattr(tool, 'name') and tool.name:
      self.name = tool.name
    # else: keep default from FunctionTool

    # Set description: similar priority
    if description is not None:
      self.description = description
    elif hasattr(tool, 'description') and tool.description:
      self.description = tool.description
    # else: keep default from FunctionTool

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool.

    Returns:
        A FunctionDeclaration object that describes the tool's interface.

    Raises:
        ValueError: If the tool schema cannot be correctly parsed.
    """
    try:
      # There are two types of tools:
      # 1. BaseTool: the tool is defined in langchain_core.tools.
      # 2. Other tools: the tool doesn't inherit any class but follow some
      #    conventions, like having a "run" method.
      # Handle BaseTool type (preferred Langchain approach)
      if isinstance(self._langchain_tool, LangchainBaseTool):
        tool_wrapper = Tool(
            name=self.name,
            func=self.func,
            description=self.description,
        )

        # Add schema if available
        if (
            hasattr(self._langchain_tool, 'args_schema')
            and self._langchain_tool.args_schema
        ):
          tool_wrapper.args_schema = self._langchain_tool.args_schema

          return _automatic_function_calling_util.build_function_declaration_for_langchain(
              False,
              self.name,
              self.description,
              tool_wrapper.func,
              tool_wrapper.args,
          )

      # Need to provide a way to override the function names and descriptions
      # as the original function names are mostly ".run" and the descriptions
      # may not meet users' needs
      function_decl = super()._get_declaration()
      function_decl.name = self.name
      function_decl.description = self.description
      return function_decl

    except Exception as e:
      raise ValueError(
          f'Failed to build function declaration for Langchain tool: {e}'
      ) from e

  @override
  @classmethod
  def from_config(
      cls: type[LangchainTool], config: ToolArgsConfig, config_abs_path: str
  ) -> LangchainTool:
    from ..agents import config_agent_utils

    langchain_tool_config = LangchainToolConfig.model_validate(
        config.model_dump()
    )
    tool = config_agent_utils.resolve_fully_qualified_name(
        langchain_tool_config.tool
    )
    name = langchain_tool_config.name
    description = langchain_tool_config.description
    return cls(tool, name=name, description=description)


class LangchainToolConfig(BaseToolConfig):
  tool: str
  """The fully qualified path of the Langchain tool instance."""

  name: str = ''
  """The name of the tool."""

  description: str = ''
  """The description of the tool."""
