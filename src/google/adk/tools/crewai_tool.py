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

from google.genai import types
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig

try:
  from crewai.tools import BaseTool as CrewaiBaseTool
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        "Crewai Tools require Python 3.10+. Please upgrade your Python version."
    ) from e
  else:
    raise ImportError(
        "Crewai Tools require pip install 'google-adk[extensions]'."
    ) from e


class CrewaiTool(FunctionTool):
  """Use this class to wrap a CrewAI tool.

  If the original tool name and description are not suitable, you can override
  them in the constructor.
  """

  tool: CrewaiBaseTool
  """The wrapped CrewAI tool."""

  def __init__(self, tool: CrewaiBaseTool, *, name: str, description: str):
    super().__init__(tool.run)
    self.tool = tool
    if name:
      self.name = name
    elif tool.name:
      # Right now, CrewAI tool name contains white spaces. White spaces are
      # not supported in our framework. So we replace them with "_".
      self.name = tool.name.replace(" ", "_").lower()
    if description:
      self.description = description
    elif tool.description:
      self.description = tool.description

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool."""
    function_declaration = _automatic_function_calling_util.build_function_declaration_for_params_for_crewai(
        False,
        self.name,
        self.description,
        self.func,
        self.tool.args_schema.model_json_schema(),
    )
    return function_declaration

  @override
  @classmethod
  def from_config(
      cls: type[CrewaiTool], config: ToolArgsConfig, config_abs_path: str
  ) -> CrewaiTool:
    from ..agents import config_agent_utils

    crewai_tool_config = CrewaiToolConfig.model_validate(config.model_dump())
    tool = config_agent_utils.resolve_fully_qualified_name(
        crewai_tool_config.tool
    )
    name = crewai_tool_config.name
    description = crewai_tool_config.description
    return cls(tool, name=name, description=description)


class CrewaiToolConfig(BaseToolConfig):
  tool: str
  """The fully qualified path of the CrewAI tool instance."""

  name: str = ""
  """The name of the tool."""

  description: str = ""
  """The description of the tool."""
