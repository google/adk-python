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
from typing import Callable

from google.genai import types
from typing_extensions import override

from . import _automatic_function_calling_util
from .function_tool import FunctionTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig
from .tool_context import ToolContext

try:
  from crewai.tools import BaseTool as CrewaiBaseTool
except ImportError as e:
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
      self.name = tool.name.replace(' ', '_').lower()
    if description:
      self.description = description
    elif tool.description:
      self.description = tool.description

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Override run_async to handle CrewAI-specific parameter filtering.

    CrewAI tools use **kwargs pattern, so we need special parameter filtering
    logic that allows all parameters to pass through while removing only
    reserved parameters like 'self' and 'tool_context'.

    Note: 'tool_context' is removed from the initial args dictionary to prevent
    duplicates, but is re-added if the function signature explicitly requires it
    as a parameter.
    """
    # Preprocess arguments (includes Pydantic model conversion)
    args_to_call = self._preprocess_args(args)

    signature = inspect.signature(self.func)
    valid_params = {param for param in signature.parameters}

    # Check if function accepts **kwargs
    has_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )

    if has_kwargs:
      # For functions with **kwargs, we pass all arguments. We defensively
      # remove arguments like `self` that are managed by the framework and not
      # intended to be passed through **kwargs.
      args_to_call.pop('self', None)
      # We also remove `tool_context` that might have been passed in `args`,
      # as it will be explicitly injected later if it's a valid parameter.
      args_to_call.pop('tool_context', None)
    else:
      # For functions without **kwargs, use the original filtering.
      args_to_call = {
          k: v for k, v in args_to_call.items() if k in valid_params
      }

    # Inject tool_context if it's an explicit parameter. This will add it
    # or overwrite any value that might have been passed in `args`.
    if 'tool_context' in valid_params:
      args_to_call['tool_context'] = tool_context

    # Check for missing mandatory arguments
    mandatory_args = self._get_mandatory_args()
    missing_mandatory_args = [
        arg for arg in mandatory_args if arg not in args_to_call
    ]

    if missing_mandatory_args:
      missing_mandatory_args_str = '\n'.join(missing_mandatory_args)
      error_str = f"""Invoking `{self.name}()` failed as the following mandatory input parameters are not present:
{missing_mandatory_args_str}
You could retry calling this tool, but it is IMPORTANT for you to provide all the mandatory parameters."""
      return {'error': error_str}

    return await self._invoke_callable(self.func, args_to_call)

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

  name: str = ''
  """The name of the tool."""

  description: str = ''
  """The description of the tool."""
