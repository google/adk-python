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
import logging
from typing import Any
from typing import Dict
from typing import Optional

from google.genai import types
from griptape.tools import BaseTool as GriptapeBaseTool
from typing_extensions import override

from .function_tool import FunctionTool
from .tool_configs import BaseToolConfig
from .tool_configs import ToolArgsConfig


class GriptapeTool(FunctionTool):
  """Adapter class that wraps a Griptape tool for use with ADK.

  This adapter converts Griptape tools into a format compatible with Google's
  generative AI function calling interface. It wraps the tool's activity methods
  (those decorated with @activity).

  The original tool's name and description can be overridden if needed.

  Args:
      tool: A Griptape BaseTool to wrap
      activity_method: Optional specific activity method name to wrap. If not provided,
                      the first method with @activity decorator will be used.
      name: Optional override for the tool's name
      description: Optional override for the tool's description

  Examples::

      from griptape.tools import WebScraperTool
      from google.adk.tools.griptape_tool import GriptapeTool

      scraper_tool = WebScraperTool()
      wrapped_tool = GriptapeTool(scraper_tool)

      # Or wrap a specific activity method:
      wrapped_tool = GriptapeTool(scraper_tool, activity_method="scrape_url")
  """

  _griptape_tool: GriptapeBaseTool
  """The wrapped griptape tool."""

  def __init__(
      self,
      tool: GriptapeBaseTool,
      activity_method: Optional[str] = None,
      name: Optional[str] = None,
      description: Optional[str] = None,
  ):
    # Check if tool is a Griptape BaseTool
    if not isinstance(tool, GriptapeBaseTool):
      raise ValueError('Tool must be a Griptape BaseTool.')

    # Find the activity method to wrap
    if activity_method:
      if not hasattr(tool, activity_method):
        raise AttributeError(f"Tool does not have method '{activity_method}'")
      func = getattr(tool, activity_method)
    else:
      # Find the first method with @activity decorator
      func = None
      for attr_name in dir(tool):
        attr = getattr(tool, attr_name)
        if callable(attr) and hasattr(attr, 'config'):
          func = attr
          break

      if func is None:
        raise AttributeError(
            'No activity method found. Specify activity_method parameter or'
            ' ensure tool has @activity decorated methods.'
        )

    # Extract parameter information from the Griptape schema
    schema_config = getattr(func, 'config', {})
    schema = schema_config.get('schema')

    # Parse schema to extract parameters and types
    param_info = self._parse_griptape_schema(schema)

    # Create a dynamic wrapper function
    wrapper_func = self._create_dynamic_wrapper(func, param_info)

    # Copy over function metadata
    wrapper_func.__name__ = func.__name__
    wrapper_func.__doc__ = func.__doc__

    super().__init__(wrapper_func)

    self._griptape_tool = tool
    self._activity_method = func

    # Set name: priority is 1) explicitly provided name, 2) activity method name, 3) tool's name, 4) class name
    if name is not None:
      self.name = name
    else:
      # Always prefer the activity method name for clarity
      self.name = func.__name__

    # Set description: similar priority
    if description is not None:
      self.description = description
    elif hasattr(func, 'config') and 'description' in func.config:
      self.description = func.config['description']
    # else: keep default from FunctionTool

  def _parse_griptape_schema(self, schema) -> Dict[str, Any]:
    """Parse Griptape schema to extract parameter information."""
    if not schema:
      return {}

    param_info = {}

    try:
      # Handle schema.Schema objects from Griptape
      if hasattr(schema, 'schema'):
        schema_dict = schema.schema
      else:
        schema_dict = schema

      for key, value_type in schema_dict.items():
        # Extract parameter name from Literal objects
        if hasattr(key, 'schema') and hasattr(key, 'literal'):
          param_name = key.literal
          param_description = getattr(key, 'description', '')
        else:
          param_name = str(key)
          param_description = ''

        # Map Griptape types to Python types
        if value_type == int:
          python_type = int
        elif value_type == float:
          python_type = float
        elif value_type == str:
          python_type = str
        elif value_type == bool:
          python_type = bool
        else:
          # Default to Any for complex types
          python_type = Any

        param_info[param_name] = {
            'type': python_type,
            'description': param_description,
        }

    except (AttributeError, KeyError, TypeError) as e:
      # If schema parsing fails, fall back to empty params
      logging.warning(f'Could not parse Griptape schema: {e}')
      return {}

    return param_info

  def _create_dynamic_wrapper(self, func, param_info: Dict[str, Any]):
    """Create a dynamic wrapper function with proper signature."""

    # Create parameter list for the dynamic function
    param_names = list(param_info.keys())

    # Create the wrapper function dynamically
    def wrapper_func(**kwargs):
      # Filter kwargs to only include expected parameters
      params = {k: v for k, v in kwargs.items() if k in param_names}

      # Call the original Griptape activity method
      result = func(params)

      # Convert Griptape artifacts to string for ADK compatibility
      if hasattr(result, 'value'):
        return result.value
      else:
        return str(result)

    # Build function signature dynamically
    parameters = []
    for param_name, param_data in param_info.items():
      param_type = param_data['type']
      param = inspect.Parameter(
          param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type
      )
      parameters.append(param)

    # Create new signature
    new_signature = inspect.Signature(parameters)
    wrapper_func.__signature__ = new_signature

    # Add type annotations for better ADK compatibility
    annotations = {name: info['type'] for name, info in param_info.items()}
    wrapper_func.__annotations__ = annotations

    return wrapper_func

  @override
  def _get_declaration(self) -> types.FunctionDeclaration:
    """Build the function declaration for the tool.

    Returns:
        A FunctionDeclaration object that describes the tool's interface.

    Raises:
        ValueError: If the tool schema cannot be correctly parsed.
    """
    try:
      # Get the base function declaration from the parent class
      function_decl = super()._get_declaration()

      # Override with Griptape-specific name and description
      function_decl.name = self.name
      function_decl.description = self.description

      return function_decl

    except (AttributeError, KeyError, TypeError) as e:
      raise ValueError(
          f'Failed to build function declaration for Griptape tool: {e}'
      ) from e

  @override
  @classmethod
  def from_config(
      cls: type[GriptapeTool], config: ToolArgsConfig, config_abs_path: str
  ) -> GriptapeTool:
    from ..agents import config_agent_utils

    griptape_tool_config = GriptapeToolConfig.model_validate(
        config.model_dump()
    )
    tool = config_agent_utils.resolve_fully_qualified_name(
        griptape_tool_config.tool
    )
    activity_method = griptape_tool_config.activity_method
    name = griptape_tool_config.name
    description = griptape_tool_config.description
    return cls(
        tool,
        activity_method=activity_method,
        name=name,
        description=description,
    )


class GriptapeToolConfig(BaseToolConfig):
  tool: str
  """The fully qualified path of the Griptape tool instance."""

  activity_method: Optional[str] = None
  """Optional specific activity method name to wrap. If not provided, the first method with @activity decorator will be used."""

  name: str = ''
  """The name of the tool."""

  description: str = ''
  """The description of the tool."""
