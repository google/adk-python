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

"""Tool registry for managing and discovering tools."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Set, Type, TypeVar

from .base_tool import BaseTool

T = TypeVar('T', bound=BaseTool)


class ToolRegistry:
  """A registry for managing and discovering tools.
  
  The ToolRegistry keeps track of all available tools and provides
  mechanisms for discovering and instantiating them dynamically.
  
  Attributes:
    _tools: Dictionary mapping tool names to tool classes.
    _categories: Dictionary mapping category names to sets of tool names.
  """
  
  _instance = None
  
  def __new__(cls):
    if cls._instance is None:
      cls._instance = super(ToolRegistry, cls).__new__(cls)
      cls._instance._tools: Dict[str, Type[BaseTool]] = {}
      cls._instance._categories: Dict[str, Set[str]] = {}
    return cls._instance
  
  def register_tool(
      self, tool_cls: Type[T], *, categories: Optional[List[str]] = None
  ) -> Type[T]:
    """Register a tool class with the registry.
    
    Args:
      tool_cls: The tool class to register.
      categories: Optional list of categories to associate with this tool.
      
    Returns:
      The registered tool class (for decorator pattern support).
    """
    # Get the name from the class (not an instance)
    # We'll use a dummy instance just to get the name
    dummy_args = inspect.signature(tool_cls.__init__).parameters
    init_params = {}
    for param_name, param in dummy_args.items():
      if param_name != 'self' and param.default == inspect.Parameter.empty:
        # For required parameters, provide dummy values
        if param.annotation == str:
          init_params[param_name] = f"dummy_{param_name}"
        elif param.annotation == int:
          init_params[param_name] = 0
        elif param.annotation == bool:
          init_params[param_name] = False
        else:
          # For complex types, we'll skip and handle specially
          pass
    
    # Special handling for known tools
    if tool_cls.__name__ == "AgentTool":
      # AgentTool has a required agent parameter - we'll handle this specially
      from ..agents.base_agent import BaseAgent
      class DummyAgent(BaseAgent):
        name = "dummy_agent"
        description = "dummy agent for registration"
      
      init_params["agent"] = DummyAgent()
    
    try:
      # Try to create a dummy instance to get the name
      dummy_instance = tool_cls(**init_params)
      tool_name = dummy_instance.name
    except Exception:
      # If we can't create an instance, use the class name as fallback
      tool_name = tool_cls.__name__
    
    self._tools[tool_name] = tool_cls
    
    # Add to categories
    if categories:
      for category in categories:
        if category not in self._categories:
          self._categories[category] = set()
        self._categories[category].add(tool_name)
        
    return tool_cls
  
  def get_tool_class(self, tool_name: str) -> Optional[Type[BaseTool]]:
    """Get a tool class by name.
    
    Args:
      tool_name: The name of the tool to retrieve.
      
    Returns:
      The tool class, or None if not found.
    """
    return self._tools.get(tool_name)
  
  def get_tools_by_category(self, category: str) -> List[Type[BaseTool]]:
    """Get all tool classes in a specific category.
    
    Args:
      category: The category to retrieve tools for.
      
    Returns:
      List of tool classes in the specified category.
    """
    tool_names = self._categories.get(category, set())
    return [self._tools[name] for name in tool_names if name in self._tools]
  
  def all_tools(self) -> List[Type[BaseTool]]:
    """Get all registered tool classes.
    
    Returns:
      List of all registered tool classes.
    """
    return list(self._tools.values())
  
  def all_categories(self) -> List[str]:
    """Get all registered categories.
    
    Returns:
      List of all registered categories.
    """
    return list(self._categories.keys())


# Global registry instance
registry = ToolRegistry()


def register_tool(categories: Optional[List[str]] = None):
  """Decorator for registering tools with the registry.
  
  Args:
    categories: Optional list of categories to associate with this tool.
    
  Returns:
    Decorator function that registers the tool class.
  """
  def decorator(cls):
    return registry.register_tool(cls, categories=categories)
  return decorator
