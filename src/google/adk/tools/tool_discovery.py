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

"""Utility for discovering and managing tools based on user context."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Type

from .base_tool import BaseTool
from .tool_registry import registry


class ToolDiscovery:
  """A utility class for discovering appropriate tools based on context.
  
  This class helps in selecting the most appropriate tools for a given
  user context or requirement, using the tool registry.
  """
  
  @classmethod
  def get_tools_for_task(
      cls, 
      task_description: str, 
      categories: Optional[List[str]] = None,
      max_tools: int = 5
  ) -> List[BaseTool]:
    """Get appropriate tools for a given task description.
    
    This method uses a simple approach to match tools with a task description.
    A more sophisticated implementation could use semantic search or LLM-based
    matching.
    
    Args:
      task_description: Description of the task to perform.
      categories: Optional list of categories to filter tools.
      max_tools: Maximum number of tools to return.
      
    Returns:
      List of instantiated tool objects appropriate for the task.
    """
    candidate_tools = []
    
    # Start with category filtering if specified
    if categories:
      for category in categories:
        candidate_tools.extend(registry.get_tools_by_category(category))
    else:
      candidate_tools = registry.all_tools()
    
    # Score tools based on relevance to task description
    # This is a simple implementation - could be enhanced with embeddings
    scored_tools = []
    for tool_cls in candidate_tools:
      # Create a dummy instance to get name and description
      try:
        # This is a simplified approach - in practice, you'd need more
        # sophisticated instantiation logic or get metadata without instantiating
        dummy_instance = cls._create_dummy_instance(tool_cls)
        if not dummy_instance:
          continue
        
        # Calculate a simple relevance score based on word overlap
        task_words = set(task_description.lower().split())
        desc_words = set(dummy_instance.description.lower().split())
        name_words = set(dummy_instance.name.lower().split('_'))
        
        # Simple scoring mechanism
        score = len(task_words.intersection(desc_words)) + len(task_words.intersection(name_words))
        scored_tools.append((score, tool_cls))
      except Exception:
        # Skip tools that can't be easily instantiated
        continue
    
    # Sort by score (descending) and take top N
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    top_tool_classes = [tool_cls for _, tool_cls in scored_tools[:max_tools]]
    
    # Instantiate the tools with default parameters
    # In practice, this would need more sophisticated instantiation logic
    tools = []
    for tool_cls in top_tool_classes:
      tool = cls._create_instance(tool_cls)
      if tool:
        tools.append(tool)
    
    return tools
  
  @classmethod
  def _create_dummy_instance(cls, tool_cls: Type[BaseTool]) -> Optional[BaseTool]:
    """Create a dummy instance of a tool class for metadata inspection.
    
    This is a simplified implementation. A real implementation would need
    to handle various tool constructors appropriately.
    
    Args:
      tool_cls: The tool class to instantiate.
      
    Returns:
      A dummy instance of the tool, or None if instantiation fails.
    """
    import inspect
    
    # Get constructor parameters
    params = inspect.signature(tool_cls.__init__).parameters
    kwargs = {}
    
    # Fill in required parameters with dummy values
    for name, param in params.items():
      if name == 'self':
        continue
      if param.default == inspect.Parameter.empty:
        # For required parameters, provide dummy values
        if param.annotation == str:
          kwargs[name] = f"dummy_{name}"
        elif param.annotation == int:
          kwargs[name] = 0
        elif param.annotation == bool:
          kwargs[name] = False
          
    # Special handling for known tools
    if tool_cls.__name__ == "AgentTool":
      from ..agents.base_agent import BaseAgent
      class DummyAgent(BaseAgent):
        name = "dummy_agent"
        description = "dummy agent for tool discovery"
      
      kwargs["agent"] = DummyAgent()
        
    try:
      return tool_cls(**kwargs)
    except Exception:
      return None
      
  @classmethod
  def _create_instance(cls, tool_cls: Type[BaseTool]) -> Optional[BaseTool]:
    """Create a real instance of a tool class.
    
    In practice, this would need to handle tool-specific initialization
    parameters and dependencies.
    
    Args:
      tool_cls: The tool class to instantiate.
      
    Returns:
      An instance of the tool, or None if instantiation fails.
    """
    # This implementation is intentionally minimal
    # In a real implementation, you'd handle each tool type specifically
    # based on its initialization requirements
    
    # For demonstration purposes, we'll just return the dummy instance
    return cls._create_dummy_instance(tool_cls)
