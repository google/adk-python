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
import importlib
import logging
import sys
from typing import Any
from typing import TYPE_CHECKING

# The TYPE_CHECKING block is needed for autocomplete to work.
if TYPE_CHECKING:
  from ..auth.auth_tool import AuthToolArguments
  from .agent_tool import AgentTool
  from .api_registry import ApiRegistry
  from .apihub_tool.apihub_toolset import APIHubToolset
  from .base_tool import BaseTool
  from .discovery_engine_search_tool import DiscoveryEngineSearchTool
  from .enterprise_search_tool import enterprise_web_search_tool as enterprise_web_search
  from .example_tool import ExampleTool
  from .exit_loop_tool import exit_loop
  from .function_tool import FunctionTool
  from .get_user_choice_tool import get_user_choice_tool as get_user_choice
  from .google_maps_grounding_tool import google_maps_grounding
  from .google_search_tool import google_search
  from .load_artifacts_tool import load_artifacts_tool as load_artifacts
  from .load_memory_tool import load_memory_tool as load_memory
  from .long_running_tool import LongRunningFunctionTool
  from .preload_memory_tool import preload_memory_tool as preload_memory
  from .tool_context import ToolContext
  from .transfer_to_agent_tool import transfer_to_agent
  from .transfer_to_agent_tool import TransferToAgentTool
  from .url_context_tool import url_context
  from .vertex_ai_search_tool import VertexAiSearchTool

# If you are adding a new tool to this file, please make sure you add it to the
# lazy mapping to avoid expensive imports. If the tool is not using any third
# party dependencies, please feel free to import it eagerly at the top of this
# file.
_LAZY_MAPPING = {
    'AuthToolArguments': ('..auth.auth_tool', 'AuthToolArguments'),
    'AgentTool': ('.agent_tool', 'AgentTool'),
    'APIHubToolset': ('.apihub_tool.apihub_toolset', 'APIHubToolset'),
    'BaseTool': ('.base_tool', 'BaseTool'),
    'DiscoveryEngineSearchTool': (
        '.discovery_engine_search_tool',
        'DiscoveryEngineSearchTool',
    ),
    'enterprise_web_search': (
        '.enterprise_search_tool',
        'enterprise_web_search_tool',
    ),
    'ExampleTool': ('.example_tool', 'ExampleTool'),
    'exit_loop': ('.exit_loop_tool', 'exit_loop'),
    'FunctionTool': ('.function_tool', 'FunctionTool'),
    'get_user_choice': ('.get_user_choice_tool', 'get_user_choice_tool'),
    'google_maps_grounding': (
        '.google_maps_grounding_tool',
        'google_maps_grounding',
    ),
    'google_search': ('.google_search_tool', 'google_search'),
    'load_artifacts': ('.load_artifacts_tool', 'load_artifacts_tool'),
    'load_memory': ('.load_memory_tool', 'load_memory_tool'),
    'LongRunningFunctionTool': (
        '.long_running_tool',
        'LongRunningFunctionTool',
    ),
    'preload_memory': ('.preload_memory_tool', 'preload_memory_tool'),
    'ToolContext': ('.tool_context', 'ToolContext'),
    'transfer_to_agent': ('.transfer_to_agent_tool', 'transfer_to_agent'),
    'TransferToAgentTool': (
        '.transfer_to_agent_tool',
        'TransferToAgentTool',
    ),
    'url_context': ('.url_context_tool', 'url_context'),
    'VertexAiSearchTool': ('.vertex_ai_search_tool', 'VertexAiSearchTool'),
    'MCPToolset': ('.mcp_tool.mcp_toolset', 'MCPToolset'),
    'McpToolset': ('.mcp_tool.mcp_toolset', 'McpToolset'),
    'ApiRegistry': ('.api_registry', 'ApiRegistry'),
}

__all__ = list(_LAZY_MAPPING.keys())


def __getattr__(name: str) -> Any:
  """Lazy loads tools to avoid expensive imports."""
  if name not in _LAZY_MAPPING:
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

  module_path, attr_name = _LAZY_MAPPING[name]
  # __name__ is `google.adk.tools` and we are doing a relative import
  # from there.
  module = importlib.import_module(module_path, __name__)
  attr = getattr(module, attr_name)
  globals()[name] = attr
  return attr


# __dir__ is used to expose all public interfaces to keep mocking with autoscope
# working.
def __dir__() -> list[str]:
  return list(globals().keys()) + __all__
