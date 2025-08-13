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
import logging
import sys

from ..auth.auth_tool import AuthToolArguments
from .agent_tool import AgentTool
from .apihub_tool.apihub_toolset import APIHubToolset
from .base_tool import BaseTool
from .enterprise_search_tool import enterprise_web_search_tool as enterprise_web_search
from .example_tool import ExampleTool
from .exit_loop_tool import exit_loop
from .function_tool import FunctionTool
from .get_user_choice_tool import get_user_choice_tool as get_user_choice
from .google_search_tool import google_search
from .load_artifacts_tool import load_artifacts_tool as load_artifacts
from .load_memory_tool import load_memory_tool as load_memory
from .long_running_tool import LongRunningFunctionTool
from .preload_memory_tool import preload_memory_tool as preload_memory
from .tool_context import ToolContext
from .transfer_to_agent_tool import transfer_to_agent
from .url_context_tool import url_context
from .vertex_ai_search_tool import VertexAiSearchTool

__all__ = [
    'AgentTool',
    'APIHubToolset',
    'AuthToolArguments',
    'BaseTool',
    'enterprise_web_search',
    'google_search',
    'url_context',
    'VertexAiSearchTool',
    'ExampleTool',
    'exit_loop',
    'FunctionTool',
    'get_user_choice',
    'load_artifacts',
    'load_memory',
    'LongRunningFunctionTool',
    'preload_memory',
    'ToolContext',
    'transfer_to_agent',
]


if sys.version_info < (3, 10):
  logger = logging.getLogger('google_adk.' + __name__)
  logger.warning(
      'MCP requires Python 3.10 or above. Please upgrade your Python'
      ' version in order to use it.'
  )
else:
  from .mcp_tool.mcp_toolset import MCPToolset

  __all__.extend([
      'MCPToolset',
  ])
