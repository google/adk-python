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

__all__ = []

try:
  from .conversion_utils import adk_to_mcp_tool_type
  from .conversion_utils import gemini_to_json_schema
  from .mcp_session_manager import SseConnectionParams
  from .mcp_session_manager import StdioConnectionParams
  from .mcp_session_manager import StreamableHTTPConnectionParams
  from .mcp_tool import MCPTool
  from .mcp_tool import McpTool
  from .mcp_toolset import MCPToolset
  from .mcp_toolset import McpToolset

  __all__.extend([
      'adk_to_mcp_tool_type',
      'gemini_to_json_schema',
      'McpTool',
      'MCPTool',
      'McpToolset',
      'MCPToolset',
      'SseConnectionParams',
      'StdioConnectionParams',
      'StreamableHTTPConnectionParams',
  ])

except ImportError as e:
  import logging

  logger = logging.getLogger('google_adk.' + __name__)
  logger.debug('MCP Tool is not installed')
  logger.debug(e)
