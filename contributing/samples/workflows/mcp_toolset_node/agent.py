# Copyright 2026 Google LLC
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


import os

from google.adk import Workflow
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.workflow import ToolsetNode
from mcp import StdioServerParameters

_allowed_path = os.path.dirname(os.path.abspath(__file__))

# The MCP server is only contacted while the workflow runs, so the toolset can
# be declared here even though listing its tools requires a live connection.
filesystem_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                '-y',
                '@modelcontextprotocol/server-filesystem',
                _allowed_path,
            ],
        ),
        timeout=15,
    ),
    tool_filter=['read_file', 'list_directory'],
)


def build_args(node_input: str):
  """Turns the user's message into arguments for the MCP tool."""
  filename = node_input.strip() or 'README.md'
  return {'path': os.path.join(_allowed_path, filename)}


def summarize(node_input: dict):
  """Formats the MCP tool's response for display."""
  if node_input.get('isError'):
    return f'The MCP server reported an error: {node_input}'
  texts = [
      part.get('text', '')
      for part in node_input.get('content', [])
      if part.get('type') == 'text'
  ]
  body = '\n'.join(texts)
  return f'Read {len(body)} characters from the MCP server:\n\n{body}'


root_agent = Workflow(
    name='mcp_toolset_node_sample',
    edges=[
        (
            'START',
            build_args,
            ToolsetNode(toolset=filesystem_toolset, tool_name='read_file'),
            summarize,
        ),
    ],
)
