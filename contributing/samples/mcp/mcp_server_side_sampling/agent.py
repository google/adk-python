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

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# This example uses the OpenAI API for both the agent and the server.
# Ensure your OPENAI_API_KEY is available as an environment variable.
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
  raise ValueError('The OPENAI_API_KEY environment variable must be set.')

# Configure the StdioServerParameters to start the mcp_server.py script
# as a subprocess. The OPENAI_API_KEY is passed to the server's environment.
server_params = StdioServerParameters(
    command='python',
    args=['mcp_server.py'],
    env={'OPENAI_API_KEY': api_key},
)

# Create the ADK MCPToolset, which connects to the FastMCP server.
# The `tool_filter` ensures that only the 'analyze_sentiment' tool is exposed
# to the agent.
mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=server_params,
    ),
    tool_filter=['analyze_sentiment'],
)

# Define the ADK agent that uses the MCP toolset.
root_agent = LlmAgent(
    model=LiteLlm(model='openai/gpt-4o'),
    name='SentimentAgent',
    instruction=(
        'You are an expert at analyzing text sentiment. Use the'
        ' analyze_sentiment tool to classify user input.'
    ),
    tools=[mcp_toolset],
)
