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

from google.adk.agents import Agent
from google.adk.tools import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# Create MCP toolset
# This uses the simple-tool MCP server via stdio
# The server will be launched automatically using uvx from the subdirectory
mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=[
                "--from",
                "git+https://github.com/modelcontextprotocol/python-sdk.git#subdirectory=examples/servers/simple-tool",
                "mcp-simple-tool",
            ],
        ),
        timeout=10.0,
    )
)

# Create sub-agent with MCP tools
# This agent has direct access to MCP tools
sub_agent = Agent(
    name="mcp_helper",
    model="gemini-2.5-flash",
    description=(
        "A helpful assistant with access to MCP tools for fetching websites."
    ),
    instruction="""You are a helpful assistant with access to MCP tools.

When the user asks for help:
1. Explain what tools you have available (website fetching)
2. Use the appropriate tool if needed
3. Provide clear and helpful responses

You have access to a website fetcher tool via MCP. Use it to fetch and return website content.""",
    tools=[mcp_toolset],
)

# Wrap sub-agent as an AgentTool
# This allows the main agent to delegate tasks to the sub-agent
# The sub-agent has access to MCP tools for fetching websites
mcp_agent_tool = AgentTool(agent=sub_agent)

# Create main agent
# This agent can delegate to the sub-agent via AgentTool
root_agent = Agent(
    name="main_agent",
    model="gemini-2.5-flash",
    description="Main agent that can delegate to a sub-agent with MCP tools.",
    instruction="""You are a helpful assistant. You have access to a sub-agent (mcp_helper)
that has MCP tools for fetching websites.

When the user asks for help:
- If they need to fetch a website, call the mcp_helper tool
- Otherwise, respond directly

Always be helpful and explain what you're doing.""",
    tools=[mcp_agent_tool],
)
