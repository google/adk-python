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

"""Sample agent demonstrating Agent Registry discovery."""

import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.agent_registry import AgentRegistry

# Project and location can be set via environment variables:
# GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

# Initialize Agent Registry client
registry = AgentRegistry(project_id=project_id, location=location)

print(f"Listing agents in {project_id}/{location}...")
agents = registry.list_agents()
for agent in agents.get("agents", []):
  print(f"- Agent: {agent.get('displayName')} ({agent.get('name')})")

print(f"\nListing MCP servers in {project_id}/{location}...")
mcp_servers = registry.list_mcp_servers()
for server in mcp_servers.get("mcpServers", []):
  print(f"- MCP Server: {server.get('displayName')} ({server.get('name')})")

# Example of using a specific agent or MCP server from the registry:
# (Note: These names should be full resource names as returned by list methods)

# 1. Using a Remote A2A Agent as a sub-agent
# TODO: Replace AGENT_NAME with your agent name
remote_agent = registry.get_remote_a2a_agent(
    f"projects/{project_id}/locations/{location}/agents/AGENT_NAME"
)

# 2. Using an MCP Server in a toolset
# TODO: Replace MCP_SERVER_NAME with your MCP server name
mcp_toolset = registry.get_mcp_toolset(
    f"projects/{project_id}/locations/{location}/mcpServers/MCP_SERVER_NAME"
)

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="discovery_agent",
    instruction=(
        "You have access to tools and sub-agents discovered via Registry."
    ),
    tools=[mcp_toolset],
    sub_agents=[remote_agent],
)
