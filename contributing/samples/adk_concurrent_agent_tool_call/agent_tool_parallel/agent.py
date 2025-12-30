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

# pylint: disable=g-importing-member

import os
import sys

SAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if SAMPLES_DIR not in sys.path:
  sys.path.append(SAMPLES_DIR)

from adk_concurrent_agent_tool_call.mock_tools import MockMcpToolset
from google.adk import Agent
from google.adk.tools.agent_tool import AgentTool

# Create a MCP toolset for the sub-agent
sub_agent_mcp_toolset = MockMcpToolset()

sub_agent_system_prompt = """
You are a helpful sub-agent that can use tools to help users.
When asked to use the mcp_tool, you should call it.
"""

# Create a sub-agent that uses the MCP toolset
sub_agent = Agent(
    model="gemini-2.5-flash",
    name="sub_agent",
    description=(
        "A sub-agent that uses a MCP toolset for testing parallel AgentTool"
        " calls."
    ),
    instruction=sub_agent_system_prompt,
    tools=[sub_agent_mcp_toolset],
)

# Create the root agent that uses AgentTool to call the sub-agent
root_agent_system_prompt = """
You are a helpful assistant that can call sub-agents as tools.
When asked to use the sub_agent tool, you should call it.
You can call multiple sub_agent tools in parallel if needed.
"""

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description=(
        "A root agent that calls sub-agents via AgentTool for testing parallel"
        " execution."
    ),
    instruction=root_agent_system_prompt,
    tools=[AgentTool(agent=sub_agent)],
)
