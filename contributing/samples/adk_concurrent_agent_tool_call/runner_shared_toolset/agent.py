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

# Create a MCP toolset
mcp_toolset = MockMcpToolset()

system_prompt = """
You are a helpful assistant that can use tools to help users.
When asked to use the mcp_tool, you should call it.
"""

root_agent = Agent(
    model="gemini-2.5-flash",
    name="parallel_agent",
    description="An agent that uses a MCP toolset for testing runner close behavior.",
    instruction=system_prompt,
    tools=[mcp_toolset],
)

