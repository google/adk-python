# agent.py
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

import os
import json
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Load environment variables from .env if present
load_dotenv()

# Retrieve GitHub Personal Access Token
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_PERSONAL_ACCESS_TOKEN environment variable is not set")

# Docker command for the GitHub MCP server
docker_command = "docker"
# Basic Docker arguments to launch the MCP server
docker_args = [
    "run", "-i", "--rm",
    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
    "ghcr.io/github/github-mcp-server"
]

# Instantiate the ADK LLM Agent with GitHub MCP tool
github_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="github_agent",
    instruction=(
        "You are a GitHub assistant. "
        "Use the provided MCP tool to fetch repositories, issues, PRs, and perform GitHub API operations. "
        "Ask clarifying questions only when essential details are missing."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioServerParameters(
                command=docker_command,
                args=docker_args,
                env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
            )
        )
    ],
)