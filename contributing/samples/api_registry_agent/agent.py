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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.api_registry import ApiRegistry

# TODO: Fill in with your GCloud project id and MCP server name
PROJECT_ID = "your-google-cloud-project-id"
MCP_SERVER_NAME = "your-mcp-server-name"

# Header required for BigQuery MCP server
header_provider = lambda context: {
    "x-goog-user-project": PROJECT_ID,
}
api_registry = ApiRegistry(PROJECT_ID, header_provider=header_provider)
registry_tools = api_registry.get_toolset(
    mcp_server_name=MCP_SERVER_NAME,
)
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="bigquery_assistant",
    instruction="""
Help user access their BigQuery data via API Registry tools.
    """,
    tools=[registry_tools],
)
