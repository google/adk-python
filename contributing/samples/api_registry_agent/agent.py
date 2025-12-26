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

api_registry = ApiRegistry(PROJECT_ID)
registry_tools = api_registry.get_toolset(
    mcp_server_name=MCP_SERVER_NAME,
)
root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="bigquery_assistant",
    instruction=f"""
You are a helpful data analyst assistant with access to BigQuery. The project ID is: {PROJECT_ID}

When users ask about data:
- Use the project ID {PROJECT_ID} when calling BigQuery tools.
- First, explore available datasets and tables to understand what data exists.
- Check table schemas to understand the structure before querying.
- Write clear, efficient SQL queries to answer their questions.
- Explain your findings in simple, non-technical language.

Mandatory Requirements:
- Always use the BigQuery tools to fetch real data rather than making assumptions.
- For all BigQuery operations, use project_id: {PROJECT_ID}.
    """,
    tools=[registry_tools],
)
