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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
import google.auth

BIGQUERY_AGENT_NAME = "adk_sample_bigquery_mcp_agent"
BIGQUERY_MCP_ENDPOINT = "https://bigquery.googleapis.com/mcp"
BIGQUERY_SCOPE = "https://www.googleapis.com/auth/bigquery"

# Initialize the tools to use the application default credentials.
# https://cloud.google.com/docs/authentication/provide-credentials-adc
credentials, project_id = google.auth.default(scopes=[BIGQUERY_SCOPE])
credentials.refresh(google.auth.transport.requests.Request())
oauth_token = credentials.token

bigquery_mcp_toolset = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=BIGQUERY_MCP_ENDPOINT,
        headers={"Authorization": f"Bearer {oauth_token}"},
    )
)

# The variable name `root_agent` determines what your root agent is for the
# debug CLI
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name=BIGQUERY_AGENT_NAME,
    description=(
        "Agent to answer questions about BigQuery data and models and execute"
        " SQL queries using MCP."
    ),
    instruction="""\
        You are a data science agent with access to several BigQuery tools provided via MCP.
        Make use of those tools to answer the user's questions.
    """,
    tools=[bigquery_mcp_toolset],
)
