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

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters

load_dotenv()

CB_CONNECTION_STRING = os.getenv("CB_CONNECTION_STRING")
CB_USERNAME = os.getenv("CB_USERNAME")
CB_PASSWORD = os.getenv("CB_PASSWORD")

if not CB_CONNECTION_STRING:
  raise ValueError(
      "CB_CONNECTION_STRING environment variable not set. "
      "Please create a .env file with this variable."
  )
if not CB_USERNAME or not CB_PASSWORD:
  raise ValueError(
      "CB_USERNAME and CB_PASSWORD environment variables must be set. "
      "Please create a .env file with these variables."
  )

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="couchbase_agent",
    description=(
        "An agent that interacts with Couchbase databases using the"
        " Couchbase MCP server."
    ),
    instruction=(
        "You are a Couchbase database assistant. "
        "Use the provided tools to check cluster health, explore the data "
        "model (buckets, scopes, collections, and document schemas), "
        "run SQL++ queries, and perform key-value document operations. "
        "Use SQL++ syntax (not standard "
        "SQL) for queries. Always discover the data model before writing "
        "queries. Ask clarifying questions when unsure about bucket, scope, "
        "or collection names."
    ),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uvx",
                    args=["couchbase-mcp-server"],
                    env={
                        "CB_CONNECTION_STRING": CB_CONNECTION_STRING,
                        "CB_USERNAME": CB_USERNAME,
                        "CB_PASSWORD": CB_PASSWORD,
                        # Prevents write operations; set to "false" to enable
                        "CB_MCP_READ_ONLY_MODE": "true",
                    },
                ),
                timeout=60,
            ),
        )
    ],
)
