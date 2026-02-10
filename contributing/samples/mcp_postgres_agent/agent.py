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
from google.adk.tools.mcp_tool import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.genai.types import GenerateContentConfig
from mcp import StdioServerParameters

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
  raise ValueError(
      "POSTGRES_CONNECTION_STRING environment variable not set. "
      "Please create a .env file with this variable."
  )

root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="postgres_agent",
    instruction=(
        "You are a PostgreSQL database assistant. "
        "Use the provided tools to query, manage, and interact with "
        "the PostgreSQL database. Ask clarifying questions when unsure."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="uvx",
                    args=["postgres-mcp", "--access-mode=unrestricted"],
                    env={"DATABASE_URI": POSTGRES_CONNECTION_STRING},
                ),
                timeout=60,
            ),
        )
    ],
    generate_content_config=GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
    ),
)
