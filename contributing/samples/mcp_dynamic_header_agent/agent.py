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


from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='tenant_agent',
    instruction="""You are a helpful assistant that helps users get tenant
 information. Call the get_tenant_data tool when the user asks for tenant data.""",
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url='http://localhost:3000/mcp',
            ),
            tool_filter=['get_tenant_data'],
            header_provider=lambda ctx: {'X-Tenant-ID': 'tenant1'},
        )
    ],
)
