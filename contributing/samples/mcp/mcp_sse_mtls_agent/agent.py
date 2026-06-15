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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.mcp_instruction_provider import McpInstructionProvider
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset

connection_params = SseConnectionParams(
    url=os.environ.get('MCP_SERVER_URL', 'https://localhost:3000/sse'),
    headers={'Accept': 'text/event-stream'},
)

root_agent = LlmAgent(
    name='enterprise_assistant',
    model='gemini-2.5-flash',
    instruction=McpInstructionProvider(
        connection_params=connection_params,
        prompt_name='file_system_prompt',
    ),
    tools=[
        MCPToolset(
            connection_params=connection_params,
            tool_filter=[
                'read_file',
                'list_directory',
                'get_cwd',
            ],
        )
    ],
)
