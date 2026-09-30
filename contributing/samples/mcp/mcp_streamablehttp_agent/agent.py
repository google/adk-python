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
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

root_agent = LlmAgent(
    name='enterprise_assistant',
    instruction="""\
Help user accessing their file systems.
    """,
    tools=[
        McpToolset(
            connection_params=StreamableHTTPServerParams(
                url='http://localhost:3000/mcp',
            ),
            # don't want agent to do write operation
            # you can also do below
            # tool_filter=lambda tool, ctx=None: tool.name
            # not in [
            #     'write_file',
            #     'edit_file',
            #     'create_directory',
            #     'move_file',
            # ],
            tool_filter=[
                'read_file',
                'list_directory',
                'get_cwd',
            ],
            use_mcp_resources=True,
        )
    ],
)
