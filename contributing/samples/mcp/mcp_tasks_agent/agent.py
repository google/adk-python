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

"""Sample agent for the MCP Tasks extension.

An operation that takes minutes cannot be held open across an HTTP write
timeout, and the usual workaround -- splitting one tool into start, status and
result, and letting the model run the polling loop -- costs a model turn per
poll and only happens if the prompt says so.

With `enable_tasks=True`, the server decides per call that the work is
long-running and answers with a task handle; the toolset polls it to
completion and returns the result. The agent below sees one tool and one
result, exactly as if the call had answered inline.

Needs MCP SDK 2.x installed (`mcp>=2,<3`); the extension seam this rides on
does not exist in 1.x.

Start the server first, in another terminal:

  python contributing/samples/mcp/mcp_tasks_agent/task_server.py

Then:

  adk run contributing/samples/mcp/mcp_tasks_agent

Ask it to run the slow operation. The call takes about twenty seconds and
comes back as an ordinary tool result.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="tasks_agent",
    instruction="""\
You run long operations for the user with the `slow_operation` tool.

Call it once and report what it returns. It takes a while; that is expected,
and you do not need to poll or call it again.""",
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="http://localhost:3000/mcp",
                # The connection budget, not a cap on the operation: a task is
                # polled over as many requests as it takes.
                timeout=10.0,
            ),
            # Accept a task handle instead of a blocking call, when the server
            # offers one. Off by default; servers without the extension are
            # unaffected.
            enable_tasks=True,
        )
    ],
)
