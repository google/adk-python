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
from pathlib import Path
import tempfile

from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

WORKPAPER_PATH = Path(
    os.getenv(
        "BILIG_WORKPAPER_PATH",
        str(Path(tempfile.gettempdir()) / "adk-bilig-quote.workpaper.json"),
    )
)

workpaper_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npm",
            args=[
                "exec",
                "--yes",
                "--package",
                "@bilig/workpaper@0.40.42",
                "--",
                "bilig-workpaper-mcp",
                "--workpaper",
                str(WORKPAPER_PATH),
                "--init-demo-workpaper",
                "--writable",
            ],
        ),
        timeout=60.0,
    )
)

root_agent = LlmAgent(
    name="bilig_workpaper_agent",
    description=(
        "An agent that edits a formula WorkPaper through a Bilig MCP server "
        "and verifies recalculated readback."
    ),
    instruction=f"""\
You are a workbook automation assistant with access to a Bilig WorkPaper MCP server.

The WorkPaper path is:
{WORKPAPER_PATH}

Use the tools to:
1. Inspect the Inputs and Summary sheets before changing anything.
2. Edit only input cells when the user asks to change workbook assumptions.
3. Read dependent formula cells after each edit and report the calculated values.
4. Confirm whether the WorkPaper JSON document persisted and restored readback matched.
5. Export the WorkPaper document when the user needs a handoff artifact.

For the demo quote workbook, changing Inputs!B3 to 0.4 should make expected
customers 8 and expected ARR 96000.
""",
    tools=[workpaper_toolset],
)
