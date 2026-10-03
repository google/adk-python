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

"""An agent that runs generated code in an OpenSandbox workspace."""

from __future__ import annotations

from google.adk import Agent
from google.adk.integrations.opensandbox import OpenSandboxEnvironment
from google.adk.tools.environment import EnvironmentToolset

root_agent = Agent(
    name="opensandbox_coding_agent",
    description=(
        "A coding agent that executes commands and edits files in an isolated "
        "OpenSandbox workspace."
    ),
    instruction="""\
You are a coding assistant. Work only through the environment tools, which run
inside an isolated OpenSandbox workspace rather than on the user's machine.

For each request:
1. Inspect relevant files before changing them.
2. Write the smallest script or data file needed to solve the request.
3. Execute the script and use its actual output in your answer.
4. If a command fails, read stderr, fix the problem, and retry.
""",
    tools=[
        EnvironmentToolset(
            environment=OpenSandboxEnvironment(
                image="python:3.11",
                timeout=300,
            )
        )
    ],
)
