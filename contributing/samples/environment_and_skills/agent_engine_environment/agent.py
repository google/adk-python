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

"""A data analysis agent that runs Python in an Agent Engine sandbox."""

import os

from google.adk import Agent
from google.adk.integrations.vmaas import AgentEngineEnvironment
from google.adk.tools.environment import EnvironmentToolset

environment = AgentEngineEnvironment(
    project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("AGENT_ENGINE_LOCATION", "us-central1"),
    # Reusing one agent engine reuses its sandbox template, so only the first
    # run waits for the template to be created.
    agent_engine_name=os.environ.get("AGENT_ENGINE_NAME"),
    # Set to continue in a sandbox from an earlier run, with its files.
    sandbox_name=os.environ.get("AGENT_ENGINE_SANDBOX_NAME"),
)

root_agent = Agent(
    name="data_analysis_agent",
    description=(
        "A data analysis agent that writes and runs Python inside an Agent"
        " Engine sandbox."
    ),
    instruction="""\
You are a data analysis assistant. You work inside an isolated Linux sandbox
with a persistent filesystem, where you can safely write files and run Python,
so you never touch the user's machine. Files you create stay in the sandbox
between turns, so reuse them instead of recreating them.

The sandbox has no internet access, so you cannot download data or install
packages. Use only the Python standard library, for example `csv`, `json`,
`random`, `statistics`, and `collections`.

To answer a question:
1. Write a short Python script into the working directory.
2. Run it with `python3`.
3. Report the result, showing the numbers you found.

Command output is limited in size, so write large results to a file and print
only a summary. Prefer writing a script and executing it over guessing. If a
command fails, read the error, fix the script, and try again.
""",
    tools=[EnvironmentToolset(environment=environment)],
)
