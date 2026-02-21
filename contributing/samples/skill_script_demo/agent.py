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

"""Sample agent with multiple skills and script execution.

This agent loads the Anthropic mcp-builder skill and a python-helper
skill, exercising all skill tools: list_skills, load_skill,
load_skill_resource, and execute_skill_script.

WARNING: This sample uses UnsafeLocalCodeExecutor, which runs scripts
in the host process with no sandboxing. For production use, prefer
ContainerCodeExecutor or VertexAICodeExecutor for isolation.
"""

import pathlib

from google.adk import Agent
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

mcp_builder_skill = load_skill_from_dir(_SKILLS_DIR / "mcp-builder")
python_helper_skill = load_skill_from_dir(_SKILLS_DIR / "python-helper")

# WARNING: UnsafeLocalCodeExecutor runs code in the host process.
# For production, use ContainerCodeExecutor or VertexAICodeExecutor.
my_skill_toolset = SkillToolset(
    skills=[mcp_builder_skill, python_helper_skill],
    code_executor=UnsafeLocalCodeExecutor(),
)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="skill_script_demo_agent",
    description=(
        "An agent that demonstrates skill script execution using"
        " the mcp-builder and python-helper skills."
    ),
    tools=[my_skill_toolset],
)
