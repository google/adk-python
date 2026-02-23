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

"""SkillsBench evaluation agent with SkillToolset and Gemini Flash.

This agent loads all skills from the skills/ directory and uses
SkillToolset to provide list_skills, load_skill, load_skill_resource,
and execute_skill_script tools. It is designed to be evaluated against
the SkillsBench benchmark tasks.

WARNING: This agent uses UnsafeLocalCodeExecutor for script execution.
For production use, prefer ContainerCodeExecutor or VertexAICodeExecutor.
"""

import pathlib

from google.adk import Agent
from google.adk.code_executors.unsafe_local_code_executor import UnsafeLocalCodeExecutor
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

_SKILL_NAMES = [
    "csv-aggregation",
    "json-transform",
    "html-extraction",
    "rest-client",
    "regex-replace",
    "function-scaffold",
    "statistical-calc",
    "log-parsing",
]

_skills = [load_skill_from_dir(_SKILLS_DIR / name) for name in _SKILL_NAMES]

skill_toolset = SkillToolset(
    skills=_skills,
    code_executor=UnsafeLocalCodeExecutor(),
)

root_agent = Agent(
    model="gemini-3-flash-preview",
    name="skillsbench_agent",
    description=(
        "An agent that completes tasks by discovering and using"
        " available skills from the SkillsBench benchmark."
    ),
    instruction=(
        "You are an agent that completes tasks by discovering and using"
        " available skills. Follow this workflow:\n"
        "1. Use list_skills to find relevant skills for the task.\n"
        "2. Use load_skill to read the skill's instructions carefully.\n"
        "3. Use load_skill_resource to examine references or sample data"
        " if available.\n"
        "4. Use execute_skill_script to run the skill's scripts with"
        " appropriate arguments.\n"
        "5. Interpret the output and present a clear answer.\n\n"
        "Always check skill instructions before executing scripts."
    ),
    tools=[skill_toolset],
)
