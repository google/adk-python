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

"""Example agent demonstrating SkillToolset with directory-loaded skills and LocalEnvironment."""

from __future__ import annotations

import pathlib

from google.adk import Agent
from google.adk.environment import LocalEnvironment
from google.adk.skills import load_skills_from_dir
from google.adk.tools.skill_toolset import SkillToolset

skills = load_skills_from_dir(pathlib.Path(__file__).parent / "skills")

# Initialize SkillToolset with LocalEnvironment (no CodeExecutor needed)
skill_toolset = SkillToolset(
    skills=skills,
    environment=LocalEnvironment(),
)

root_agent = Agent(
    name="local_env_skill_agent",
    description=(
        "An agent that executes skill scripts within a local environment."
    ),
    instruction=(
        "You are a helpful assistant equipped with calculation and text"
        " formatting skills. When requested to calculate or format text, use"
        " your available skills."
    ),
    tools=[skill_toolset],
)
