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

"""Sample agent demonstrating the use of GCPSkillRegistry."""

from google.adk import Agent
from google.adk.integrations.skill_registry import GCPSkillRegistry
from google.adk.tools.skill_toolset import SkillToolset

# Initialize GCP Skill Registry
registry = GCPSkillRegistry(
    project_id="your-project-id", location="us-central1"
)

# Initialize SkillToolset with registry
skill_toolset = SkillToolset(skills=[], registry=registry)

root_agent = Agent(
    model="gemini-2.5-flash",
    name="skill_registry_agent",
    description=(
        "An agent that can discover and load skills from GCP Skill Registry."
    ),
    instruction=(
        "Use search_skills to find skills and load_skill to load them if"
        " needed."
    ),
    tools=[skill_toolset],
)
