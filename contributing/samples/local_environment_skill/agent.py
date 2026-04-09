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
import pathlib

from google.adk import Agent
from google.adk.environment import LocalEnvironment
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.environment import EnvironmentToolset
from google.genai import types


class GetTimezoneTool(BaseTool):
  """A tool to get the timezone for a given location."""

  def __init__(self):
    super().__init__(
        name="get_timezone",
        description="Returns the timezone for a given location.",
    )

  def _get_declaration(self) -> types.FunctionDeclaration | None:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get the timezone for.",
                },
            },
            "required": ["location"],
        },
    )

  async def run_async(self, *, args: dict, tool_context) -> str:
    return f"The timezone for {args['location']} is UTC+00:00."


def get_wind_speed(location: str) -> str:
  """Returns the current wind speed for a given location."""
  return f"The wind speed in {location} is 10 mph."


BASE_INSTRUCTION = (
    "You are a helpful AI assistant that can use the local environment to"
    " execute commands and file I/O."
)

SKILL_USAGE_INSTRUCTION = """\
[SKILLS ACCESS]
You have access to specialized skills stored in the environment's `skills/` folder.
Each skill is a folder containing a `SKILL.md` file with instructions.

[MANDATORY PROCEDURE]
Before declaring that you cannot perform a task or answer a question (especially for domain-specific queries like weather), you MUST:
1. Use the `Execute` tool to search for all available skills by running: `find skills -name SKILL.md`
2. Review the list of found skills to see if any are relevant to the user's request.
3. If a relevant skill is found, use the `ReadFile` tool to read its `SKILL.md` file.
4. Follow the instructions in that file to complete the request.
   *CRITICAL NOTE ON PATHS:* All file and script paths mentioned inside a `SKILL.md` file (e.g., `references/...` or `scripts/...`) are RELATIVE to that specific skill's folder. You MUST resolve them by prepending the skill's folder path (e.g., if the skill is at `skills/weather-skill/`, you must read `skills/weather-skill/references/weather_info.md`).

Failure to check the `skills/` directory before stating you cannot help is unacceptable.\
"""


root_agent = Agent(
    model="gemini-2.5-pro",
    name="local_environment_skill_agent",
    description=(
        "An agent that uses local environment tools to load and use skills."
    ),
    instruction=f"{BASE_INSTRUCTION}\n\n{SKILL_USAGE_INSTRUCTION}",
    tools=[
        EnvironmentToolset(
            environment=LocalEnvironment(
                working_dir=pathlib.Path(__file__).parent
            ),
        ),
        GetTimezoneTool(),
        get_wind_speed,
    ],
)
