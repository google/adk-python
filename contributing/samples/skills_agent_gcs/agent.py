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

"""Example agent demonstrating the use of SkillToolset with GCS.

Go to parent directory and run with `adk web --host=0.0.0.0`.
"""

import asyncio
import logging

from google.adk import Agent
from google.adk import Runner
from google.adk.plugins import LoggingPlugin
from google.adk.skills import list_skills_in_gcs_dir
from google.adk.skills import load_skill_from_gcs_dir
from google.adk.tools.skill_toolset import SkillToolset

# Define the GCS bucket and skills prefix
BUCKET_NAME = "sample-skills"
SKILLS_PREFIX = "static-skills"

logging.info("Loading skills from gs://%s/%s...", BUCKET_NAME, SKILLS_PREFIX)

# List and load skills from GCS
skills = []
try:
  available_skills = list_skills_in_gcs_dir(
      bucket_name=BUCKET_NAME, skills_base_path=SKILLS_PREFIX
  )
  for skill_id in available_skills.keys():
    skills.append(
        load_skill_from_gcs_dir(
            bucket_name=BUCKET_NAME,
            skills_base_path=SKILLS_PREFIX,
            skill_id=skill_id,
        )
    )
  logging.info("Loaded %d skills successfully.", len(skills))
except Exception as e:  # pylint: disable=broad-exception-caught
  logging.error("Failed to load skills from GCS: %s", e)

# Create the SkillToolset
my_skill_toolset = SkillToolset(skills=skills)

# Create the Agent
root_agent = Agent(
    model="gemini-3-flash-preview",
    name="skill_user_agent",
    description="An agent that can use specialized skills loaded from GCS.",
    tools=[
        my_skill_toolset,
    ],
)


async def main():
  # Initialize the plugins
  logging_plugin = LoggingPlugin()

  # Create a Runner
  runner = Runner(
      agents=[root_agent],
      plugins=[logging_plugin],
  )

  # Example run
  print("Agent initialized with GCS skills. Sending a test prompt...")
  # You can replace this with an interactive loop if needed.
  responses = await runner.run(
      user_input="Hello! What skills do you have access to?"
  )

  if responses and responses[-1].content and responses[-1].content.parts:
    print(f"\nResponse: {responses[-1].content.parts[0].text}")


if __name__ == "__main__":
  asyncio.run(main())
