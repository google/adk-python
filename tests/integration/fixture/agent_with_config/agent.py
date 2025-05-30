# Copyright 2025 Google LLC
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

from google.adk import Agent
from google.genai import types

new_message = types.Content(
    role="user",
    parts=[types.Part.from_text(text="Count a number")],
)

google_agent_1 = Agent(
    model="gemini-1.5-flash",
    name="agent_1",
    description="The first agent in the team.",
    instruction="Just say 1",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
    ),
)

google_agent_2 = Agent(
    model="gemini-1.5-flash",
    name="agent_2",
    description="The second agent in the team.",
    instruction="Just say 2",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        safety_settings=[{
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH",
        }],
    ),
)

google_agent_3 = Agent(
    model="gemini-1.5-flash",
    name="agent_3",
    description="The third agent in the team.",
    instruction="Just say 3",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5,
        safety_settings=[{
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }],
    ),
)

google_agent_with_instruction_in_config = Agent(
    model="gemini-1.5-flash",
    name="agent",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5, system_instruction="Count 1"
    ),
)


def function():
  pass


google_agent_with_tools_in_config = Agent(
    model="gemini-1.5-flash",
    name="agent",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5, tools=[function]
    ),
)

google_agent_with_response_schema_in_config = Agent(
    model="gemini-1.5-flash",
    name="agent",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.5, response_schema={"key": "value"}
    ),
)
