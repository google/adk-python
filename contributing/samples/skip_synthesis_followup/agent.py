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
from google.adk.tools import AgentTool
from google.adk.agents import LlmAgent
from .prompts import explaination_instruction, followup_instruction
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from typing import List

class FollowupsPayload(BaseModel):
    questions: List[str] = Field(
        description="List of 3 short follow-up questions as strings."
    )

# Follow-up questions agent
followup_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="followup_agent",
    description="Generates 3 follow-up questions to spark curiosity and deepen understanding after explaining concepts. Creates questions covering application, comparison, and exploration. DO NOT call when student is stuck on problems or during practice sessions.",
    output_schema=FollowupsPayload,
    include_contents="none",
    instruction=followup_instruction,
    generate_content_config=genai_types.GenerateContentConfig(
        temperature=0.0,
    ),
)

# Convert agent to tool
followup_agent_tool = AgentTool(
    agent=followup_agent,
    skip_synthesis=True
)

explainer_agent = Agent(
    name="explainer_agent",
    model="gemini-2.5-flash",
    description="An agent that explains topics.",
    instruction=explanation_instruction,
    tools=[followup_agent_tool]
)

# Root agent is the explainer
root_agent = explainer_agent