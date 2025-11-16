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

"""Sample agent showing LiteLLM structured output support."""

from __future__ import annotations

from google.adk import Agent
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel
from pydantic import Field


class CitySummary(BaseModel):
  """Simple structure used to verify LiteLLM JSON schema handling."""

  city: str = Field(description="Name of the city being described.")
  highlights: list[str] = Field(
      description="Bullet points summarising the city's key highlights.",
  )
  recommended_visit_length_days: int = Field(
      description="Recommended number of days for a typical visit.",
  )


root_agent = Agent(
    name="litellm_structured_output_agent",
    model=LiteLlm(model="gemini-2.5-flash"),
    description="Generates structured travel recommendations for a given city.",
    instruction="""
Produce a JSON object that follows the CitySummary schema.
Only include fields that appear in the schema and ensure highlights
contains short bullet points.
""".strip(),
    output_schema=CitySummary,
)
