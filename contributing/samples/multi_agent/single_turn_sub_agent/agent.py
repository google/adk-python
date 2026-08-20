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

from google.adk import Agent
from pydantic import BaseModel
from pydantic import Field


class UserPreferences(BaseModel):
  budget: int = Field(description="The user's maximum budget in USD")
  primary_use: str = Field(
      description=(
          "What the user primarily uses their phone for (e.g., photography,"
          " gaming, basics)"
      )
  )
  preferred_size: str = Field(
      description="Preferred phone size (e.g., small, large, any)"
  )


class PhoneRecommendation(BaseModel):
  """Output schema for the phone recommendation."""

  model_name: str
  price: float
  reason: str


def check_phone_price(model_name: str) -> float:
  """Mock tool to check the current price of a Pixel phone model."""
  prices = {
      "Pixel 10a": 499.0,
      "Pixel 10": 799.0,
      "Pixel 10 Pro": 999.0,
      "Pixel 10 Pro XL": 1199.0,
      "Pixel 10 Pro Fold": 1799.0,
  }
  # Simple mock logic, defaulting to 799 if not found exactly
  for key, value in prices.items():
    if key.lower() in model_name.lower():
      return value
  return 799.0


phone_recommender = Agent(
    name="phone_recommender",
    mode="single_turn",
    input_schema=UserPreferences,
    output_schema=PhoneRecommendation,
    tools=[check_phone_price],
    instruction=("""\
You are an expert Google Pixel hardware recommender.
Based on the provided UserPreferences, recommend exactly one Pixel phone model.
You must use the `check_phone_price` tool to find the exact current price of the model you are recommending before you finish your task.
Only recommend these phones: Pixel 10a, Pixel 10, Pixel 10 Pro, Pixel 10 Pro XL, Pixel 10 Pro Fold.
    """),
    description="Recommends a Pixel phone based on preferences.",
)


root_agent = Agent(
    name="root_agent",
    sub_agents=[phone_recommender],
    instruction=("""\
You are a helpful phone sales associate.
If the user is asking for a phone recommendation, use the `phone_recommender` to get a structured recommendation.
Once the recommender finishes, present the model, price, and reason to the user in a friendly way.
    """),
)
