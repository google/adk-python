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

from __future__ import annotations

from google.adk import Agent
from pydantic import BaseModel
from pydantic import Field


class CityQuery(BaseModel):
  city: str = Field(
      description='The name of the city to query weather for, e.g. San Jose'
  )


class WeatherInfo(BaseModel):
  temperature: str = Field(description='The temperature in Celsius')
  conditions: str = Field(description='The weather condition, e.g. Sunny')


weather_agent = Agent(
    name='weather_agent',
    mode='single_turn',
    input_schema=CityQuery,
    output_schema=WeatherInfo,
    instruction="""\
Provide weather information for the requested city.

For San Jose, return temperature: 26 C, conditions: Sunny.
For Cupertino, return temperature: 16 C, conditions: Foggy.
For any other city, return temperature: unknown, conditions: unknown.
""",
)

root_agent = Agent(
    name='root_agent',
    instruction="""\
You are a helpful weather concierge assistant. Use the weather_agent tool to get weather information for the user's city, and then answer the user in a friendly manner.
""",
    sub_agents=[weather_agent],
)
