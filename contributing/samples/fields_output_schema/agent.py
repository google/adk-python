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
from pydantic import BaseModel


class WeatherData(BaseModel):
  temperature: str
  humidity: str
  wind_speed: str


root_agent = Agent(
    name='root_agent',
    model='gemini-2.0-flash',
    instruction="""\
Answer user's questions based on the data you have.

If you don't have the data, you can just say you don't know.

Here are the data you have for San Jose

* temperature: 26 C
* humidity: 20%
* wind_speed: 29 mph

Here are the data you have for Cupertino

* temperature: 16 C
* humidity: 10%
* wind_speed: 13 mph

""",
    output_schema=WeatherData,
    output_key='weather_data',
)
