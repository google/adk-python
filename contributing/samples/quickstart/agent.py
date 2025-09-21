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

from google.adk.agents.llm_agent import Agent


def get_weather(city: str) -> dict:
  """Retrieves the current weather report for a specified city.

  Args:
      city (str): The name of the city for which to retrieve the weather report.

  Returns:
      dict: status and result or error msg.
  """
  if city.lower() == "new york":
    return {
        "status": "success",
        "report": (
            "The weather in New York is sunny with a temperature of 25 degrees"
            " Celsius (77 degrees Fahrenheit)."
        ),
    }
  else:
    return {
        "status": "error",
        "error_message": f"Weather information for '{city}' is not available.",
    }


def get_current_time(city: str) -> dict:
  """Returns the current time in a specified city.

  Args:
      city (str): The name of the city for which to retrieve the current time.

  Returns:
      dict: status and result or error msg.
  """
  import datetime
  from zoneinfo import ZoneInfo

  if city.lower() == "new york":
    tz_identifier = "America/New_York"
  else:
    return {
        "status": "error",
        "error_message": (
            f"Sorry, I don't have timezone information for {city}."
        ),
    }

  tz = ZoneInfo(tz_identifier)
  now = datetime.datetime.now(tz)
  report = (
      f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
  )
  return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "I can answer your questions about the time and weather in a city."
    ),
    tools=[get_weather, get_current_time],
)
