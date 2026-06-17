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

"""Agent definition for testing the Interactions API integration."""

from google.adk.agents.llm_agent import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools.google_search_tool import GoogleSearchTool


def get_current_weather(city: str) -> dict:
  """Get the current weather for a city.

  This is a mock implementation for testing purposes.

  Args:
    city: The name of the city to get weather for.

  Returns:
    A dictionary containing weather information.
  """
  # Mock weather data for testing
  weather_data = {
      "new york": {"temperature": 72, "condition": "Sunny", "humidity": 45},
      "london": {"temperature": 59, "condition": "Cloudy", "humidity": 78},
      "tokyo": {
          "temperature": 68,
          "condition": "Partly Cloudy",
          "humidity": 60,
      },
      "paris": {"temperature": 64, "condition": "Rainy", "humidity": 85},
      "sydney": {"temperature": 77, "condition": "Clear", "humidity": 55},
  }

  city_lower = city.lower()
  if city_lower in weather_data:
    data = weather_data[city_lower]
    return {
        "city": city,
        "temperature_f": data["temperature"],
        "condition": data["condition"],
        "humidity": data["humidity"],
    }
  else:
    return {
        "city": city,
        "temperature_f": 70,
        "condition": "Unknown",
        "humidity": 50,
        "note": "Weather data not available, using defaults",
    }


# Main agent with google_search built-in tool and custom function tools
#
# NOTE: code_executor is not compatible with function calling mode because the model
# tries to call a function (e.g., run_code) instead of outputting code in markdown.
root_agent = Agent(
    model=Gemini(
        model="gemini-3.1-flash-lite",
        use_interactions_api=True,
    ),
    name="interactions_test_agent",
    description="An agent for testing the Interactions API integration",
    instruction="""You are a helpful assistant that can:

1. Search the web for information using google_search
2. Get weather information using get_current_weather

When users ask for information that requires searching, use google_search.
When users ask about weather, use get_current_weather.

Be concise and helpful in your responses. Always confirm what you did.
""",
    tools=[
        GoogleSearchTool(),
        get_current_weather,
    ],
)
