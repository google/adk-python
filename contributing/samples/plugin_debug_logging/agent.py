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

"""Sample agent demonstrating DebugLoggingPlugin usage.

This sample shows how to use the DebugLoggingPlugin to capture complete
debug information (LLM requests/responses, tool calls, events, session state)
to a YAML file for debugging purposes.

Usage:
  adk run contributing/samples/plugin_debug_logging

After running, check the generated `adk_debug.yaml` file for detailed logs.
"""

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.plugins import DebugLoggingPlugin


def get_weather(city: str) -> dict[str, Any]:
  """Get the current weather for a city.

  Args:
    city: The name of the city to get weather for.

  Returns:
    A dictionary containing weather information.
  """
  # Simulated weather data
  weather_data = {
      "new york": {"temperature": 22, "condition": "sunny", "humidity": 45},
      "london": {"temperature": 15, "condition": "cloudy", "humidity": 70},
      "tokyo": {"temperature": 28, "condition": "humid", "humidity": 85},
      "paris": {"temperature": 18, "condition": "rainy", "humidity": 80},
  }

  city_lower = city.lower()
  if city_lower in weather_data:
    data = weather_data[city_lower]
    return {
        "city": city,
        "temperature_celsius": data["temperature"],
        "condition": data["condition"],
        "humidity_percent": data["humidity"],
    }
  else:
    return {
        "city": city,
        "error": f"Weather data not available for {city}",
    }


def calculate(expression: str) -> dict[str, Any]:
  """Evaluate a simple mathematical expression.

  Args:
    expression: A mathematical expression to evaluate (e.g., "2 + 2").

  Returns:
    A dictionary containing the result or error.
  """
  try:
    # Only allow safe mathematical operations
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
      return {"error": "Invalid characters in expression"}

    result = eval(expression)  # Safe due to character restriction
    return {"expression": expression, "result": result}
  except Exception as e:
    return {"expression": expression, "error": str(e)}


# Sample queries to try:
# - "What's the weather in Tokyo?"
# - "Calculate 15 * 7 + 3"
# - "What's the weather in London and calculate 100 / 4"
root_agent = LlmAgent(
    name="debug_demo_agent",
    description="A demo agent that shows DebugLoggingPlugin capabilities",
    instruction="""You are a helpful assistant that can:
1. Get weather information for cities (New York, London, Tokyo, Paris)
2. Perform simple calculations

When asked about weather, use the get_weather tool.
When asked to calculate, use the calculate tool.
Be concise in your responses.""",
    model="gemini-2.0-flash",
    tools=[get_weather, calculate],
)


# Create the app with DebugLoggingPlugin
# The plugin will write detailed debug information to adk_debug.yaml
app = App(
    name="plugin_debug_logging",
    root_agent=root_agent,
    plugins=[
        # DebugLoggingPlugin captures complete interaction data to a YAML file
        # Options:
        #   output_path: Path to output file (default: "adk_debug.yaml")
        #   include_session_state: Include session state snapshot (default: True)
        #   include_system_instruction: Include full system instruction (default: True)
        DebugLoggingPlugin(
            output_path="adk_debug.yaml",
            include_session_state=True,
            include_system_instruction=True,
        ),
    ],
)
