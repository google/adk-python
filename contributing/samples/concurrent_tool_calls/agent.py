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

from datetime import datetime
import random
from google.adk import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.planners import PlanReActPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types


async def get_weather(city: str, tool_context: ToolContext) -> str:
  """Get weather information for a specified city.

  Args:
    city: The name of the city to get weather information for.

  Returns:
    A string containing weather information for the city.
  """
  print('@get_weather is starting', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
  import asyncio
  # Use async sleep for non-blocking delay to simulate a real-world weather API call.
  await asyncio.sleep(3)
  
  # Mock weather data for demonstration
  weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "stormy"]
  temperature = random.randint(-10, 35)  # Temperature in Celsius
  condition = random.choice(weather_conditions)
  humidity = random.randint(30, 90)
  
  weather_info = f"Weather in {city}: {condition}, {temperature}°C, humidity {humidity}%"
  
  if not 'weather_queries' in tool_context.state:
    tool_context.state['weather_queries'] = []

  tool_context.state['weather_queries'] = tool_context.state['weather_queries'] + [{"city": city, "weather": weather_info}]
  return weather_info

def before_tool_cb(tool, args, tool_context):
  print('@before_tool_cb1', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))

def after_tool_cb(tool, args, tool_context, tool_response):
  print('@after_tool_cb1', datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))


root_agent = Agent(
    model='gemini-2.0-flash',
    name='weather_agent',
    description=(
        'Weather information agent that can provide current weather conditions'
        ' for different cities around the world.'
    ),
    instruction="""
      You provide weather information for cities and answer questions about weather conditions.
      You can check weather for different cities around the world.
      You can use multiple tools in parallel by calling functions in parallel (in one request and in one round).
      It is ok to discuss previous weather queries and compare weather conditions between cities.
      When you are asked to check weather for a city, you must call the get_weather tool with the city name. Be sure to pass in a string with the city name.
      
      IMPORTANT: When you are asked to check weather for multiple cities (e.g., "check weather in New York, London, and Tokyo"), you MUST make ALL the get_weather function calls in parallel (simultaneously in one turn) to provide a faster response. Do NOT wait between weather queries.
    """,
    tools=[
        get_weather,
    ],
    before_tool_callback=[before_tool_cb],
    after_tool_callback=[after_tool_cb],
)
