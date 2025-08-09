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

"""Sample agent for testing parallel function calling."""

import asyncio
import time
from typing import List

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext


async def get_weather(city: str, tool_context: ToolContext) -> dict:
  """Get the current weather for a city.

  Args:
    city: The name of the city to get weather for.

  Returns:
    A dictionary with weather information.
  """
  # Simulate some async processing time (non-blocking)
  await asyncio.sleep(2)

  # Mock weather data
  weather_data = {
      'New York': {'temp': 72, 'condition': 'sunny', 'humidity': 45},
      'London': {'temp': 60, 'condition': 'cloudy', 'humidity': 80},
      'Tokyo': {'temp': 68, 'condition': 'rainy', 'humidity': 90},
      'San Francisco': {'temp': 65, 'condition': 'foggy', 'humidity': 85},
      'Paris': {'temp': 58, 'condition': 'overcast', 'humidity': 70},
      'Sydney': {'temp': 75, 'condition': 'sunny', 'humidity': 60},
  }

  result = weather_data.get(
      city,
      {
          'temp': 70,
          'condition': 'unknown',
          'humidity': 50,
          'note': (
              f'Weather data not available for {city}, showing default values'
          ),
      },
  )

  # Store in context for testing thread safety
  if 'weather_requests' not in tool_context.state:
    tool_context.state['weather_requests'] = []
  tool_context.state['weather_requests'].append(
      {'city': city, 'timestamp': time.time(), 'result': result}
  )

  return {
      'city': city,
      'temperature': result['temp'],
      'condition': result['condition'],
      'humidity': result['humidity'],
      **({'note': result['note']} if 'note' in result else {}),
  }


async def get_currency_rate(
    from_currency: str, to_currency: str, tool_context: ToolContext
) -> dict:
  """Get the exchange rate between two currencies.

  Args:
    from_currency: The source currency code (e.g., 'USD').
    to_currency: The target currency code (e.g., 'EUR').

  Returns:
    A dictionary with exchange rate information.
  """
  # Simulate async processing time
  await asyncio.sleep(1.5)

  # Mock exchange rates
  rates = {
      ('USD', 'EUR'): 0.85,
      ('USD', 'GBP'): 0.75,
      ('USD', 'JPY'): 110.0,
      ('EUR', 'USD'): 1.18,
      ('EUR', 'GBP'): 0.88,
      ('GBP', 'USD'): 1.33,
      ('GBP', 'EUR'): 1.14,
      ('JPY', 'USD'): 0.009,
  }

  rate = rates.get((from_currency, to_currency), 1.0)

  # Store in context for testing thread safety
  if 'currency_requests' not in tool_context.state:
    tool_context.state['currency_requests'] = []
  tool_context.state['currency_requests'].append({
      'from': from_currency,
      'to': to_currency,
      'rate': rate,
      'timestamp': time.time(),
  })

  return {
      'from_currency': from_currency,
      'to_currency': to_currency,
      'exchange_rate': rate,
      'timestamp': time.time(),
  }


async def calculate_distance(
    city1: str, city2: str, tool_context: ToolContext
) -> dict:
  """Calculate the distance between two cities.

  Args:
    city1: The first city.
    city2: The second city.

  Returns:
    A dictionary with distance information.
  """
  # Simulate async processing time (non-blocking)
  await asyncio.sleep(1)

  # Mock distances (in kilometers)
  city_coords = {
      'New York': (40.7128, -74.0060),
      'London': (51.5074, -0.1278),
      'Tokyo': (35.6762, 139.6503),
      'San Francisco': (37.7749, -122.4194),
      'Paris': (48.8566, 2.3522),
      'Sydney': (-33.8688, 151.2093),
  }

  # Simple distance calculation (mock)
  if city1 in city_coords and city2 in city_coords:
    coord1 = city_coords[city1]
    coord2 = city_coords[city2]
    # Simplified distance calculation
    distance = int(
        ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5
        * 111
    )  # rough km conversion
  else:
    distance = 5000  # default distance

  # Store in context for testing thread safety
  if 'distance_requests' not in tool_context.state:
    tool_context.state['distance_requests'] = []
  tool_context.state['distance_requests'].append({
      'city1': city1,
      'city2': city2,
      'distance': distance,
      'timestamp': time.time(),
  })

  return {
      'city1': city1,
      'city2': city2,
      'distance_km': distance,
      'distance_miles': int(distance * 0.621371),
  }


async def get_population(cities: List[str], tool_context: ToolContext) -> dict:
  """Get population information for multiple cities.

  Args:
    cities: A list of city names.

  Returns:
    A dictionary with population data for each city.
  """
  # Simulate async processing time proportional to number of cities (non-blocking)
  await asyncio.sleep(len(cities) * 0.5)

  # Mock population data
  populations = {
      'New York': 8336817,
      'London': 9648110,
      'Tokyo': 13960000,
      'San Francisco': 873965,
      'Paris': 2161000,
      'Sydney': 5312163,
  }

  results = {}
  for city in cities:
    results[city] = populations.get(city, 1000000)  # default 1M if not found

  # Store in context for testing thread safety
  if 'population_requests' not in tool_context.state:
    tool_context.state['population_requests'] = []
  tool_context.state['population_requests'].append(
      {'cities': cities, 'results': results, 'timestamp': time.time()}
  )

  return {
      'populations': results,
      'total_population': sum(results.values()),
      'cities_count': len(cities),
  }


root_agent = Agent(
    model='gemini-2.0-flash',
    name='parallel_function_test_agent',
    description=(
        'Agent for testing parallel function calling performance and thread'
        ' safety.'
    ),
    instruction="""
    You are a helpful assistant that can provide information about weather, currency rates, 
    distances between cities, and population data. You have access to multiple tools and 
    should use them efficiently.
    
    When users ask for information about multiple cities or multiple types of data, 
    you should call multiple functions in parallel to provide faster responses.
    
    For example:
    - If asked about weather in multiple cities, call get_weather for each city in parallel
    - If asked about weather and currency rates, call both functions in parallel
    - If asked to compare cities, you might need weather, population, and distance data in parallel
    
    Always aim to be efficient and call multiple functions simultaneously when possible.
    Be informative and provide clear, well-structured responses.
  """,
    tools=[
        get_weather,
        get_currency_rate,
        calculate_distance,
        get_population,
    ],
)
