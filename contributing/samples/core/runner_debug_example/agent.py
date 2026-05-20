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

"""Example agent for demonstrating run_debug helper method."""

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext


def get_weather(city: str, tool_context: ToolContext) -> str:
  """Get weather information for a city.

  Args:
      city: Name of the city to get weather for.
      tool_context: Tool context for session state.

  Returns:
      Weather information as a string.
  """
  # Store query history in session state
  if "weather_queries" not in tool_context.state:
    tool_context.state["weather_queries"] = [city]
  else:
    tool_context.state["weather_queries"] = tool_context.state[
        "weather_queries"
    ] + [city]

  # Mock weather data for demonstration
  weather_data = {
      "San Francisco": "Foggy, 15°C (59°F)",
      "New York": "Sunny, 22°C (72°F)",
      "London": "Rainy, 12°C (54°F)",
      "Tokyo": "Clear, 25°C (77°F)",
      "Paris": "Cloudy, 18°C (64°F)",
  }

  return weather_data.get(
      city, f"Weather data not available for {city}. Try a major city."
  )


def get_stock_price(ticker: str) -> str:
  """Get the current stock price for a given ticker symbol.

  This tool demonstrates how function calls are displayed in run_debug().

  Args:
      ticker: Stock ticker symbol (e.g., GOOGL, AAPL, MSFT).

  Returns:
      Stock price information as a string.
  """
  prices = {
      "GOOGL": "175.50 USD",
      "AAPL": "225.00 USD",
      "MSFT": "420.00 USD",
      "AMZN": "190.00 USD",
      "NVDA": "125.00 USD",
  }
  ticker = ticker.upper()
  if ticker in prices:
    return f"Price for {ticker}: {prices[ticker]}"
  return f"Stock ticker {ticker} not found in database."


root_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="agent",
    description="A helpful assistant demonstrating run_debug() helper method",
    instruction="""You are a helpful assistant that can:
    1. Provide weather information for major cities
    2. Provide stock prices for major tech companies
    3. Remember previous queries in the conversation

    When users ask about weather, use the get_weather tool.
    When users ask for stock prices, use the get_stock_price tool.
    Be friendly and conversational.""",
    tools=[get_weather, get_stock_price],
)
