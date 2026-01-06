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

"""Shared agent definition for BQ vs Cloud Logging comparison demo.

This agent demonstrates typical patterns that generate diverse telemetry:
- LLM requests/responses (token usage tracking)
- Tool calls (success/failure rates)
- Multi-modal content handling (image analysis)

Run with either:
- BigQuery Analytics: python run_with_bq_analytics.py
- Cloud Logging: python run_with_cloud_logging.py
"""

from __future__ import annotations

import os
import random
from typing import Any

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.google_llm import Gemini
from google.adk.tools.function_tool import FunctionTool

# Default Vertex AI settings
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", os.getenv("PROJECT_ID", ""))
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")


# =============================================================================
# TOOLS - Designed to generate diverse telemetry data
# =============================================================================


def get_weather(city: str) -> dict[str, Any]:
  """Get current weather for a city.

  This tool demonstrates successful tool calls with structured output.
  """
  weather_data = {
      "Tokyo": {"temp_c": 22, "condition": "Partly cloudy", "humidity": 65},
      "Paris": {"temp_c": 18, "condition": "Sunny", "humidity": 55},
      "New York": {"temp_c": 25, "condition": "Clear", "humidity": 70},
      "Sydney": {"temp_c": 28, "condition": "Sunny", "humidity": 60},
      "London": {"temp_c": 15, "condition": "Rainy", "humidity": 80},
  }
  if city in weather_data:
    return {"city": city, **weather_data[city], "status": "success"}
  return {"city": city, "error": "City not found", "status": "not_found"}


def calculate_trip_cost(
    city: str,
    days: int,
    hotel_class: str = "mid-range",
) -> dict[str, Any]:
  """Calculate estimated trip cost.

  This tool demonstrates numeric calculations and multiple parameters.
  """
  daily_rates = {
      "budget": {"hotel": 80, "food": 40, "transport": 20},
      "mid-range": {"hotel": 150, "food": 70, "transport": 40},
      "luxury": {"hotel": 400, "food": 150, "transport": 80},
  }
  rates = daily_rates.get(hotel_class, daily_rates["mid-range"])
  total = days * sum(rates.values())
  return {
      "city": city,
      "days": days,
      "hotel_class": hotel_class,
      "daily_breakdown": rates,
      "total_estimate_usd": total,
  }


def flaky_api_call(operation: str) -> dict[str, Any]:
  """Simulates an unreliable external API (fails 30% of the time).

  This tool is intentionally flaky to demonstrate error tracking and
  tool failure rate analysis.
  """
  if random.random() < 0.3:
    raise RuntimeError(
        f"External API temporarily unavailable for operation: {operation}"
    )
  return {
      "operation": operation,
      "status": "success",
      "data": f"Result for {operation}",
  }


def analyze_sentiment(text: str) -> dict[str, Any]:
  """Analyze sentiment of provided text.

  Demonstrates longer text processing for token usage analysis.
  """
  word_count = len(text.split())
  # Simple mock sentiment analysis
  positive_words = {"good", "great", "excellent", "amazing", "wonderful"}
  negative_words = {"bad", "terrible", "awful", "poor", "disappointing"}
  words = set(text.lower().split())
  pos_count = len(words & positive_words)
  neg_count = len(words & negative_words)

  if pos_count > neg_count:
    sentiment = "positive"
  elif neg_count > pos_count:
    sentiment = "negative"
  else:
    sentiment = "neutral"

  return {
      "sentiment": sentiment,
      "word_count": word_count,
      "positive_indicators": pos_count,
      "negative_indicators": neg_count,
  }


# Create FunctionTool instances
weather_tool = FunctionTool(get_weather)
trip_cost_tool = FunctionTool(calculate_trip_cost)
flaky_tool = FunctionTool(flaky_api_call)
sentiment_tool = FunctionTool(analyze_sentiment)


# =============================================================================
# AGENT DEFINITION
# =============================================================================

MODEL = Gemini(model="gemini-2.0-flash")

root_agent = Agent(
    name="telemetry_demo_agent",
    model=MODEL,
    instruction="""You are a helpful travel assistant that can:
1. Check weather in major cities using the get_weather tool
2. Calculate trip costs using the calculate_trip_cost tool
3. Analyze text sentiment using the analyze_sentiment tool
4. Make external API calls using the flaky_api_call tool (this may fail sometimes)

When helping users, actively use your tools to provide accurate information.
If a tool fails, acknowledge the error and try again or suggest alternatives.

Keep responses concise and actionable.""",
    description="A demo agent for comparing BigQuery Analytics vs Cloud Logging",
    tools=[
        weather_tool,
        trip_cost_tool,
        flaky_tool,
        sentiment_tool,
    ],
)


def create_app_with_plugins(plugins: list | None = None) -> App:
  """Create an App instance with optional plugins.

  Args:
    plugins: List of plugins to attach (e.g., BigQueryAgentAnalyticsPlugin)

  Returns:
    Configured App instance
  """
  return App(
      name="bq_vs_cloud_logging_demo",
      root_agent=root_agent,
      plugins=plugins or [],
  )


# Default app without plugins (for Cloud Logging via --otel_to_cloud)
app = create_app_with_plugins()
