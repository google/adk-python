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

from __future__ import annotations

"""Minimal agent wired to the BigQuery Agent Analytics plugin."""

import os
from typing import Dict
from typing import List

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.google_llm import Gemini
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)

PROJECT_ID = os.getenv("BQ_AGENT_ANALYTICS_PROJECT")
DATASET_ID = os.getenv("BQ_AGENT_ANALYTICS_DATASET")
TABLE_ID = os.getenv("BQ_AGENT_ANALYTICS_TABLE", "agent_events")

# Default Vertex AI settings if env vars are not provided.
os.environ.setdefault("VERTEXAI_PROJECT", "test-project-0728-467323")
os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")
# google.genai expects these env vars to auto-switch to Vertex AI with ADC.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project-0728-467323")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

if not PROJECT_ID or not DATASET_ID:
  raise ValueError(
      "Set BQ_AGENT_ANALYTICS_PROJECT and BQ_AGENT_ANALYTICS_DATASET before "
      "running this agent to enable BigQuery analytics logging."
  )

analytics_plugin = BigQueryAgentAnalyticsPlugin(
    project_id=PROJECT_ID,
    dataset_id=DATASET_ID,
    table_id=TABLE_ID,
    config=BigQueryLoggerConfig(
        event_allowlist=[
            "USER_MESSAGE_RECEIVED",
            "LLM_REQUEST",
            "LLM_RESPONSE",
            "TOOL_STARTING",
            "TOOL_COMPLETED",
            "MODEL_RESPONSE",
            "TOOL_CALL",
            "TOOL_RESULT",
            "ERROR",
        ],
        max_content_length=400,
    ),
)

def pick_city(top_n: int = 3) -> List[str]:
  """Return a short list of recommended cities."""
  cities = ["Tokyo", "Paris", "New York", "Sydney", "Singapore", "Toronto"]
  return cities[:top_n]

def city_highlights(city: str) -> Dict[str, str]:
  """Return quick highlights for a city."""
  highlights = {
      "Tokyo": "Sushi, Akihabara tech, efficient transit.",
      "Paris": "Museums, pastries, walkable boulevards.",
      "New York": "Broadway, diverse food, skyline views.",
      "Sydney": "Harbour, beaches, outdoor cafes.",
      "Singapore": "Hawker food, gardens, clean and safe.",
      "Toronto": "CN Tower, neighbourhood food, waterfront.",
  }
  return {"city": city, "highlights": highlights.get(city, "Explore freely.")}

def estimate_trip_budget(city: str, days: int, budget_per_day: float) -> Dict[str, str]:
  """Rough budget calculator."""
  total = days * budget_per_day
  return {
      "city": city,
      "days": str(days),
      "budget_per_day": f"${budget_per_day:,.0f}",
      "estimated_total": f"${total:,.0f}",
      "note": "Assumes lodging+food+local transit; adjust for flights.",
  }

city_tool = FunctionTool(pick_city)
city_highlights_tool = FunctionTool(city_highlights)
budget_tool = FunctionTool(estimate_trip_budget)

gemini = Gemini(model="gemini-2.5-flash")
search_agent = create_google_search_agent(model=gemini)
google_search_tool = GoogleSearchAgentTool(agent=search_agent)

root_agent = Agent(
    name="bq_agent_analytics_demo",
    model=gemini,
    instruction=(
        "You are a concise assistant. Prefer to use tools when asked for trip "
        "ideas, highlights, or cost estimates. Keep answers short and "
        "actionable."
    ),
    description="A minimal agent that logs events to BigQuery.",
    tools=[
        city_tool,
        city_highlights_tool,
        budget_tool,
        google_search_tool,
    ],
)

app = App(
    name="bq_agent_analytics_demo",
    root_agent=root_agent,
    plugins=[analytics_plugin],
)
