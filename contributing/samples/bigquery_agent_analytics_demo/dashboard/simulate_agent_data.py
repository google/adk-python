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

"""Simulate agent interactions to populate BigQuery analytics data.

This script runs a series of simulated conversations with the demo agent
to generate realistic analytics data for the dashboard.

Usage:
    python simulate_agent_data.py --num-sessions 10

Environment Variables:
    BQ_AGENT_ANALYTICS_PROJECT: Google Cloud project ID
    BQ_AGENT_ANALYTICS_DATASET: BigQuery dataset ID
    BQ_AGENT_ANALYTICS_TABLE: BigQuery table name (default: agent_events_v2)
    VERTEXAI_PROJECT: Vertex AI project ID
    VERTEXAI_LOCATION: Vertex AI location
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import uuid
from typing import Any

# Ensure the parent directory is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk import Agent
from google.adk.apps import App
from google.adk.models.google_llm import Gemini
from google.adk.plugins.bigquery_agent_analytics_plugin import (
    BigQueryAgentAnalyticsPlugin,
    BigQueryLoggerConfig,
)
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.genai import types


# Configuration from environment
PROJECT_ID = os.getenv("BQ_AGENT_ANALYTICS_PROJECT")
DATASET_ID = os.getenv("BQ_AGENT_ANALYTICS_DATASET")
TABLE_ID = os.getenv("BQ_AGENT_ANALYTICS_TABLE", "agent_events_v2")

# Default Vertex AI settings
os.environ.setdefault("VERTEXAI_PROJECT", PROJECT_ID or "")
os.environ.setdefault("VERTEXAI_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID or "")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")


# Sample conversation prompts for simulation
SAMPLE_PROMPTS = [
    # Trip planning conversations
    "What are the top 3 cities to visit in Asia?",
    "Give me highlights for Tokyo",
    "What's the estimated budget for 5 days in Paris with $200 per day?",
    "I want to plan a trip to New York, what should I know?",
    "Compare Tokyo and Singapore for a tech enthusiast",
    # Budget inquiries
    "How much would a week in Sydney cost at $150/day?",
    "What's a reasonable budget for Toronto?",
    "Estimate trip cost for 3 days in Paris",
    # City highlights
    "Tell me about Singapore's food scene",
    "What makes Paris special?",
    "What are the must-see places in New York?",
    # General travel questions
    "Which city has the best food?",
    "Where should I go for a tech conference?",
    "Recommend a city for first-time international travelers",
    # Multi-turn simulation prompts
    "Pick 3 cities for me",
    "Now tell me about the first one",
    "What about the budget for that city?",
]

# Sample user IDs for realistic data
SAMPLE_USER_IDS = [
    "user_alice_001",
    "user_bob_002",
    "user_charlie_003",
    "user_diana_004",
    "user_eve_005",
    "user_frank_006",
    "user_grace_007",
    "user_henry_008",
    "user_ivy_009",
    "user_jack_010",
]

# Sample agents for multi-agent simulation
AGENT_CONFIGS = [
    {
        "name": "travel_advisor",
        "instruction": (
            "You are a helpful travel advisor. Recommend cities and provide "
            "travel tips. Use tools to get city lists, highlights, and budgets."
        ),
    },
    {
        "name": "budget_planner",
        "instruction": (
            "You are a budget travel planner. Focus on cost-effective travel. "
            "Always calculate budgets when users ask about trips."
        ),
    },
    {
        "name": "city_guide",
        "instruction": (
            "You are an expert city guide. Provide detailed highlights and "
            "recommendations for cities around the world."
        ),
    },
]


# Tool functions
def pick_city(top_n: int = 3) -> list[str]:
  """Return a short list of recommended cities."""
  cities = ["Tokyo", "Paris", "New York", "Sydney", "Singapore", "Toronto"]
  return random.sample(cities, min(top_n, len(cities)))


def city_highlights(city: str) -> dict[str, str]:
  """Return quick highlights for a city."""
  highlights = {
      "Tokyo": "Sushi, Akihabara tech district, efficient transit system.",
      "Paris": "World-class museums, pastries, walkable boulevards.",
      "New York": "Broadway shows, diverse food scene, iconic skyline views.",
      "Sydney": "Beautiful harbour, beaches, outdoor cafes.",
      "Singapore": "Amazing hawker food, stunning gardens, clean and safe.",
      "Toronto": "CN Tower, diverse neighbourhoods, waterfront activities.",
  }
  return {
      "city": city,
      "highlights": highlights.get(city, "A wonderful place to explore!"),
  }


def estimate_trip_budget(
    city: str, days: int, budget_per_day: float
) -> dict[str, Any]:
  """Calculate rough trip budget."""
  total = days * budget_per_day
  return {
      "city": city,
      "days": days,
      "budget_per_day": f"${budget_per_day:,.0f}",
      "estimated_total": f"${total:,.0f}",
      "note": "Includes lodging, food, and local transit. Adjust for flights.",
  }


def failing_tool(query: str) -> str:
  """A tool that sometimes fails to simulate error scenarios."""
  if random.random() < 0.3:  # 30% failure rate
    raise ValueError(f"Failed to process query: {query}")
  return f"Processed: {query}"


def create_agent(config: dict, include_failing_tool: bool = False) -> Agent:
  """Create an agent with the given configuration."""
  gemini = Gemini(model="gemini-2.5-flash")

  tools = [
      FunctionTool(pick_city),
      FunctionTool(city_highlights),
      FunctionTool(estimate_trip_budget),
  ]

  if include_failing_tool:
    tools.append(FunctionTool(failing_tool))

  return Agent(
      name=config["name"],
      model=gemini,
      instruction=config["instruction"],
      description=f"Demo agent: {config['name']}",
      tools=tools,
  )


def create_analytics_plugin() -> BigQueryAgentAnalyticsPlugin:
  """Create the BigQuery analytics plugin."""
  if not PROJECT_ID or not DATASET_ID:
    raise ValueError(
        "Set BQ_AGENT_ANALYTICS_PROJECT and BQ_AGENT_ANALYTICS_DATASET "
        "environment variables"
    )

  return BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      table_id=TABLE_ID,
      config=BigQueryLoggerConfig(
          event_allowlist=[
              "USER_MESSAGE_RECEIVED",
              "INVOCATION_STARTING",
              "INVOCATION_COMPLETED",
              "AGENT_STARTING",
              "AGENT_COMPLETED",
              "LLM_REQUEST",
              "LLM_RESPONSE",
              "LLM_ERROR",
              "TOOL_STARTING",
              "TOOL_COMPLETED",
              "TOOL_ERROR",
          ],
          max_content_length=10000,
          batch_size=1,
      ),
  )


async def run_single_session(
    agent_config: dict,
    prompts: list[str],
    user_id: str,
    plugin: BigQueryAgentAnalyticsPlugin,
    include_failing_tool: bool = False,
) -> None:
  """Run a single conversation session."""
  agent = create_agent(agent_config, include_failing_tool)
  app = App(
      name=agent_config["name"],
      root_agent=agent,
      plugins=[plugin],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app=app,
      session_service=session_service,
  )

  session = await session_service.create_session(
      app_name=app.name,
      user_id=user_id,
  )

  print(f"  Session {session.id[:8]}... started for user {user_id}")

  for prompt in prompts:
    print(f"    User: {prompt[:50]}...")
    try:
      content = types.Content(
          role="user", parts=[types.Part.from_text(text=prompt)]
      )

      async for event in runner.run_async(
          session_id=session.id,
          user_id=user_id,
          new_message=content,
      ):
        # Process events silently
        pass

      # Small delay between messages to simulate realistic timing
      await asyncio.sleep(random.uniform(0.5, 2.0))

    except Exception as e:
      print(f"    Error during conversation: {e}")

  # Ensure logs are flushed
  await plugin.flush()
  print(f"  Session {session.id[:8]}... completed")


async def simulate_conversations(
    num_sessions: int = 10,
    prompts_per_session: tuple[int, int] = (2, 5),
    include_errors: bool = True,
) -> None:
  """Simulate multiple conversation sessions."""
  print(f"Starting simulation of {num_sessions} sessions...")
  print(f"Project: {PROJECT_ID}")
  print(f"Dataset: {DATASET_ID}")
  print(f"Table: {TABLE_ID}")
  print()

  plugin = create_analytics_plugin()

  for i in range(num_sessions):
    # Randomly select agent configuration
    agent_config = random.choice(AGENT_CONFIGS)

    # Randomly select user
    user_id = random.choice(SAMPLE_USER_IDS)

    # Randomly select prompts
    num_prompts = random.randint(*prompts_per_session)
    prompts = random.sample(SAMPLE_PROMPTS, min(num_prompts, len(SAMPLE_PROMPTS)))

    # Sometimes include failing tool to simulate errors
    include_failing_tool = include_errors and random.random() < 0.2

    print(f"Session {i + 1}/{num_sessions}:")
    print(f"  Agent: {agent_config['name']}")
    print(f"  User: {user_id}")
    print(f"  Prompts: {num_prompts}")

    try:
      await run_single_session(
          agent_config=agent_config,
          prompts=prompts,
          user_id=user_id,
          plugin=plugin,
          include_failing_tool=include_failing_tool,
      )
    except Exception as e:
      print(f"  Session failed: {e}")

    print()

    # Small delay between sessions
    await asyncio.sleep(random.uniform(1.0, 3.0))

  # Final flush and shutdown
  await plugin.shutdown()
  print("Simulation complete!")


async def inject_historical_data(
    days_back: int = 7,
    sessions_per_day: int = 20,
) -> None:
  """Inject historical data by running simulations with backdated timestamps.

  Note: This creates real conversations but the timestamps will be current.
  For true historical data, you would need to manually insert rows into BigQuery.
  """
  total_sessions = days_back * sessions_per_day
  print(f"Injecting data: ~{total_sessions} sessions over {days_back} days")
  print("Note: Timestamps will reflect actual execution time.")
  print()

  await simulate_conversations(
      num_sessions=total_sessions,
      prompts_per_session=(1, 4),
      include_errors=True,
  )


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
      description="Simulate agent interactions for analytics"
  )
  parser.add_argument(
      "--num-sessions",
      type=int,
      default=10,
      help="Number of sessions to simulate (default: 10)",
  )
  parser.add_argument(
      "--min-prompts",
      type=int,
      default=2,
      help="Minimum prompts per session (default: 2)",
  )
  parser.add_argument(
      "--max-prompts",
      type=int,
      default=5,
      help="Maximum prompts per session (default: 5)",
  )
  parser.add_argument(
      "--no-errors",
      action="store_true",
      help="Disable error simulation",
  )
  parser.add_argument(
      "--historical",
      action="store_true",
      help="Run extended simulation for historical data",
  )
  parser.add_argument(
      "--days",
      type=int,
      default=7,
      help="Days of historical data to generate (default: 7)",
  )
  parser.add_argument(
      "--sessions-per-day",
      type=int,
      default=20,
      help="Sessions per day for historical mode (default: 20)",
  )

  args = parser.parse_args()

  if args.historical:
    asyncio.run(
        inject_historical_data(
            days_back=args.days,
            sessions_per_day=args.sessions_per_day,
        )
    )
  else:
    asyncio.run(
        simulate_conversations(
            num_sessions=args.num_sessions,
            prompts_per_session=(args.min_prompts, args.max_prompts),
            include_errors=not args.no_errors,
        )
    )


if __name__ == "__main__":
  main()
