#!/usr/bin/env python3
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

"""End-to-end validation: TRANSFER_A2A classification in BigQuery rows.

This script validates the fix for issue #5073 by running a real agent
system with both local and remote A2A sub-agents, logging to BigQuery
via BigQueryAgentAnalyticsPlugin, and querying the resulting rows to
confirm that ``tool_origin`` is correctly classified as
``TRANSFER_A2A`` for remote agents and ``TRANSFER_AGENT`` for local
agents.

Architecture::

    orchestrator_agent (LLM)
        |
        +-- local_weather_agent  (local sub-agent)
        |
        +-- remote_math_agent    (RemoteA2aAgent -> in-process A2A server)

Usage::

    # From the repo root, with ADC configured:
    python contributing/samples/validate_transfer_a2a/validate.py

    # Custom GCP settings:
    python contributing/samples/validate_transfer_a2a/validate.py \
        --project_id=my-project --dataset_id=adk_logs

The script creates a dedicated BQ table ``transfer_a2a_validation``,
runs agent conversations, waits for rows to flush, queries the table,
and prints a PASS/FAIL verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from unittest.mock import Mock

from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import TransportProtocol as A2ATransport
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.cloud import bigquery
from google.genai import types
import httpx

# Suppress experimental A2A warnings
os.environ["ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS"] = "1"

# Use Vertex AI backend (ADC credentials) if no API key is set
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project-0728-467323")
  os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
  os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_transfer_a2a")

# ---------------------------------------------------------------------------
# 1. Remote A2A math agent (served in-process via ASGI transport)
# ---------------------------------------------------------------------------


def add_numbers(a: float, b: float) -> float:
  """Add two numbers together.

  Args:
    a: First number.
    b: Second number.

  Returns:
    The sum of the two numbers.
  """
  return a + b


remote_math_root = Agent(
    model="gemini-2.0-flash",
    name="math_agent",
    description="A math agent that can add numbers.",
    instruction=(
        "You are a math assistant. When asked to add numbers, use the"
        " add_numbers tool. Always respond with the result."
    ),
    tools=[add_numbers],
)

REMOTE_AGENT_CARD = AgentCard(
    name="math_agent",
    url="http://a2a-test-server",
    description="A math agent that can add numbers.",
    capabilities=AgentCapabilities(streaming=True),
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[],
)


class _FakeRunner(Runner):
  """Runner that delegates to a real agent but is pluggable into A2A."""

  def __init__(self, agent):
    session_service = InMemorySessionService()
    super().__init__(
        app_name="MathApp",
        agent=agent,
        session_service=session_service,
    )


def _create_a2a_server_app():
  """Create an in-process A2A FastAPI app for the math agent."""
  runner = _FakeRunner(remote_math_root)
  executor = A2aAgentExecutor(runner=runner)
  task_store = InMemoryTaskStore()
  handler = DefaultRequestHandler(
      agent_executor=executor, task_store=task_store
  )
  app = A2AFastAPIApplication(
      agent_card=REMOTE_AGENT_CARD, http_handler=handler
  )
  return app.build()


def _create_remote_a2a_agent(fastapi_app) -> RemoteA2aAgent:
  """Create a RemoteA2aAgent wired to the in-process server."""
  client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=fastapi_app),
      base_url="http://a2a-test-server",
  )
  client_config = A2AClientConfig(
      httpx_client=client,
      streaming=False,
      polling=False,
      supported_transports=[A2ATransport.jsonrpc],
  )
  factory = A2AClientFactory(config=client_config)
  return RemoteA2aAgent(
      name="remote_math_agent",
      description="Remote A2A math agent that adds numbers.",
      agent_card=REMOTE_AGENT_CARD,
      a2a_client_factory=factory,
      use_legacy=False,
  )


# ---------------------------------------------------------------------------
# 2. Local weather agent
# ---------------------------------------------------------------------------


def get_weather(city: str) -> str:
  """Get the current weather for a city.

  Args:
    city: The city name.

  Returns:
    A weather description string.
  """
  return f"The weather in {city} is sunny, 22C."


local_weather_agent = Agent(
    model="gemini-2.0-flash",
    name="local_weather_agent",
    description="A local weather agent that reports the weather.",
    instruction=(
        "You are a weather assistant. When asked about weather, use the"
        " get_weather tool and report the result."
    ),
    tools=[get_weather],
)


# ---------------------------------------------------------------------------
# 3. Orchestrator + BQ Plugin + Runner
# ---------------------------------------------------------------------------


async def run_validation(
    project_id: str,
    dataset_id: str,
    table_id: str,
    location: str,
) -> bool:
  """Run the end-to-end validation.

  Returns True if the BQ rows contain the expected tool_origin values.
  """
  # -- Build the A2A in-process server + client --
  a2a_app = _create_a2a_server_app()
  remote_agent = _create_remote_a2a_agent(a2a_app)

  # -- Orchestrator agent --
  orchestrator = Agent(
      model="gemini-2.0-flash",
      name="orchestrator_agent",
      description="Routes tasks to the right sub-agent.",
      instruction=(
          "You are a dispatcher. You have two sub-agents:\n"
          "- local_weather_agent: handles weather questions\n"
          "- remote_math_agent: handles math / addition questions\n\n"
          "For weather questions, transfer to local_weather_agent.\n"
          "For math questions, transfer to remote_math_agent.\n"
          "Always transfer immediately; do not answer yourself."
      ),
      sub_agents=[local_weather_agent, remote_agent],
  )

  # -- BQ analytics plugin --
  bq_plugin = BigQueryAgentAnalyticsPlugin(
      project_id=project_id,
      dataset_id=dataset_id,
      table_id=table_id,
      location=location,
  )

  # -- Session + Runner --
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="transfer_a2a_validation",
      agent=orchestrator,
      session_service=session_service,
      plugins=[bq_plugin],
  )

  session = await session_service.create_session(
      app_name="transfer_a2a_validation",
      user_id="validation_user",
  )

  # -- Run two conversations: one for each transfer type --
  conversations = [
      ("What is 3 + 5?", "math"),
      ("What is the weather in Tokyo?", "weather"),
  ]

  for user_msg, label in conversations:
    logger.info("--- Sending: %r  (expect %s transfer) ---", user_msg, label)
    new_message = types.Content(
        parts=[types.Part.from_text(text=user_msg)],
        role="user",
    )
    events = []
    async for event in runner.run_async(
        user_id="validation_user",
        session_id=session.id,
        new_message=new_message,
    ):
      if event.content and event.content.parts:
        text_parts = [p.text for p in event.content.parts if p.text]
        if text_parts:
          logger.info("  [%s] %s", event.author, " ".join(text_parts)[:120])
      events.append(event)
    logger.info("  => %d events produced", len(events))

  # -- Flush and shut down the plugin --
  logger.info("Shutting down BQ plugin (flushing remaining rows)...")
  await bq_plugin.shutdown()
  # Give BQ a moment to commit rows
  logger.info("Waiting 10s for BigQuery row availability...")
  await asyncio.sleep(10)

  # -- Query BigQuery for the results --
  logger.info("Querying BigQuery for tool_origin values...")
  bq_client = bigquery.Client(project=project_id, location=location)
  query = f"""
    SELECT
      event_type,
      JSON_VALUE(content, '$.tool') AS tool_name,
      JSON_VALUE(content, '$.tool_origin') AS tool_origin,
      JSON_VALUE(content, '$.args.agent_name') AS agent_name,
      timestamp
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE
      event_type IN ('TOOL_STARTING', 'TOOL_COMPLETED')
      AND JSON_VALUE(content, '$.tool') = 'transfer_to_agent'
      AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 MINUTE)
    ORDER BY timestamp DESC
    LIMIT 20
  """
  logger.info("Running query:\n%s", query)
  rows = list(bq_client.query(query).result())

  if not rows:
    logger.error("NO ROWS FOUND. The BQ plugin may not have flushed.")
    return False

  logger.info("Found %d transfer_to_agent rows:", len(rows))
  found_a2a = False
  found_local = False
  for row in rows:
    origin = row.tool_origin
    agent_name = row.agent_name
    event_type = row.event_type
    logger.info(
        "  %s | agent_name=%s | tool_origin=%s",
        event_type,
        agent_name,
        origin,
    )
    if origin == "TRANSFER_A2A":
      found_a2a = True
    if origin == "TRANSFER_AGENT":
      found_local = True

  # -- Verdict --
  print()
  print("=" * 60)
  if found_a2a and found_local:
    print("PASS: Both TRANSFER_A2A and TRANSFER_AGENT found in BQ rows.")
    print("      The fix for #5073 is validated end-to-end.")
    print("=" * 60)
    return True
  else:
    print("FAIL: Expected both TRANSFER_A2A and TRANSFER_AGENT.")
    print(f"      TRANSFER_A2A found:    {found_a2a}")
    print(f"      TRANSFER_AGENT found:  {found_local}")
    print("=" * 60)
    return False


def main():
  parser = argparse.ArgumentParser(
      description="Validate TRANSFER_A2A classification in BigQuery."
  )
  parser.add_argument(
      "--project_id",
      default=os.environ.get(
          "GOOGLE_CLOUD_PROJECT", "test-project-0728-467323"
      ),
      help="GCP project ID.",
  )
  parser.add_argument(
      "--dataset_id",
      default="adk_logs",
      help="BigQuery dataset ID.",
  )
  parser.add_argument(
      "--table_id",
      default="transfer_a2a_validation",
      help="BigQuery table ID (created automatically).",
  )
  parser.add_argument(
      "--location",
      default="us-central1",
      help="BigQuery dataset location.",
  )
  args = parser.parse_args()

  logger.info(
      "Config: project=%s  dataset=%s  table=%s  location=%s",
      args.project_id,
      args.dataset_id,
      args.table_id,
      args.location,
  )

  success = asyncio.run(
      run_validation(
          args.project_id,
          args.dataset_id,
          args.table_id,
          args.location,
      )
  )
  sys.exit(0 if success else 1)


if __name__ == "__main__":
  main()
