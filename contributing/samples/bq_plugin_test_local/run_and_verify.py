"""Run the local test agent and verify BigQuery logging.

Usage:
  python contributing/samples/bq_plugin_test_local/run_and_verify.py

This script:
1. Creates the BQ dataset if needed
2. Runs the agent with several test queries
3. Waits for data to land in BigQuery
4. Queries the events table and auto-created views
5. Prints a summary report
"""

import asyncio
import os
import sys
import time

# Configure Vertex AI backend before any ADK imports.
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project-0728-467323"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

import random

from google.adk.agents.llm_agent import LlmAgent
from google.adk.plugins.bigquery_agent_analytics_plugin import _EVENT_VIEW_DEFS
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.cloud import bigquery
from google.genai import types

PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"
TABLE_ID = "agent_events"


# --- Tools ---
def roll_die(sides: int) -> int:
  """Roll a die and return the result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  return random.randint(1, sides)


def get_weather(city: str) -> str:
  """Get the current weather for a city.

  Args:
    city: The name of the city.

  Returns:
    A string describing the weather.
  """
  weathers = ["sunny", "cloudy", "rainy", "snowy", "windy"]
  temp = random.randint(50, 95)
  return f"The weather in {city} is {random.choice(weathers)}, {temp}F."


def calculate(expression: str) -> str:
  """Evaluate a math expression safely.

  Args:
    expression: A simple math expression string.

  Returns:
    The result as a string.
  """
  allowed = set("0123456789+-*/.() ")
  if not all(c in allowed for c in expression):
    return "Error: only basic math operators are allowed."
  try:
    result = eval(expression)  # pylint: disable=eval-used
    return str(result)
  except Exception as e:
    return f"Error: {e}"


def ensure_dataset():
  """Create the BQ dataset if it doesn't exist."""
  client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
  dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
  try:
    client.get_dataset(dataset_ref)
    print(f"Dataset {dataset_ref} already exists.")
  except Exception:
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = LOCATION
    client.create_dataset(dataset, exists_ok=True)
    print(f"Created dataset {dataset_ref}.")
  return client


async def run_agent_test():
  """Run the agent with test queries and return the session ID."""
  print("\n=== Setting up agent ===")

  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      create_views=True,
  )
  bq_plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name="bq_test_agent",
      description="Test agent for BQ plugin verification.",
      instruction=(
          "You are a helpful assistant. Use tools when asked. Be concise."
      ),
      tools=[roll_die, get_weather, calculate],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="bq_plugin_test",
      agent=agent,
      session_service=session_service,
      plugins=[bq_plugin],
  )

  session = await session_service.create_session(
      app_name="bq_plugin_test", user_id="test-user-local"
  )
  session_id = session.id
  print(f"Session ID: {session_id}")

  queries = [
      "Roll a 20-sided die for me",
      "What's the weather in San Francisco?",
      "Calculate 42 * 17 + 3",
      "Roll a 6-sided die and tell me if it's even or odd",
  ]

  for query in queries:
    print(f"\nUser: {query}")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    response_text = ""
    async for event in runner.run_async(
        user_id="test-user-local",
        session_id=session_id,
        new_message=content,
    ):
      if event.content and event.content.parts:
        for part in event.content.parts:
          if part.text:
            response_text += part.text
    print(f"Agent: {response_text[:200]}")
    await asyncio.sleep(0.5)

  # Flush and shut down
  print("\nFlushing plugin...")
  await bq_plugin.flush()
  await asyncio.sleep(2)
  await bq_plugin.shutdown()
  print("Plugin shut down.")

  return session_id


def verify_bigquery(client, session_id):
  """Query BigQuery to verify logged events and views."""
  full_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

  print("\n=== Verifying BigQuery Data ===\n")

  # 1. Check total events
  query = f"""
  SELECT COUNT(*) as total_events
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  """
  result = client.query(query).result()
  total = list(result)[0].total_events
  print(f"Total events for session: {total}")
  if total == 0:
    print("ERROR: No events found! Plugin may not be logging.")
    return False

  # 2. Check event types
  query = f"""
  SELECT event_type, COUNT(*) as cnt
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  GROUP BY event_type
  ORDER BY cnt DESC
  """
  result = client.query(query).result()
  print("\nEvent type breakdown:")
  event_types_found = set()
  for row in result:
    print(f"  {row.event_type}: {row.cnt}")
    event_types_found.add(row.event_type)

  # 3. Verify expected event types
  expected = {
      "INVOCATION_STARTING",
      "INVOCATION_COMPLETED",
      "AGENT_STARTING",
      "AGENT_COMPLETED",
      "LLM_REQUEST",
      "LLM_RESPONSE",
      "TOOL_STARTING",
      "TOOL_COMPLETED",
      "USER_MESSAGE_RECEIVED",
  }
  missing = expected - event_types_found
  if missing:
    print(f"\nWARNING: Missing expected event types: {missing}")
  else:
    print("\nAll expected event types present.")

  # 4. Check a sample event for completeness
  query = f"""
  SELECT timestamp, event_type, agent, trace_id, span_id, status
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  ORDER BY timestamp
  LIMIT 5
  """
  result = client.query(query).result()
  print("\nSample events:")
  for row in result:
    print(
        f"  {row.timestamp} | {row.event_type:25s} | agent={row.agent}"
        f" | trace={row.trace_id[:8] if row.trace_id else 'N/A'}..."
        f" | status={row.status}"
    )

  # 5. Verify analytics views exist and work
  print("\n=== Verifying Analytics Views ===\n")
  views_ok = 0
  views_fail = 0
  for event_type in _EVENT_VIEW_DEFS:
    view_name = "v_" + event_type.lower()
    full_view = f"{PROJECT_ID}.{DATASET_ID}.{view_name}"
    try:
      query = f"SELECT COUNT(*) as cnt FROM `{full_view}` LIMIT 1"
      result = client.query(query).result()
      cnt = list(result)[0].cnt
      print(f"  {view_name:40s} -> {cnt} rows")
      views_ok += 1
    except Exception as e:
      print(f"  {view_name:40s} -> FAILED: {e}")
      views_fail += 1

  print(f"\nViews: {views_ok} OK, {views_fail} failed")

  # 6. Spot-check a typed view
  print("\n=== Spot-check: v_llm_response ===\n")
  try:
    query = f"""
    SELECT
      timestamp, model_version, usage_prompt_tokens,
      usage_completion_tokens, total_ms, ttft_ms
    FROM `{PROJECT_ID}.{DATASET_ID}.v_llm_response`
    WHERE session_id = '{session_id}'
    LIMIT 3
    """
    result = client.query(query).result()
    for row in result:
      print(
          f"  {row.timestamp} | model={row.model_version}"
          f" | prompt_tok={row.usage_prompt_tokens}"
          f" | completion_tok={row.usage_completion_tokens}"
          f" | total_ms={row.total_ms}"
          f" | ttft_ms={row.ttft_ms}"
      )
  except Exception as e:
    print(f"  Error querying v_llm_response: {e}")

  # 7. Spot-check tool view
  print("\n=== Spot-check: v_tool_completed ===\n")
  try:
    query = f"""
    SELECT timestamp, tool_name, tool_origin, total_ms
    FROM `{PROJECT_ID}.{DATASET_ID}.v_tool_completed`
    WHERE session_id = '{session_id}'
    LIMIT 5
    """
    result = client.query(query).result()
    for row in result:
      print(
          f"  {row.timestamp} | tool={row.tool_name}"
          f" | origin={row.tool_origin}"
          f" | total_ms={row.total_ms}"
      )
  except Exception as e:
    print(f"  Error querying v_tool_completed: {e}")

  print("\n=== Verification Complete ===")
  return views_fail == 0 and total > 0


async def main():
  print("=" * 60)
  print("BigQuery Analytics Plugin - Local End-to-End Test")
  print("=" * 60)

  # Step 1: Ensure dataset
  client = ensure_dataset()

  # Step 2: Run agent
  session_id = await run_agent_test()

  # Step 3: Wait for data to settle
  print("\nWaiting 5s for BigQuery data to settle...")
  time.sleep(5)

  # Step 4: Verify
  success = verify_bigquery(client, session_id)

  if success:
    print("\nSUCCESS: All verifications passed!")
    return 0
  else:
    print("\nFAILURE: Some verifications failed.")
    return 1


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
