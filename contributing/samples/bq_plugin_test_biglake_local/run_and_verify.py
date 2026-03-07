"""Run the BigLake Iceberg test agent locally and verify BigQuery logging.

Usage:
  python contributing/samples/bq_plugin_test_biglake_local/run_and_verify.py

This script:
1. Creates the BQ dataset if needed
2. Runs the agent with several test queries against a BigLake Iceberg table
3. Waits for data to land in BigQuery
4. Verifies:
   - Table was created with BigLakeConfiguration (Iceberg / PARQUET)
   - Schema has no JSON fields (all replaced with STRING)
   - Events were logged correctly (event types, trace IDs, content)
   - STRING content columns are queryable
   - No time partitioning (BigLake Iceberg default)
5. Prints a summary report

Prerequisites: see agent.py docstring for setup instructions.
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
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.cloud import bigquery
from google.genai import types

# ──────────────────────────────────────────────────────────────────
# CONFIGURE THESE for your environment (must match agent.py)
# ──────────────────────────────────────────────────────────────────
PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"
CONNECTION_ID = "us-central1.my-ai-connection"
BIGLAKE_STORAGE_URI = (
    "gs://test-project-0728-467323-biglake/agent_events_iceberg/"
)
TABLE_ID = "agent_events_iceberg"


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
  print("\n=== Setting up BigLake Iceberg agent ===")

  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=0.5,
      create_views=False,
      connection_id=CONNECTION_ID,
      biglake_storage_uri=BIGLAKE_STORAGE_URI,
      table_id=TABLE_ID,
  )
  bq_plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  agent = LlmAgent(
      model="gemini-2.5-flash",
      name="bq_biglake_test_agent",
      description="Test agent for BigLake Iceberg BQ plugin verification.",
      instruction=(
          "You are a helpful assistant. Use tools when asked. Be concise."
      ),
      tools=[roll_die, get_weather, calculate],
  )

  session_service = InMemorySessionService()
  runner = Runner(
      app_name="bq_biglake_test",
      agent=agent,
      session_service=session_service,
      plugins=[bq_plugin],
  )

  session = await session_service.create_session(
      app_name="bq_biglake_test", user_id="test-user-biglake"
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
        user_id="test-user-biglake",
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
  """Query BigQuery to verify BigLake Iceberg table and logged events."""
  full_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
  all_passed = True

  def check(name, passed, detail=""):
    nonlocal all_passed
    status = "PASS" if passed else "FAIL"
    if not passed:
      all_passed = False
    msg = f"  [{status}] {name}"
    if detail:
      msg += f" — {detail}"
    print(msg)

  # ── 1. Verify BigLake Configuration ──
  print("\n=== 1. BigLake Table Configuration ===\n")
  try:
    table = client.get_table(full_table)

    has_biglake = table.biglake_configuration is not None
    check("Table has BigLakeConfiguration", has_biglake)

    if has_biglake:
      cfg = table.biglake_configuration
      check(
          "file_format is PARQUET",
          cfg.file_format == "PARQUET",
          f"got {cfg.file_format}",
      )
      check(
          "table_format is ICEBERG",
          cfg.table_format == "ICEBERG",
          f"got {cfg.table_format}",
      )
      check(
          "storage_uri matches",
          cfg.storage_uri == BIGLAKE_STORAGE_URI,
          f"got {cfg.storage_uri}",
      )
      # BigQuery may return connection_id in either full resource
      # path or dot-separated format.
      check(
          "connection_id contains project and connection",
          PROJECT_ID in cfg.connection_id
          and "my-ai-connection" in cfg.connection_id,
          f"got {cfg.connection_id}",
      )
    else:
      check("BigLakeConfiguration details", False, "no config found")

    # Check no time partitioning (default for BigLake Iceberg)
    check(
        "No time partitioning (BigLake default)",
        table.time_partitioning is None,
        f"got {table.time_partitioning}",
    )
  except Exception as e:
    check("Table exists", False, str(e))
    return False

  # ── 2. Verify Schema: No JSON fields ──
  print("\n=== 2. Schema Validation (no JSON fields) ===\n")

  def find_json_fields(fields, prefix=""):
    found = []
    for f in fields:
      full_name = f"{prefix}{f.name}" if prefix else f.name
      if f.field_type == "JSON":
        found.append(full_name)
      if f.fields:
        found.extend(find_json_fields(f.fields, f"{full_name}."))
    return found

  json_fields = find_json_fields(table.schema)
  check(
      "No JSON fields in schema",
      len(json_fields) == 0,
      f"JSON fields found: {json_fields}" if json_fields else "all STRING",
  )

  # Verify the fields that were JSON/RECORD are now STRING
  expected_string_fields = {
      "content": False,
      "attributes": False,
      "latency_ms": False,
      "content_parts": False,
  }
  for f in table.schema:
    if f.name in expected_string_fields and f.field_type == "STRING":
      expected_string_fields[f.name] = True

  for field_name, is_string in expected_string_fields.items():
    check(f"'{field_name}' is STRING", is_string)

  # ── 3. Verify Events Logged ──
  print("\n=== 3. Event Logging ===\n")

  query = f"""
  SELECT COUNT(*) as total_events
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  """
  result = client.query(query).result()
  total = list(result)[0].total_events
  check(
      "Events logged for session",
      total > 0,
      f"{total} events",
  )

  if total == 0:
    print("\nNo events found — cannot continue verification.")
    return False

  # Check event types
  query = f"""
  SELECT event_type, COUNT(*) as cnt
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  GROUP BY event_type
  ORDER BY cnt DESC
  """
  result = client.query(query).result()
  print("\n  Event type breakdown:")
  event_types_found = set()
  for row in result:
    print(f"    {row.event_type}: {row.cnt}")
    event_types_found.add(row.event_type)

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
  check(
      "All expected event types present",
      len(missing) == 0,
      f"missing: {missing}" if missing else f"{len(expected)} types",
  )

  # ── 4. Verify STRING content is queryable ──
  print("\n=== 4. STRING Content Queryability ===\n")

  # content is now STRING; verify we can query it
  query = f"""
  SELECT
    event_type,
    SAFE.PARSE_JSON(content) IS NOT NULL AS content_is_valid_json,
    SAFE.PARSE_JSON(attributes) IS NOT NULL AS attrs_is_valid_json
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
    AND content IS NOT NULL
  LIMIT 5
  """
  try:
    result = client.query(query).result()
    rows = list(result)
    check(
        "STRING content is queryable",
        len(rows) > 0,
        f"{len(rows)} rows with content",
    )
    # Verify content is valid JSON stored as STRING
    valid_json_count = sum(1 for r in rows if r.content_is_valid_json)
    check(
        "Content is valid JSON in STRING column",
        valid_json_count > 0,
        f"{valid_json_count}/{len(rows)} valid",
    )
  except Exception as e:
    check("STRING content query", False, str(e))

  # ── 5. Verify trace IDs and sample events ──
  print("\n=== 5. Sample Events ===\n")

  query = f"""
  SELECT timestamp, event_type, agent, trace_id, span_id, status
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  ORDER BY timestamp
  LIMIT 5
  """
  result = client.query(query).result()
  for row in result:
    print(
        f"    {row.timestamp} | {row.event_type:25s} | agent={row.agent}"
        f" | trace={row.trace_id[:8] if row.trace_id else 'N/A'}..."
        f" | status={row.status}"
    )

  # ── Summary ──
  print("\n" + "=" * 60)
  if all_passed:
    print("SUCCESS: All BigLake Iceberg verifications passed!")
  else:
    print("FAILURE: Some verifications failed. See above.")
  print("=" * 60)
  return all_passed


async def main():
  print("=" * 60)
  print("BigQuery Analytics Plugin — BigLake Iceberg E2E Test")
  print("=" * 60)

  # Step 1: Ensure dataset
  client = ensure_dataset()

  # Step 2: Optionally drop existing table for a clean test.
  # Pass --fresh to drop and recreate.
  full_table = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
  if "--fresh" in sys.argv:
    try:
      client.delete_table(full_table, not_found_ok=True)
      print(f"Dropped existing table {full_table} for clean test.")
      # BigLake Iceberg tables need time after creation before the
      # Storage Write API _default stream becomes available.  Wait
      # for the table to be created in _lazy_setup, then give it
      # extra time.
      print("Will wait 30s after table creation for BigLake readiness.")
    except Exception as e:
      print(f"Could not drop table (may not exist): {e}")
  else:
    print(f"Using existing table {full_table} (pass --fresh to recreate).")

  # Step 3: Run agent
  session_id = await run_agent_test()

  # Step 4: Wait for data to settle.  BigLake Iceberg needs longer.
  wait_secs = 30 if "--fresh" in sys.argv else 8
  print(f"\nWaiting {wait_secs}s for BigQuery data to settle...")
  time.sleep(wait_secs)

  # Step 5: Verify
  success = verify_bigquery(client, session_id)

  return 0 if success else 1


if __name__ == "__main__":
  sys.exit(asyncio.run(main()))
