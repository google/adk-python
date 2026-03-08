"""Deploy and test the BigLake Iceberg BQ plugin agent on Agent Engine.

Usage:
  python contributing/samples/bq_plugin_test_biglake_agent_engine/deploy.py

This script:
1. Deploys the agent to Vertex AI Agent Engine
2. Runs test queries
3. Waits for events to land in BigQuery
4. Verifies the BigLake Iceberg table configuration and logged events

Prerequisites: see bq_plugin_test_biglake_local/agent.py for setup.
"""

import os
import random
import sys
import time

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.cloud import bigquery
import vertexai
from vertexai import agent_engines

# Path to local wheel — build with: uv build --wheel
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_DIST_DIR = os.path.join(_REPO_ROOT, "dist")


def _find_local_wheel():
  """Find the latest google_adk wheel in dist/."""
  import glob

  pattern = os.path.join(_DIST_DIR, "google_adk-*-py3-none-any.whl")
  wheels = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
  return wheels[0] if wheels else None

# ──────────────────────────────────────────────────────────────────
# CONFIGURE THESE for your environment
# ──────────────────────────────────────────────────────────────────
PROJECT_ID = "test-project-0728-467323"
LOCATION = "us-central1"
DATASET_ID = "adk_logs"
CONNECTION_ID = "us-central1.my-ai-connection"
BIGLAKE_STORAGE_URI = (
    "gs://test-project-0728-467323-biglake/agent_events_iceberg_ae/"
)
TABLE_ID = "agent_events_iceberg_ae"


# --- Tools (defined at module level for cloudpickle) ---
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


def deploy():
  """Deploy the agent to Agent Engine."""
  vertexai.init(
      project=PROJECT_ID,
      location=LOCATION,
      staging_bucket=f"gs://{PROJECT_ID}-adk-staging",
  )

  # Define agent, plugin, and app inline to avoid cloudpickle
  # referencing external modules that won't exist on the remote
  # container.
  root_agent = Agent(
      model="gemini-2.5-flash",
      name="bq_biglake_ae_test_agent",
      description=(
          "Test agent for Agent Engine with BigQuery analytics and"
          " BigLake Iceberg."
      ),
      instruction=(
          "You are a helpful assistant that can roll dice and check"
          " weather. Use the appropriate tool. Be concise."
      ),
      tools=[roll_die, get_weather],
  )

  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=1.0,
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

  app = App(
      name="bq_biglake_test_agent_engine",
      root_agent=root_agent,
      plugins=[bq_plugin],
  )

  # Wrap the App in AdkApp which provides the query/stream
  # interface required by Agent Engine.
  adk_app = agent_engines.AdkApp(app=app, enable_tracing=True)

  # Use local wheel if available, otherwise fall back to PyPI.
  extra_packages = []
  _LOCAL_WHEEL = _find_local_wheel()
  if _LOCAL_WHEEL:
    print(f"Using local wheel: {_LOCAL_WHEEL}")
    # Copy wheel to /tmp/whl/ so the tarball member path is short
    # and predictable. After extraction to /code/, the wheel will be
    # at /code/tmp/whl/<filename>.whl.
    import shutil

    whl_name = os.path.basename(_LOCAL_WHEEL)
    staging_dir = "/tmp/whl"
    os.makedirs(staging_dir, exist_ok=True)
    staged_wheel = os.path.join(staging_dir, whl_name)
    shutil.copy2(_LOCAL_WHEEL, staged_wheel)

    extra_packages = [staged_wheel]
    # Reference the wheel via its extracted path so pip installs it.
    # The tarball extracts to /code/, making the wheel available at
    # ./tmp/whl/<filename>.whl relative to the pip working directory.
    reqs = [
        f"./tmp/whl/{whl_name}",
        "google-cloud-aiplatform",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "google-cloud-storage",
        "pyarrow",
    ]
  else:
    print("Local wheel not found, using PyPI google-adk[bigquery].")
    reqs = [
        "google-adk[bigquery]",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "google-cloud-storage",
        "pyarrow",
    ]

  print("Deploying BigLake Iceberg agent to Agent Engine...")
  agent_engine = agent_engines.create(
      agent_engine=adk_app,
      display_name="bq-biglake-iceberg-test",
      requirements=reqs,
      extra_packages=extra_packages,
  )
  print(f"Deployed! Resource name: {agent_engine.resource_name}")
  return agent_engine


def test_agent(agent_engine):
  """Send test queries to the deployed agent."""
  print("\n--- Testing deployed BigLake Iceberg agent ---")

  session = agent_engine.create_session(user_id="test-user-biglake-ae")
  print(f"Session ID: {session['id']}")

  queries = [
      "Roll a 20-sided die",
      "What's the weather in Tokyo?",
      "Roll a 6-sided die",
  ]

  for query in queries:
    print(f"\nUser: {query}")
    for event in agent_engine.stream_query(
        user_id="test-user-biglake-ae",
        session_id=session["id"],
        message=query,
    ):
      if isinstance(event, dict):
        content = event.get("content")
        if content and isinstance(content, dict):
          for part in content.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
              print(f"Agent: {part['text']}")
      elif hasattr(event, "content") and event.content:
        for part in event.content.parts:
          if part.text:
            print(f"Agent: {part.text}")
    time.sleep(1)

  print("\n--- Agent Engine test complete ---")
  return session["id"]


def verify_bigquery(session_id):
  """Verify the BigLake Iceberg table and logged events."""
  client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
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

  for field_name in ("content", "attributes", "latency_ms", "content_parts"):
    f = next((f for f in table.schema if f.name == field_name), None)
    check(
        f"'{field_name}' is STRING",
        f is not None and f.field_type == "STRING",
    )

  # ── 3. Verify Events Logged ──
  print("\n=== 3. Event Logging ===\n")

  query = f"""
  SELECT COUNT(*) as total_events
  FROM `{full_table}`
  WHERE session_id = '{session_id}'
  """
  result = client.query(query).result()
  total = list(result)[0].total_events
  check("Events logged for session", total > 0, f"{total} events")

  if total > 0:
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
        "LLM_REQUEST",
        "LLM_RESPONSE",
        "TOOL_STARTING",
        "TOOL_COMPLETED",
    }
    missing = expected - event_types_found
    check(
        "Expected event types present",
        len(missing) == 0,
        f"missing: {missing}" if missing else f"{len(expected)} types",
    )

  # ── 4. Verify STRING content queryable ──
  print("\n=== 4. STRING Content Queryability ===\n")
  try:
    query = f"""
    SELECT
      event_type,
      SAFE.PARSE_JSON(content) IS NOT NULL AS content_is_valid_json
    FROM `{full_table}`
    WHERE session_id = '{session_id}'
      AND content IS NOT NULL
    LIMIT 5
    """
    result = client.query(query).result()
    rows = list(result)
    check(
        "STRING content is queryable",
        len(rows) > 0,
        f"{len(rows)} rows with content",
    )
  except Exception as e:
    check("STRING content query", False, str(e))

  # ── Summary ──
  print("\n" + "=" * 60)
  if all_passed:
    print("SUCCESS: All BigLake Iceberg verifications passed!")
  else:
    print("FAILURE: Some verifications failed. See above.")
  print("=" * 60)
  return all_passed


if __name__ == "__main__":
  agent_engine = deploy()
  session_id = test_agent(agent_engine)

  print("\nWaiting 10s for BigQuery data to settle...")
  time.sleep(10)

  success = verify_bigquery(session_id)
  sys.exit(0 if success else 1)
