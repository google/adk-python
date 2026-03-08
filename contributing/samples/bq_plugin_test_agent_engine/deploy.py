"""Deploy and test the BQ plugin agent on Agent Engine.

Usage:
  python contributing/samples/bq_plugin_test_agent_engine/deploy.py
"""

import os
import random
import time

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
import vertexai
from vertexai import agent_engines

PROJECT_ID = "test-project-0728-467323"
LOCATION = "us-central1"
DATASET_ID = "adk_logs"

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
      name="bq_plugin_ae_test_agent",
      description="Test agent for Agent Engine with BigQuery analytics.",
      instruction=(
          "You are a helpful assistant that can roll dice and check"
          " weather. Use the appropriate tool. Be concise."
      ),
      tools=[roll_die, get_weather],
  )

  bq_config = BigQueryLoggerConfig(
      batch_size=1,
      batch_flush_interval=1.0,
      create_views=True,
  )

  bq_plugin = BigQueryAgentAnalyticsPlugin(
      project_id=PROJECT_ID,
      dataset_id=DATASET_ID,
      config=bq_config,
      location=LOCATION,
  )

  app = App(
      name="bq_plugin_test_agent_engine",
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
    import shutil

    whl_name = os.path.basename(_LOCAL_WHEEL)
    staging_dir = "/tmp/whl"
    os.makedirs(staging_dir, exist_ok=True)
    staged_wheel = os.path.join(staging_dir, whl_name)
    shutil.copy2(_LOCAL_WHEEL, staged_wheel)

    extra_packages = [staged_wheel]
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

  print("Deploying agent to Agent Engine...")
  agent_engine = agent_engines.create(
      agent_engine=adk_app,
      display_name="bq-plugin-test",
      requirements=reqs,
      extra_packages=extra_packages,
  )
  print(f"Deployed! Resource name: {agent_engine.resource_name}")
  return agent_engine


def test_agent(agent_engine):
  """Send test queries to the deployed agent."""
  print("\n--- Testing deployed agent ---")

  session = agent_engine.create_session(user_id="test-user-ae")
  print(f"Session ID: {session['id']}")

  queries = [
      "Roll a 20-sided die",
      "What's the weather in Tokyo?",
      "Roll a 6-sided die",
  ]

  for query in queries:
    print(f"\nUser: {query}")
    for event in agent_engine.stream_query(
        user_id="test-user-ae",
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
  """Verify the native BigQuery table and logged events."""
  from google.cloud import bigquery

  client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
  full_table = f"{PROJECT_ID}.{DATASET_ID}.agent_events"
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

  # ── 1. Table Configuration ──
  print("\n=== 1. Native Table Configuration ===\n")
  try:
    table = client.get_table(full_table)

    check(
        "No BigLakeConfiguration (native table)",
        table.biglake_configuration is None,
    )
    check(
        "Time partitioning on timestamp",
        table.time_partitioning is not None,
        f"got {table.time_partitioning}",
    )
  except Exception as e:
    check("Table exists", False, str(e))
    return False

  # ── 2. Events Logged ──
  print("\n=== 2. Event Logging ===\n")

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

  # ── Summary ──
  print("\n" + "=" * 60)
  if all_passed:
    print("SUCCESS: All native BigQuery verifications passed!")
  else:
    print("FAILURE: Some verifications failed. See above.")
  print("=" * 60)
  return all_passed


if __name__ == "__main__":
  import sys

  agent_engine = deploy()
  session_id = test_agent(agent_engine)

  print("\nWaiting 10s for BigQuery data to settle...")
  time.sleep(10)

  success = verify_bigquery(session_id)
  sys.exit(0 if success else 1)
