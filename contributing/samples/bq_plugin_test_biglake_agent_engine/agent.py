"""Agent Engine agent for BigQuery Analytics Plugin with BigLake Iceberg.

Validates that the BigLake Iceberg support works when deployed to
Vertex AI Agent Engine.

Prerequisites: see bq_plugin_test_biglake_local/agent.py for setup.
"""

import random

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig

# ──────────────────────────────────────────────────────────────────
# CONFIGURE THESE for your environment
# ──────────────────────────────────────────────────────────────────
PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"
CONNECTION_ID = "us-central1.my-ai-connection"
BIGLAKE_STORAGE_URI = (
    "gs://test-project-0728-467323-biglake/agent_events_iceberg_ae/"
)
TABLE_ID = "agent_events_iceberg_ae"


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


# --- Agent ---
root_agent = Agent(
    model="gemini-2.5-flash",
    name="bq_biglake_ae_test_agent",
    description=(
        "Test agent for Agent Engine with BigQuery analytics and BigLake"
        " Iceberg."
    ),
    instruction="""You are a helpful assistant that can:
1. Roll dice of various sizes (use roll_die tool)
2. Check weather for cities (use get_weather tool)

Always use the appropriate tool. Be concise.
""",
    tools=[roll_die, get_weather],
)

# --- BigQuery Analytics Plugin (BigLake Iceberg) ---
bq_config = BigQueryLoggerConfig(
    batch_size=1,
    batch_flush_interval=1.0,
    create_views=True,
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

# --- App (required for Agent Engine deployment) ---
app = App(
    name="bq_biglake_test_agent_engine",
    root_agent=root_agent,
    plugins=[bq_plugin],
)
