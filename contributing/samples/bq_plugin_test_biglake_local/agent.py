"""Local test agent for BigQuery Analytics Plugin with BigLake Iceberg.

Validates that the BigLake Iceberg support (PR #4750) works end-to-end:
  - JSON fields are replaced with STRING in the schema
  - BigLakeConfiguration is set on the table
  - Events are written via legacy streaming (insert_rows_json)
  - Analytics views work on STRING columns (JSON_VALUE is polymorphic)

Prerequisites:
  1. A BigQuery Cloud Resource connection in your project.
     Create one via:
       bq mk --connection --connection_type=CLOUD_RESOURCE \
         --location=us-central1 my-biglake-conn
  2. A GCS bucket for Iceberg table storage.
  3. Grant the connection's service account the Storage Object Admin
     role on the GCS bucket.

Usage:
  # Update PROJECT_ID, CONNECTION_ID, BIGLAKE_STORAGE_URI below, then:
  adk web contributing/samples/bq_plugin_test_biglake_local
"""

import random

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.adk.tools.tool_context import ToolContext

# ──────────────────────────────────────────────────────────────────
# CONFIGURE THESE for your environment
# ──────────────────────────────────────────────────────────────────
PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"
# BigLake connection in "location.connection_name" format.
CONNECTION_ID = "us-central1.my-ai-connection"
# GCS path where Iceberg metadata & data files will be stored.
BIGLAKE_STORAGE_URI = (
    "gs://test-project-0728-467323-biglake/agent_events_iceberg/"
)
# Table name — separate from the default to avoid conflicts.
TABLE_ID = "agent_events_iceberg"


# --- Tools ---
def roll_die(sides: int, tool_context: ToolContext) -> int:
  """Roll a die and return the result.

  Args:
    sides: The integer number of sides the die has.

  Returns:
    An integer of the result of rolling the die.
  """
  result = random.randint(1, sides)
  if "rolls" not in tool_context.state:
    tool_context.state["rolls"] = []
  tool_context.state["rolls"] = tool_context.state["rolls"] + [result]
  return result


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


# --- Agent ---
root_agent = Agent(
    model="gemini-2.5-flash",
    name="bq_biglake_test_agent",
    description=(
        "A test agent for verifying BigQuery analytics plugin with"
        " BigLake Iceberg support."
    ),
    instruction="""You are a helpful assistant that can:
1. Roll dice of various sizes (use roll_die tool)
2. Check weather for cities (use get_weather tool)
3. Do basic math calculations (use calculate tool)

Always use the appropriate tool when asked. Be concise in responses.
When rolling dice, track results in state.
""",
    tools=[roll_die, get_weather, calculate],
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

# --- App ---
app = App(
    name="bq_biglake_test_local",
    root_agent=root_agent,
    plugins=[bq_plugin],
)
