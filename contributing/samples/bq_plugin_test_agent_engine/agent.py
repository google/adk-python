"""Agent Engine deployment test for BigQuery Agent Analytics Plugin.

Deploy with:
  python contributing/samples/bq_plugin_test_agent_engine/deploy.py

This agent uses the App pattern required for Agent Engine deployment
with the BigQuery analytics plugin.
"""

import random

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig

PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"


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
    name="bq_plugin_ae_test_agent",
    description="Test agent for Agent Engine with BigQuery analytics.",
    instruction="""You are a helpful assistant that can:
1. Roll dice of various sizes (use roll_die tool)
2. Check weather for cities (use get_weather tool)

Always use the appropriate tool. Be concise.
""",
    tools=[roll_die, get_weather],
)

# --- BigQuery Analytics Plugin ---
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

# --- App (required for Agent Engine deployment) ---
app = App(
    name="bq_plugin_test_agent_engine",
    root_agent=root_agent,
    plugins=[bq_plugin],
)
