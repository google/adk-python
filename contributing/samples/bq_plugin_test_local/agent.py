"""Local test agent for BigQuery Agent Analytics Plugin.

Tests fork-safety (PID tracking) and auto-created analytics views.
Uses Gemini 2.5 Flash and logs all events to BigQuery.
"""

import random

from google.adk import Agent
from google.adk.apps import App
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryLoggerConfig
from google.adk.tools.tool_context import ToolContext

PROJECT_ID = "test-project-0728-467323"
DATASET_ID = "adk_logs"
LOCATION = "us-central1"


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
    name="bq_plugin_test_agent",
    description=(
        "A test agent for verifying BigQuery analytics plugin with"
        " tools and state tracking."
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

# --- App ---
app = App(
    name="bq_plugin_test_local",
    root_agent=root_agent,
    plugins=[bq_plugin],
)
