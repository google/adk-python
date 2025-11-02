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

"""Example agent for demonstrating run_debug helper method."""

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext


def get_weather(city: str, tool_context: ToolContext) -> str:
  """Get weather information for a city.

  Args:
      city: Name of the city to get weather for.
      tool_context: Tool context for session state.

  Returns:
      Weather information as a string.
  """
  # Store query history in session state
  if "weather_queries" not in tool_context.state:
    tool_context.state["weather_queries"] = [city]
  else:
    tool_context.state["weather_queries"] = tool_context.state[
        "weather_queries"
    ] + [city]

  # Mock weather data for demonstration
  weather_data = {
      "San Francisco": "Foggy, 15°C (59°F)",
      "New York": "Sunny, 22°C (72°F)",
      "London": "Rainy, 12°C (54°F)",
      "Tokyo": "Clear, 25°C (77°F)",
      "Paris": "Cloudy, 18°C (64°F)",
  }

  return weather_data.get(
      city, f"Weather data not available for {city}. Try a major city."
  )


def calculate(expression: str) -> str:
  """Safely evaluate a mathematical expression.

  This tool demonstrates how function calls are displayed in run_debug().

  Args:
      expression: Mathematical expression to evaluate.

  Returns:
      Result of the calculation as a string.
  """
  import ast
  import operator

  # Supported operators for safe evaluation
  operators = {
      ast.Add: operator.add,
      ast.Sub: operator.sub,
      ast.Mult: operator.mul,
      ast.Div: operator.truediv,
      ast.Pow: operator.pow,
      ast.USub: operator.neg,
  }

  def _eval(node):
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.Expression):
      return _eval(node.body)
    elif isinstance(node, ast.Constant):  # Python 3.8+
      return node.value
    elif isinstance(node, ast.Num):  # For older Python versions
      return node.n
    elif isinstance(node, ast.BinOp):
      op = operators.get(type(node.op))
      if op:
        return op(_eval(node.left), _eval(node.right))
      else:
        raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
    elif isinstance(node, ast.UnaryOp):
      op = operators.get(type(node.op))
      if op:
        return op(_eval(node.operand))
      else:
        raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
    else:
      raise ValueError(f"Unsupported expression type: {type(node).__name__}")

  try:
    # Parse the expression into an AST
    tree = ast.parse(expression, mode="eval")
    # Safely evaluate the AST
    result = _eval(tree)
    return f"Result: {result}"
  except (SyntaxError, ValueError) as e:
    return f"Error: {str(e)}"
  except ZeroDivisionError:
    return "Error: Division by zero"
  except Exception as e:
    return f"Error: {str(e)}"


root_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="agent",
    description="A helpful assistant demonstrating run_debug() helper method",
    instruction="""You are a helpful assistant that can:
    1. Provide weather information for major cities
    2. Perform mathematical calculations
    3. Remember previous queries in the conversation

    When users ask about weather, use the get_weather tool.
    When users ask for calculations, use the calculate tool.
    Be friendly and conversational.""",
    tools=[get_weather, calculate],
)
