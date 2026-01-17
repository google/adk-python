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

"""Sample CodingAgent demonstrating code generation with tool usage.

This sample shows how to create a CodingAgent that can:
- Generate Python code to solve tasks
- Call tools as Python functions from within the generated code
- Execute code in a sandboxed container environment
- Provide final answers after multi-step reasoning

Prerequisites:
- Docker must be installed and running
- Set GOOGLE_API_KEY or configure Vertex AI credentials

Usage:
    adk run contributing/samples/coding_agent
    adk web contributing/samples

Example queries:
- "What is 15% of 847?"
- "Calculate the compound interest on $10,000 at 5% annual rate for 3 years"
- "Search for the latest Python release and summarize the key features"
"""

from google.adk.agents import CodingAgent
from google.adk.code_executors import ContainerCodeExecutor


# Define sample tools that the CodingAgent can use
def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression.

    Args:
      expression: A mathematical expression to evaluate (e.g., "2 + 2 * 3").

    Returns:
      Dictionary with the result or error message.
    """
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e), "expression": expression}


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for information.

    Args:
      query: The search query.
      max_results: Maximum number of results to return.

    Returns:
      Dictionary with search results.
    """
    # This is a mock implementation for demonstration
    # In production, you would integrate with a real search API
    return {
        "query": query,
        "results": [
            {
                "title": f"Result {i + 1} for: {query}",
                "snippet": f"This is a sample result snippet for '{query}'...",
                "url": f"https://example.com/result{i + 1}",
            }
            for i in range(min(max_results, 3))
        ],
        "total_results": max_results,
    }


def read_file(path: str) -> dict:
    """Read contents of a file.

    Args:
      path: Path to the file to read.

    Returns:
      Dictionary with file contents or error.
    """
    # This is a mock implementation for demonstration
    # In production, you would implement actual file reading with proper security
    mock_files = {
        "data.csv": {
            "content": "name,amount\nAlice,100\nBob,200\nCharlie,150",
            "rows": [
                {"name": "Alice", "amount": "100"},
                {"name": "Bob", "amount": "200"},
                {"name": "Charlie", "amount": "150"},
            ],
        },
        "config.json": {
            "content": '{"setting": "value"}',
            "data": {"setting": "value"},
        },
    }

    if path in mock_files:
        return {"path": path, **mock_files[path]}
    return {"error": f"File not found: {path}", "path": path}


def get_current_time() -> dict:
    """Get the current date and time.

    Returns:
      Dictionary with current timestamp information.
    """
    from datetime import datetime

    now = datetime.now()
    return {
        "timestamp": now.isoformat(),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.strftime("%A"),
    }


# Create the CodingAgent with tools
root_agent = CodingAgent(
    name="code_assistant",
    description=(
        "An AI assistant that solves tasks by writing and executing Python code. "
        "It can perform calculations, search for information, read files, and more."
    ),
    model="gemini-2.5-flash",
    instruction="""
You are a helpful coding assistant that solves problems by writing Python code.

When given a task:
1. Think about what tools and computations you need
2. Write clear, well-commented Python code
3. Use the available tools as needed
4. Print intermediate results to verify your work
5. Call final_answer() with your result

Always show your reasoning through code comments and print statements.
If a task cannot be completed with the available tools, explain why.
""",
    tools=[
        calculator,
        web_search,
        read_file,
        get_current_time,
    ],
    # Use ContainerCodeExecutor for sandboxed execution
    # Note: Docker must be installed and running
    code_executor=ContainerCodeExecutor(
        image="python:3.11-slim",
    ),
    max_iterations=10,
    error_retry_attempts=2,
    stateful=False,
)
