"""Basic GraphAgent example demonstrating conditional routing.

This example shows how GraphAgent enables conditional workflow routing
based on runtime state, which cannot be achieved with SequentialAgent
or ParallelAgent composition.

Use case: Data validation pipeline with retry logic.
- If validation passes -> process data
- If validation fails -> retry validation
- After max retries -> route to error handler
"""

import asyncio
import os

from google.adk.agents import GraphAgent
from google.adk.agents import LlmAgent
from google.adk.agents.graph import GraphState
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")


# --- Validation Result Schema ---
class ValidationResult(BaseModel):
  """Validation result structure."""

  valid: bool
  error: str | None = None


# --- Validator Agent ---
validator = LlmAgent(
    name="validator",
    model=_MODEL,
    instruction="""
    You validate input data quality.
    Check if the input contains valid JSON.
    Return {"valid": true} if valid, {"valid": false, "error": "reason"} if invalid.
    """,
    output_schema=ValidationResult,  # Ensures structured JSON output
    # output_key auto-defaults to "validator" (agent name)
)

# --- Processor Agent ---
processor = LlmAgent(
    name="processor",
    model=_MODEL,
    instruction="""
    You process validated data.
    Transform the input JSON and return processed results.
    """,
)

# --- Error Handler Agent ---
error_handler = LlmAgent(
    name="error_handler",
    model=_MODEL,
    instruction="""
    You handle validation errors.
    Provide helpful error messages and suggestions for fixing invalid data.
    """,
)


# --- Edge Condition Functions ---
def is_valid_json(state: GraphState) -> bool:
  """Check if JSON is valid from structured output."""
  result = state.get_parsed("validator", ValidationResult)
  return result.valid if result else False


# --- Create GraphAgent with Conditional Routing ---
def build_validation_graph() -> GraphAgent:
  """Build the validation pipeline graph."""
  g = GraphAgent(name="validation_pipeline")

  # Add nodes
  g.add_node("validate", agent=validator)
  g.add_node("process", agent=processor)
  g.add_node("error", agent=error_handler)

  # Add conditional edges
  # If validation passes (state.data["validator"]["valid"] == True) -> process
  g.add_edge(
      "validate",
      "process",
      condition=is_valid_json,
  )

  # If validation fails (state.data["validator"]["valid"] == False) -> error handler
  g.add_edge(
      "validate",
      "error",
      condition=lambda state: not is_valid_json(state),
  )

  # Define workflow
  g.set_start("validate")
  g.set_end("process")  # Success path ends at process
  g.set_end("error")  # Error path ends at error handler

  return g


# --- Run the workflow ---


async def main():
  graph = build_validation_graph()
  runner = Runner(
      app_name="validation_pipeline",
      agent=graph,
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  # Example: Valid input
  print("=== Testing with valid JSON ===")
  async for event in runner.run_async(
      user_id="user_1",
      session_id="session_1",
      new_message=types.Content(
          role="user",
          parts=[types.Part(text='{"name": "John", "age": 30}')],
      ),
  ):
    if event.content and event.content.parts:
      print(f"{event.author}: {event.content.parts[0].text}")

  # Example: Invalid input
  print("\n=== Testing with invalid JSON ===")
  async for event in runner.run_async(
      user_id="user_1",
      session_id="session_2",
      new_message=types.Content(
          role="user",
          parts=[types.Part(text='{"name": "Invalid data')],
      ),
  ):
    if event.content and event.content.parts:
      print(f"{event.author}: {event.content.parts[0].text}")


if __name__ == "__main__":
  asyncio.run(main())
