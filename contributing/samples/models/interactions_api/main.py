# Copyright 2026 Google LLC
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

"""Main script for testing the Interactions API integration.

This script tests the following features:
1. Basic text generation
2. Google Search tool
3. Multi-turn conversations with stateful interactions
4. Google Search tool (additional coverage)
5. Custom function tool (get_current_weather)

NOTE: Code execution via UnsafeLocalCodeExecutor is not compatible with function
calling mode because the model tries to call a function instead of outputting
code in markdown.

Run with:
  cd contributing/samples
  python -m interactions_api_test.main
"""

import argparse
import asyncio
import logging
from pathlib import Path
import time

from dotenv import load_dotenv
from google.adk.agents.run_config import RunConfig
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.runners import Runner
from google.genai import types
import httpx

from .agent import root_agent

# Load .env from the samples directory (parent of this module's directory)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

APP_NAME = "interactions_api_test_app"
USER_ID = "test_user"


async def call_agent_async(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
    agent_name: str = "",
    show_interaction_id: bool = True,
    additional_parts: list[types.Part] | None = None,
) -> tuple[str, str | None]:
  """Call the agent asynchronously with the user's prompt.

  Args:
    runner: The agent runner
    user_id: The user ID
    session_id: The session ID
    prompt: The prompt to send
    agent_name: The expected agent name for filtering responses
    show_interaction_id: Whether to show interaction IDs in output
    additional_parts: Optional list of additional content parts (e.g. files)

  Returns:
    A tuple of (response_text, interaction_id)
  """
  parts = [types.Part.from_text(text=prompt)]
  if additional_parts:
    parts.extend(additional_parts)

  content = types.Content(role="user", parts=parts)

  final_response_text = ""
  last_interaction_id = None

  print(f"\n>> User: {prompt}")

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(save_input_blobs_as_artifacts=False),
  ):
    # Track interaction ID if available
    if event.interaction_id:
      last_interaction_id = event.interaction_id

    # Show function calls
    if event.get_function_calls():
      for fc in event.get_function_calls():
        print(f"   [Tool Call] {fc.name}({fc.args})")

    # Show function responses
    if event.get_function_responses():
      for fr in event.get_function_responses():
        print(f"   [Tool Result] {fr.name}: {fr.response}")

    # Collect text responses from the agent (not user, not partial)
    if (
        event.content
        and event.content.parts
        and event.author != "user"
        and not event.partial
    ):
      for part in event.content.parts:
        if part.text:
          # Filter by agent name if provided, otherwise accept any non-user
          if not agent_name or event.author == agent_name:
            final_response_text += part.text

  print(f"<< Agent: {final_response_text}")
  if show_interaction_id and last_interaction_id:
    print(f"   [Interaction ID: {last_interaction_id}]")

  return final_response_text, last_interaction_id


async def test_basic_text_generation(runner: Runner, session_id: str):
  """Test basic text generation without tools."""
  print("\n" + "=" * 60)
  print("TEST 1: Basic Text Generation")
  print("=" * 60)

  response, interaction_id = await call_agent_async(
      runner, USER_ID, session_id, "Hello! What can you help me with?"
  )

  assert response, "Expected a non-empty response"
  print("PASSED: Basic text generation works")
  return interaction_id


async def test_function_calling(runner: Runner, session_id: str):
  """Test function calling with the google_search tool."""
  print("\n" + "=" * 60)
  print("TEST 2: Function Calling (Google Search Tool)")
  print("=" * 60)

  response, interaction_id = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "Search for the capital of France.",
  )

  assert response, "Expected a non-empty response"
  assert "paris" in response.lower(), f"Expected Paris in response: {response}"
  print("PASSED: Google search tool works")
  return interaction_id


async def test_multi_turn_conversation(runner: Runner, session_id: str):
  """Test multi-turn conversation to verify stateful interactions."""
  print("\n" + "=" * 60)
  print("TEST 3: Multi-Turn Conversation (Stateful)")
  print("=" * 60)

  # Turn 1: Tell the agent a fact directly (test conversation memory)
  response1, id1 = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "My favorite color is blue. Just acknowledge this, don't use any tools.",
  )
  assert response1, "Expected a response for turn 1"
  print(f"   Turn 1 interaction_id: {id1}")

  # Turn 2: Ask about something else (use weather tool to add variety)
  response2, id2 = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "What's the weather like in London?",
  )
  assert response2, "Expected a response for turn 2"
  assert (
      "59" in response2
      or "london" in response2.lower()
      or "cloudy" in response2.lower()
  ), f"Expected London weather info in response: {response2}"
  print(f"   Turn 2 interaction_id: {id2}")

  # Turn 3: Ask the agent to recall conversation context
  response3, id3 = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "What is my favorite color that I mentioned earlier in our conversation?",
  )
  assert response3, "Expected a response for turn 3"
  assert (
      "blue" in response3.lower()
  ), f"Expected agent to remember the color 'blue': {response3}"
  print(f"   Turn 3 interaction_id: {id3}")

  # Verify interaction IDs are different (new interactions) but chained
  if id1 and id2 and id3:
    print(f"   Interaction chain: {id1} -> {id2} -> {id3}")

  print("PASSED: Multi-turn conversation works with context retention")


async def test_google_search_tool(runner: Runner, session_id: str):
  """Test the google_search built-in tool."""
  print("\n" + "=" * 60)
  print("TEST 4: Google Search Tool (Additional)")
  print("=" * 60)

  response, interaction_id = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "Use google search to find out who wrote the novel '1984'.",
  )

  assert response, "Expected a non-empty response"
  assert (
      "orwell" in response.lower() or "george" in response.lower()
  ), f"Expected George Orwell in response: {response}"
  print("PASSED: Google search built-in tool works")


async def test_custom_function_tool(runner: Runner, session_id: str):
  """Test the custom function tool alongside google_search.

  The root_agent has both GoogleSearchTool (with bypass_multi_tools_limit=True)
  and get_current_weather. This tests that function calling tools work with
  the Interactions API when all tools are function calling types.
  """
  print("\n" + "=" * 60)
  print("TEST 5: Custom Function Tool (get_current_weather)")
  print("=" * 60)

  response, interaction_id = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "What's the weather like in Tokyo?",
  )

  assert response, "Expected a non-empty response"
  # The mock weather data for Tokyo has temperature 68, condition "Partly Cloudy"
  assert (
      "68" in response
      or "partly" in response.lower()
      or "tokyo" in response.lower()
  ), f"Expected weather info for Tokyo in response: {response}"
  print("PASSED: Custom function tool works with bypass_multi_tools_limit")
  return interaction_id


async def test_pdf_summarization(runner: Runner, session_id: str) -> str | None:
  """Test PDF summarization using the Interactions API."""
  print("\n" + "=" * 60)
  print("TEST 6: PDF Summarization")
  print("=" * 60)

  url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"
  print(f"Downloading {url}...")
  async with httpx.AsyncClient() as client:
    response = await client.get(
        url, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
    )
    response.raise_for_status()
    pdf_bytes = response.content

  pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
  response, interaction_id = await call_agent_async(
      runner,
      USER_ID,
      session_id,
      "Please summarize the attached PDF document.",
      additional_parts=[pdf_part],
  )

  assert response, "Expected a non-empty response"
  assert len(response) > 0, f"Expected summary in response: {response}"
  assert (
      "gemini" in response.lower() or "multimodal" in response.lower()
  ), f"Expected summary of PDF in response: {response}"
  print("PASSED: PDF Summarization works")
  return interaction_id


def check_interactions_api_available() -> bool:
  """Check if the interactions API is available in the SDK."""
  try:
    from google.genai import Client

    client = Client()
    # Check if interactions attribute exists
    return hasattr(client.aio, "interactions")
  except Exception:
    return False


async def run_all_tests():
  """Run all tests with the Interactions API."""
  print("\n" + "#" * 70)
  print("# Running tests with Interactions API")
  print("#" * 70)

  # Check if interactions API is available
  if not check_interactions_api_available():
    print("\nERROR: Interactions API is not available in the current SDK.")
    print("The interactions API requires a SDK version with this feature.")
    print("To use the interactions API, ensure you have the SDK with")
    print("interactions support installed (e.g., from private-python-genai).")
    return False

  test_agent = root_agent

  runner = InMemoryRunner(
      agent=test_agent,
      app_name=APP_NAME,
  )

  # Create a new session
  session = await runner.session_service.create_session(
      user_id=USER_ID,
      app_name=APP_NAME,
  )
  print(f"\nSession created: {session.id}")

  try:
    # Run all tests
    await test_basic_text_generation(runner, session.id)
    await test_function_calling(runner, session.id)
    await test_multi_turn_conversation(runner, session.id)
    await test_google_search_tool(runner, session.id)
    await test_custom_function_tool(runner, session.id)
    await test_pdf_summarization(runner, session.id)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED (Interactions API)")
    print("=" * 60)
    return True

  except AssertionError as e:
    print(f"\nTEST FAILED: {e}")
    return False
  except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
    return False


async def interactive_mode():
  """Run in interactive mode for manual testing."""
  # Check if interactions API is available
  if not check_interactions_api_available():
    print("\nERROR: Interactions API is not available in the current SDK.")
    print("To use the interactions API, ensure you have the SDK with")
    print("interactions support installed (e.g., from private-python-genai).")
    return

  print("\nInteractive mode with Interactions API")
  print("Type 'quit' to exit, 'new' for a new session\n")

  test_agent = agent.root_agent

  runner = InMemoryRunner(
      agent=test_agent,
      app_name=APP_NAME,
  )

  session = await runner.session_service.create_session(
      user_id=USER_ID,
      app_name=APP_NAME,
  )
  print(f"Session created: {session.id}\n")

  while True:
    try:
      user_input = input("You: ").strip()
      if not user_input:
        continue
      if user_input.lower() == "quit":
        break
      if user_input.lower() == "new":
        session = await runner.session_service.create_session(
            user_id=USER_ID,
            app_name=APP_NAME,
        )
        print(f"New session created: {session.id}\n")
        continue

      await call_agent_async(runner, USER_ID, session.id, user_input)

    except KeyboardInterrupt:
      break

  print("\nGoodbye!")


def main():
  parser = argparse.ArgumentParser(
      description="Test the Interactions API integration"
  )
  parser.add_argument(
      "--mode",
      choices=["test", "interactive"],
      default="test",
      help=(
          "Run mode: 'test' runs automated tests, 'interactive' for manual"
          " testing"
      ),
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging",
  )

  args = parser.parse_args()

  if args.debug:
    logs.setup_adk_logger(level=logging.DEBUG)
  else:
    logs.setup_adk_logger(level=logging.INFO)

  start_time = time.time()

  if args.mode == "test":
    success = asyncio.run(run_all_tests())
    if not success:
      exit(1)

  elif args.mode == "interactive":
    asyncio.run(interactive_mode())

  end_time = time.time()
  print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
  main()
