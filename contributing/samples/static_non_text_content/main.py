"""Static non-text content sample agent main script."""

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

import argparse
import asyncio
import logging
import sys
import time

from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

from . import agent

APP_NAME = "static_non_text_content_demo"
USER_ID = "demo_user"

logs.setup_adk_logger(level=logging.INFO)


async def call_agent_async(
    runner, user_id: str, session_id: str, prompt: str
) -> str:
  """Helper function to call agent and return final response."""
  from google.adk.agents.run_config import RunConfig
  from google.genai import types

  content = types.Content(
      role="user", parts=[types.Part.from_text(text=prompt)]
  )

  final_response_text = ""
  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=content,
      run_config=RunConfig(save_input_blobs_as_artifacts=False),
  ):
    if event.content and event.content.parts:
      if text := "".join(part.text or "" for part in event.content.parts):
        if event.author != "user":
          final_response_text += text

  return final_response_text or "No response received"


def process_arguments():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "A demo script that tests static instructions with non-text content."
      ),
      epilog=(
          "Example usage: \n\tpython -m static_non_text_content.main --prompt"
          " 'What can you see in the reference chart?'\n\tpython -m"
          " static_non_text_content.main --prompt 'What is the Gemma research"
          " paper about?'\n\tpython -m static_non_text_content.main  # Runs"
          " default test prompts\n\tadk run"
          " contributing/samples/static_non_text_content  # Interactive mode\n"
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  parser.add_argument(
      "--prompt",
      type=str,
      help=(
          "Single prompt to send to the agent. If not provided, runs"
          " default test prompts."
      ),
  )

  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging to see internal processing details.",
  )

  return parser.parse_args()


async def run_default_test_prompts(runner):
  """Run default test prompts to demonstrate static content features."""
  from google.adk.utils.variant_utils import get_google_llm_variant
  from google.adk.utils.variant_utils import GoogleLLMVariant

  api_variant = get_google_llm_variant()

  print("=== Static Non-Text Content Demo Agent - Default Test Prompts ===")
  print(
      "Running test prompts to demonstrate inline_data and file_data"
      " features..."
  )
  print(f"API Variant: {api_variant}")
  print(
      "Use 'adk run contributing/samples/static_non_text_content' for"
      " interactive mode.\n"
  )

  # Create session
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  # Test prompts that specifically exercise the static content features
  test_prompts = [
      "What reference materials do you have access to?",
      "Can you describe the sample chart that was provided to you?",
      "What does the contributing guide document say about best practices?",
      (
          "How do the inline image and file references in your instructions "
          "help you answer questions?"
      ),
  ]

  # Add Vertex AI specific prompt to test GCS file reference
  if api_variant == GoogleLLMVariant.VERTEX_AI:
    test_prompts.append(
        "What is the Gemma research paper about and what are its key "
        "contributions?"
    )

  for i, prompt in enumerate(test_prompts, 1):
    print(f"Test {i}/{len(test_prompts)}: {prompt}")
    print("-" * 60)

    try:
      response = await call_agent_async(runner, USER_ID, session.id, prompt)
      print(f"Response: {response}")
    except (ConnectionError, TimeoutError, ValueError) as e:
      print(f"Error: {e}")

    print(f"\n{'=' * 60}\n")


async def single_prompt_mode(runner, prompt: str):
  """Run the agent with a single prompt."""
  print("=== Static Non-Text Content Demo Agent - Single Prompt Mode ===")
  print(f"Prompt: {prompt}")
  print("-" * 50)

  # Create session
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  response = await call_agent_async(runner, USER_ID, session.id, prompt)
  print(f"Agent Response:\n{response}")


async def main():
  args = process_arguments()

  if args.debug:
    logs.setup_adk_logger(level=logging.DEBUG)
    print("Debug logging enabled. You'll see internal processing details.\n")

  print("Initializing Static Non-Text Content Demo Agent...")
  print(f"Agent: {agent.root_agent.name}")
  print(f"Model: {agent.root_agent.model}")
  print(f"Description: {agent.root_agent.description}")

  # Show information about static instruction content
  if agent.root_agent.static_instruction:
    static_parts = agent.root_agent.static_instruction.parts
    text_parts = sum(1 for part in static_parts if part.text)
    image_parts = sum(1 for part in static_parts if part.inline_data)
    file_parts = sum(1 for part in static_parts if part.file_data)

    print("Static instruction contains:")
    print(f"  - {text_parts} text parts")
    print(f"  - {image_parts} inline image(s)")
    print(f"  - {file_parts} file reference(s)")

  print("-" * 50)

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  if args.prompt:
    await single_prompt_mode(runner, args.prompt)
  else:
    await run_default_test_prompts(runner)


if __name__ == "__main__":
  start_time = time.time()
  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\nExiting...")
  except Exception as e:
    print(f"Unexpected error: {e}", file=sys.stderr)
    sys.exit(1)
  finally:
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")
