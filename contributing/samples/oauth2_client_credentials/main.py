"""WeatherAssistant Agent main script.

This script demonstrates OAuth2 client credentials flow using a practical
weather assistant agent with AuthenticatedFunctionTool.
"""

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

import argparse
import asyncio
import logging
import sys
import time

import agent
from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

APP_NAME = "weather_assistant_app"
USER_ID = "weather_user"

logs.setup_adk_logger(level=logging.INFO)


def process_arguments():
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description=(
          "WeatherAssistant Agent - demonstrates OAuth2 client credentials"
          " authentication transparently through weather queries."
      ),
      epilog=(
          "Example usage:\n\tpython main.py"
          ' "What\'s the weather in Tokyo?"\n\n'
          "For interactive usage, use ADK commands:\n"
          "\tadk run .\n"
          "\tadk web .\n"
      ),
      formatter_class=argparse.RawTextHelpFormatter,
  )

  parser.add_argument(
      "message",
      type=str,
      help=(
          "Ask the weather assistant a question or request weather information."
      ),
  )

  return parser.parse_args()


async def process_message(runner, session_id, message):
  """Process a single message with the weather assistant."""
  print(f"🌤️  Weather Assistant: ")

  response = await call_agent_async(runner, USER_ID, session_id, message)
  print(f"{response}\n")


async def call_agent_async(runner, user_id, session_id, prompt):
  """Helper function to call agent asynchronously."""
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

  return final_response_text


async def main():
  """Main function."""
  # Load environment variables from .env file
  load_dotenv()

  args = process_arguments()

  print("🌤️  WeatherAssistant Agent")
  print("=" * 40)
  print("Ask me about weather in any city around the world!")
  print("(OAuth2 client credentials authentication happens transparently)\n")

  # Create runner and session
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  try:
    await process_message(runner, session.id, args.message)

  except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    return 1

  return 0


if __name__ == "__main__":
  start_time = time.time()
  print(
      "⏰ Started at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
  )
  print("-" * 50)

  try:
    exit_code = asyncio.run(main())
  except KeyboardInterrupt:
    print("\n⏹️  Interrupted by user")
    exit_code = 1

  end_time = time.time()
  print("-" * 50)
  print(
      "⏰ Finished at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
  )
  print(f"⌛ Total execution time: {end_time - start_time:.2f} seconds")

  sys.exit(exit_code)
