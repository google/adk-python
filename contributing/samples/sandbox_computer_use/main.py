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

"""Main script to run the sandbox computer use agent.

This script demonstrates how to run the sandbox computer use agent
programmatically using the InMemoryRunner.

Prerequisites:
  1. Set environment variables:
     - GOOGLE_CLOUD_PROJECT: Your GCP project ID
     - VMAAS_SERVICE_ACCOUNT: Your service account email with
       roles/iam.serviceAccountTokenCreator permission

Usage:
  cd contributing/samples
  python -m sandbox_computer_use.main
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types

# Import the agent module
from . import agent

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def run_prompt(
    runner: InMemoryRunner,
    session: Session,
    user_id: str,
    message: str,
) -> None:
  """Run a single prompt and print the response.

  Args:
    runner: The agent runner.
    session: The session to use.
    user_id: The user ID.
    message: The user message.
  """
  content = types.Content(
      role="user", parts=[types.Part.from_text(text=message)]
  )
  print(f"\n** User says: {message}")
  print("-" * 40)

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session.id,
      new_message=content,
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"** {event.author}: {part.text}")
        elif hasattr(part, "inline_data") and part.inline_data:
          # Screenshot received
          print(f"** {event.author}: [Screenshot received]")


async def main():
  """Main function to run the sandbox computer use agent."""
  # Validate environment
  project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
  service_account = os.environ.get("VMAAS_SERVICE_ACCOUNT")

  if not project_id:
    print("ERROR: GOOGLE_CLOUD_PROJECT environment variable is not set.")
    print("Please set it to your GCP project ID.")
    return

  if not service_account:
    print("ERROR: VMAAS_SERVICE_ACCOUNT environment variable is not set.")
    print(
        "Please set it to your service account email with"
        " roles/iam.serviceAccountTokenCreator permission."
    )
    return

  print("=" * 60)
  print("Sandbox Computer Use Agent Demo")
  print("=" * 60)
  print(f"Project: {project_id}")
  print(f"Service Account: {service_account}")
  print("=" * 60)

  app_name = "sandbox_computer_use_demo"
  user_id = "demo_user"

  # Create runner and session
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  print(f"\nSession created: {session.id}")
  print("\nStarting agent interaction...")

  start_time = time.time()

  # Example interaction: Navigate and describe
  await run_prompt(
      runner,
      session,
      user_id,
      "Navigate to https://www.google.com and tell me what you see.",
  )

  # Example interaction: Search for something
  await run_prompt(
      runner,
      session,
      user_id,
      "Search for 'Vertex AI Agent Engine' and tell me the first result.",
  )

  end_time = time.time()

  print("\n" + "=" * 60)
  print(f"Demo completed in {end_time - start_time:.2f} seconds")
  print("=" * 60)

  # Print session state to show sandbox info
  session = await runner.session_service.get_session(
      app_name=app_name, user_id=user_id, session_id=session.id
  )
  print("\nSession state (sandbox info):")
  for key, value in session.state.items():
    if key.startswith("_vmaas_"):
      # Mask token for security
      if "token" in key.lower():
        print(f"  {key}: [REDACTED]")
      else:
        print(f"  {key}: {value}")


if __name__ == "__main__":
  asyncio.run(main())
