#!/usr/bin/env python3
"""Simple test script for Rewind Session agent."""

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

import asyncio
import logging

import agent
from google.adk.agents.run_config import RunConfig
from google.adk.cli.utils import logs
from google.adk.events.event import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

APP_NAME = "rewind_test_app"
USER_ID = "test_user"

logs.setup_adk_logger(level=logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)


# ANSI color codes for terminal output
COLOR_RED = "\x1b[31m"
COLOR_BLUE = "\x1b[34m"
COLOR_YELLOW = "\x1b[33m"
COLOR_BOLD = "\x1b[1m"
RESET = "\x1b[0m"


def highlight(text: str) -> str:
  """Adds color highlights to tool responses and agent text."""
  text = str(text)
  return (
      text.replace("'red'", f"'{COLOR_RED}red{RESET}'")
      .replace('"red"', f'"{COLOR_RED}red{RESET}"')
      .replace("'blue'", f"'{COLOR_BLUE}blue{RESET}'")
      .replace('"blue"', f'"{COLOR_BLUE}blue{RESET}"')
      .replace("'version1'", f"'{COLOR_BOLD}{COLOR_YELLOW}version1{RESET}'")
      .replace("'version2'", f"'{COLOR_BOLD}{COLOR_YELLOW}version2{RESET}'")
  )


async def call_agent_async(
    runner: InMemoryRunner, user_id: str, session_id: str, prompt: str
) -> list[Event]:
  """Helper function to call the agent and return events."""
  print(f"\n👤 User: {prompt}")
  content = types.Content(
      role="user", parts=[types.Part.from_text(text=prompt)]
  )
  events = []
  try:
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
        run_config=RunConfig(),
    ):
      events.append(event)
      if event.content and event.author and event.author != "user":
        for part in event.content.parts:
          if part.text:
            print(f"  🤖 Agent: {highlight(part.text)}")
          elif part.function_call:
            print(f"    🛠️ Tool Call: {part.function_call.name}")
          elif part.function_response:
            print(
                "    📦 Tool Response:"
                f" {highlight(part.function_response.response)}"
            )
  except Exception as e:
    print(f"❌ Error during agent call: {e}")
    raise
  return events


async def main():
  """Demonstrates session rewind."""
  print("🚀 Testing Rewind Session Feature")
  print("=" * 50)

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  # Create a session
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )
  print(f"Created session: {session.id}")

  # 1. Initial agent calls to set state and artifact
  print("\n\n===== INITIALIZING STATE AND ARTIFACT =====")
  await call_agent_async(
      runner, USER_ID, session.id, "set state `color` to red"
  )
  await call_agent_async(
      runner, USER_ID, session.id, "save artifact file1 with content version1"
  )

  # 2. Check current state and artifact
  print("\n\n===== STATE BEFORE UPDATE =====")
  await call_agent_async(
      runner, USER_ID, session.id, "what is the value of state `color`?"
  )
  await call_agent_async(runner, USER_ID, session.id, "load artifact file1")

  # 3. Update state and artifact - THIS IS THE POINT WE WILL REWIND BEFORE
  print("\n\n===== UPDATING STATE AND ARTIFACT =====")
  events_update_state = await call_agent_async(
      runner, USER_ID, session.id, "update state key color to blue"
  )
  rewind_invocation_id = events_update_state[0].invocation_id
  print(f"Will rewind before invocation: {rewind_invocation_id}")

  await call_agent_async(
      runner, USER_ID, session.id, "save artifact file1 with content version2"
  )

  # 4. Check state and artifact after update
  print("\n\n===== STATE AFTER UPDATE =====")
  await call_agent_async(
      runner, USER_ID, session.id, "what is the value of state key color?"
  )
  await call_agent_async(runner, USER_ID, session.id, "load artifact file1")

  # 5. Perform rewind
  print(f"\n\n===== REWINDING SESSION to before {rewind_invocation_id} =====")
  await runner.rewind_async(
      user_id=USER_ID,
      session_id=session.id,
      rewind_before_invocation_id=rewind_invocation_id,
  )
  print("✅ Rewind complete.")

  # 6. Check state and artifact after rewind
  print("\n\n===== STATE AFTER REWIND =====")
  await call_agent_async(
      runner, USER_ID, session.id, "what is the value of state `color`?"
  )
  await call_agent_async(runner, USER_ID, session.id, "load artifact file1")

  print("\n" + "=" * 50)
  print("✨ Rewind testing complete!")
  print(
      "🔧 If rewind was successful, color should be 'red' and file1 content"
      " should contain 'version1' in the final check."
  )


if __name__ == "__main__":
  asyncio.run(main())
