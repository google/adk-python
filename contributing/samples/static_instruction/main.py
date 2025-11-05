"""Bingo Digital Pet main script.

This script demonstrates static instruction functionality through a digital pet
that has different moods based on feeding time stored in session state.
"""

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

import asyncio
import logging
import time

from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner

from . import agent

APP_NAME = "bingo_digital_pet_app"
USER_ID = "pet_owner"

logs.setup_adk_logger(level=logging.DEBUG)


async def call_agent_async(
    runner, user_id, session_id, prompt, state_delta=None
):
  """Call the agent asynchronously with state delta support."""
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
      state_delta=state_delta,
      run_config=RunConfig(save_input_blobs_as_artifacts=False),
  ):
    if event.content and event.content.parts:
      if text := "".join(part.text or "" for part in event.content.parts):
        if event.author != "user":
          final_response_text += text

  return final_response_text


async def test_hunger_states(runner):
  """Test different hunger states by simulating feeding times."""
  print("Testing Bingo's different hunger states...\n")

  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  # Simulate different hunger scenarios
  current_time = time.time()
  hunger_scenarios = [
      {
          "description": "Newly created pet (hungry)",
          "last_fed": None,
          "prompt": "Hi Bingo! I just got you as my new digital pet!",
      },
      {
          "description": "Just fed (full and content)",
          "last_fed": current_time,  # Just now
          "prompt": "How are you feeling after that meal, Bingo?",
      },
      {
          "description": "Fed 4 seconds ago (satisfied)",
          "last_fed": current_time - 4,  # 4 seconds ago
          "prompt": "Want to play a game with me?",
      },
      {
          "description": "Fed 10 seconds ago (a little hungry)",
          "last_fed": current_time - 10,  # 10 seconds ago
          "prompt": "How are you doing, buddy?",
      },
      {
          "description": "Fed 20 seconds ago (hungry)",
          "last_fed": current_time - 20,  # 20 seconds ago
          "prompt": "Bingo, what's on your mind?",
      },
      {
          "description": "Fed 30 seconds ago (very hungry)",
          "last_fed": current_time - 30,  # 30 seconds ago
          "prompt": "Hey Bingo, how are you feeling?",
      },
      {
          "description": "Fed 60 seconds ago (starving)",
          "last_fed": current_time - 60,  # 60 seconds ago
          "prompt": "Bingo? Are you okay?",
      },
  ]

  for i, scenario in enumerate(hunger_scenarios, 1):
    print(f"{'='*80}")
    print(f"SCENARIO #{i}: {scenario['description']}")
    print(f"{'='*80}")

    # Set up state delta with the simulated feeding time
    state_delta = {}
    if scenario["last_fed"] is not None:
      state_delta["last_fed_timestamp"] = scenario["last_fed"]

    print(f"You: {scenario['prompt']}")

    response = await call_agent_async(
        runner,
        USER_ID,
        session.id,
        scenario["prompt"],
        state_delta if state_delta else None,
    )
    print(f"Bingo: {response}\n")

    # Short delay between scenarios
    if i < len(hunger_scenarios):
      await asyncio.sleep(1)


async def main():
  """Main function to run Bingo the digital pet."""
  # Load environment variables from .env file
  load_dotenv()

  print("🐕 Initializing Bingo the Digital Pet...")
  print(f"Pet Name: {agent.root_agent.name}")
  print(f"Model: {agent.root_agent.model}")
  print(
      "Static Personality Configured:"
      f" {agent.root_agent.static_instruction is not None}"
  )
  print(
      "Dynamic Mood System Configured:"
      f" {agent.root_agent.instruction is not None}"
  )
  print()

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  # Run hunger state demonstration
  await test_hunger_states(runner)


if __name__ == "__main__":
  start_time = time.time()
  print(
      "🐕 Starting Bingo Digital Pet Session at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(start_time))}"
  )
  print("-" * 80)

  asyncio.run(main())

  print("-" * 80)
  end_time = time.time()
  print(
      "🐕 Pet session ended at"
      f" {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(end_time))}"
  )
  print(f"Total playtime: {end_time - start_time:.2f} seconds")
  print("Thanks for spending time with Bingo! 🐾")
