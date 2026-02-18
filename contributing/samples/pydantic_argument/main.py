#!/usr/bin/env python3
"""Simple test script for Pydantic argument agent."""

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

from google.adk.agents.run_config import RunConfig
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic_argument import agent

APP_NAME = "pydantic_test_app"
USER_ID = "test_user"

logs.setup_adk_logger(level=logging.INFO)


async def call_agent_async(runner, user_id, session_id, prompt):
  """Helper function to call the agent and return response."""
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
    if hasattr(event, "content") and event.content:
      final_response_text += event.content

  return final_response_text


async def main():
  print("🚀 Testing Pydantic Argument Feature")
  print("=" * 50)

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=APP_NAME,
  )

  # Create a session
  session = await runner.session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID
  )

  test_prompts = [
      # Test Optional[Pydantic] type handling (UserProfile + Optional[UserPreferences])
      (
          "Create an account for Alice, 25 years old, email: alice@example.com,"
          " with dark theme and Spanish language preferences"
      ),
      (
          "Create a user account for Bob, age 30, no email, "
          "with light theme, French language, and notifications disabled"
      ),
      (
          "Make an account for Charlie, 28 years old, email: charlie@test.com, "
          "but use default preferences"
      ),
      # Test Union type handling (Union[UserProfile, CompanyProfile])
      (
          "Create a profile for Tech Corp company, software industry, "
          "with 150 employees and website techcorp.com"
      ),
      (
          "Create an entity profile for Diana, 32 years old, "
          "email diana@example.com"
      ),
  ]

  for i, prompt in enumerate(test_prompts, 1):
    print(f"\n📝 Test {i}: {prompt}")
    print("-" * 40)

    try:
      response = await call_agent_async(runner, USER_ID, session.id, prompt)
      print(f"✅ Response: {response}")
    except Exception as e:
      print(f"❌ Error: {e}")

  print("\n" + "=" * 50)
  print("✨ Testing complete!")
  print("🔧 Features demonstrated:")
  print("   • JSON dict → Pydantic model conversion (UserProfile)")
  print("   • Optional type handling (Optional[UserPreferences])")
  print("   • Union type handling (Union[UserProfile, CompanyProfile])")
  print("   • Automatic model validation and conversion")
  print("   • No manual isinstance() checks needed!")


if __name__ == "__main__":
  asyncio.run(main())
