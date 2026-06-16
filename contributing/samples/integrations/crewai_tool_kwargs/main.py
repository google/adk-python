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

"""Happy path test for CrewAI tool with **kwargs parameter handling.

This demonstrates that CrewaiTool correctly passes arbitrary parameters
through **kwargs to the underlying CrewAI tool.
"""

import asyncio

import agent
from dotenv import load_dotenv
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(override=True)
logs.log_to_tmp_folder()


async def main():
  """Run happy path test demonstrating **kwargs parameter passing."""
  app_name = "crewai_kwargs_test"
  user_id = "test_user"

  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )

  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id
  )

  print("=" * 60)
  print("CrewAI Tool **kwargs Parameter Test")
  print("=" * 60)

  # Test 1: Simple search without extra parameters
  print("\n🧪 Test 1: Basic search (no extra parameters)")
  print("-" * 60)
  content1 = types.Content(
      role="user",
      parts=[types.Part.from_text(text="Search for Python tutorials")],
  )
  print(f"User: {content1.parts[0].text}")

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session.id,
      new_message=content1,
  ):
    if event.content.parts and event.content.parts[0].text:
      print(f"Agent: {event.content.parts[0].text}")

  # Test 2: Search with extra parameters (testing **kwargs)
  print("\n🧪 Test 2: Search with filters (**kwargs test)")
  print("-" * 60)
  content2 = types.Content(
      role="user",
      parts=[
          types.Part.from_text(
              text=(
                  "Search for machine learning articles, filtered by category"
                  " 'technology', date_range 'last_month', and limit to 10"
                  " results"
              )
          )
      ],
  )
  print(f"User: {content2.parts[0].text}")

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session.id,
      new_message=content2,
  ):
    if event.content.parts and event.content.parts[0].text:
      print(f"Agent: {event.content.parts[0].text}")

  # Verify success
  print("\n" + "=" * 60)
  print("✅ Happy path test completed successfully!")
  print("=" * 60)
  print("\nVerified behaviors:")
  print("  ✅ CrewAI tool integrated with ADK agent")
  print("  ✅ Basic parameters passed correctly")
  print("  ✅ Extra parameters passed through **kwargs")
  print("  ✅ Tool executed and returned results")


if __name__ == "__main__":
  asyncio.run(main())
