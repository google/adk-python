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

"""Demonstrates the run_debug() helper method for simplified agent interaction."""

import asyncio

from google.adk.runners import InMemoryRunner

from . import agent


async def example_minimal():
  """Minimal usage - just 2 lines for debugging."""
  print("------------------------------------")
  print("Example 1: Minimal Debug Usage")
  print("------------------------------------")

  # Create runner
  runner = InMemoryRunner(agent=agent.root_agent)

  # Debug with just 2 lines
  await runner.run_debug("What's the weather in San Francisco?")


async def example_multiple_messages():
  """Debug with multiple messages in sequence."""
  print("\n------------------------------------")
  print("Example 2: Multiple Messages")
  print("------------------------------------")

  runner = InMemoryRunner(agent=agent.root_agent)

  # Pass multiple messages as a list
  await runner.run_debug([
      "Hi there!",
      "What's the weather in Tokyo?",
      "How about New York?",
      "Calculate 15 * 7 + 3",
  ])


async def example_conversation_persistence():
  """Demonstrate conversation persistence during debugging."""
  print("\n------------------------------------")
  print("Example 3: Session Persistence")
  print("------------------------------------")

  runner = InMemoryRunner(agent=agent.root_agent)

  # First interaction
  await runner.run_debug("Hi, I'm planning a trip to Europe")

  # Second interaction - continues same session
  await runner.run_debug("What's the weather in Paris?")

  # Third interaction - agent remembers context
  await runner.run_debug("And London?")

  # Fourth interaction - referring to previous messages
  await runner.run_debug("Which city had better weather?")


async def example_separate_sessions():
  """Debug with multiple separate sessions."""
  print("\n------------------------------------")
  print("Example 4: Separate Sessions")
  print("------------------------------------")

  runner = InMemoryRunner(agent=agent.root_agent)

  # Alice's session
  print("\n-- Alice's session --")
  await runner.run_debug(
      "What's the weather in San Francisco?",
      user_id="alice",
      session_id="alice_debug",
  )

  # Bob's session (separate)
  print("\n-- Bob's session --")
  await runner.run_debug(
      "Calculate 100 / 5", user_id="bob", session_id="bob_debug"
  )

  # Continue Alice's session
  print("\n-- Back to Alice's session --")
  await runner.run_debug(
      "Should I bring an umbrella?",
      user_id="alice",
      session_id="alice_debug",
  )


async def example_with_tools():
  """Demonstrate tool calls and responses with verbose flag."""
  print("\n------------------------------------")
  print("Example 5: Tool Calls (verbose flag)")
  print("------------------------------------")

  runner = InMemoryRunner(agent=agent.root_agent)

  print("\n-- Default (verbose=False) - Clean output --")
  # Without verbose: Only shows final agent responses
  await runner.run_debug([
      "What's the weather in Tokyo?",
      "Calculate (42 * 3.14) + 10",
  ])

  print("\n-- With verbose=True - Detailed output --")
  # With verbose: Shows tool calls as [Calling tool: ...] and [Tool result: ...]
  await runner.run_debug(
      [
          "What's the weather in Paris?",
          "Calculate 100 / 5",
      ],
      verbose=True,
  )


async def example_capture_events():
  """Capture events for inspection during debugging."""
  print("\n------------------------------------")
  print("Example 6: Capture Events (No Print)")
  print("------------------------------------")

  runner = InMemoryRunner(agent=agent.root_agent)

  # Capture events without printing for inspection
  events = await runner.run_debug(
      ["Get weather for London", "Calculate 42 * 3.14"],
      quiet=True,
  )

  # Inspect the captured events
  print(f"Captured {len(events)} events")
  for i, event in enumerate(events):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          print(f"  Event {i+1}: {event.author} - Text: {len(part.text)} chars")
        elif part.function_call:
          print(
              f"  Event {i+1}: {event.author} - Tool call:"
              f" {part.function_call.name}"
          )
        elif part.function_response:
          print(f"  Event {i+1}: {event.author} - Tool response received")


async def example_with_run_config():
  """Demonstrate using RunConfig for advanced settings."""
  print("\n------------------------------------")
  print("Example 7: Advanced Configuration")
  print("------------------------------------")

  from google.adk.agents.run_config import RunConfig

  runner = InMemoryRunner(agent=agent.root_agent)

  # Custom configuration - RunConfig supports:
  # - support_cfc: Control function calling behavior
  # - response_modalities: Output modalities (for LIVE API)
  # - speech_config: Speech settings (for LIVE API)
  config = RunConfig(
      support_cfc=False,  # Disable controlled function calling
  )

  await runner.run_debug(
      "Explain what tools you have available", run_config=config
  )


async def example_comparison():
  """Show before/after comparison of boilerplate reduction."""
  print("\n------------------------------------")
  print("Example 8: Before vs After Comparison")
  print("------------------------------------")

  print("\nBefore (7-8 lines of boilerplate):")
  print("""
  from google.adk.sessions import InMemorySessionService
  from google.genai import types

  APP_NAME = "default"
  USER_ID = "default"
  session_service = InMemorySessionService()
  runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
  session = await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id="default"
  )
  content = types.Content(role="user", parts=[types.Part.from_text("Hi")])
  async for event in runner.run_async(
      user_id=USER_ID, session_id=session.id, new_message=content
  ):
      if event.content and event.content.parts:
          print(event.content.parts[0].text)
  """)

  print("\nAfter (just 2 lines):")
  print("""
  runner = InMemoryRunner(agent=agent)
  await runner.run_debug("Hi")
  """)

  print("\nThat's a 75% reduction in boilerplate.")


async def main():
  """Run all debug examples."""
  print("ADK run_debug() Helper Method Examples")
  print("=======================================")
  print("Demonstrating all capabilities:\n")
  print("1. Minimal usage (2 lines)")
  print("2. Multiple messages")
  print("3. Session persistence")
  print("4. Separate sessions")
  print("5. Tool calls")
  print("6. Event capture")
  print("7. Advanced configuration")
  print("8. Before/after comparison")

  await example_minimal()
  await example_multiple_messages()
  await example_conversation_persistence()
  await example_separate_sessions()
  await example_with_tools()
  await example_capture_events()
  await example_with_run_config()
  await example_comparison()

  print("\n=======================================")
  print("All examples completed.")
  print("\nHow different part types appear:")
  print("  Text: agent > Hello world (always shown)")
  print("\nWith verbose=True only:")
  print("  Tool call: agent > [Calling tool: calculate({'expression': '2+2'})]")
  print("  Tool result: agent > [Tool result: Result: 4]")
  print("\nNote: When models have code execution enabled (verbose=True):")
  print("  Code exec: agent > [Executing python code...]")
  print("  Code output: agent > [Code output: Result: 42]")
  print("  Inline data: agent > [Inline data: image/png]")
  print("  File ref: agent > [File: gs://bucket/file.pdf]")


if __name__ == "__main__":
  asyncio.run(main())
