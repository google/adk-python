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

"""Main entry point for testing the News Podcast Agent."""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.genai import types

import agent

# Load environment variables
load_dotenv(override=True)


async def main():
    """Test the News Podcast Agent with sample interactions."""
    app_name = "news_podcast_app"
    user_id = "demo_user"

    # Initialize runner
    runner = InMemoryRunner(
        agent=agent.root_agent,
        app_name=app_name,
    )

    # Create session
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )
    print(f"Created session: {session.id}")

    async def run_prompt(session_id: str, message: str):
        """Run a prompt and print the response."""
        print(f"\n📧 User: {message}")
        content = types.Content(
            role="user", parts=[types.Part.from_text(text=message)]
        )

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.content.parts and event.content.parts[0].text:
                print(f"🤖 {event.author}: {event.content.parts[0].text}")

    # Test interactions
    print("\n" + "=" * 60)
    print("News Podcast Agent Demo")
    print("=" * 60)

    # Test 1: Process newsletters
    await run_prompt(
        session.id,
        "Process my newsletters from the last 24 hours and create a podcast",
    )

    # Test 2: Check status
    await run_prompt(session.id, "What newsletters did you find?")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

