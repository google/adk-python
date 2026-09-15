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
import os
import sys

import agent
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)

async def main():
    # Check for API credentials to fail gracefully with a clear message
    if 'GOOGLE_API_KEY' not in os.environ and 'GOOGLE_CLOUD_PROJECT' not in os.environ:
        print("Error: Missing required LLM credentials.")
        print("Please set the GOOGLE_API_KEY environment variable before running this sample.")
        sys.exit(1)

    app_name = 'stateless_elicitation_app'
    user_id = 'user_example'
    
    runner = InMemoryRunner(
        agent=agent.root_agent,
        app_name=app_name,
    )
    
    session = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id
    )

    async def run_turn(prompt: str):
        print(f"\n=== User Turn: '{prompt}' ===")
        content = types.Content(
            role='user', 
            parts=[types.Part.from_text(text=prompt)]
        )
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"** Agent Response: {part.text.strip()}")
                    if part.function_call:
                        print(f"** Agent Tool Call: {part.function_call.name}")
                        print(f"   Arguments: {part.function_call.args}")
        
        # Fetch the updated session to inspect the stateless hidden_context
        updated_session = await runner.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        )
        if updated_session.hidden_context:
            print("\n--- Stateless Hidden Context ---")
            print(updated_session.hidden_context)
            print("---------------------------------")

    # Turn 1: Ambiguous query (triggers elicitation)
    await run_turn("Book a hotel in Tokyo")
    
    # Turn 2: Resolving query (completes the flow)
    await run_turn("I am John Doe, and my check-in date is 2026-07-01")

if __name__ == '__main__':
    asyncio.run(main())
