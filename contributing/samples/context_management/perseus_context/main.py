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

"""Runner for the Perseus context agent sample.

Demonstrates a multi-turn conversation where the agent uses Perseus-resolved
workspace context to answer questions about the current project.

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types

import agent

load_dotenv(override=True)


async def main() -> None:
  app_name = "perseus_context_app"
  user_id = "user1"

  runner = InMemoryRunner(
      app_name=app_name,
      agent=agent.root_agent,
  )

  async def run_prompt(session: Session, new_message: str) -> Session:
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=new_message)]
    )
    print(f"\n** User: {new_message}")

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content and event.content.parts and event.content.parts[0].text:
        print(f"** {event.author}: {event.content.parts[0].text}")

    from typing import cast
    return cast(
        Session,
        await runner.session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session.id
        ),
    )

  # Create a session and set Perseus directives.
  session = await runner.session_service.create_session(
      app_name=app_name, user_id=user_id,
      state={
          "_perseus_workspace": os.getcwd(),
          "_perseus_directives": "@file AGENTS.md @file README.md",
      },
  )

  print("=== Perseus Context Agent Demo ===\n")

  # Ask questions that require workspace knowledge.
  session = await run_prompt(
      session, "What project is this? What's in the AGENTS.md file?"
  )

  session = await run_prompt(
      session, "What are the key skills available for this project?"
  )

  print("\n=== Demo complete ===")


if __name__ == "__main__":
  asyncio.run(main())
