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

"""Example demonstrating PostgreSQL session persistence with DatabaseSessionService."""

import asyncio
import os

import agent
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)


async def main():
  """Main function demonstrating PostgreSQL session persistence."""
  postgres_url = os.environ.get("POSTGRES_URL")
  if not postgres_url:
    raise ValueError(
        "POSTGRES_URL environment variable not set. "
        "Please create a .env file with"
        " POSTGRES_URL=postgresql+asyncpg://user:password@localhost:5432/adk_sessions"
    )

  app_name = "postgres_session_demo"
  user_id = "demo_user"
  session_id = "persistent-session"

  # Initialize PostgreSQL-backed session service
  session_service = DatabaseSessionService(postgres_url)

  runner = Runner(
      app_name=app_name,
      agent=agent.root_agent,
      session_service=session_service,
  )

  # Try to get existing session or create new one
  session = await session_service.get_session(
      app_name=app_name,
      user_id=user_id,
      session_id=session_id,
  )

  if session:
    print(f"Resuming existing session: {session.id}")
    print(f"Previous events count: {len(session.events)}")
  else:
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    print(f"Created new session: {session.id}")

  async def run_prompt(session: Session, new_message: str):
    """Send a prompt to the agent and print the response."""
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=new_message)]
    )
    print(f"User: {new_message}")
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=content,
    ):
      if event.content and event.content.parts and event.content.parts[0].text:
        print(f"{event.author}: {event.content.parts[0].text}")

  print("------------------------------------")
  await run_prompt(session, "What time is it? Please remember this.")
  print("------------------------------------")
  await run_prompt(session, "What did I just ask you?")
  print("------------------------------------")

  print("\nSession persisted to PostgreSQL. Run again to see event history.")


if __name__ == "__main__":
  asyncio.run(main())
