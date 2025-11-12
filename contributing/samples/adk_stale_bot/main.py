import asyncio
from adk_stale_bot.agent import root_agent
from google.adk.runners import InMemoryRunner
import time
from google.genai import types

APP_NAME = "adk_stale_agent_app"
USER_ID = "adk_stale_agent_user"

async def main():
    """Initializes and runs the stale issue agent."""
    print(f"--- Starting Stale Agent at {time.ctime()} ---")
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    session = await runner.session_service.create_session(user_id=USER_ID, app_name=APP_NAME)

    prompt_text = "Find and process all open issues and PRs to manage staleness according to your rules."
    print(f"Agent Prompt: {prompt_text}\n")

    # --- THE FIX IS ON THIS LINE ---
    # Create a properly structured message object.
    prompt_message = types.Content(role="user", parts=[types.Part(text=prompt_text)])

    # The agent will now reason and use its tools to fulfill the request
    async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=prompt_message):
        if event.content and event.content.parts and hasattr(event.content.parts[0], "text"):
            # Print the agent's "thoughts" and actions for logging purposes
            print(f"** {event.author} (ADK): {event.content.parts[0].text}")

    print(f"\n--- Stale Agent Finished at {time.ctime()} ---")

if __name__ == "__main__":
    asyncio.run(main())