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

from adk_stale_agent.agent import root_agent
from adk_stale_agent.settings import OWNER
from adk_stale_agent.settings import REPO
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.genai import types

logs.setup_adk_logger(level=logging.INFO)
logger = logging.getLogger("google_adk." + __name__)

APP_NAME = "adk_stale_agent_app"
USER_ID = "adk_stale_agent_user"


async def main():
  """Initializes and runs the stale issue agent."""
  logger.info("--- Starting Stale Agent Run ---")
  runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
  session = await runner.session_service.create_session(
      user_id=USER_ID, app_name=APP_NAME
  )

  prompt_text = (
      "Find and process all open issues to manage staleness according to your"
      " rules."
  )
  logger.info(f"Initial Agent Prompt: {prompt_text}\n")
  prompt_message = types.Content(
      role="user", parts=[types.Part(text=prompt_text)]
  )

  async for event in runner.run_async(
      user_id=USER_ID, session_id=session.id, new_message=prompt_message
  ):
    if (
        event.content
        and event.content.parts
        and hasattr(event.content.parts[0], "text")
    ):
      # Print the agent's "thoughts" and actions for logging purposes
      logger.debug(f"** {event.author} (ADK): {event.content.parts[0].text}")

  logger.info(f"--- Stale Agent Run Finished---")


if __name__ == "__main__":
  start_time = time.time()
  logger.info(f"Initializing stale agent for repository: {OWNER}/{REPO}")
  logger.info("-" * 80)

  asyncio.run(main())

  logger.info("-" * 80)
  end_time = time.time()
  duration = end_time - start_time
  logger.info(f"Script finished in {duration:.2f} seconds.")
