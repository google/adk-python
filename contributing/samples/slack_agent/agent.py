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

from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.slack import SlackRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from slack_bolt.app.async_app import AsyncApp


async def main():
  # 1. Setup your ADK agent
  agent = LlmAgent(
      name="slack_agent",
      instruction=(
          "You are a helpful Slack bot powered by Google ADK. Be concise and"
          " friendly."
      ),
  )

  # 2. Setup ADK Runner
  runner = Runner(
      agent=agent,
      app_name="slack_app",
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  # 3. Setup Slack Bolt App
  # Ensure you have SLACK_BOT_TOKEN and SLACK_APP_TOKEN in your environment
  slack_app = AsyncApp(token=os.environ.get("SLACK_BOT_TOKEN"))

  # 4. Initialize SlackRunner
  slack_runner = SlackRunner(runner=runner, slack_app=slack_app)

  # 5. Start the Slack bot (using Socket Mode)
  app_token = os.environ.get("SLACK_APP_TOKEN")
  if not app_token:
    print("SLACK_APP_TOKEN not found. Please set it for Socket Mode.")
    return

  print("Starting Slack bot...")
  await slack_runner.start(app_token=app_token)


if __name__ == "__main__":
  asyncio.run(main())
