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
from google.adk.integrations.google_chat import GoogleChatRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService


async def main():
  # 1. Setup your ADK agent
  agent = LlmAgent(
      name="google_chat_agent",
      instruction=(
          "You are a helpful Google Chat app powered by Google ADK. Be concise"
          " and friendly."
      ),
  )

  # 2. Setup ADK Runner
  runner = Runner(
      agent=agent,
      app_name="google_chat_app",
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  # 3. Initialize GoogleChatRunner. Replies are posted with Application
  # Default Credentials, which must belong to the Chat app's service account.
  chat_runner = GoogleChatRunner(runner=runner)

  # 4. Serve the Chat app endpoint. CHAT_AUDIENCE is the Authentication
  # Audience configured for the Chat app (see README.md); leaving it unset
  # turns off request verification for local testing.
  print("Starting Google Chat app...")
  await chat_runner.start(
      audience=os.environ.get("CHAT_AUDIENCE"),
      service_account_email=os.environ.get(
          "CHAT_SERVICE_ACCOUNT_EMAIL", "chat@system.gserviceaccount.com"
      ),
      host=os.environ.get("HOST", "localhost"),
      port=int(os.environ.get("PORT", "8080")),
  )


if __name__ == "__main__":
  asyncio.run(main())
