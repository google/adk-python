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
from google.adk.integrations.teams import TeamsRunner
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from microsoft_agents.activity import load_configuration_from_env
from microsoft_agents.authentication.msal import MsalConnectionManager
from microsoft_agents.hosting.aiohttp import CloudAdapter
from microsoft_agents.hosting.core import AgentApplication
from microsoft_agents.hosting.core import MemoryStorage
from microsoft_agents.hosting.core import TurnState


async def main():
  # 1. Setup your ADK agent
  agent = LlmAgent(
      name="teams_agent",
      instruction=(
          "You are a helpful Microsoft Teams bot powered by Google ADK. Be"
          " concise and friendly."
      ),
  )

  # 2. Setup ADK Runner
  runner = Runner(
      agent=agent,
      app_name="teams_app",
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  # 3. Setup the Microsoft 365 Agents SDK app from CONNECTIONS__* environment
  # variables (see README.md)
  sdk_config = load_configuration_from_env(os.environ)
  connection_manager = MsalConnectionManager(**sdk_config)
  agent_app = AgentApplication[TurnState](
      storage=MemoryStorage(),
      adapter=CloudAdapter(connection_manager=connection_manager),
      connection_manager=connection_manager,
      **sdk_config,
  )

  # 4. Initialize TeamsRunner
  teams_runner = TeamsRunner(runner=runner, agent_app=agent_app)

  # 5. Serve the messaging endpoint at http://localhost:3978/api/messages
  print("Starting Teams bot...")
  await teams_runner.start(
      connection_manager.get_default_connection_configuration(),
      host=os.environ.get("HOST", "localhost"),
      port=int(os.environ.get("PORT", "3978")),
  )


if __name__ == "__main__":
  asyncio.run(main())
