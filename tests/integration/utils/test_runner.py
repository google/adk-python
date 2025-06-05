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
import importlib
from typing import Optional

from google.adk import Agent
from google.adk import Runner
from google.adk.artifacts import BaseArtifactService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.events import Event
from google.adk.sessions import BaseSessionService
from google.adk.sessions import InMemorySessionService
from google.adk.sessions import Session
from google.genai import types


class TestRunner:
  """Agents runner for testing."""
  
  # Prevent pytest from collecting this as a test class
  __test__ = False

  app_name = "test_app"
  user_id = "test_user"

  def __init__(
      self,
      agent: Agent,
      artifact_service: BaseArtifactService = InMemoryArtifactService(),
      session_service: BaseSessionService = InMemorySessionService(),
  ) -> None:
    self.agent = agent
    self.agent_client = Runner(
        app_name=self.app_name,
        agent=agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )
    self.session_service = session_service
    self.current_session_id = None
    self._session_initialized = False

  async def _ensure_session(self) -> str:
    """Ensure a session is created and return the session ID."""
    if not self._session_initialized:
      session = await self.session_service.create_session(
          app_name=self.app_name, user_id=self.user_id
      )
      self.current_session_id = session.id
      self._session_initialized = True
    return self.current_session_id

  async def new_session_async(self, session_id: Optional[str] = None) -> None:
    session = await self.session_service.create_session(
        app_name=self.app_name, user_id=self.user_id, session_id=session_id
    )
    self.current_session_id = session.id
    self._session_initialized = True

  def new_session(self, session_id: Optional[str] = None) -> None:
    """Create a new session (sync version)."""
    return asyncio.get_event_loop().run_until_complete(self.new_session_async(session_id))

  async def run_async(self, prompt: str) -> list[Event]:
    await self._ensure_session()
    current_session = await self.session_service.get_session(
        app_name=self.app_name,
        user_id=self.user_id,
        session_id=self.current_session_id,
    )
    assert current_session is not None

    return list(
        self.agent_client.run(
            user_id=current_session.user_id,
            session_id=current_session.id,
            new_message=types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        )
    )

  def run(self, prompt: str) -> list[Event]:
    """Run the agent with a prompt (sync version)."""
    return asyncio.get_event_loop().run_until_complete(self.run_async(prompt))

  async def get_current_session_async(self) -> Optional[Session]:
    await self._ensure_session()
    return await self.session_service.get_session(
        app_name=self.app_name,
        user_id=self.user_id,
        session_id=self.current_session_id,
    )

  def get_current_session(self) -> Optional[Session]:
    """Get current session (sync version)."""
    return asyncio.get_event_loop().run_until_complete(self.get_current_session_async())

  async def get_events_async(self) -> list[Event]:
    session = await self.get_current_session_async()
    return session.events

  def get_events(self) -> list[Event]:
    """Get events from current session (sync version)."""
    return asyncio.get_event_loop().run_until_complete(self.get_events_async())

  @classmethod
  def from_agent_name(cls, agent_name: str):
    agent_module_path = f"tests.integration.fixture.{agent_name}"
    agent_module = importlib.import_module(agent_module_path)
    agent: Agent = agent_module.agent.root_agent
    return cls(agent)

  async def get_current_agent_name_async(self) -> str:
    session = await self.get_current_session_async()
    return self.agent_client._find_agent_to_run(session, self.agent).name

  def get_current_agent_name(self) -> str:
    """Get current agent name (sync version)."""
    return asyncio.get_event_loop().run_until_complete(self.get_current_agent_name_async())

  # Sync wrapper methods for backward compatibility
  def run_sync(self, prompt: str) -> list[Event]:
    """Synchronous wrapper for run method."""
    return asyncio.get_event_loop().run_until_complete(self.run(prompt))

  def get_events_sync(self) -> list[Event]:
    """Synchronous wrapper for get_events method."""
    return asyncio.get_event_loop().run_until_complete(self.get_events())

  def get_current_agent_name_sync(self) -> str:
    """Synchronous wrapper for get_current_agent_name method."""
    return asyncio.get_event_loop().run_until_complete(self.get_current_agent_name())

  def new_session_sync(self, session_id: Optional[str] = None) -> None:
    """Synchronous wrapper for new_session method."""
    return asyncio.get_event_loop().run_until_complete(self.new_session(session_id))
