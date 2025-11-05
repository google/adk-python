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

import importlib
from typing import Optional

from google.adk import Agent
from google.adk import Runner
from google.adk.artifacts.base_artifact_service import BaseArtifactService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types


class TestRunner:
  """Agents runner for testing."""

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
    self.current_session_id = session_service.create_session(
        app_name=self.app_name, user_id=self.user_id
    ).id

  def new_session(self, session_id: Optional[str] = None) -> None:
    self.current_session_id = self.session_service.create_session(
        app_name=self.app_name, user_id=self.user_id, session_id=session_id
    ).id

  def run(self, prompt: str) -> list[Event]:
    current_session = self.session_service.get_session(
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

  def get_current_session(self) -> Optional[Session]:
    return self.session_service.get_session(
        app_name=self.app_name,
        user_id=self.user_id,
        session_id=self.current_session_id,
    )

  def get_events(self) -> list[Event]:
    return self.get_current_session().events

  @classmethod
  def from_agent_name(cls, agent_name: str):
    agent_module_path = f"tests.integration.fixture.{agent_name}"
    agent_module = importlib.import_module(agent_module_path)
    agent: Agent = agent_module.agent.root_agent
    return cls(agent)

  def get_current_agent_name(self) -> str:
    return self.agent_client._find_agent_to_run(
        self.get_current_session(), self.agent
    ).name
