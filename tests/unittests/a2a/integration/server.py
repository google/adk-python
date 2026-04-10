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

"""A2A Server for integration tests."""

from unittest.mock import AsyncMock
from unittest.mock import Mock

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentSkill
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import include_artifacts_in_a2a_event_interceptor
from google.adk.agents.base_agent import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


class FakeRunner(Runner):
  """A Fake Runner that delegates run_async to a provided function."""

  def __init__(self, run_async_fn):
    agent = Mock(spec=BaseAgent)
    agent.name = "FakeAgent"

    session_service = InMemorySessionService()
    super().__init__(
        app_name="FakeApp",
        agent=agent,
        session_service=session_service,
    )
    self.run_async_fn = run_async_fn

    mock_artifact_service = Mock()
    mock_artifact_service.load_artifact = AsyncMock(
        return_value=types.Part(text="artifact content")
    )
    self.artifact_service = mock_artifact_service

  async def run_async(self, **kwargs):
    async for event in self.run_async_fn(**kwargs):
      yield event


agent_card = AgentCard(
    name="remote_agent",
    url="http://test",
    description="A fun fact generator agent",
    capabilities=AgentCapabilities(
        streaming=True,
        extensions=[{"uri": "https://a2a-adk/a2a-extension/new-integration"}],
    ),
    version="0.0.1",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    skills=[],
)


def create_server_app(
    run_async_fn=None, config: A2aAgentExecutorConfig | None = None
):
  """Creates an A2A FastAPI application with a mocked runner.

  Args:
    run_async_fn: A generator function that takes **kwargs and yields Event
      objects.
    include_artifacts: Whether to include artifacts in A2A events.

  Returns:
    A FastAPI application instance.
  """
  runner = FakeRunner(run_async_fn)
  executor = A2aAgentExecutor(runner=runner, config=config)
  task_store = InMemoryTaskStore()
  handler = DefaultRequestHandler(
      agent_executor=executor, task_store=task_store
  )

  app = A2AFastAPIApplication(agent_card=agent_card, http_handler=handler)
  return app.build()
