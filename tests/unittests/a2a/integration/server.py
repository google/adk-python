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

try:
  from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
  from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
  from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
except ImportError:
  A2AFastAPIApplication = None
  DefaultRequestHandler = None
  InMemoryTaskStore = None
from google.adk.a2a import _compat
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
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


if _compat.IS_A2A_V1:
  agent_card = _compat.parse_agent_card({
      "name": "remote_agent",
      "description": "A fun fact generator agent",
      "version": "0.0.1",
      "supported_interfaces": [
          {"url": "http://test", "protocol_binding": "JSONRPC"}
      ],
      "default_input_modes": ["text/plain"],
      "default_output_modes": ["text/plain"],
  })
else:
  agent_card = _compat.parse_agent_card({
      "name": "remote_agent",
      "url": "http://test",
      "description": "A fun fact generator agent",
      "capabilities": {"streaming": True},
      "version": "0.0.1",
      "defaultInputModes": ["text/plain"],
      "defaultOutputModes": ["text/plain"],
      "skills": [],
  })


def create_server_app(
    run_async_fn=None,
    config: A2aAgentExecutorConfig | None = None,
    task_store=None,
):
  """Creates an A2A FastAPI application with a mocked runner.

  Args:
    run_async_fn: A generator function that takes **kwargs and yields Event
      objects.
    config: Optional executor configuration.
    task_store: Optional task store instance. Defaults to InMemoryTaskStore.

  Returns:
    A FastAPI application instance.
  """
  runner = FakeRunner(run_async_fn)
  executor = A2aAgentExecutor(runner=runner, config=config)
  if task_store is None:
    task_store = InMemoryTaskStore()
  handler = DefaultRequestHandler(
      agent_executor=executor, task_store=task_store
  )

  app = A2AFastAPIApplication(agent_card=agent_card, http_handler=handler)
  return app.build()


class _FixedContentArtifactService(InMemoryArtifactService):
  """``InMemoryArtifactService`` whose ``load_artifact`` always returns content."""

  async def load_artifact(self, **kwargs):
    return types.Part(text="artifact content")


class FakeRunnerV1(Runner):
  """A Fake Runner for 1.x with a real in-memory artifact service."""

  def __init__(self, run_async_fn):
    agent = Mock(spec=BaseAgent)
    agent.name = "FakeAgent"
    super().__init__(
        app_name="FakeApp",
        agent=agent,
        session_service=InMemorySessionService(),
        artifact_service=_FixedContentArtifactService(),
    )
    self.run_async_fn = run_async_fn

  async def run_async(self, **kwargs):
    async for event in self.run_async_fn(**kwargs):
      yield event


def create_server_app_v1(
    run_async_fn=None, config: A2aAgentExecutorConfig | None = None
):
  """Creates a 1.x Starlette app hosting an A2A executor (JSON-RPC routes).

  Mirrors ``create_server_app`` but uses the 1.x route-factory path via
  ``_compat.attach_a2a_routes_to_app`` instead of the 0.3-only
  ``A2AFastAPIApplication``. Returns the Starlette app; callers must drive the
  app's lifespan so the routes are attached before sending requests.
  """
  from a2a.server.tasks import InMemoryTaskStore as TaskStore
  from starlette.applications import Starlette

  runner = FakeRunnerV1(run_async_fn)
  executor = A2aAgentExecutor(runner=runner, config=config)
  app = Starlette()
  _compat.attach_a2a_routes_to_app(
      app,
      agent_card=agent_card,
      agent_executor=executor,
      task_store=TaskStore(),
  )
  return app
