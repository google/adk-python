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

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes
from a2a.server.routes import create_jsonrpc_routes
from a2a.server.routes import create_rest_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentInterface
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT
from a2a.utils.constants import TransportProtocol
from fastapi import FastAPI
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import include_artifacts_in_a2a_event_interceptor
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types


class _MockArtifactService(InMemoryArtifactService):
  """Artifact service that returns mock content for any artifact load."""

  async def load_artifact(self, **kwargs):
    return types.Part(text="artifact content")


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
    # Use a subclassed artifact service so pydantic InvocationContext validation
    # passes and load_artifact returns mock content for integration tests.
    self.artifact_service = _MockArtifactService()

  async def run_async(self, **kwargs):
    async for event in self.run_async_fn(**kwargs):
      yield event


# Build agent card using proto-based API
agent_card = AgentCard(
    name="remote_agent",
    description="A fun fact generator agent",
    capabilities=AgentCapabilities(streaming=True),
    version="0.0.1",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
)
agent_card.supported_interfaces.append(
    AgentInterface(
        url="http://test",
        protocol_binding=TransportProtocol.JSONRPC,
        protocol_version=PROTOCOL_VERSION_CURRENT,
    )
)


def create_server_app(
    run_async_fn=None,
    config: A2aAgentExecutorConfig | None = None,
    task_store=None,
):
  """Creates an A2A FastAPI application with a mocked runner."""
  runner = FakeRunner(run_async_fn)
  # use_legacy=False + force_new_version=True forces the new executor impl
  # which correctly handles streaming via artifact_update events
  executor = A2aAgentExecutor(
      runner=runner, config=config, use_legacy=False, force_new_version=True
  )
  if task_store is None:
    task_store = InMemoryTaskStore()

  handler = DefaultRequestHandler(
      agent_executor=executor,
      task_store=task_store,
      agent_card=agent_card,
  )

  app = FastAPI()
  add_a2a_routes_to_fastapi(
      app,
      agent_card_routes=create_agent_card_routes(agent_card),
      jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"),
      rest_routes=create_rest_routes(handler),
  )
  return app
