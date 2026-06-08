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

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator
from typing import Callable

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import PushNotificationConfigStore
from a2a.server.tasks import TaskStore
from a2a.types import AgentCard
from starlette.applications import Starlette

from ...agents.base_agent import BaseAgent
from ...artifacts.in_memory_artifact_service import InMemoryArtifactService
from ...auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ...memory.in_memory_memory_service import InMemoryMemoryService
from ...runners import Runner
from ...sessions.in_memory_session_service import InMemorySessionService
from ..executor.a2a_agent_executor import A2aAgentExecutor
from ..executor.config import A2aAgentExecutorConfig
from ..experimental import a2a_experimental
from .agent_card_builder import AgentCardBuilder


def _load_agent_card(
    agent_card: AgentCard | str | None,
) -> AgentCard | None:
  """Load agent card from various sources.

  Args:
      agent_card: AgentCard object, path to JSON file, or None

  Returns:
      AgentCard object or None if no agent card provided

  Raises:
      ValueError: If loading agent card from file fails
  """
  if agent_card is None:
    return None

  if isinstance(agent_card, str):
    # Load agent card from file path
    import json
    from pathlib import Path

    try:
      path = Path(agent_card)
      with path.open("r", encoding="utf-8") as f:
        agent_card_data = json.load(f)
        return AgentCard(**agent_card_data)
    except Exception as e:
      raise ValueError(
          f"Failed to load agent card from {agent_card}: {e}"
      ) from e
  else:
    return agent_card


@a2a_experimental
def to_a2a(
    agent: BaseAgent,
    *,
    host: str = "localhost",
    port: int = 8000,
    protocol: str = "http",
    agent_card: AgentCard | str | None = None,
    push_config_store: PushNotificationConfigStore | None = None,
    task_store: TaskStore | None = None,
    runner: Runner | None = None,
    lifespan: Callable[[Starlette], AsyncIterator[None]] | None = None,
    agent_executor_factory: Callable[[Runner], A2aAgentExecutor] | None = None,
) -> Starlette:
  """Convert an ADK agent to a A2A Starlette application.

  Args:
      agent: The ADK agent to convert
      host: The host for the A2A RPC URL (default: "localhost")
      port: The port for the A2A RPC URL (default: 8000)
      protocol: The protocol for the A2A RPC URL (default: "http")
      agent_card: Optional pre-built AgentCard object or path to agent card
        JSON. If not provided, will be built automatically from the agent.
      push_config_store: Optional A2A push notification config store. If not
        provided, an in-memory store will be created so push-notification config
        RPC methods are supported.
      task_store: Optional A2A task store for persisting task state. If not
        provided, an in-memory store will be created.
      runner: Optional pre-built Runner object. If not provided, a default
        runner will be created using in-memory services.
      lifespan: Optional async context manager for Starlette lifespan events.
        Use this to run startup/shutdown logic (e.g. initializing database
        connections or loading resources). The context manager receives the
        Starlette app instance and can set state on ``app.state``.
      agent_executor_factory: Optional factory function that creates an instance
        of A2aAgentExecutor. If not provided, a default A2aAgentExecutor will be
        created.

  Returns:
      A Starlette application that can be run with uvicorn

  Example:
      agent = MyAgent()
      app = to_a2a(agent, host="localhost", port=8000, protocol="http")
      # Then run with: uvicorn module:app --host localhost --port 8000

      # Or with custom agent card:
      app = to_a2a(agent, agent_card=my_custom_agent_card)

      # Or with lifespan:
      @asynccontextmanager
      async def lifespan(app):
          app.state.db = await init_db()
          yield
          await app.state.db.close()

      app = to_a2a(agent, lifespan=lifespan)

      # Or with a persistent task store (the caller owns engine disposal):
      from a2a.server.tasks import DatabaseTaskStore
      from sqlalchemy.ext.asyncio import create_async_engine

      engine = create_async_engine("postgresql+asyncpg://...")
      task_store = DatabaseTaskStore(engine=engine)

      @asynccontextmanager
      async def lifespan(app):
          yield
          await engine.dispose()

      app = to_a2a(agent, task_store=task_store, lifespan=lifespan)
  """
  # Set up ADK logging to ensure logs are visible when using uvicorn directly
  adk_logger = logging.getLogger("google_adk")
  adk_logger.setLevel(logging.INFO)

  def create_runner() -> Runner:
    """Create a runner for the agent."""
    return Runner(
        app_name=agent.name or "adk_agent",
        agent=agent,
        # Use minimal services - in a real implementation these could be configured
        artifact_service=InMemoryArtifactService(),
        session_service=InMemorySessionService(),
        memory_service=InMemoryMemoryService(),
        credential_service=InMemoryCredentialService(),
    )

  # Create A2A components
  if task_store is None:
    task_store = InMemoryTaskStore()

  agent_executor = (
      agent_executor_factory(runner or create_runner())
      if agent_executor_factory is not None
      else A2aAgentExecutor(runner=runner or create_runner)
  )

  if push_config_store is None:
    push_config_store = InMemoryPushNotificationConfigStore()

  request_handler = DefaultRequestHandler(
      agent_executor=agent_executor,
      task_store=task_store,
      push_config_store=push_config_store,
  )

  # Use provided agent card or build one from the agent
  rpc_url = f"{protocol}://{host}:{port}/"
  provided_agent_card = _load_agent_card(agent_card)

  card_builder = AgentCardBuilder(
      agent=agent,
      rpc_url=rpc_url,
  )

  # Build the agent card and configure A2A routes
  async def setup_a2a(app: Starlette):
    # Use provided agent card or build one asynchronously
    if provided_agent_card is not None:
      final_agent_card = provided_agent_card
    else:
      final_agent_card = await card_builder.build()

    # Create the A2A Starlette application
    a2a_app = A2AStarletteApplication(
        agent_card=final_agent_card,
        http_handler=request_handler,
    )

    # Add A2A routes to the main app
    a2a_app.add_routes_to_app(
        app,
    )

  # Compose a lifespan that runs A2A setup and the user's lifespan
  @asynccontextmanager
  async def _combined_lifespan(
      app: Starlette,
  ) -> AsyncIterator[None]:
    await setup_a2a(app)
    if lifespan:
      async with lifespan(app):
        yield
    else:
      yield

  # Create a Starlette app with the composed lifespan
  app = Starlette(lifespan=_combined_lifespan)

  return app
