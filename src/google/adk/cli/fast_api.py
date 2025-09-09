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

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any
from typing import Mapping
from typing import Optional

import click
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import TracerProvider
from starlette.types import Lifespan
from watchdog.observers import Observer

from ..artifacts.gcs_artifact_service import GcsArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..memory.vertex_ai_memory_bank_service import VertexAiMemoryBankService
from ..runners import Runner
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.vertex_ai_session_service import VertexAiSessionService
from ..utils.feature_decorator import working_in_progress
from .adk_web_server import AdkWebServer
from .utils import envs
from .utils import evals
from .utils.agent_change_handler import AgentChangeEventHandler
from .utils.agent_loader import AgentLoader

logger = logging.getLogger("google_adk." + __name__)


def get_fast_api_app(
    *,
    agents_dir: str,
    session_service_uri: Optional[str] = None,
    session_db_kwargs: Optional[Mapping[str, Any]] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    eval_storage_uri: Optional[str] = None,
    allow_origins: Optional[list[str]] = None,
    web: bool,
    a2a: bool = False,
    host: str = "127.0.0.1",
    port: int = 8000,
    trace_to_cloud: bool = False,
    reload_agents: bool = False,
    lifespan: Optional[Lifespan[FastAPI]] = None,
) -> FastAPI:
  # Set up eval managers.
  if eval_storage_uri:
    gcs_eval_managers = evals.create_gcs_eval_managers_from_uri(
        eval_storage_uri
    )
    eval_sets_manager = gcs_eval_managers.eval_sets_manager
    eval_set_results_manager = gcs_eval_managers.eval_set_results_manager
  else:
    eval_sets_manager = LocalEvalSetsManager(agents_dir=agents_dir)
    eval_set_results_manager = LocalEvalSetResultsManager(agents_dir=agents_dir)

  def _parse_agent_engine_resource_name(agent_engine_id_or_resource_name):
    if not agent_engine_id_or_resource_name:
      raise click.ClickException(
          "Agent engine resource name or resource id can not be empty."
      )

    # "projects/my-project/locations/us-central1/reasoningEngines/1234567890",
    if "/" in agent_engine_id_or_resource_name:
      # Validate resource name.
      if len(agent_engine_id_or_resource_name.split("/")) != 6:
        raise click.ClickException(
            "Agent engine resource name is mal-formatted. It should be of"
            " format :"
            " projects/{project_id}/locations/{location}/reasoningEngines/{resource_id}"
        )
      project = agent_engine_id_or_resource_name.split("/")[1]
      location = agent_engine_id_or_resource_name.split("/")[3]
      agent_engine_id = agent_engine_id_or_resource_name.split("/")[-1]
    else:
      envs.load_dotenv_for_agent("", agents_dir)
      project = os.environ["GOOGLE_CLOUD_PROJECT"]
      location = os.environ["GOOGLE_CLOUD_LOCATION"]
      agent_engine_id = agent_engine_id_or_resource_name
    return project, location, agent_engine_id

  # Build the Memory service
  if memory_service_uri:
    if memory_service_uri.startswith("rag://"):
      from ..memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService

      rag_corpus = memory_service_uri.split("://")[1]
      if not rag_corpus:
        raise click.ClickException("Rag corpus can not be empty.")
      envs.load_dotenv_for_agent("", agents_dir)
      memory_service = VertexAiRagMemoryService(
          rag_corpus=f'projects/{os.environ["GOOGLE_CLOUD_PROJECT"]}/locations/{os.environ["GOOGLE_CLOUD_LOCATION"]}/ragCorpora/{rag_corpus}'
      )
    elif memory_service_uri.startswith("agentengine://"):
      agent_engine_id_or_resource_name = memory_service_uri.split("://")[1]
      project, location, agent_engine_id = _parse_agent_engine_resource_name(
          agent_engine_id_or_resource_name
      )
      memory_service = VertexAiMemoryBankService(
          project=project,
          location=location,
          agent_engine_id=agent_engine_id,
      )
    else:
      raise click.ClickException(
          "Unsupported memory service URI: %s" % memory_service_uri
      )
  else:
    memory_service = InMemoryMemoryService()

  # Build the Session service
  if session_service_uri:
    if session_service_uri.startswith("agentengine://"):
      agent_engine_id_or_resource_name = session_service_uri.split("://")[1]
      project, location, agent_engine_id = _parse_agent_engine_resource_name(
          agent_engine_id_or_resource_name
      )
      session_service = VertexAiSessionService(
          project=project,
          location=location,
          agent_engine_id=agent_engine_id,
      )
    else:
      from ..sessions.database_session_service import DatabaseSessionService

      # Database session additional settings
      if session_db_kwargs is None:
        session_db_kwargs = {}
      session_service = DatabaseSessionService(
          db_url=session_service_uri, **session_db_kwargs
      )
  else:
    session_service = InMemorySessionService()

  # Build the Artifact service
  if artifact_service_uri:
    if artifact_service_uri.startswith("gs://"):
      gcs_bucket = artifact_service_uri.split("://")[1]
      artifact_service = GcsArtifactService(bucket_name=gcs_bucket)
    else:
      raise click.ClickException(
          "Unsupported artifact service URI: %s" % artifact_service_uri
      )
  else:
    artifact_service = InMemoryArtifactService()

  # Build  the Credential service
  credential_service = InMemoryCredentialService()

  # initialize Agent Loader
  agent_loader = AgentLoader(agents_dir)

  adk_web_server = AdkWebServer(
      agent_loader=agent_loader,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
      credential_service=credential_service,
      eval_sets_manager=eval_sets_manager,
      eval_set_results_manager=eval_set_results_manager,
      agents_dir=agents_dir,
  )

  # Callbacks & other optional args for when constructing the FastAPI instance
  extra_fast_api_args = {}

  if trace_to_cloud:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    def register_processors(provider: TracerProvider) -> None:
      envs.load_dotenv_for_agent("", agents_dir)
      if project_id := os.environ.get("GOOGLE_CLOUD_PROJECT", None):
        processor = export.BatchSpanProcessor(
            CloudTraceSpanExporter(project_id=project_id)
        )
        provider.add_span_processor(processor)
      else:
        logger.warning(
            "GOOGLE_CLOUD_PROJECT environment variable is not set. Tracing will"
            " not be enabled."
        )

    extra_fast_api_args.update(
        register_processors=register_processors,
    )

  if reload_agents:

    def setup_observer(observer: Observer, adk_web_server: AdkWebServer):
      agent_change_handler = AgentChangeEventHandler(
          agent_loader=agent_loader,
          runners_to_clean=adk_web_server.runners_to_clean,
          current_app_name_ref=adk_web_server.current_app_name_ref,
      )
      observer.schedule(agent_change_handler, agents_dir, recursive=True)
      observer.start()

    def tear_down_observer(observer: Observer, _: AdkWebServer):
      observer.stop()
      observer.join()

    extra_fast_api_args.update(
        setup_observer=setup_observer,
        tear_down_observer=tear_down_observer,
    )

  if web:
    BASE_DIR = Path(__file__).parent.resolve()
    ANGULAR_DIST_PATH = BASE_DIR / "browser"
    extra_fast_api_args.update(
        web_assets_dir=ANGULAR_DIST_PATH,
    )

  app = adk_web_server.get_fast_api_app(
      lifespan=lifespan,
      allow_origins=allow_origins,
      **extra_fast_api_args,
  )

  @working_in_progress("builder_save is not ready for use.")
  @app.post("/builder/save", response_model_exclude_none=True)
  async def builder_build(files: list[UploadFile]) -> bool:
    base_path = Path.cwd() / agents_dir

    for file in files:
      try:
        # File name format: {app_name}/{agent_name}.yaml
        if not file.filename:
          logger.exception("Agent name is missing in the input files")
          return False

        agent_name, filename = file.filename.split("/")

        agent_dir = os.path.join(base_path, agent_name)
        os.makedirs(agent_dir, exist_ok=True)
        file_path = os.path.join(agent_dir, filename)

        with open(file_path, "wb") as buffer:
          shutil.copyfileobj(file.file, buffer)

      except Exception as e:
        logger.exception("Error in builder_build: %s", e)
        return False

    return True

  @working_in_progress("builder_get is not ready for use.")
  @app.get(
      "/builder/app/{app_name}",
      response_model_exclude_none=True,
      response_class=PlainTextResponse,
  )
  async def get_agent_builder(app_name: str, file_path: Optional[str] = None):
    base_path = Path.cwd() / agents_dir
    agent_dir = base_path / app_name
    if not file_path:
      file_name = "root_agent.yaml"
      root_file_path = agent_dir / file_name
      if not root_file_path.is_file():
        return ""
      else:
        return FileResponse(
            path=root_file_path,
            media_type="application/x-yaml",
            filename="${app_name}.yaml",
            headers={"Cache-Control": "no-store"},
        )
    else:
      agent_file_path = agent_dir / file_path
      if not agent_file_path.is_file():
        return ""
      else:
        return FileResponse(
            path=agent_file_path,
            media_type="application/x-yaml",
            filename=file_path,
            headers={"Cache-Control": "no-store"},
        )

  if a2a:
    try:
      from a2a.server.apps import A2AStarletteApplication
      from a2a.server.request_handlers import DefaultRequestHandler
      from a2a.server.tasks import InMemoryTaskStore
      from a2a.types import AgentCard
      from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

      from ..a2a.executor.a2a_agent_executor import A2aAgentExecutor

    except ImportError as e:
      import sys

      if sys.version_info < (3, 10):
        raise ImportError(
            "A2A requires Python 3.10 or above. Please upgrade your Python"
            " version."
        ) from e
      else:
        raise e
    # locate all a2a agent apps in the agents directory
    base_path = Path.cwd() / agents_dir
    # the root agents directory should be an existing folder
    if base_path.exists() and base_path.is_dir():
      a2a_task_store = InMemoryTaskStore()

      def create_a2a_runner_loader(captured_app_name: str):
        """Factory function to create A2A runner with proper closure."""

        async def _get_a2a_runner_async() -> Runner:
          return await adk_web_server.get_runner_async(captured_app_name)

        return _get_a2a_runner_async

      for p in base_path.iterdir():
        # only folders with an agent.json file representing agent card are valid
        # a2a agents
        if (
            p.is_file()
            or p.name.startswith((".", "__pycache__"))
            or not (p / "agent.json").is_file()
        ):
          continue

        app_name = p.name
        logger.info("Setting up A2A agent: %s", app_name)

        try:
          agent_executor = A2aAgentExecutor(
              runner=create_a2a_runner_loader(app_name),
          )

          request_handler = DefaultRequestHandler(
              agent_executor=agent_executor, task_store=a2a_task_store
          )

          with (p / "agent.json").open("r", encoding="utf-8") as f:
            data = json.load(f)
            agent_card = AgentCard(**data)

          a2a_app = A2AStarletteApplication(
              agent_card=agent_card,
              http_handler=request_handler,
          )

          routes = a2a_app.routes(
              rpc_url=f"/a2a/{app_name}",
              agent_card_url=f"/a2a/{app_name}{AGENT_CARD_WELL_KNOWN_PATH}",
          )

          for new_route in routes:
            app.router.routes.append(new_route)

          logger.info("Successfully configured A2A agent: %s", app_name)

        except Exception as e:
          logger.error("Failed to setup A2A agent %s: %s", app_name, e)
          # Continue with other agents even if one fails

  return app
