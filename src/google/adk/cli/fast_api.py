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

import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
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

from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
from ..runners import Runner
from .adk_web_server import AdkWebServer
from .service_registry import load_services_module
from .utils import envs
from .utils import evals
from .utils.agent_change_handler import AgentChangeEventHandler
from .utils.agent_loader import AgentLoader
from .utils.service_factory import create_artifact_service_from_options
from .utils.service_factory import create_memory_service_from_options
from .utils.service_factory import create_session_service_from_options

logger = logging.getLogger("google_adk." + __name__)

_LAZY_SERVICE_IMPORTS: dict[str, str] = {
    "AgentLoader": ".utils.agent_loader",
    "LocalEvalSetResultsManager": "..evaluation.local_eval_set_results_manager",
    "LocalEvalSetsManager": "..evaluation.local_eval_sets_manager",
}


def __getattr__(name: str):
  """Lazily import defaults so patching in tests keeps working."""
  if name not in _LAZY_SERVICE_IMPORTS:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

  module = importlib.import_module(_LAZY_SERVICE_IMPORTS[name], __package__)
  attr = getattr(module, name)
  globals()[name] = attr
  return attr


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
    url_prefix: Optional[str] = None,
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    reload_agents: bool = False,
    lifespan: Optional[Lifespan[FastAPI]] = None,
    extra_plugins: Optional[list[str]] = None,
    logo_text: Optional[str] = None,
    logo_image_url: Optional[str] = None,
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

  # initialize Agent Loader
  agent_loader = AgentLoader(agents_dir)
  # Load services.py from agents_dir for custom service registration.
  load_services_module(agents_dir)

  # Build the Memory service
  try:
    memory_service = create_memory_service_from_options(
        base_dir=agents_dir,
        memory_service_uri=memory_service_uri,
    )
  except ValueError as exc:
    raise click.ClickException(str(exc)) from exc

  # Build the Session service
  session_service = create_session_service_from_options(
      base_dir=agents_dir,
      session_service_uri=session_service_uri,
      session_db_kwargs=session_db_kwargs,
  )

  # Build the Artifact service
  try:
    artifact_service = create_artifact_service_from_options(
        base_dir=agents_dir,
        artifact_service_uri=artifact_service_uri,
        strict_uri=True,
    )
  except ValueError as exc:
    raise click.ClickException(str(exc)) from exc

  # Build  the Credential service
  credential_service = InMemoryCredentialService()

  adk_web_server = AdkWebServer(
      agent_loader=agent_loader,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
      credential_service=credential_service,
      eval_sets_manager=eval_sets_manager,
      eval_set_results_manager=eval_set_results_manager,
      agents_dir=agents_dir,
      extra_plugins=extra_plugins,
      logo_text=logo_text,
      logo_image_url=logo_image_url,
      url_prefix=url_prefix,
  )

  # Callbacks & other optional args for when constructing the FastAPI instance
  extra_fast_api_args = {}

  # TODO - Remove separate trace_to_cloud logic once otel_to_cloud stops being
  # EXPERIMENTAL.
  if trace_to_cloud and not otel_to_cloud:
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
      otel_to_cloud=otel_to_cloud,
      **extra_fast_api_args,
  )

  @app.post("/builder/save", response_model_exclude_none=True)
  async def builder_build(
      files: list[UploadFile], tmp: Optional[bool] = False
  ) -> bool:
    base_path = Path.cwd() / agents_dir
    for file in files:
      if not file.filename:
        logger.exception("Agent name is missing in the input files")
        return False
      agent_name, filename = file.filename.split("/")
      agent_dir = os.path.join(base_path, agent_name)
      try:
        # File name format: {app_name}/{agent_name}.yaml
        if tmp:
          agent_dir = os.path.join(agent_dir, "tmp/" + agent_name)
          os.makedirs(agent_dir, exist_ok=True)
          file_path = os.path.join(agent_dir, filename)
          with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        else:
          source_dir = os.path.join(agent_dir, "tmp/" + agent_name)
          destination_dir = agent_dir
          for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            destination_item = os.path.join(destination_dir, item)
            if os.path.isdir(source_item):
              shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            # Check if the item is a file
            elif os.path.isfile(source_item):
              shutil.copy2(source_item, destination_item)
      except Exception as e:
        logger.exception("Error in builder_build: %s", e)
        return False

    return True

  @app.post("/builder/app/{app_name}/cancel", response_model_exclude_none=True)
  async def builder_cancel(app_name: str) -> bool:
    base_path = Path.cwd() / agents_dir
    agent_dir = os.path.join(base_path, app_name)
    destination_dir = os.path.join(agent_dir, "tmp/" + app_name)
    source_dir = agent_dir
    source_items = set(os.listdir(source_dir))
    try:
      for item in os.listdir(destination_dir):
        if item in source_items:
          continue
        # If it doesn't exist in the source, delete it from the destination
        item_path = os.path.join(destination_dir, item)
        if os.path.isdir(item_path):
          shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
          os.remove(item_path)

      for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir, item)
        if item == "tmp" and os.path.isdir(source_item):
          continue
        if os.path.isdir(source_item):
          shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
        # Check if the item is a file
        elif os.path.isfile(source_item):
          shutil.copy2(source_item, destination_item)
    except Exception as e:
      logger.exception("Error in builder_build: %s", e)
      return False
    return True

  @app.get(
      "/builder/app/{app_name}",
      response_model_exclude_none=True,
      response_class=PlainTextResponse,
  )
  async def get_agent_builder(
      app_name: str,
      file_path: Optional[str] = None,
      tmp: Optional[bool] = False,
  ):
    base_path = Path.cwd() / agents_dir
    agent_dir = base_path / app_name
    if tmp:
      agent_dir = agent_dir / "tmp"
      agent_dir = agent_dir / app_name
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
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCard
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

    from ..a2a.executor.a2a_agent_executor import A2aAgentExecutor

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
