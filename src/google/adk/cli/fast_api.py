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
    use_local_storage: bool = True,
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
    auto_create_session: bool = False,
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
      use_local_storage=use_local_storage,
  )

  # Build the Artifact service
  try:
    artifact_service = create_artifact_service_from_options(
        base_dir=agents_dir,
        artifact_service_uri=artifact_service_uri,
        strict_uri=True,
        use_local_storage=use_local_storage,
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
      auto_create_session=auto_create_session,
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

  agents_base_path = (Path.cwd() / agents_dir).resolve()

  def _get_app_root(app_name: str) -> Path:
    if app_name in ("", ".", ".."):
      raise ValueError(f"Invalid app name: {app_name!r}")
    if Path(app_name).name != app_name or "\\" in app_name:
      raise ValueError(f"Invalid app name: {app_name!r}")
    app_root = (agents_base_path / app_name).resolve()
    if not app_root.is_relative_to(agents_base_path):
      raise ValueError(f"Invalid app name: {app_name!r}")
    return app_root

  def _normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")

  def _has_parent_reference(path: str) -> bool:
    return any(part == ".." for part in path.split("/"))

  def _parse_upload_filename(filename: Optional[str]) -> tuple[str, str]:
    if not filename:
      raise ValueError("Upload filename is missing.")
    filename = _normalize_relative_path(filename)
    if "/" not in filename:
      raise ValueError(f"Invalid upload filename: {filename!r}")
    app_name, rel_path = filename.split("/", 1)
    if not app_name or not rel_path:
      raise ValueError(f"Invalid upload filename: {filename!r}")
    if rel_path.startswith("/"):
      raise ValueError(f"Absolute upload path rejected: {filename!r}")
    if _has_parent_reference(rel_path):
      raise ValueError(f"Path traversal rejected: {filename!r}")
    return app_name, rel_path

  def _parse_file_path(file_path: str) -> str:
    file_path = _normalize_relative_path(file_path)
    if not file_path:
      raise ValueError("file_path is missing.")
    if file_path.startswith("/"):
      raise ValueError(f"Absolute file_path rejected: {file_path!r}")
    if _has_parent_reference(file_path):
      raise ValueError(f"Path traversal rejected: {file_path!r}")
    return file_path

  def _resolve_under_dir(root_dir: Path, rel_path: str) -> Path:
    file_path = root_dir / rel_path
    resolved_root_dir = root_dir.resolve()
    resolved_file_path = file_path.resolve()
    if not resolved_file_path.is_relative_to(resolved_root_dir):
      raise ValueError(f"Path escapes root_dir: {rel_path!r}")
    return file_path

  def _get_tmp_agent_root(app_root: Path, app_name: str) -> Path:
    tmp_agent_root = app_root / "tmp" / app_name
    resolved_tmp_agent_root = tmp_agent_root.resolve()
    if not resolved_tmp_agent_root.is_relative_to(app_root):
      raise ValueError(f"Invalid tmp path for app: {app_name!r}")
    return tmp_agent_root

  def copy_dir_contents(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for source_path in source_dir.iterdir():
      if source_path.name == "tmp":
        continue

      dest_path = dest_dir / source_path.name
      if source_path.is_dir():
        if dest_path.exists() and dest_path.is_file():
          dest_path.unlink()
        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
      elif source_path.is_file():
        if dest_path.exists() and dest_path.is_dir():
          shutil.rmtree(dest_path)
        shutil.copy2(source_path, dest_path)

  def cleanup_tmp(app_name: str) -> bool:
    try:
      app_root = _get_app_root(app_name)
    except ValueError as exc:
      logger.exception("Error in cleanup_tmp: %s", exc)
      return False

    try:
      tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
    except ValueError as exc:
      logger.exception("Error in cleanup_tmp: %s", exc)
      return False

    try:
      shutil.rmtree(tmp_agent_root)
    except FileNotFoundError:
      pass
    except OSError as exc:
      logger.exception("Error deleting tmp agent root: %s", exc)
      return False

    tmp_dir = app_root / "tmp"
    resolved_tmp_dir = tmp_dir.resolve()
    if not resolved_tmp_dir.is_relative_to(app_root):
      logger.error(
          "Refusing to delete tmp outside app_root: %s", resolved_tmp_dir
      )
      return False

    try:
      tmp_dir.rmdir()
    except OSError:
      pass

    return True

  def ensure_tmp_exists(app_name: str) -> bool:
    try:
      app_root = _get_app_root(app_name)
    except ValueError as exc:
      logger.exception("Error in ensure_tmp_exists: %s", exc)
      return False

    if not app_root.is_dir():
      return False

    try:
      tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
    except ValueError as exc:
      logger.exception("Error in ensure_tmp_exists: %s", exc)
      return False

    if tmp_agent_root.exists():
      return True

    try:
      tmp_agent_root.mkdir(parents=True, exist_ok=True)
      copy_dir_contents(app_root, tmp_agent_root)
    except OSError as exc:
      logger.exception("Error in ensure_tmp_exists: %s", exc)
      return False

    return True

  @app.post("/builder/save", response_model_exclude_none=True)
  async def builder_build(
      files: list[UploadFile], tmp: Optional[bool] = False
  ) -> bool:
    try:
      if tmp:
        app_names = set()
        uploads = []
        for file in files:
          app_name, rel_path = _parse_upload_filename(file.filename)
          app_names.add(app_name)
          uploads.append((rel_path, file))

        if len(app_names) != 1:
          logger.error(
              "Exactly one app name is required, found: %s", sorted(app_names)
          )
          return False

        app_name = next(iter(app_names))
        app_root = _get_app_root(app_name)
        tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
        tmp_agent_root.mkdir(parents=True, exist_ok=True)

        for rel_path, file in uploads:
          destination_path = _resolve_under_dir(tmp_agent_root, rel_path)
          destination_path.parent.mkdir(parents=True, exist_ok=True)
          with destination_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return True

      app_names = set()
      uploads = []
      for file in files:
        app_name, rel_path = _parse_upload_filename(file.filename)
        app_names.add(app_name)
        uploads.append((rel_path, file))

      if len(app_names) != 1:
        logger.error(
            "Exactly one app name is required, found: %s", sorted(app_names)
        )
        return False

      app_name = next(iter(app_names))
      app_root = _get_app_root(app_name)
      app_root.mkdir(parents=True, exist_ok=True)

      tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
      if tmp_agent_root.is_dir():
        copy_dir_contents(tmp_agent_root, app_root)

      for rel_path, file in uploads:
        destination_path = _resolve_under_dir(app_root, rel_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("wb") as buffer:
          shutil.copyfileobj(file.file, buffer)

      return cleanup_tmp(app_name)
    except ValueError as exc:
      logger.exception("Error in builder_build: %s", exc)
      return False
    except OSError as exc:
      logger.exception("Error in builder_build: %s", exc)
      return False

  @app.post("/builder/app/{app_name}/cancel", response_model_exclude_none=True)
  async def builder_cancel(app_name: str) -> bool:
    return cleanup_tmp(app_name)

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
    try:
      app_root = _get_app_root(app_name)
    except ValueError as exc:
      logger.exception("Error in get_agent_builder: %s", exc)
      return ""

    agent_dir = app_root
    if tmp:
      if not ensure_tmp_exists(app_name):
        return ""
      agent_dir = app_root / "tmp" / app_name

    if not file_path:
      rel_path = "root_agent.yaml"
    else:
      try:
        rel_path = _parse_file_path(file_path)
      except ValueError as exc:
        logger.exception("Error in get_agent_builder: %s", exc)
        return ""

    try:
      agent_file_path = _resolve_under_dir(agent_dir, rel_path)
    except ValueError as exc:
      logger.exception("Error in get_agent_builder: %s", exc)
      return ""

    if not agent_file_path.is_file():
      return ""

    return FileResponse(
        path=agent_file_path,
        media_type="application/x-yaml",
        filename=file_path or f"{app_name}.yaml",
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
