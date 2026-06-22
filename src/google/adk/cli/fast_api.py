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
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Optional

import click
from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import Request
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from fastapi.responses import StreamingResponse
from opentelemetry import context
from opentelemetry import trace
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.types import Lifespan
from watchdog.observers import Observer

from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..runners import Runner
from ..telemetry._agent_engine import get_propagated_context
from ..telemetry._agent_engine import TopSpanProcessor
from .api_server import ApiServer
from .cli_deploy import _AGENT_ENGINE_CLASS_METHODS
from .dev_server import DevServer
from .service_registry import load_services_module
from .utils import envs
from .utils.agent_change_handler import AgentChangeEventHandler
from .utils.agent_loader import is_single_agent_directory
from .utils.base_agent_loader import BaseAgentLoader
from .utils.service_factory import _create_task_store_from_options
from .utils.service_factory import create_artifact_service_from_options
from .utils.service_factory import create_memory_service_from_options
from .utils.service_factory import create_session_service_from_options

_ALLOWED_AGENT_ENGINE_CLASS_METHODS = frozenset(
    method["name"] for method in _AGENT_ENGINE_CLASS_METHODS
)


class _QueryRequest(BaseModel):
  input: dict[str, Any] | None = None
  class_method: str | None = None


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


def _register_builder_endpoints(app: FastAPI, web: bool, agents_dir: str):
  """Registers builder endpoints if web is enabled and multipart is installed."""
  if not web:
    return
  try:
    import multipart  # noqa: F401
  except ImportError:
    logger.warning(
        "python-multipart not installed. Builder UI endpoints will not be"
        " available."
    )
    return

  import shutil

  import yaml

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

  _ALLOWED_EXTENSIONS = frozenset({".yaml", ".yml"})

  _BLOCKED_YAML_KEYS = frozenset({"args"})

  def _check_yaml_for_blocked_keys(content: bytes, filename: str) -> None:
    try:
      docs = list(yaml.safe_load_all(content))
    except yaml.YAMLError as exc:
      raise ValueError(f"Invalid YAML in {filename!r}: {exc}") from exc

    def _walk(node: Any) -> None:
      if isinstance(node, dict):
        for key, value in node.items():
          if key in _BLOCKED_YAML_KEYS:
            raise ValueError(
                f"Blocked key {key!r} found in {filename!r}. "
                f"The '{key}' field is not allowed in builder uploads "
                "because it can execute arbitrary code."
            )
          _walk(value)
      elif isinstance(node, list):
        for item in node:
          _walk(item)

    for doc in docs:
      _walk(doc)

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
    ext = os.path.splitext(rel_path)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
      raise ValueError(
          f"File type not allowed: {rel_path!r}"
          f" (allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))})"
      )
    return app_name, rel_path

  def _parse_file_path(file_path: str) -> str:
    file_path = _normalize_relative_path(file_path)
    if not file_path:
      raise ValueError("file_path is missing.")
    if file_path.startswith("/"):
      raise ValueError(f"Absolute file_path rejected: {file_path!r}")
    if _has_parent_reference(file_path):
      raise ValueError(f"Path traversal rejected: {file_path!r}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
      raise ValueError(
          f"File type not allowed: {file_path!r}"
          f" (allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))})"
      )
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
      files: list[UploadFile] = File(...), tmp: Optional[bool] = False
  ) -> bool:
    try:
      app_names: set[str] = set()
      uploads: list[tuple[str, bytes]] = []
      for file in files:
        app_name, rel_path = _parse_upload_filename(file.filename)
        app_names.add(app_name)
        content = await file.read()
        uploads.append((rel_path, content))

      if len(app_names) != 1:
        logger.error(
            "Exactly one app name is required, found: %s",
            sorted(app_names),
        )
        return False

      app_name = next(iter(app_names))

      for rel_path, content in uploads:
        _check_yaml_for_blocked_keys(content, f"{app_name}/{rel_path}")

      if tmp:
        app_root = _get_app_root(app_name)
        tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
        tmp_agent_root.mkdir(parents=True, exist_ok=True)

        for rel_path, content in uploads:
          destination_path = _resolve_under_dir(tmp_agent_root, rel_path)
          destination_path.parent.mkdir(parents=True, exist_ok=True)
          destination_path.write_bytes(content)

        return True

      app_root = _get_app_root(app_name)
      app_root.mkdir(parents=True, exist_ok=True)

      tmp_agent_root = _get_tmp_agent_root(app_root, app_name)
      if tmp_agent_root.is_dir():
        copy_dir_contents(tmp_agent_root, app_root)

      for rel_path, content in uploads:
        destination_path = _resolve_under_dir(app_root, rel_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(content)

      return cleanup_tmp(app_name)
    except ValueError as exc:
      logger.exception("Error in builder_build: %s", exc)
      raise HTTPException(status_code=400, detail=str(exc))
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


def get_fast_api_app(
    *,
    agents_dir: str,
    agent_loader: BaseAgentLoader | None = None,
    session_service_uri: str | None = None,
    session_db_kwargs: Mapping[str, Any] | None = None,
    artifact_service_uri: str | None = None,
    memory_service_uri: str | None = None,
    use_local_storage: bool = True,
    eval_storage_uri: str | None = None,
    allow_origins: list[str] | None = None,
    web: bool,
    a2a: bool = False,
    task_store_uri: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    url_prefix: str | None = None,
    trace_to_cloud: bool = False,
    otel_to_cloud: bool = False,
    reload_agents: bool = False,
    lifespan: Lifespan[FastAPI] | None = None,
    extra_plugins: list[str] | None = None,
    logo_text: str | None = None,
    logo_image_url: str | None = None,
    auto_create_session: bool = False,
    trigger_sources: list[Literal["pubsub", "eventarc"]] | None = None,
    default_llm_model: str | None = None,
    gemini_enterprise_app_name: str | None = None,
    express_mode: bool = False,
) -> FastAPI:
  """Constructs and returns a FastAPI application for serving ADK agents.

  This function orchestrates the initialization of core ADK services (Session,
  Artifact, Memory, and Credential) based on the provided configuration,
  configures the ADK Web Server, and optionally enables advanced features
  like Agent-to-Agent (A2A) protocol support and cloud telemetry.

  Args:
    agents_dir: The root directory containing agent definitions. This path is
      used to discover agents, load custom service registrations (via
      services.py/yaml), and as a base for local storage.
    agent_loader: An optional custom loader for retrieving agent instances. If
      not provided, a default AgentLoader targeting agents_dir is used.
    session_service_uri: A URI defining the backend for session persistence.
      Supports schemes like 'memory://', 'sqlite://', 'postgresql://',
      'mysql://', or 'agentengine://'. Defaults to per-agent local SQLite
      storage if None.
    session_db_kwargs: Optional keyword arguments for custom session service
      initialization. These are passed to the service factory along with the
      URI.
    artifact_service_uri: URI for the artifact service. Uses local artifact
      service if None.
    memory_service_uri: URI for the memory service. Uses local memory service if
      None.
    use_local_storage: Whether to use local storage for session and artifacts.
    eval_storage_uri: URI for evaluation storage. If provided, uses GCS
      managers.
    allow_origins: List of allowed origins for CORS.
    web: Whether to enable the web UI and serve its assets.
    a2a: Whether to enable Agent-to-Agent (A2A) protocol support.
    task_store_uri: URI for the A2A task store. Uses in-memory task store if
      None. Only used when ``a2a=True``.
    host: Host address for the server (defaults to 127.0.0.1).
    port: Port number for the server (defaults to 8000).
    url_prefix: Optional prefix for all URL routes.
    trace_to_cloud: Whether to export traces to Google Cloud Trace.
    otel_to_cloud: Whether to export OpenTelemetry data to Google Cloud.
    reload_agents: Whether to watch for file changes and reload agents.
    lifespan: Optional FastAPI lifespan context manager.
    extra_plugins: List of extra plugin names to load.
    logo_text: Text to display in the web UI logo area.
    logo_image_url: URL for an image to display in the web UI logo area.
    auto_create_session: Whether to automatically create a session when not
      found.
    trigger_sources: List of trigger sources to enable (e.g. ["pubsub",
      "eventarc"]). When set, registers /trigger/* endpoints for batch and
      event-driven agent invocations. None disables all trigger endpoints.
    default_llm_model: Default LLM model to use for the agent.
    gemini_enterprise_app_name: The Gemini Enterprise app name to use for the
      agent.
    express_mode: Whether to enable express mode.

  Returns:
    The configured FastAPI application instance.
  """

  # Detect single agent mode
  agents_path = Path(agents_dir).resolve()
  is_single_agent = is_single_agent_directory(agents_path)

  original_agents_dir = agents_dir
  single_agent_name = None
  if is_single_agent:
    single_agent_name = agents_path.name
    agents_dir = str(agents_path.parent)

  # Set up eval managers.
  if eval_storage_uri:
    from .utils import evals

    gcs_eval_managers = evals.create_gcs_eval_managers_from_uri(
        eval_storage_uri
    )
    eval_sets_manager = gcs_eval_managers.eval_sets_manager
    eval_set_results_manager = gcs_eval_managers.eval_set_results_manager
  else:
    this_module = sys.modules[__name__]
    eval_sets_manager = this_module.LocalEvalSetsManager(agents_dir=agents_dir)
    eval_set_results_manager = this_module.LocalEvalSetResultsManager(
        agents_dir=agents_dir
    )

  # initialize Agent Loader if not passed as argument
  this_module = sys.modules[__name__]
  if agent_loader is None:
    agent_loader = this_module.AgentLoader(original_agents_dir)
  elif is_single_agent and isinstance(agent_loader, this_module.AgentLoader):
    agent_loader._set_single_agent_mode(single_agent_name, agents_dir)

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

  # Instantiate the appropriate server class based on web option
  # If web=True, use DevServer (includes all endpoints: production + dev)
  # If web=False, use ApiServer (production-safe endpoints only)
  ServerClass = DevServer if web else ApiServer

  adk_web_server = ServerClass(
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
      trigger_sources=trigger_sources,
      default_llm_model=default_llm_model,
  )

  # In single agent mode, use that agent as the default app.
  if is_single_agent:
    adk_web_server.default_app_name = single_agent_name

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

    def setup_observer(observer: Observer, adk_web_server: ApiServer):
      agent_change_handler = AgentChangeEventHandler(
          agent_loader=agent_loader,
          runners_to_clean=adk_web_server.runners_to_clean,
          current_app_name_ref=adk_web_server.current_app_name_ref,
      )
      observer.schedule(agent_change_handler, agents_dir, recursive=True)
      observer.start()

    def tear_down_observer(observer: Observer, _: ApiServer):
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

  # Create the task store early so its engine can be disposed via the
  # lifespan, preventing connection pool leaks on shutdown.
  a2a_task_store = None
  if a2a:
    base_path = Path.cwd() / agents_dir
    if base_path.exists() and base_path.is_dir():
      a2a_task_store = _create_task_store_from_options(
          task_store_uri=task_store_uri,
      )

  if a2a_task_store is not None and hasattr(a2a_task_store, "engine"):
    outer_lifespan = lifespan

    @asynccontextmanager
    async def _a2a_lifespan(app_instance: FastAPI):
      try:
        if outer_lifespan:
          async with outer_lifespan(app_instance) as ctx:
            yield ctx
        else:
          yield
      finally:
        logger.info("Disposing A2A task store engine")
        await a2a_task_store.engine.dispose()

    lifespan = _a2a_lifespan

  app = adk_web_server.get_fast_api_app(
      lifespan=lifespan,
      allow_origins=allow_origins,
      otel_to_cloud=otel_to_cloud,
      **extra_fast_api_args,
  )

  # --- Builder endpoints (agent editor UI) ---
  _register_builder_endpoints(app, web, agents_dir)

  if a2a and a2a_task_store is not None:
    from a2a.server.apps import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryPushNotificationConfigStore
    from a2a.types import AgentCard
    from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

    from ..a2a.executor.a2a_agent_executor import A2aAgentExecutor

    # locate all a2a agent apps in the agents directory
    base_path = Path.cwd() / agents_dir
    # the root agents directory should be an existing folder
    if base_path.exists() and base_path.is_dir():

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

          push_config_store = InMemoryPushNotificationConfigStore()

          request_handler = DefaultRequestHandler(
              agent_executor=agent_executor,
              task_store=a2a_task_store,
              push_config_store=push_config_store,
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

  if gemini_enterprise_app_name:
    if gemini_enterprise_app_name not in agent_loader.list_agents():
      raise ValueError(
          f"App {gemini_enterprise_app_name} not found in dir: {agents_dir}"
      )

    import inspect
    import json

    from google.adk.agents import Agent
    import google.auth
    from pydantic import ValidationError as _ValidationError
    from vertexai import agent_engines

    # The tmp agent will be replaced by the adk server's runner and services.
    # It is specified here because it is a required argument to AdkApp.
    adk_app = agent_engines.AdkApp(agent=Agent(name="tmp"))
    if express_mode:
      api_key = os.environ.get("GOOGLE_API_KEY", None)
      if not api_key:
        raise ValueError(
            "No GOOGLE_API_KEY found in environment variables for express mode."
        )
      adk_app._tmpl_attrs["project"] = None
      adk_app._tmpl_attrs["location"] = None
      adk_app._tmpl_attrs["express_mode_api_key"] = api_key
    else:
      _, project_id = google.auth.default()
      location = os.environ.get(
          "GOOGLE_CLOUD_AGENT_ENGINE_LOCATION",
          os.environ.get("GOOGLE_CLOUD_LOCATION", None),
      )
      if not project_id or not location:
        raise ValueError(
            "No GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION found in"
            " environment variables."
        )
      adk_app._tmpl_attrs["project"] = project_id
      adk_app._tmpl_attrs["location"] = location
      adk_app._tmpl_attrs["express_mode_api_key"] = None
    adk_app._tmpl_attrs["runner"] = None
    adk_app._tmpl_attrs["app_name"] = gemini_enterprise_app_name
    adk_app._tmpl_attrs["session_service"] = session_service
    adk_app._tmpl_attrs["memory_service"] = memory_service
    adk_app._tmpl_attrs["artifact_service"] = artifact_service

    def _encode_chunk_to_json(chunk: Any) -> str | None:
      """Encodes a chunk to a JSON string with a newline."""
      try:
        json_chunk = jsonable_encoder(chunk)
        return f"{json.dumps(json_chunk)}\n"
      except Exception:
        logging.exception("Failed to encode chunk")
        return None

    async def json_generator(output: AsyncIterator[Any]) -> AsyncIterator[str]:
      async for chunk in output:
        encoded_chunk = _encode_chunk_to_json(chunk)
        if encoded_chunk is None:
          break
        yield encoded_chunk

    async def _invoke_callable_or_raise(
        invocation_callable: Callable[..., Any],
        invocation_payload: dict[str, Any],
    ) -> Any:
      if inspect.iscoroutinefunction(invocation_callable):
        return await invocation_callable(**invocation_payload)
      elif inspect.isasyncgenfunction(invocation_callable):
        return invocation_callable(**invocation_payload)
      else:
        return await run_in_threadpool(
            invocation_callable, **invocation_payload
        )

    # Implement a FastAPI middleware to extract and attach OpenTelemetry trace
    # context from a custom Google-Agent-Engine-Traceparent header in incoming
    # requests. This enables distributed tracing.
    tracer_provider = trace.get_tracer_provider()
    if isinstance(tracer_provider, TracerProvider):
      tracer_provider.add_span_processor(TopSpanProcessor())
    else:
      logging.warning(
          "OpenTelemetry tracing is not enabled. Please set the"
          " `OTEL_PYTHON_TRACER_PROVIDER` environment variable to enable"
          " tracing."
      )

    @app.middleware("http")
    async def context_propagation(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
      ctx = get_propagated_context(request)
      token = context.attach(ctx)
      try:
        response = await call_next(request)
        return response
      finally:
        context.detach(token)

    @app.post(
        "/api/reasoning_engine",
        response_model_exclude_none=True,
        response_class=JSONResponse,
    )
    async def query(request: Request):
      try:
        body = await request.json()
      except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
      try:
        parsed = _QueryRequest.model_validate(body)
      except _ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors())
      if not adk_app._tmpl_attrs.get("runner"):
        adk_app._tmpl_attrs["runner"] = await adk_web_server.get_runner_async(
            app_name=gemini_enterprise_app_name
        )
      if parsed.class_method is None:
        raise HTTPException(
            status_code=400, detail="class_method cannot be None"
        )
      if parsed.class_method not in _ALLOWED_AGENT_ENGINE_CLASS_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"class_method {parsed.class_method} is not allowed",
        )
      method = getattr(adk_app, parsed.class_method)
      output = await _invoke_callable_or_raise(method, parsed.input or {})

      try:
        json_serialized_content = jsonable_encoder({"output": output})
      except ValueError as encoding_error:
        logging.exception(
            "FastAPI could not JSON-encode the response from invocation method"
            " %s. Error: %s. Invocation method's original response: %r",
            parsed.class_method,
            encoding_error,
            output,
        )
        raise
      return JSONResponse(content=json_serialized_content)

    @app.post(
        "/api/stream_reasoning_engine",
        response_model_exclude_none=True,
        response_class=StreamingResponse,
    )
    async def stream_query(request: Request):
      try:
        body = await request.json()
      except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
      try:
        parsed = _QueryRequest.model_validate(body)
      except _ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors())
      if not adk_app._tmpl_attrs.get("runner"):
        adk_app._tmpl_attrs["runner"] = await adk_web_server.get_runner_async(
            app_name=gemini_enterprise_app_name
        )
      if parsed.class_method is None:
        raise HTTPException(
            status_code=400, detail="class_method cannot be None"
        )
      if parsed.class_method not in _ALLOWED_AGENT_ENGINE_CLASS_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"class_method {parsed.class_method} is not allowed",
        )
      method = getattr(adk_app, parsed.class_method)
      output = await _invoke_callable_or_raise(method, parsed.input or {})

      if inspect.isgenerator(output):

        async def _aiter_from_iter(iterator):
          while True:
            try:
              chunk = await run_in_threadpool(next, iterator)
              yield chunk
            except StopIteration:
              break

        content_iter = _aiter_from_iter(output)
      else:
        content_iter = output

      return StreamingResponse(
          content=json_generator(content_iter),
          media_type="application/json",
      )

  return app
