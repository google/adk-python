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

"""
Api server with all production ADK endpoints.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib
import json
import logging
import os
import re
import sys
import time
import traceback
import typing
from typing import Any
from typing import Callable
from typing import List
from typing import Literal
from typing import Optional

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi import Request
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.genai import types
from opentelemetry import trace
import opentelemetry.sdk.environment_variables as otel_env
from opentelemetry.sdk.trace import export as export_lib
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider
from pydantic import Field
from pydantic import ValidationError
from starlette.types import Lifespan
from typing_extensions import deprecated
from typing_extensions import override
from watchdog.observers import Observer
import yaml

from ..agents.base_agent import BaseAgent
from ..agents.live_request_queue import LiveRequest
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import LlmAgent
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..apps.app import App
from ..artifacts.base_artifact_service import ArtifactVersion
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..errors.already_exists_error import AlreadyExistsError
from ..errors.input_validation_error import InputValidationError
from ..errors.session_not_found_error import SessionNotFoundError
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..plugins.base_plugin import BasePlugin
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.agent_info import AgentInfo
from ..utils.agent_info import get_agents_dict
from ..utils.context_utils import Aclosing
from ..utils.feature_decorator import experimental
from ..version import __version__
from .cli_eval import EVAL_SESSION_ID_PREFIX
from .utils import cleanup
from .utils import common
from .utils import envs
from .utils.base_agent_loader import BaseAgentLoader
from .utils.shared_value import SharedValue

logger = logging.getLogger("google_adk." + __name__)

_REGEX_PREFIX = "regex:"


def _parse_cors_origins(
    allow_origins: list[str],
) -> tuple[list[str], Optional[str]]:
  """Parse allow_origins into literal origins and a combined regex pattern.

  Args:
    allow_origins: List of origin strings. Entries prefixed with 'regex:' are
      treated as regex patterns; all others are treated as literal origins.

  Returns:
    A tuple of (literal_origins, combined_regex) where combined_regex is None
    if no regex patterns were provided, or a single pattern joining all regex
    patterns with '|'.
  """
  literal_origins = []
  regex_patterns = []
  for origin in allow_origins:
    if origin.startswith(_REGEX_PREFIX):
      pattern = origin[len(_REGEX_PREFIX) :]
      if pattern:
        regex_patterns.append(pattern)
    else:
      literal_origins.append(origin)

  combined_regex = "|".join(regex_patterns) if regex_patterns else None
  return literal_origins, combined_regex


def _is_origin_allowed(
    origin: str,
    allowed_literal_origins: list[str],
    allowed_origin_regex: Optional[re.Pattern[str]],
) -> bool:
  """Check whether the given origin matches the allowed origins."""
  if "*" in allowed_literal_origins:
    return True
  if origin in allowed_literal_origins:
    return True
  if allowed_origin_regex is not None:
    return allowed_origin_regex.fullmatch(origin) is not None
  return False


def _normalize_origin_scheme(scheme: str) -> str:
  """Normalize request schemes to the browser Origin scheme space."""
  if scheme == "ws":
    return "http"
  if scheme == "wss":
    return "https"
  return scheme


def _strip_optional_quotes(value: str) -> str:
  """Strip a single pair of wrapping quotes from a header value."""
  if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
    return value[1:-1]
  return value


def _get_scope_header(
    scope: dict[str, Any], header_name: bytes
) -> Optional[str]:
  """Return the first matching header value from an ASGI scope."""
  for candidate_name, candidate_value in scope.get("headers", []):
    if candidate_name == header_name:
      return candidate_value.decode("latin-1").split(",", 1)[0].strip()
  return None


def _get_request_origin(scope: dict[str, Any]) -> Optional[str]:
  """Compute the effective origin for the current HTTP/WebSocket request."""
  forwarded = _get_scope_header(scope, b"forwarded")
  if forwarded is not None:
    proto = None
    host = None
    for element in forwarded.split(",", 1)[0].split(";"):
      if "=" not in element:
        continue
      name, value = element.split("=", 1)
      if name.strip().lower() == "proto":
        proto = _strip_optional_quotes(value.strip())
      elif name.strip().lower() == "host":
        host = _strip_optional_quotes(value.strip())
    if proto is not None and host is not None:
      return f"{_normalize_origin_scheme(proto)}://{host}"

  host = _get_scope_header(scope, b"x-forwarded-host")
  if host is None:
    host = _get_scope_header(scope, b"host")
  if host is None:
    return None

  proto = _get_scope_header(scope, b"x-forwarded-proto")
  if proto is None:
    proto = scope.get("scheme", "http")
  return f"{_normalize_origin_scheme(proto)}://{host}"


def _is_request_origin_allowed(
    origin: str,
    scope: dict[str, Any],
    allowed_literal_origins: list[str],
    allowed_origin_regex: Optional[re.Pattern[str]],
    has_configured_allowed_origins: bool,
) -> bool:
  """Validate an Origin header against explicit config or same-origin."""
  if has_configured_allowed_origins and _is_origin_allowed(
      origin, allowed_literal_origins, allowed_origin_regex
  ):
    return True

  request_origin = _get_request_origin(scope)
  if request_origin is None:
    return False
  return origin == request_origin


_SAFE_HTTP_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class _OriginCheckMiddleware:
  """ASGI middleware that blocks cross-origin state-changing requests."""

  def __init__(
      self,
      app: Any,
      has_configured_allowed_origins: bool,
      allowed_origins: list[str],
      allowed_origin_regex: Optional[re.Pattern[str]],
  ) -> None:
    self._app = app
    self._has_configured_allowed_origins = has_configured_allowed_origins
    self._allowed_origins = allowed_origins
    self._allowed_origin_regex = allowed_origin_regex

  async def __call__(
      self,
      scope: dict[str, Any],
      receive: Any,
      send: Any,
  ) -> None:
    if scope["type"] != "http":
      await self._app(scope, receive, send)
      return

    method = scope.get("method", "GET")
    if method in _SAFE_HTTP_METHODS:
      await self._app(scope, receive, send)
      return

    origin = _get_scope_header(scope, b"origin")
    if origin is None:
      await self._app(scope, receive, send)
      return

    if _is_request_origin_allowed(
        origin,
        scope,
        self._allowed_origins,
        self._allowed_origin_regex,
        self._has_configured_allowed_origins,
    ):
      await self._app(scope, receive, send)
      return

    response_body = b"Forbidden: origin not allowed"
    await send({
        "type": "http.response.start",
        "status": 403,
        "headers": [
            (b"content-type", b"text/plain"),
            (b"content-length", str(len(response_body)).encode()),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": response_body,
    })


class _DefaultAppRewriteMiddleware:
  """ASGI middleware that rewrites URLs to inject default app name if set and missing."""

  _PRODUCTION_PATH_PATTERNS = [
      re.compile(r"^/users/"),
      re.compile(r"^/app-info$"),
      re.compile(r"^/trigger/"),
  ]

  def __init__(self, app: Any, default_app_name: Optional[str] = None) -> None:
    self._app = app
    self._default_app_name = default_app_name

  async def __call__(
      self,
      scope: dict[str, Any],
      receive: Any,
      send: Any,
  ) -> None:
    if scope["type"] in ("http", "websocket"):
      if self._default_app_name:
        path: str = scope.get("path", "")

        if any(
            pattern.match(path) for pattern in self._PRODUCTION_PATH_PATTERNS
        ):
          scope["path"] = f"/apps/{self._default_app_name}{path}"

        if "raw_path" in scope:
          scope["raw_path"] = scope["path"].encode("latin-1")

    await self._app(scope, receive, send)


class ApiServerSpanExporter(export_lib.SpanExporter):

  def __init__(self, trace_dict):
    self.trace_dict = trace_dict

  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export_lib.SpanExportResult:
    for span in spans:
      if (
          span.name == "call_llm"
          or span.name == "send_data"
          or span.name.startswith("execute_tool")
      ):
        attributes = dict(span.attributes)
        attributes["trace_id"] = span.get_span_context().trace_id
        attributes["span_id"] = span.get_span_context().span_id
        if attributes.get("gcp.vertex.agent.event_id", None):
          self.trace_dict[attributes["gcp.vertex.agent.event_id"]] = attributes
    return export_lib.SpanExportResult.SUCCESS

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


class InMemoryExporter(export_lib.SpanExporter):

  def __init__(self, trace_dict):
    super().__init__()
    self._spans = []
    self.trace_dict = trace_dict

  @override
  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export_lib.SpanExportResult:
    for span in spans:
      trace_id = span.context.trace_id
      attributes = dict(span.attributes)
      session_id = attributes.get(
          "gcp.vertex.agent.session_id", None
      ) or attributes.get("gen_ai.conversation.id", None)
      if session_id:
        trace_ids = self.trace_dict.setdefault(session_id, [])
        if trace_id not in trace_ids:
          trace_ids.append(trace_id)
    self._spans.extend(spans)
    return export_lib.SpanExportResult.SUCCESS

  @override
  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True

  def get_finished_spans(self, session_id: str):
    trace_ids = self.trace_dict.get(session_id, None)
    if trace_ids is None or not trace_ids:
      return []
    return [x for x in self._spans if x.context.trace_id in trace_ids]

  def clear(self):
    self._spans.clear()


class RunAgentRequest(common.BaseModel):
  app_name: Optional[str] = None
  user_id: str
  session_id: str
  new_message: Optional[types.Content] = None
  streaming: bool = False
  state_delta: Optional[dict[str, Any]] = None
  # for long-running function resume requests (e.g., OAuth callback)
  function_call_event_id: Optional[str] = None
  # for resume long-running functions
  invocation_id: Optional[str] = None
  custom_metadata: Optional[dict[str, Any]] = None


class CreateSessionRequest(common.BaseModel):
  session_id: Optional[str] = Field(
      default=None,
      description=(
          "The ID of the session to create. If not provided, a random session"
          " ID will be generated."
      ),
  )
  state: Optional[dict[str, Any]] = Field(
      default=None, description="The initial state of the session."
  )
  events: Optional[list[Event]] = Field(
      default=None,
      description="A list of events to initialize the session with.",
  )


class SaveArtifactRequest(common.BaseModel):
  """Request payload for saving a new artifact."""

  filename: str = Field(description="Artifact filename.")
  artifact: types.Part = Field(
      description="Artifact payload encoded as google.genai.types.Part."
  )
  custom_metadata: Optional[dict[str, Any]] = Field(
      default=None,
      description="Optional metadata to associate with the artifact version.",
  )


class UpdateMemoryRequest(common.BaseModel):
  """Request to add a session to the memory service."""

  session_id: str
  """The ID of the session to add to memory."""


class UpdateSessionRequest(common.BaseModel):
  """Request to update session state without running the agent."""

  state_delta: dict[str, Any]
  """The state changes to apply to the session."""


class AppInfo(common.BaseModel):
  name: str
  root_agent_name: str
  description: str
  language: Literal["yaml", "python"]
  is_computer_use: bool = False
  agents: Optional[dict[str, AgentInfo]] = None


class ListAppsResponse(common.BaseModel):
  apps: list[AppInfo]


def _setup_telemetry(
    otel_to_cloud: bool = False,
    internal_exporters: Optional[list[SpanProcessor]] = None,
):
  # TODO - remove the else branch here once maybe_set_otel_providers is no
  # longer experimental.
  if otel_to_cloud:
    _setup_gcp_telemetry(internal_exporters=internal_exporters)
  elif _otel_env_vars_enabled():
    _setup_telemetry_from_env(internal_exporters=internal_exporters)
  else:
    # Old logic - to be removed when above leaves experimental.
    tracer_provider = TracerProvider()
    if internal_exporters is not None:
      for exporter in internal_exporters:
        tracer_provider.add_span_processor(exporter)
    trace.set_tracer_provider(tracer_provider=tracer_provider)


def _otel_env_vars_enabled() -> bool:
  return any([
      os.getenv(endpoint_var)
      for endpoint_var in [
          otel_env.OTEL_EXPORTER_OTLP_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
          otel_env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT,
      ]
  ])


def _setup_gcp_telemetry(
    internal_exporters: list[SpanProcessor] = None,
):
  if typing.TYPE_CHECKING:
    from ..telemetry.setup import OTelHooks

  otel_hooks_to_add: list[OTelHooks] = []

  if internal_exporters:
    from ..telemetry.setup import OTelHooks

    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  import google.auth

  from ..telemetry.google_cloud import get_gcp_exporters
  from ..telemetry.google_cloud import get_gcp_resource
  from ..telemetry.setup import maybe_set_otel_providers

  credentials, project_id = google.auth.default()

  otel_hooks_to_add.append(
      get_gcp_exporters(
          # TODO - use trace_to_cloud here as well once otel_to_cloud is no
          # longer experimental.
          enable_cloud_tracing=True,
          # TODO - re-enable metrics once errors during shutdown are fixed.
          enable_cloud_metrics=False,
          enable_cloud_logging=True,
          google_auth=(credentials, project_id),
      )
  )
  otel_resource = get_gcp_resource(project_id)

  maybe_set_otel_providers(
      otel_hooks_to_setup=otel_hooks_to_add,
      otel_resource=otel_resource,
  )
  _setup_instrumentation_lib_if_installed()


def _setup_telemetry_from_env(
    internal_exporters: list[SpanProcessor] = None,
):
  from ..telemetry.setup import maybe_set_otel_providers

  otel_hooks_to_add = []

  if internal_exporters:
    from ..telemetry.setup import OTelHooks

    # Register ADK-specific exporters in trace provider.
    otel_hooks_to_add.append(OTelHooks(span_processors=internal_exporters))

  maybe_set_otel_providers(otel_hooks_to_setup=otel_hooks_to_add)
  _setup_instrumentation_lib_if_installed()


def _setup_instrumentation_lib_if_installed():
  # Set instrumentation to enable emitting OTel data from GenAISDK
  # Currently the instrumentation lib is in extras dependencies, make sure to
  # warn the user if it's not installed.
  try:
    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

    GoogleGenAiSdkInstrumentor().instrument()
  except ImportError:
    logger.warning(
        "Unable to import GoogleGenAiSdkInstrumentor - some"
        " telemetry will be disabled. Make sure to install google-adk[otel-gcp]"
    )


class ApiServer:
  """Helper class for setting up and running the ADK web server on FastAPI.

  You construct this class with all the Services required to run ADK agents and
  can then call the get_fast_api_app method to get a FastAPI app instance that
  can will use your provided service instances, static assets, and agent loader.
  If you pass in a web_assets_dir, the static assets will be served under
  /dev-ui in addition to the API endpoints created by default.

  You can add additional API endpoints by modifying the FastAPI app
  instance returned by get_fast_api_app as this class exposes the agent runners
  and most other bits of state retained during the lifetime of the server.

  Attributes:
      agent_loader: An instance of BaseAgentLoader for loading agents.
      session_service: An instance of BaseSessionService for managing sessions.
      memory_service: An instance of BaseMemoryService for managing memory.
      artifact_service: An instance of BaseArtifactService for managing
        artifacts.
      credential_service: An instance of BaseCredentialService for managing
        credentials.
      eval_sets_manager: An instance of EvalSetsManager for managing evaluation
        sets.
      eval_set_results_manager: An instance of EvalSetResultsManager for
        managing evaluation set results.
      agents_dir: Root directory containing subdirs for agents with those
        containing resources (e.g. .env files, eval sets, etc.) for the agents.
      extra_plugins: A list of fully qualified names of extra plugins to load.
      logo_text: Text to display in the logo of the UI.
      logo_image_url: URL of an image to display as logo of the UI.
      runners_to_clean: Set of runner names marked for cleanup.
      current_app_name_ref: A shared reference to the latest ran app name.
      runner_dict: A dict of instantiated runners for each app.
  """

  def __init__(
      self,
      *,
      agent_loader: BaseAgentLoader,
      session_service: BaseSessionService,
      memory_service: BaseMemoryService,
      artifact_service: BaseArtifactService,
      credential_service: BaseCredentialService,
      eval_sets_manager: EvalSetsManager,
      eval_set_results_manager: EvalSetResultsManager,
      agents_dir: str,
      extra_plugins: Optional[list[str]] = None,
      logo_text: Optional[str] = None,
      logo_image_url: Optional[str] = None,
      url_prefix: Optional[str] = None,
      auto_create_session: bool = False,
      trigger_sources: Optional[list[str]] = None,
      default_llm_model: Optional[str] = None,
  ):
    self.agent_loader = agent_loader
    self.session_service = session_service
    self.memory_service = memory_service
    self.artifact_service = artifact_service
    self.credential_service = credential_service
    self.eval_sets_manager = eval_sets_manager
    self.eval_set_results_manager = eval_set_results_manager
    self.agents_dir = agents_dir
    self.extra_plugins = extra_plugins or []
    self.logo_text = logo_text
    self.logo_image_url = logo_image_url
    # Internal properties we want to allow being modified from callbacks.
    self.runners_to_clean: set[str] = set()
    self.current_app_name_ref: SharedValue[str] = SharedValue(value="")
    self.runner_dict = {}
    self.url_prefix = url_prefix
    self.auto_create_session = auto_create_session
    self.trigger_sources = trigger_sources
    self.default_llm_model = default_llm_model
    self.default_app_name = os.getenv("ADK_DEFAULT_APP_NAME")

  async def get_runner_async(self, app_name: str) -> Runner:
    """Returns the cached runner for the given app."""
    # Handle cleanup
    if app_name in self.runners_to_clean:
      self.runners_to_clean.remove(app_name)
      runner = self.runner_dict.pop(app_name, None)
      await cleanup.close_runners(list([runner]))

    # Return cached runner if exists
    if app_name in self.runner_dict:
      return self.runner_dict[app_name]

    # Create new runner
    envs.load_dotenv_for_agent(os.path.basename(app_name), self.agents_dir)
    agent_or_app = self.agent_loader.load_agent(app_name)

    if self.default_llm_model:
      from .cli import _override_default_llm_model

      _override_default_llm_model(self.default_llm_model)

    # Instantiate extra plugins if configured
    extra_plugins_instances = self._instantiate_extra_plugins()

    plugins_yaml_path = os.path.join(self.agents_dir, app_name, "plugins.yaml")
    bq_analytics_config = None
    if os.path.exists(plugins_yaml_path):
      with open(plugins_yaml_path, "r", encoding="utf-8") as f:
        plugins_config = yaml.safe_load(f)
        if plugins_config and isinstance(plugins_config, dict):
          bq_analytics_config = plugins_config.get("bigquery_agent_analytics")

    # All YAML agents are treated as visual builder agents.
    is_visual_builder_agent = os.path.exists(
        os.path.join(self.agents_dir, app_name, "root_agent.yaml")
    )

    def _maybe_add_bq_plugin(plugins: list[BasePlugin]) -> list[BasePlugin]:
      if bq_analytics_config and all([
          bq_analytics_config.get("project_id"),
          bq_analytics_config.get("dataset_id"),
          bq_analytics_config.get("dataset_location"),
      ]):
        from ..plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin

        plugins.append(
            BigQueryAgentAnalyticsPlugin(
                project_id=bq_analytics_config.get("project_id"),
                dataset_id=bq_analytics_config.get("dataset_id"),
                table_id=bq_analytics_config.get("table_id"),
                location=bq_analytics_config.get("dataset_location"),
            )
        )
      return plugins

    if isinstance(agent_or_app, App):
      # Combine existing plugins with extra plugins
      plugins = _maybe_add_bq_plugin(
          agent_or_app.plugins + extra_plugins_instances
      )
      agent_or_app.plugins = plugins
      agentic_app = agent_or_app
    elif isinstance(agent_or_app, BaseAgent):
      plugins = _maybe_add_bq_plugin(extra_plugins_instances)
      agentic_app = App(
          name=app_name,
          root_agent=agent_or_app,
          plugins=plugins,
      )
    else:
      # BaseNode (non-agent)
      agentic_app = App(
          name=app_name,
          root_agent=agent_or_app,
          plugins=extra_plugins_instances,
      )

    # If the root agent was loaded from YAML, we treat it as being from Visual Builder
    if is_visual_builder_agent:
      object.__setattr__(agentic_app, "_is_visual_builder_app", True)

    runner = self._create_runner(agentic_app)
    self.runner_dict[app_name] = runner
    return runner

  def _get_root_agent(self, agent_or_app: BaseAgent | App) -> BaseAgent:
    """Extract root agent from either a BaseAgent or App object."""
    if isinstance(agent_or_app, App):
      return agent_or_app.root_agent
    return agent_or_app

  def _create_runner(self, agentic_app: App) -> Runner:
    """Create a runner with common services."""
    return Runner(
        app=agentic_app,
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        credential_service=self.credential_service,
        auto_create_session=self.auto_create_session,
    )

  def _instantiate_extra_plugins(self) -> list[BasePlugin]:
    """Instantiate extra plugins from the configured list.

    Returns:
      List of instantiated BasePlugin objects.
    """
    extra_plugins_instances = []
    for qualified_name in self.extra_plugins:
      try:
        plugin_obj = self._import_plugin_object(qualified_name)
        if isinstance(plugin_obj, BasePlugin):
          extra_plugins_instances.append(plugin_obj)
        elif issubclass(plugin_obj, BasePlugin):
          extra_plugins_instances.append(plugin_obj(name=qualified_name))
      except Exception as e:
        logger.error("Failed to load plugin %s: %s", qualified_name, e)
    return extra_plugins_instances

  def _import_plugin_object(self, qualified_name: str) -> Any:
    """Import a plugin object (class or instance) from a fully qualified name.

    Args:
      qualified_name: Fully qualified name (e.g.,
        'my_package.my_plugin.MyPlugin')

    Returns:
      The imported object, which can be either a class or an instance.

    Raises:
      ImportError: If the module cannot be imported.
      AttributeError: If the object doesn't exist in the module.
    """
    module_name, obj_name = qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)

  def _setup_runtime_config(self, web_assets_dir: str):
    """Sets up the runtime config for the web server."""
    # Read existing runtime config file.
    runtime_config_path = os.path.join(
        web_assets_dir, "assets", "config", "runtime-config.json"
    )
    runtime_config = {}
    try:
      with open(runtime_config_path, "r") as f:
        runtime_config = json.load(f)
    except FileNotFoundError:
      logger.info(
          "File not found: %s. A new runtime config file will be created.",
          runtime_config_path,
      )
    except json.JSONDecodeError:
      logger.warning(
          "Failed to decode JSON from %s. The file content will be"
          " overwritten.",
          runtime_config_path,
      )
    runtime_config["backendUrl"] = self.url_prefix if self.url_prefix else ""

    # Set custom logo config.
    if self.logo_text or self.logo_image_url:
      if not self.logo_text or not self.logo_image_url:
        raise ValueError(
            "Both --logo-text and --logo-image-url must be defined when using"
            " logo config."
        )
      runtime_config["logo"] = {
          "text": self.logo_text,
          "imageUrl": self.logo_image_url,
      }
    elif "logo" in runtime_config:
      del runtime_config["logo"]

    # Write the runtime config file.
    try:
      os.makedirs(os.path.dirname(runtime_config_path), exist_ok=True)
      with open(runtime_config_path, "w") as f:
        json.dump(runtime_config, f, indent=2)
    except IOError as e:
      logger.error(
          "Failed to write runtime config file %s: %s", runtime_config_path, e
      )

  async def _create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: Optional[str] = None,
      state: Optional[dict[str, Any]] = None,
  ) -> Session:
    try:
      session = await self.session_service.create_session(
          app_name=app_name,
          user_id=user_id,
          state=state,
          session_id=session_id,
      )
      logger.info("New session created: %s", session.id)
      return session
    except AlreadyExistsError as e:
      raise HTTPException(
          status_code=409, detail=f"Session already exists: {session_id}"
      ) from e
    except Exception as e:
      logger.error(
          "Internal server error during session creation: %s", e, exc_info=True
      )
      raise HTTPException(status_code=500, detail=str(e)) from e

  def get_fast_api_app(
      self,
      lifespan: Optional[Lifespan[FastAPI]] = None,
      allow_origins: Optional[list[str]] = None,
      web_assets_dir: Optional[str] = None,
      setup_observer: Callable[
          [Observer, "ApiServer"], None
      ] = lambda o, s: None,
      tear_down_observer: Callable[
          [Observer, "ApiServer"], None
      ] = lambda o, s: None,
      register_processors: Callable[[TracerProvider], None] = lambda o: None,
      otel_to_cloud: bool = False,
      with_ui: bool = False,
  ):
    """Creates a FastAPI app for the ADK web server.

    By default it'll just return a FastAPI instance with the API server
    endpoints,
    but if you specify a web_assets_dir, it'll also serve the static web assets
    from that directory.

    Args:
      lifespan: The lifespan of the FastAPI app.
      allow_origins: The origins that are allowed to make cross-origin requests.
        Entries can be literal origins (e.g., 'https://example.com') or regex
        patterns prefixed with 'regex:' (e.g.,
        'regex:https://.*\\.example\\.com').
      web_assets_dir: The directory containing the web assets to serve.
      setup_observer: Callback for setting up the file system observer.
      tear_down_observer: Callback for cleaning up the file system observer.
      register_processors: Callback for additional Span processors to be added
        to the TracerProvider.
      otel_to_cloud: Whether to enable Cloud Trace and Cloud Logging
        integrations.

    Returns:
      A FastAPI app instance.
    """
    trace_dict = {}
    session_trace_dict = {}
    self._trace_dict = trace_dict
    self._session_trace_dict = session_trace_dict

    # Set up a file system watcher to detect changes in the agents directory.
    observer = Observer()
    setup_observer(observer, self)

    @asynccontextmanager
    async def internal_lifespan(app: FastAPI):
      try:
        if lifespan:
          async with lifespan(app) as lifespan_context:
            yield lifespan_context
        else:
          yield
      finally:
        tear_down_observer(observer, self)
        # Create tasks for all runner closures to run concurrently
        await cleanup.close_runners(list(self.runner_dict.values()))

    memory_exporter = InMemoryExporter(session_trace_dict)
    self._memory_exporter = memory_exporter

    _setup_telemetry(
        otel_to_cloud=otel_to_cloud,
        internal_exporters=[
            export_lib.SimpleSpanProcessor(ApiServerSpanExporter(trace_dict)),
            export_lib.SimpleSpanProcessor(memory_exporter),
        ],
    )
    if web_assets_dir:
      self._setup_runtime_config(web_assets_dir)

    tracer_provider = trace.get_tracer_provider()
    register_processors(tracer_provider)

    # Run the FastAPI server.
    app = FastAPI(lifespan=internal_lifespan)

    has_configured_allowed_origins = bool(allow_origins)
    if allow_origins:
      literal_origins, combined_regex = _parse_cors_origins(allow_origins)
      compiled_origin_regex = (
          re.compile(combined_regex) if combined_regex is not None else None
      )
      app.add_middleware(
          CORSMiddleware,
          allow_origins=literal_origins,
          allow_origin_regex=combined_regex,
          allow_credentials=True,
          allow_methods=["*"],
          allow_headers=["*"],
      )
    else:
      literal_origins = []
      compiled_origin_regex = None

    app.add_middleware(
        _OriginCheckMiddleware,
        has_configured_allowed_origins=has_configured_allowed_origins,
        allowed_origins=literal_origins,
        allowed_origin_regex=compiled_origin_regex,
    )

    app.add_middleware(
        _DefaultAppRewriteMiddleware,
        default_app_name=self.default_app_name,
    )

    # Register production endpoints (22 total)
    self._register_production_endpoints(
        app,
        trace_dict,
        memory_exporter,
        literal_origins,
        compiled_origin_regex,
        has_configured_allowed_origins,
    )

    if web_assets_dir:
      import mimetypes

      mimetypes.add_type("application/javascript", ".js", True)
      mimetypes.add_type("text/javascript", ".js", True)

      redirect_dev_ui_url = (
          self.url_prefix + "/dev-ui/" if self.url_prefix else "/dev-ui/"
      )

      @app.get("/dev-ui/config")
      async def get_ui_config():
        return {
            "logo_text": self.logo_text,
            "logo_image_url": self.logo_image_url,
        }

      @app.get("/")
      async def redirect_root_to_dev_ui():
        return RedirectResponse(redirect_dev_ui_url)

      @app.get("/dev-ui")
      async def redirect_dev_ui_add_slash():
        return RedirectResponse(redirect_dev_ui_url)

      app.mount(
          "/dev-ui/",
          StaticFiles(directory=web_assets_dir, html=True, follow_symlink=True),
          name="static",
      )

    # Register /trigger/* endpoints when enabled.
    if self.trigger_sources:
      from .trigger_routes import TriggerRouter

      trigger_router = TriggerRouter(self, trigger_sources=self.trigger_sources)
      trigger_router.register(app)

    return app

  def _register_production_endpoints(
      self,
      app: FastAPI,
      trace_dict: dict,
      memory_exporter: Any,
      literal_origins: list[str],
      compiled_origin_regex: Optional[re.Pattern[str]],
      has_configured_allowed_origins: bool,
  ):
    """Register all core production-safe endpoints."""

    @app.get("/health")
    async def health() -> dict[str, str]:
      return {"status": "ok"}

    @app.get("/version")
    async def version() -> dict[str, str]:
      return {
          "version": __version__,
          "language": "python",
          "language_version": (
              f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
          ),
      }

    @app.get("/list-apps")
    async def list_apps(
        detailed: bool = Query(
            default=False, description="Return detailed app information"
        )
    ) -> list[str] | ListAppsResponse:
      if detailed:
        apps_info = self.agent_loader.list_agents_detailed()
        return ListAppsResponse(apps=[AppInfo(**app) for app in apps_info])
      return self.agent_loader.list_agents()

    @experimental
    @app.get("/apps/{app_name}/app-info", response_model_exclude_none=True)
    async def get_adk_app_info(app_name: str) -> AppInfo:
      """Returns the detailed info for a given ADK app."""
      agent_or_app = self.agent_loader.load_agent(app_name)
      root_agent = self._get_root_agent(agent_or_app)
      if isinstance(root_agent, LlmAgent):
        return AppInfo(
            name=app_name,
            root_agent_name=root_agent.name,
            description=root_agent.description,
            language="python",
            agents=await get_agents_dict(root_agent),
        )
      else:
        raise HTTPException(
            status_code=400, detail="Root agent is not an LlmAgent"
        )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
    )
    async def get_session(
        app_name: str, user_id: str, session_id: str
    ) -> Session:
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")
      self.current_app_name_ref.value = app_name
      return session

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
    )
    async def list_sessions(app_name: str, user_id: str) -> list[Session]:
      list_sessions_response = await self.session_service.list_sessions(
          app_name=app_name, user_id=user_id
      )
      return [
          session
          for session in list_sessions_response.sessions
          # Remove sessions that were generated as a part of Eval.
          if not session.id.startswith(EVAL_SESSION_ID_PREFIX)
      ]

    @deprecated(
        "Please use create_session instead. This will be removed in future"
        " releases."
    )
    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
    )
    async def create_session_with_id(
        app_name: str,
        user_id: str,
        session_id: str,
        state: Optional[dict[str, Any]] = None,
    ) -> Session:
      return await self._create_session(
          app_name=app_name,
          user_id=user_id,
          state=state,
          session_id=session_id,
      )

    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
    )
    async def create_session(
        app_name: str,
        user_id: str,
        req: Optional[CreateSessionRequest] = None,
    ) -> Session:
      if not req:
        return await self._create_session(app_name=app_name, user_id=user_id)

      session = await self._create_session(
          app_name=app_name,
          user_id=user_id,
          state=req.state,
          session_id=req.session_id,
      )

      if req.events:
        for event in req.events:
          await self.session_service.append_event(session=session, event=event)

      return session

    @app.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def delete_session(
        app_name: str, user_id: str, session_id: str
    ) -> None:
      await self.session_service.delete_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    @app.patch(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
        response_model_exclude_none=True,
    )
    async def update_session(
        app_name: str,
        user_id: str,
        session_id: str,
        req: UpdateSessionRequest,
    ) -> Session:
      """Updates session state without running the agent.

      Args:
          app_name: The name of the application.
          user_id: The ID of the user.
          session_id: The ID of the session to update.
          req: The patch request containing state changes.

      Returns:
          The updated session.

      Raises:
          HTTPException: If the session is not found.
      """
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")

      # Create an event to record the state change
      import uuid

      from ..events.event import Event
      from ..events.event import EventActions

      state_update_event = Event(
          invocation_id="p-" + str(uuid.uuid4()),
          author="user",
          actions=EventActions(state_delta=req.state_delta),
      )

      # Append the event to the session
      # This will automatically update the session state through __update_session_state
      await self.session_service.append_event(
          session=session, event=state_update_event
      )

      return session

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
        response_model_exclude_none=True,
    )
    async def load_artifact(
        app_name: str,
        user_id: str,
        session_id: str,
        artifact_name: str,
        version: Optional[int] = Query(None),
    ) -> Optional[types.Part]:
      artifact = await self.artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version,
      )
      if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
      return artifact

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/metadata",
        response_model=list[ArtifactVersion],
        response_model_exclude_none=True,
    )
    async def list_artifact_versions_metadata(
        app_name: str,
        user_id: str,
        session_id: str,
        artifact_name: str,
    ) -> list[ArtifactVersion]:
      return await self.artifact_service.list_artifact_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}",
        response_model_exclude_none=True,
    )
    async def load_artifact_version(
        app_name: str,
        user_id: str,
        session_id: str,
        artifact_name: str,
        version_id: int,
    ) -> Optional[types.Part]:
      artifact = await self.artifact_service.load_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version_id,
      )
      if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
      return artifact

    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
        response_model=ArtifactVersion,
        response_model_exclude_none=True,
    )
    async def save_artifact(
        app_name: str,
        user_id: str,
        session_id: str,
        req: SaveArtifactRequest,
    ) -> ArtifactVersion:
      """Request payload for saving a new artifact."""
      try:
        version = await self.artifact_service.save_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=req.filename,
            artifact=req.artifact,
            custom_metadata=req.custom_metadata,
        )
      except InputValidationError as ive:
        raise HTTPException(status_code=400, detail=str(ive)) from ive
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error(
            "Internal error while saving artifact %s for app=%s user=%s"
            " session=%s: %s",
            req.filename,
            app_name,
            user_id,
            session_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
      artifact_version = await self.artifact_service.get_artifact_version(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=req.filename,
          version=version,
      )
      if artifact_version is None:
        raise HTTPException(
            status_code=500, detail="Artifact metadata unavailable"
        )
      return artifact_version

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}/metadata",
        response_model=ArtifactVersion,
        response_model_exclude_none=True,
    )
    async def get_artifact_version_metadata(
        app_name: str,
        user_id: str,
        session_id: str,
        artifact_name: str,
        version_id: int,
    ) -> ArtifactVersion:
      artifact_version = await self.artifact_service.get_artifact_version(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
          version=version_id,
      )
      if not artifact_version:
        raise HTTPException(
            status_code=404, detail="Artifact version not found"
        )
      return artifact_version

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
        response_model_exclude_none=True,
    )
    async def list_artifact_names(
        app_name: str, user_id: str, session_id: str
    ) -> list[str]:
      return await self.artifact_service.list_artifact_keys(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
        response_model_exclude_none=True,
    )
    async def list_artifact_versions(
        app_name: str, user_id: str, session_id: str, artifact_name: str
    ) -> list[int]:
      return await self.artifact_service.list_versions(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.delete(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
    )
    async def delete_artifact(
        app_name: str, user_id: str, session_id: str, artifact_name: str
    ) -> None:
      await self.artifact_service.delete_artifact(
          app_name=app_name,
          user_id=user_id,
          session_id=session_id,
          filename=artifact_name,
      )

    @app.patch("/apps/{app_name}/users/{user_id}/memory")
    async def patch_memory(
        app_name: str, user_id: str, update_memory_request: UpdateMemoryRequest
    ) -> None:
      """Adds all events from a given session to the memory service.

      Args:
          app_name: The name of the application.
          user_id: The ID of the user.
          update_memory_request: The memory request for the update

      Raises:
          HTTPException: If the memory service is not configured or the request
          is invalid.
      """
      if not self.memory_service:
        raise HTTPException(
            status_code=400, detail="Memory service is not configured."
        )
      if (
          update_memory_request is None
          or update_memory_request.session_id is None
      ):
        raise HTTPException(
            status_code=400, detail="Update memory request is invalid."
        )

      session = await self.session_service.get_session(
          app_name=app_name,
          user_id=user_id,
          session_id=update_memory_request.session_id,
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")
      await self.memory_service.add_session_to_memory(session)

    def _set_telemetry_context_if_needed(runner: Runner):
      """Helper to set contextvars for the current request task."""
      app = getattr(runner, "app", None)
      from ..utils._telemetry_context import _is_visual_builder

      if app and getattr(app, "_is_visual_builder_app", False):
        _is_visual_builder.set(True)
      else:
        _is_visual_builder.set(False)

    @app.post("/run", response_model_exclude_none=True)
    async def run_agent(req: RunAgentRequest, request: Request) -> list[Event]:
      app_name = req.app_name or self.default_app_name
      if not app_name:
        raise HTTPException(
            status_code=400,
            detail="app_name is required when ADK_DEFAULT_APP_NAME is not set",
        )
      req.app_name = app_name
      self.current_app_name_ref.value = req.app_name
      runner = await self.get_runner_async(req.app_name)
      _set_telemetry_context_if_needed(runner)
      run_config = (
          RunConfig(custom_metadata=req.custom_metadata)
          if req.custom_metadata
          else None
      )

      async def worker():
        try:
          async with Aclosing(
              runner.run_async(
                  user_id=req.user_id,
                  session_id=req.session_id,
                  new_message=req.new_message,
                  state_delta=req.state_delta,
                  invocation_id=req.invocation_id,
                  run_config=run_config,
              )
          ) as agen:
            return [event async for event in agen]
        except SessionNotFoundError as e:
          raise HTTPException(status_code=404, detail=str(e)) from e

      worker_task = asyncio.create_task(worker())

      async def monitor():
        try:
          while True:
            message = await request.receive()
            if message.get("type") == "http.disconnect":
              logger.warning(
                  "Client disconnected. Aborting agent run for session %s.",
                  req.session_id,
              )
              worker_task.cancel()
              break
        except asyncio.CancelledError:
          pass
        except Exception as e:  # pylint: disable=broad-exception-caught
          logger.error("Exception in disconnect monitor: %s", e, exc_info=True)

      monitor_task = asyncio.create_task(monitor())

      try:
        events = await worker_task
        logger.info("Generated %s events in agent run", len(events))
        logger.debug("Events generated: %s", events)
        return events
      except asyncio.CancelledError:
        if await request.is_disconnected():
          return Response(status_code=499)
        raise
      finally:
        monitor_task.cancel()

    @app.post("/run_sse")
    async def run_agent_sse(req: RunAgentRequest) -> StreamingResponse:
      app_name = req.app_name or self.default_app_name
      if not app_name:
        raise HTTPException(
            status_code=400,
            detail="app_name is required when ADK_DEFAULT_APP_NAME is not set",
        )
      req.app_name = app_name
      self.current_app_name_ref.value = req.app_name
      stream_mode = StreamingMode.SSE if req.streaming else StreamingMode.NONE
      runner = await self.get_runner_async(req.app_name)
      _set_telemetry_context_if_needed(runner)

      # Validate session existence before starting the stream.
      # We check directly here instead of eagerly advancing the
      # runner's async generator with anext(), because splitting
      # generator consumption across two asyncio Tasks (request
      # handler vs StreamingResponse) breaks OpenTelemetry context
      # detachment.
      if not runner.auto_create_session:
        session = await self.session_service.get_session(
            app_name=req.app_name,
            user_id=req.user_id,
            session_id=req.session_id,
        )
        if not session:
          raise HTTPException(
              status_code=404,
              detail=f"Session not found: {req.session_id}",
          )

      # Convert the events to properly formatted SSE
      async def event_generator():
        async with Aclosing(
            runner.run_async(
                user_id=req.user_id,
                session_id=req.session_id,
                new_message=req.new_message,
                state_delta=req.state_delta,
                run_config=RunConfig(
                    streaming_mode=stream_mode,
                    custom_metadata=req.custom_metadata,
                ),
                invocation_id=req.invocation_id,
            )
        ) as agen:
          try:
            async for event in agen:
              # ADK Web renders artifacts from `actions.artifactDelta`
              # during part processing *and* during action processing
              # 1) the original event with `artifactDelta` cleared (content)
              # 2) a content-less "action-only" event carrying `artifactDelta`
              events_to_stream = [event]
              if (
                  not req.function_call_event_id
                  and event.actions.artifact_delta
                  and event.content
                  and event.content.parts
              ):
                content_event = event.model_copy(deep=True)
                content_event.actions.artifact_delta = {}
                artifact_event = event.model_copy(deep=True)
                artifact_event.content = None
                events_to_stream = [content_event, artifact_event]

              for event_to_stream in events_to_stream:
                sse_event = event_to_stream.model_dump_json(
                    exclude_none=True,
                    by_alias=True,
                )
                logger.debug(
                    "Generated event in agent run streaming: %s", sse_event
                )
                yield f"data: {sse_event}\n\n"
          except Exception as e:
            logger.exception("Error in event_generator: %s", e)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

      # Returns a streaming response with the proper media type for SSE
      return StreamingResponse(
          event_generator(),
          media_type="text/event-stream",
      )

    @app.websocket("/run_live")
    async def run_agent_live(
        websocket: WebSocket,
        user_id: str,
        session_id: str,
        app_name: Optional[str] = Query(default=None),
        modalities: List[Literal["TEXT", "AUDIO"]] = Query(
            default=["AUDIO"]
        ),  # Only allows "TEXT" or "AUDIO"
        proactive_audio: bool | None = Query(default=None),
        enable_affective_dialog: bool | None = Query(default=None),
        enable_session_resumption: bool | None = Query(default=None),
        save_live_blob: bool = Query(default=False),
    ) -> None:
      resolved_app_name = app_name or self.default_app_name
      if not resolved_app_name:
        await websocket.close(
            code=1008,
            reason=(
                "app_name query parameter is required when ADK_DEFAULT_APP_NAME"
                " is not set"
            ),
        )
        return
      app_name = resolved_app_name

      ws_origin = websocket.headers.get("origin")
      if ws_origin is not None and not _is_request_origin_allowed(
          ws_origin,
          websocket.scope,
          literal_origins,
          compiled_origin_regex,
          has_configured_allowed_origins,
      ):
        await websocket.close(code=1008, reason="Origin not allowed")
        return

      await websocket.accept()
      self.current_app_name_ref.value = app_name
      runner_for_context = await self.get_runner_async(app_name)
      _set_telemetry_context_if_needed(runner_for_context)

      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        await websocket.close(code=1002, reason="Session not found")
        return

      live_request_queue = LiveRequestQueue()

      async def forward_events():
        runner = await self.get_runner_async(app_name)
        run_config = RunConfig(
            response_modalities=modalities,
            proactivity=(
                types.ProactivityConfig(proactive_audio=proactive_audio)
                if proactive_audio is not None
                else None
            ),
            enable_affective_dialog=enable_affective_dialog,
            session_resumption=(
                types.SessionResumptionConfig(
                    transparent=enable_session_resumption
                )
                if enable_session_resumption is not None
                else None
            ),
            save_live_blob=save_live_blob,
        )
        async with Aclosing(
            runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
            )
        ) as agen:
          async for event in agen:
            await websocket.send_text(
                event.model_dump_json(exclude_none=True, by_alias=True)
            )

      async def process_messages():
        try:
          while True:
            data = await websocket.receive_text()
            # Validate and send the received message to the live queue.
            live_request_queue.send(LiveRequest.model_validate_json(data))
        except ValidationError as ve:
          logger.error("Validation error in process_messages: %s", ve)

      # Run both tasks concurrently and cancel all if one fails.
      tasks = [
          asyncio.create_task(forward_events()),
          asyncio.create_task(process_messages()),
      ]
      done, pending = await asyncio.wait(
          tasks, return_when=asyncio.FIRST_EXCEPTION
      )
      try:
        # This will re-raise any exception from the completed tasks.
        for task in done:
          task.result()
      except WebSocketDisconnect:
        # Disconnection could happen when receive or send text via websocket
        logger.info("Client disconnected during live session.")
      except Exception as e:
        logger.exception("Error during live websocket communication: %s", e)
        traceback.print_exc()
        WEBSOCKET_INTERNAL_ERROR_CODE = 1011
        WEBSOCKET_MAX_BYTES_FOR_REASON = 123
        await websocket.close(
            code=WEBSOCKET_INTERNAL_ERROR_CODE,
            reason=str(e)[:WEBSOCKET_MAX_BYTES_FOR_REASON],
        )
      finally:
        for task in pending:
          task.cancel()
