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

import asyncio
from contextlib import asynccontextmanager
import logging
import os
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.genai import types
import graphviz
from opentelemetry import trace
from opentelemetry.sdk.trace import export as export_lib
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from pydantic import Field
from pydantic import ValidationError
from starlette.types import Lifespan
from typing_extensions import override
from watchdog.observers import Observer

from . import agent_graph
from ..agents.live_request_queue import LiveRequest
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.run_config import RunConfig
from ..agents.run_config import StreamingMode
from ..artifacts.base_artifact_service import BaseArtifactService
from ..auth.credential_service.base_credential_service import BaseCredentialService
from ..errors.not_found_error import NotFoundError
from ..evaluation.base_eval_service import InferenceConfig
from ..evaluation.base_eval_service import InferenceRequest
from ..evaluation.constants import MISSING_EVAL_DEPENDENCIES_MESSAGE
from ..evaluation.eval_case import EvalCase
from ..evaluation.eval_case import SessionInput
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_metrics import MetricInfo
from ..evaluation.eval_result import EvalSetResult
from ..evaluation.eval_set_results_manager import EvalSetResultsManager
from ..evaluation.eval_sets_manager import EvalSetsManager
from ..events.event import Event
from ..memory.base_memory_service import BaseMemoryService
from ..runners import Runner
from ..sessions.base_session_service import BaseSessionService
from ..sessions.session import Session
from ..utils.context_utils import Aclosing
from .cli_eval import EVAL_SESSION_ID_PREFIX
from .cli_eval import EvalStatus
from .utils import cleanup
from .utils import common
from .utils import envs
from .utils import evals
from .utils.base_agent_loader import BaseAgentLoader
from .utils.shared_value import SharedValue
from .utils.state import create_empty_state

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"

TAG_DEBUG = "Debug"
TAG_EVALUATION = "Evaluation"


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
      if span.name == "call_llm":
        attributes = dict(span.attributes)
        session_id = attributes.get("gcp.vertex.agent.session_id", None)
        if session_id:
          if session_id not in self.trace_dict:
            self.trace_dict[session_id] = [trace_id]
          else:
            self.trace_dict[session_id] += [trace_id]
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
  app_name: str
  user_id: str
  session_id: str
  new_message: types.Content
  streaming: bool = False
  state_delta: Optional[dict[str, Any]] = None


class AddSessionToEvalSetRequest(common.BaseModel):
  eval_id: str
  session_id: str
  user_id: str


class RunEvalRequest(common.BaseModel):
  eval_ids: list[str]  # if empty, then all evals in the eval set are run.
  eval_metrics: list[EvalMetric]


class RunEvalResult(common.BaseModel):
  eval_set_file: str
  eval_set_id: str
  eval_id: str
  final_eval_status: EvalStatus
  eval_metric_results: list[tuple[EvalMetric, EvalMetricResult]] = Field(
      deprecated=True,
      default=[],
      description=(
          "This field is deprecated, use overall_eval_metric_results instead."
      ),
  )
  overall_eval_metric_results: list[EvalMetricResult]
  eval_metric_result_per_invocation: list[EvalMetricResultPerInvocation]
  user_id: str
  session_id: str


class GetEventGraphResult(common.BaseModel):
  dot_src: str


class AdkWebServer:
  """Helper class for setting up and running the ADK web server on FastAPI.

  You construct this class with all the Services required to run ADK agents and
  can then call the get_fast_api_app method to get a FastAPI app instance that
  can will use your provided service instances, static assets, and agent loader.
  If you pass in a web_assets_dir, the static assets will be served under
  /dev-ui in addition to the API endpoints created by default.

  You can add add additional API endpoints by modifying the FastAPI app
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
  ):
    self.agent_loader = agent_loader
    self.session_service = session_service
    self.memory_service = memory_service
    self.artifact_service = artifact_service
    self.credential_service = credential_service
    self.eval_sets_manager = eval_sets_manager
    self.eval_set_results_manager = eval_set_results_manager
    self.agents_dir = agents_dir
    # Internal propeties we want to allow being modified from callbacks.
    self.runners_to_clean: set[str] = set()
    self.current_app_name_ref: SharedValue[str] = SharedValue(value="")
    self.runner_dict = {}

  async def get_runner_async(self, app_name: str) -> Runner:
    """Returns the runner for the given app."""
    if app_name in self.runners_to_clean:
      self.runners_to_clean.remove(app_name)
      runner = self.runner_dict.pop(app_name, None)
      await cleanup.close_runners(list([runner]))

    envs.load_dotenv_for_agent(os.path.basename(app_name), self.agents_dir)
    if app_name in self.runner_dict:
      return self.runner_dict[app_name]
    root_agent = self.agent_loader.load_agent(app_name)
    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=self.artifact_service,
        session_service=self.session_service,
        memory_service=self.memory_service,
        credential_service=self.credential_service,
    )
    self.runner_dict[app_name] = runner
    return runner

  def get_fast_api_app(
      self,
      lifespan: Optional[Lifespan[FastAPI]] = None,
      allow_origins: Optional[list[str]] = None,
      web_assets_dir: Optional[str] = None,
      setup_observer: Callable[
          [Observer, "AdkWebServer"], None
      ] = lambda o, s: None,
      tear_down_observer: Callable[
          [Observer, "AdkWebServer"], None
      ] = lambda o, s: None,
      register_processors: Callable[[TracerProvider], None] = lambda o: None,
  ):
    """Creates a FastAPI app for the ADK web server.

    By default it'll just return a FastAPI instance with the API server
    endpoints,
    but if you specify a web_assets_dir, it'll also serve the static web assets
    from that directory.

    Args:
      lifespan: The lifespan of the FastAPI app.
      allow_origins: The origins that are allowed to make cross-origin requests.
      web_assets_dir: The directory containing the web assets to serve.
      setup_observer: Callback for setting up the file system observer.
      tear_down_observer: Callback for cleaning up the file system observer.
      register_processors: Callback for additional Span processors to be added
        to the TracerProvider.

    Returns:
      A FastAPI app instance.
    """
    # Properties we don't need to modify from callbacks
    trace_dict = {}
    session_trace_dict = {}
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

    # Set up tracing in the FastAPI server.
    provider = TracerProvider()
    provider.add_span_processor(
        export_lib.SimpleSpanProcessor(ApiServerSpanExporter(trace_dict))
    )
    memory_exporter = InMemoryExporter(session_trace_dict)
    provider.add_span_processor(export_lib.SimpleSpanProcessor(memory_exporter))

    register_processors(provider)

    trace.set_tracer_provider(provider)

    # Run the FastAPI server.
    app = FastAPI(lifespan=internal_lifespan)

    if allow_origins:
      app.add_middleware(
          CORSMiddleware,
          allow_origins=allow_origins,
          allow_credentials=True,
          allow_methods=["*"],
          allow_headers=["*"],
      )

    @app.get("/list-apps")
    async def list_apps() -> list[str]:
      return self.agent_loader.list_agents()

    @app.get("/debug/trace/{event_id}", tags=[TAG_DEBUG])
    async def get_trace_dict(event_id: str) -> Any:
      event_dict = trace_dict.get(event_id, None)
      if event_dict is None:
        raise HTTPException(status_code=404, detail="Trace not found")
      return event_dict

    @app.get("/debug/trace/session/{session_id}", tags=[TAG_DEBUG])
    async def get_session_trace(session_id: str) -> Any:
      spans = memory_exporter.get_finished_spans(session_id)
      if not spans:
        return []
      return [
          {
              "name": s.name,
              "span_id": s.context.span_id,
              "trace_id": s.context.trace_id,
              "start_time": s.start_time,
              "end_time": s.end_time,
              "attributes": dict(s.attributes),
              "parent_span_id": s.parent.span_id if s.parent else None,
          }
          for s in spans
      ]

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
      if (
          await self.session_service.get_session(
              app_name=app_name, user_id=user_id, session_id=session_id
          )
          is not None
      ):
        raise HTTPException(
            status_code=400, detail=f"Session already exists: {session_id}"
        )
      session = await self.session_service.create_session(
          app_name=app_name, user_id=user_id, state=state, session_id=session_id
      )
      logger.info("New session created: %s", session_id)
      return session

    @app.post(
        "/apps/{app_name}/users/{user_id}/sessions",
        response_model_exclude_none=True,
    )
    async def create_session(
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        events: Optional[list[Event]] = None,
    ) -> Session:
      session = await self.session_service.create_session(
          app_name=app_name, user_id=user_id, state=state
      )

      if events:
        for event in events:
          await self.session_service.append_event(session=session, event=event)

      logger.info("New session created")
      return session

    @app.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
    async def delete_session(
        app_name: str, user_id: str, session_id: str
    ) -> None:
      await self.session_service.delete_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )

    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def create_eval_set(
        app_name: str,
        eval_set_id: str,
    ):
      """Creates an eval set, given the id."""
      try:
        self.eval_sets_manager.create_eval_set(app_name, eval_set_id)
      except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        ) from ve

    @app.get(
        "/apps/{app_name}/eval_sets",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_sets(app_name: str) -> list[str]:
      """Lists all eval sets for the given app."""
      try:
        return self.eval_sets_manager.list_eval_sets(app_name)
      except NotFoundError as e:
        logger.warning(e)
        return []

    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def add_session_to_eval_set(
        app_name: str, eval_set_id: str, req: AddSessionToEvalSetRequest
    ):
      # Get the session
      session = await self.session_service.get_session(
          app_name=app_name, user_id=req.user_id, session_id=req.session_id
      )
      assert session, "Session not found."

      # Convert the session data to eval invocations
      invocations = evals.convert_session_to_eval_invocations(session)

      # Populate the session with initial session state.
      initial_session_state = create_empty_state(
          self.agent_loader.load_agent(app_name)
      )

      new_eval_case = EvalCase(
          eval_id=req.eval_id,
          conversation=invocations,
          session_input=SessionInput(
              app_name=app_name,
              user_id=req.user_id,
              state=initial_session_state,
          ),
          creation_timestamp=time.time(),
      )

      try:
        self.eval_sets_manager.add_eval_case(
            app_name, eval_set_id, new_eval_case
        )
      except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve

    @app.get(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_evals_in_eval_set(
        app_name: str,
        eval_set_id: str,
    ) -> list[str]:
      """Lists all evals in an eval set."""
      eval_set_data = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

      if not eval_set_data:
        raise HTTPException(
            status_code=400, detail=f"Eval set `{eval_set_id}` not found."
        )

      return sorted([x.eval_id for x in eval_set_data.eval_cases])

    @app.get(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> EvalCase:
      """Gets an eval case in an eval set."""
      eval_case_to_find = self.eval_sets_manager.get_eval_case(
          app_name, eval_set_id, eval_case_id
      )

      if eval_case_to_find:
        return eval_case_to_find

      raise HTTPException(
          status_code=404,
          detail=(
              f"Eval set `{eval_set_id}` or Eval `{eval_case_id}` not found."
          ),
      )

    @app.put(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def update_eval(
        app_name: str,
        eval_set_id: str,
        eval_case_id: str,
        updated_eval_case: EvalCase,
    ):
      if (
          updated_eval_case.eval_id
          and updated_eval_case.eval_id != eval_case_id
      ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Eval id in EvalCase should match the eval id in the API route."
            ),
        )

      # Overwrite the value. We are either overwriting the same value or an empty
      # field.
      updated_eval_case.eval_id = eval_case_id
      try:
        self.eval_sets_manager.update_eval_case(
            app_name, eval_set_id, updated_eval_case
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.delete(
        "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
        tags=[TAG_EVALUATION],
    )
    async def delete_eval(
        app_name: str, eval_set_id: str, eval_case_id: str
    ) -> None:
      try:
        self.eval_sets_manager.delete_eval_case(
            app_name, eval_set_id, eval_case_id
        )
      except NotFoundError as nfe:
        raise HTTPException(status_code=404, detail=str(nfe)) from nfe

    @app.post(
        "/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def run_eval(
        app_name: str, eval_set_id: str, req: RunEvalRequest
    ) -> list[RunEvalResult]:
      """Runs an eval given the details in the eval request."""
      # Create a mapping from eval set file to all the evals that needed to be
      # run.
      try:
        from ..evaluation.local_eval_service import LocalEvalService
        from .cli_eval import _collect_eval_results
        from .cli_eval import _collect_inferences

        eval_set = self.eval_sets_manager.get_eval_set(app_name, eval_set_id)

        if not eval_set:
          raise HTTPException(
              status_code=400, detail=f"Eval set `{eval_set_id}` not found."
          )

        root_agent = self.agent_loader.load_agent(app_name)

        eval_case_results = []

        eval_service = LocalEvalService(
            root_agent=root_agent,
            eval_sets_manager=self.eval_sets_manager,
            eval_set_results_manager=self.eval_set_results_manager,
            session_service=self.session_service,
            artifact_service=self.artifact_service,
        )
        inference_request = InferenceRequest(
            app_name=app_name,
            eval_set_id=eval_set.eval_set_id,
            eval_case_ids=req.eval_ids,
            inference_config=InferenceConfig(),
        )
        inference_results = await _collect_inferences(
            inference_requests=[inference_request], eval_service=eval_service
        )

        eval_case_results = await _collect_eval_results(
            inference_results=inference_results,
            eval_service=eval_service,
            eval_metrics=req.eval_metrics,
        )
      except ModuleNotFoundError as e:
        logger.exception("%s", e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

      run_eval_results = []
      for eval_case_result in eval_case_results:
        run_eval_results.append(
            RunEvalResult(
                eval_set_file=eval_case_result.eval_set_file,
                eval_set_id=eval_set_id,
                eval_id=eval_case_result.eval_id,
                final_eval_status=eval_case_result.final_eval_status,
                overall_eval_metric_results=eval_case_result.overall_eval_metric_results,
                eval_metric_result_per_invocation=eval_case_result.eval_metric_result_per_invocation,
                user_id=eval_case_result.user_id,
                session_id=eval_case_result.session_id,
            )
        )

      return run_eval_results

    @app.get(
        "/apps/{app_name}/eval_results/{eval_result_id}",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def get_eval_result(
        app_name: str,
        eval_result_id: str,
    ) -> EvalSetResult:
      """Gets the eval result for the given eval id."""
      try:
        return self.eval_set_results_manager.get_eval_set_result(
            app_name, eval_result_id
        )
      except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve)) from ve
      except ValidationError as ve:
        raise HTTPException(status_code=500, detail=str(ve)) from ve

    @app.get(
        "/apps/{app_name}/eval_results",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_results(app_name: str) -> list[str]:
      """Lists all eval results for the given app."""
      return self.eval_set_results_manager.list_eval_set_results(app_name)

    @app.get(
        "/apps/{app_name}/eval_metrics",
        response_model_exclude_none=True,
        tags=[TAG_EVALUATION],
    )
    async def list_eval_metrics(app_name: str) -> list[MetricInfo]:
      """Lists all eval metrics for the given app."""
      try:
        from ..evaluation.metric_evaluator_registry import DEFAULT_METRIC_EVALUATOR_REGISTRY

        # Right now we ignore the app_name as eval metrics are not tied to the
        # app_name, but they could be moving forward.
        return DEFAULT_METRIC_EVALUATOR_REGISTRY.get_registered_metrics()
      except ModuleNotFoundError as e:
        logger.exception("%s\n%s", MISSING_EVAL_DEPENDENCIES_MESSAGE, e)
        raise HTTPException(
            status_code=400, detail=MISSING_EVAL_DEPENDENCIES_MESSAGE
        ) from e

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

    @app.post("/run", response_model_exclude_none=True)
    async def run_agent(req: RunAgentRequest) -> list[Event]:
      session = await self.session_service.get_session(
          app_name=req.app_name, user_id=req.user_id, session_id=req.session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")
      runner = await self.get_runner_async(req.app_name)
      async with Aclosing(
          runner.run_async(
              user_id=req.user_id,
              session_id=req.session_id,
              new_message=req.new_message,
          )
      ) as agen:
        events = [event async for event in agen]
      logger.info("Generated %s events in agent run", len(events))
      logger.debug("Events generated: %s", events)
      return events

    @app.post("/run_sse")
    async def run_agent_sse(req: RunAgentRequest) -> StreamingResponse:
      # SSE endpoint
      session = await self.session_service.get_session(
          app_name=req.app_name, user_id=req.user_id, session_id=req.session_id
      )
      if not session:
        raise HTTPException(status_code=404, detail="Session not found")

      # Convert the events to properly formatted SSE
      async def event_generator():
        try:
          stream_mode = (
              StreamingMode.SSE if req.streaming else StreamingMode.NONE
          )
          runner = await self.get_runner_async(req.app_name)
          async with Aclosing(
              runner.run_async(
                  user_id=req.user_id,
                  session_id=req.session_id,
                  new_message=req.new_message,
                  state_delta=req.state_delta,
                  run_config=RunConfig(streaming_mode=stream_mode),
              )
          ) as agen:
            async for event in agen:
              # Format as SSE data
              sse_event = event.model_dump_json(
                  exclude_none=True, by_alias=True
              )
              logger.debug(
                  "Generated event in agent run streaming: %s", sse_event
              )
              yield f"data: {sse_event}\n\n"
        except Exception as e:
          logger.exception("Error in event_generator: %s", e)
          # You might want to yield an error event here
          yield f'data: {{"error": "{str(e)}"}}\n\n'

      # Returns a streaming response with the proper media type for SSE
      return StreamingResponse(
          event_generator(),
          media_type="text/event-stream",
      )

    @app.get(
        "/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
        response_model_exclude_none=True,
        tags=[TAG_DEBUG],
    )
    async def get_event_graph(
        app_name: str, user_id: str, session_id: str, event_id: str
    ):
      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      session_events = session.events if session else []
      event = next((x for x in session_events if x.id == event_id), None)
      if not event:
        return {}

      function_calls = event.get_function_calls()
      function_responses = event.get_function_responses()
      root_agent = self.agent_loader.load_agent(app_name)
      dot_graph = None
      if function_calls:
        function_call_highlights = []
        for function_call in function_calls:
          from_name = event.author
          to_name = function_call.name
          function_call_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_call_highlights
          )
      elif function_responses:
        function_responses_highlights = []
        for function_response in function_responses:
          from_name = function_response.name
          to_name = event.author
          function_responses_highlights.append((from_name, to_name))
          dot_graph = await agent_graph.get_agent_graph(
              root_agent, function_responses_highlights
          )
      else:
        from_name = event.author
        to_name = ""
        dot_graph = await agent_graph.get_agent_graph(
            root_agent, [(from_name, to_name)]
        )
      if dot_graph and isinstance(dot_graph, graphviz.Digraph):
        return GetEventGraphResult(dot_src=dot_graph.source)
      else:
        return {}

    @app.websocket("/run_live")
    async def run_agent_live(
        websocket: WebSocket,
        app_name: str,
        user_id: str,
        session_id: str,
        modalities: List[Literal["TEXT", "AUDIO"]] = Query(
            default=["TEXT", "AUDIO"]
        ),  # Only allows "TEXT" or "AUDIO"
    ) -> None:
      await websocket.accept()

      session = await self.session_service.get_session(
          app_name=app_name, user_id=user_id, session_id=session_id
      )
      if not session:
        # Accept first so that the client is aware of connection establishment,
        # then close with a specific code.
        await websocket.close(code=1002, reason="Session not found")
        return

      live_request_queue = LiveRequestQueue()

      async def forward_events():
        runner = await self.get_runner_async(app_name)
        async with Aclosing(
            runner.run_live(
                session=session, live_request_queue=live_request_queue
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
        logger.info("Client disconnected during process_messages.")
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

    if web_assets_dir:
      import mimetypes

      mimetypes.add_type("application/javascript", ".js", True)
      mimetypes.add_type("text/javascript", ".js", True)

      @app.get("/")
      async def redirect_root_to_dev_ui():
        return RedirectResponse("/dev-ui/")

      @app.get("/dev-ui")
      async def redirect_dev_ui_add_slash():
        return RedirectResponse("/dev-ui/")

      app.mount(
          "/dev-ui/",
          StaticFiles(directory=web_assets_dir, html=True, follow_symlink=True),
          name="static",
      )

    return app
