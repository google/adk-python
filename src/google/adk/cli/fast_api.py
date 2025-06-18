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
from pathlib import Path
import time
import traceback
import typing
from typing import Any
from typing import List
from typing import Literal
from typing import Optional

import click
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from google.genai import types
import graphviz
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import export
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider
from pydantic import Field
from pydantic import ValidationError
from starlette.types import Lifespan
from typing_extensions import override

from ..agents import RunConfig
from ..agents.live_request_queue import LiveRequest
from ..agents.live_request_queue import LiveRequestQueue
from ..agents.llm_agent import Agent
from ..agents.run_config import StreamingMode
from ..artifacts.gcs_artifact_service import GcsArtifactService
from ..artifacts.in_memory_artifact_service import InMemoryArtifactService
from ..auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from ..errors.not_found_error import NotFoundError
from ..evaluation.eval_case import EvalCase
from ..evaluation.eval_case import SessionInput
from ..evaluation.eval_metrics import EvalMetric
from ..evaluation.eval_metrics import EvalMetricResult
from ..evaluation.eval_metrics import EvalMetricResultPerInvocation
from ..evaluation.eval_result import EvalSetResult
from ..evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from ..evaluation.local_eval_sets_manager import LocalEvalSetsManager
from ..events.event import Event
from ..memory.in_memory_memory_service import InMemoryMemoryService
from ..memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
from ..runners import Runner
from ..sessions.database_session_service import DatabaseSessionService
from ..sessions.in_memory_session_service import InMemorySessionService
from ..sessions.session import Session
from ..sessions.vertex_ai_session_service import VertexAiSessionService
from .cli_eval import EVAL_SESSION_ID_PREFIX
from .cli_eval import EvalStatus
from .utils import cleanup
from .utils import common
from .utils import create_empty_state
from .utils import envs
from .utils import evals
from .utils.agent_loader import AgentLoader

logger = logging.getLogger("google_adk." + __name__)

_EVAL_SET_FILE_EXTENSION = ".evalset.json"


class ApiServerSpanExporter(export.SpanExporter):

  def __init__(self, trace_dict):
    self.trace_dict = trace_dict

  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export.SpanExportResult:
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
    return export.SpanExportResult.SUCCESS

  def force_flush(self, timeout_millis: int = 30000) -> bool:
    return True


class InMemoryExporter(export.SpanExporter):

  def __init__(self, trace_dict):
    super().__init__()
    self._spans = []
    self.trace_dict = trace_dict

  @override
  def export(
      self, spans: typing.Sequence[ReadableSpan]
  ) -> export.SpanExportResult:
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
    return export.SpanExportResult.SUCCESS

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


class AgentRunRequest(common.BaseModel):
  app_name: str
  user_id: str
  session_id: str
  new_message: types.Content
  streaming: bool = False


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


def get_fast_api_app(
    *,
    agents_dir: str,
    session_service_uri: Optional[str] = None,
    artifact_service_uri: Optional[str] = None,
    memory_service_uri: Optional[str] = None,
    allow_origins: Optional[list[str]] = None,
    web: bool,
    trace_to_cloud: bool = False,
    lifespan: Optional[Lifespan[FastAPI]] = None,
) -> FastAPI:
  # InMemory tracing dict.
  trace_dict: dict[str, Any] = {}
  session_trace_dict: dict[str, Any] = {}

  # Set up tracing in the FastAPI server.
  provider = TracerProvider()
  provider.add_span_processor(
      export.SimpleSpanProcessor(ApiServerSpanExporter(trace_dict))
  )
  memory_exporter = InMemoryExporter(session_trace_dict)
  provider.add_span_processor(export.SimpleSpanProcessor(memory_exporter))
  if trace_to_cloud:
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

  trace.set_tracer_provider(provider)

  @asynccontextmanager
  async def internal_lifespan(app: FastAPI):

    try:
      if lifespan:
        async with lifespan(app) as lifespan_context:
          yield lifespan_context
      else:
        yield
    finally:
      # Create tasks for all runner closures to run concurrently
      await cleanup.close_runners(list(runner_dict.values()))

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

  runner_dict = {}

  eval_sets_manager = LocalEvalSetsManager(agents_dir=agents_dir)
  eval_set_results_manager = LocalEvalSetResultsManager(agents_dir=agents_dir)

  # Build the Memory service
  if memory_service_uri:
    if memory_service_uri.startswith("rag://"):
      rag_corpus = memory_service_uri.split("://")[1]
      if not rag_corpus:
        raise click.ClickException("Rag corpus can not be empty.")
      envs.load_dotenv_for_agent("", agents_dir)
      memory_service = VertexAiRagMemoryService(
          rag_corpus=f'projects/{os.environ["GOOGLE_CLOUD_PROJECT"]}/locations/{os.environ["GOOGLE_CLOUD_LOCATION"]}/ragCorpora/{rag_corpus}'
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
      # Create vertex session service
      agent_engine_id = session_service_uri.split("://")[1]
      if not agent_engine_id:
        raise click.ClickException("Agent engine id can not be empty.")
      envs.load_dotenv_for_agent("", agents_dir)
      session_service = VertexAiSessionService(
          project=os.environ["GOOGLE_CLOUD_PROJECT"],
          location=os.environ["GOOGLE_CLOUD_LOCATION"],
          agent_engine_id=agent_engine_id,
      )
    else:
      session_service = DatabaseSessionService(db_url=session_service_uri)
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

  @app.get("/list-apps")
  def list_apps() -> list[str]:
    base_path = Path.cwd() / agents_dir
    if not base_path.exists():
      raise HTTPException(status_code=404, detail="Path not found")
    if not base_path.is_dir():
      raise HTTPException(status_code=400, detail="Not a directory")
    agent_names = [
        x
        for x in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, x))
        and not x.startswith(".")
        and x != "__pycache__"
    ]
    agent_names.sort()
    return agent_names

  @app.get("/debug/trace/{event_id}")
  def get_trace_dict(event_id: str) -> Any:
    event_dict = trace_dict.get(event_id, None)
    if event_dict is None:
      raise HTTPException(status_code=404, detail="Trace not found")
    return event_dict

  @app.get("/debug/trace/session/{session_id}")
  def get_session_trace(session_id: str) -> Any:
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
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")
    return session

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions",
      response_model_exclude_none=True,
  )
  async def list_sessions(app_name: str, user_id: str) -> list[Session]:
    list_sessions_response = await session_service.list_sessions(
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
        await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        is not None
    ):
      logger.warning("Session already exists: %s", session_id)
      raise HTTPException(
          status_code=400, detail=f"Session already exists: {session_id}"
      )
    logger.info("New session created: %s", session_id)
    return await session_service.create_session(
        app_name=app_name, user_id=user_id, state=state, session_id=session_id
    )

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
    logger.info("New session created")
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id, state=state
    )

    if events:
      for event in events:
        await session_service.append_event(session=session, event=event)

    return session

  def _get_eval_set_file_path(app_name, agents_dir, eval_set_id) -> str:
    return os.path.join(
        agents_dir,
        app_name,
        eval_set_id + _EVAL_SET_FILE_EXTENSION,
    )

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}",
      response_model_exclude_none=True,
  )
  def create_eval_set(
      app_name: str,
      eval_set_id: str,
  ):
    """Creates an eval set, given the id."""
    try:
      eval_sets_manager.create_eval_set(app_name, eval_set_id)
    except ValueError as ve:
      raise HTTPException(
          status_code=400,
          detail=str(ve),
      ) from ve

  @app.get(
      "/apps/{app_name}/eval_sets",
      response_model_exclude_none=True,
  )
  def list_eval_sets(app_name: str) -> list[str]:
    """Lists all eval sets for the given app."""
    return eval_sets_manager.list_eval_sets(app_name)

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
      response_model_exclude_none=True,
  )
  async def add_session_to_eval_set(
      app_name: str, eval_set_id: str, req: AddSessionToEvalSetRequest
  ):
    # Get the session
    session = await session_service.get_session(
        app_name=app_name, user_id=req.user_id, session_id=req.session_id
    )
    assert session, "Session not found."

    # Convert the session data to eval invocations
    invocations = evals.convert_session_to_eval_invocations(session)

    # Populate the session with initial session state.
    initial_session_state = create_empty_state(
        agent_loader.load_agent(app_name)
    )

    new_eval_case = EvalCase(
        eval_id=req.eval_id,
        conversation=invocations,
        session_input=SessionInput(
            app_name=app_name, user_id=req.user_id, state=initial_session_state
        ),
        creation_timestamp=time.time(),
    )

    try:
      eval_sets_manager.add_eval_case(app_name, eval_set_id, new_eval_case)
    except ValueError as ve:
      raise HTTPException(status_code=400, detail=str(ve)) from ve

  @app.get(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals",
      response_model_exclude_none=True,
  )
  def list_evals_in_eval_set(
      app_name: str,
      eval_set_id: str,
  ) -> list[str]:
    """Lists all evals in an eval set."""
    eval_set_data = eval_sets_manager.get_eval_set(app_name, eval_set_id)

    if not eval_set_data:
      raise HTTPException(
          status_code=400, detail=f"Eval set `{eval_set_id}` not found."
      )

    return sorted([x.eval_id for x in eval_set_data.eval_cases])

  @app.get(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
      response_model_exclude_none=True,
  )
  def get_eval(app_name: str, eval_set_id: str, eval_case_id: str) -> EvalCase:
    """Gets an eval case in an eval set."""
    eval_case_to_find = eval_sets_manager.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    if eval_case_to_find:
      return eval_case_to_find

    raise HTTPException(
        status_code=404,
        detail=f"Eval set `{eval_set_id}` or Eval `{eval_case_id}` not found.",
    )

  @app.put(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
      response_model_exclude_none=True,
  )
  def update_eval(
      app_name: str,
      eval_set_id: str,
      eval_case_id: str,
      updated_eval_case: EvalCase,
  ):
    if updated_eval_case.eval_id and updated_eval_case.eval_id != eval_case_id:
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
      eval_sets_manager.update_eval_case(
          app_name, eval_set_id, updated_eval_case
      )
    except NotFoundError as nfe:
      raise HTTPException(status_code=404, detail=str(nfe)) from nfe

  @app.delete("/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}")
  def delete_eval(app_name: str, eval_set_id: str, eval_case_id: str):
    try:
      eval_sets_manager.delete_eval_case(app_name, eval_set_id, eval_case_id)
    except NotFoundError as nfe:
      raise HTTPException(status_code=404, detail=str(nfe)) from nfe

  @app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
      response_model_exclude_none=True,
  )
  async def run_eval(
      app_name: str, eval_set_id: str, req: RunEvalRequest
  ) -> list[RunEvalResult]:
    """Runs an eval given the details in the eval request."""
    from .cli_eval import run_evals

    # Create a mapping from eval set file to all the evals that needed to be
    # run.
    eval_set = eval_sets_manager.get_eval_set(app_name, eval_set_id)

    if not eval_set:
      raise HTTPException(
          status_code=400, detail=f"Eval set `{eval_set_id}` not found."
      )

    if req.eval_ids:
      eval_cases = [e for e in eval_set.eval_cases if e.eval_id in req.eval_ids]
      eval_set_to_evals = {eval_set_id: eval_cases}
    else:
      logger.info("Eval ids to run list is empty. We will run all eval cases.")
      eval_set_to_evals = {eval_set_id: eval_set.eval_cases}

    root_agent = agent_loader.load_agent(app_name)
    run_eval_results = []
    eval_case_results = []
    try:
      async for eval_case_result in run_evals(
          eval_set_to_evals,
          root_agent,
          getattr(root_agent, "reset_data", None),
          req.eval_metrics,
          session_service=session_service,
          artifact_service=artifact_service,
      ):
        run_eval_results.append(
            RunEvalResult(
                app_name=app_name,
                eval_set_file=eval_case_result.eval_set_file,
                eval_set_id=eval_set_id,
                eval_id=eval_case_result.eval_id,
                final_eval_status=eval_case_result.final_eval_status,
                eval_metric_results=eval_case_result.eval_metric_results,
                overall_eval_metric_results=eval_case_result.overall_eval_metric_results,
                eval_metric_result_per_invocation=eval_case_result.eval_metric_result_per_invocation,
                user_id=eval_case_result.user_id,
                session_id=eval_case_result.session_id,
            )
        )
        eval_case_result.session_details = await session_service.get_session(
            app_name=app_name,
            user_id=eval_case_result.user_id,
            session_id=eval_case_result.session_id,
        )
        eval_case_results.append(eval_case_result)
    except ModuleNotFoundError as e:
      logger.exception("%s", e)
      raise HTTPException(status_code=400, detail=str(e)) from e

    eval_set_results_manager.save_eval_set_result(
        app_name, eval_set_id, eval_case_results
    )

    return run_eval_results

  @app.get(
      "/apps/{app_name}/eval_results/{eval_result_id}",
      response_model_exclude_none=True,
  )
  def get_eval_result(
      app_name: str,
      eval_result_id: str,
  ) -> EvalSetResult:
    """Gets the eval result for the given eval id."""
    try:
      return eval_set_results_manager.get_eval_set_result(
          app_name, eval_result_id
      )
    except ValueError as ve:
      raise HTTPException(status_code=404, detail=str(ve)) from ve
    except ValidationError as ve:
      raise HTTPException(status_code=500, detail=str(ve)) from ve

  @app.get(
      "/apps/{app_name}/eval_results",
      response_model_exclude_none=True,
  )
  def list_eval_results(app_name: str) -> list[str]:
    """Lists all eval results for the given app."""
    return eval_set_results_manager.list_eval_set_results(app_name)

  @app.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")
  async def delete_session(app_name: str, user_id: str, session_id: str):
    await session_service.delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

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
    artifact = await artifact_service.load_artifact(
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
    artifact = await artifact_service.load_artifact(
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
    return await artifact_service.list_artifact_keys(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

  @app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
      response_model_exclude_none=True,
  )
  async def list_artifact_versions(
      app_name: str, user_id: str, session_id: str, artifact_name: str
  ) -> list[int]:
    return await artifact_service.list_versions(
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
  ):
    await artifact_service.delete_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=artifact_name,
    )

  @app.post("/run", response_model_exclude_none=True)
  async def agent_run(req: AgentRunRequest) -> list[Event]:
    session = await session_service.get_session(
        app_name=req.app_name, user_id=req.user_id, session_id=req.session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")
    runner = await _get_runner_async(req.app_name)
    events = [
        event
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=req.new_message,
        )
    ]
    logger.info("Generated %s events in agent run: %s", len(events), events)
    return events

  @app.post("/run_sse")
  async def agent_run_sse(req: AgentRunRequest) -> StreamingResponse:
    # SSE endpoint
    session = await session_service.get_session(
        app_name=req.app_name, user_id=req.user_id, session_id=req.session_id
    )
    if not session:
      raise HTTPException(status_code=404, detail="Session not found")

    # Convert the events to properly formatted SSE
    async def event_generator():
      try:
        stream_mode = StreamingMode.SSE if req.streaming else StreamingMode.NONE
        runner = await _get_runner_async(req.app_name)
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=req.new_message,
            run_config=RunConfig(streaming_mode=stream_mode),
        ):
          # Format as SSE data
          sse_event = event.model_dump_json(exclude_none=True, by_alias=True)
          logger.info("Generated event in agent run streaming: %s", sse_event)
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
  )
  async def get_event_graph(
      app_name: str, user_id: str, session_id: str, event_id: str
  ):
    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    session_events = session.events if session else []
    event = next((x for x in session_events if x.id == event_id), None)
    if not event:
      return {}

    from . import agent_graph

    function_calls = event.get_function_calls()
    function_responses = event.get_function_responses()
    root_agent = agent_loader.load_agent(app_name)
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
  async def agent_live_run(
      websocket: WebSocket,
      app_name: str,
      user_id: str,
      session_id: str,
      modalities: List[Literal["TEXT", "AUDIO"]] = Query(
          default=["TEXT", "AUDIO"]
      ),  # Only allows "TEXT" or "AUDIO"
  ) -> None:
    await websocket.accept()

    session = await session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
      # Accept first so that the client is aware of connection establishment,
      # then close with a specific code.
      await websocket.close(code=1002, reason="Session not found")
      return

    live_request_queue = LiveRequestQueue()

    async def forward_events():
      runner = await _get_runner_async(app_name)
      async for event in runner.run_live(
          session=session, live_request_queue=live_request_queue
      ):
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

  async def _get_runner_async(app_name: str) -> Runner:
    """Returns the runner for the given app."""
    envs.load_dotenv_for_agent(os.path.basename(app_name), agents_dir)
    if app_name in runner_dict:
      return runner_dict[app_name]
    root_agent = agent_loader.load_agent(app_name)
    runner = Runner(
        app_name=app_name,
        agent=root_agent,
        artifact_service=artifact_service,
        session_service=session_service,
        memory_service=memory_service,
        credential_service=credential_service,
    )
    runner_dict[app_name] = runner
    return runner

  if web:
    import mimetypes

    mimetypes.add_type("application/javascript", ".js", True)
    mimetypes.add_type("text/javascript", ".js", True)

    BASE_DIR = Path(__file__).parent.resolve()
    ANGULAR_DIST_PATH = BASE_DIR / "browser"

    @app.get("/")
    async def redirect_root_to_dev_ui():
      return RedirectResponse("/dev-ui/")

    @app.get("/dev-ui")
    async def redirect_dev_ui_add_slash():
      return RedirectResponse("/dev-ui/")

    app.mount(
        "/dev-ui/",
        StaticFiles(directory=ANGULAR_DIST_PATH, html=True),
        name="static",
    )
  return app
