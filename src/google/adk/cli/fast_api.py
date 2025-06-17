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

from contextlib import asynccontextmanager
import mimetypes
from pathlib import Path
from typing import List
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from starlette.types import Lifespan

from .adk_dev_server import AdkDevServer

# LINT.IfChange
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
  adk_dev_server = AdkDevServer(
      agents_dir=agents_dir,
      session_service_uri=session_service_uri,
      artifact_service_uri=artifact_service_uri,
      memory_service_uri=memory_service_uri,
      trace_to_cloud=trace_to_cloud,
  )

  # Set up tracing in the FastAPI server.
  trace.set_tracer_provider(adk_dev_server.provider)

  @asynccontextmanager
  async def internal_lifespan(app: FastAPI):
    try:
      if lifespan:
        async with lifespan(app) as lifespan_context:
          yield lifespan_context
      else:
        yield
    finally:
      await adk_dev_server.close()

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

  # Wire up the endpoints
  app.get("/list-apps")(adk_dev_server.list_apps)
  app.get("/debug/trace/{event_id}")(adk_dev_server.get_trace_dict)
  app.get("/debug/trace/session/{session_id}")(adk_dev_server.get_session_trace)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.get_session)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions",
      response_model_exclude_none=True,
  )(adk_dev_server.list_sessions)
  app.post(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.create_session_with_id)
  app.post(
      "/apps/{app_name}/users/{user_id}/sessions",
      response_model_exclude_none=True,
  )(adk_dev_server.create_session)
  app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.create_eval_set)
  app.get(
      "/apps/{app_name}/eval_sets",
      response_model_exclude_none=True,
  )(adk_dev_server.list_eval_sets)
  app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/add_session",
      response_model_exclude_none=True,
  )(adk_dev_server.add_session_to_eval_set)
  app.get(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals",
      response_model_exclude_none=True,
  )(adk_dev_server.list_evals_in_eval_set)
  app.get(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.get_eval)
  app.put(
      "/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.update_eval)
  app.delete("/apps/{app_name}/eval_sets/{eval_set_id}/evals/{eval_case_id}")(
      adk_dev_server.delete_eval
  )
  app.post(
      "/apps/{app_name}/eval_sets/{eval_set_id}/run_eval",
      response_model_exclude_none=True,
  )(adk_dev_server.run_eval)
  app.get(
      "/apps/{app_name}/eval_results/{eval_result_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.get_eval_result)
  app.get(
      "/apps/{app_name}/eval_results",
      response_model_exclude_none=True,
  )(adk_dev_server.list_eval_results)
  app.delete("/apps/{app_name}/users/{user_id}/sessions/{session_id}")(
      adk_dev_server.delete_session
  )
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
      response_model_exclude_none=True,
  )(adk_dev_server.load_artifact)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions/{version_id}",
      response_model_exclude_none=True,
  )(adk_dev_server.load_artifact_version)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts",
      response_model_exclude_none=True,
  )(adk_dev_server.list_artifact_names)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}/versions",
      response_model_exclude_none=True,
  )(adk_dev_server.list_artifact_versions)
  app.delete(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/artifacts/{artifact_name}",
  )(adk_dev_server.delete_artifact)
  app.post("/run", response_model_exclude_none=True)(adk_dev_server.agent_run)
  app.post("/run_sse")(adk_dev_server.agent_run_sse)
  app.get(
      "/apps/{app_name}/users/{user_id}/sessions/{session_id}/events/{event_id}/graph",
      response_model_exclude_none=True,
  )(adk_dev_server.get_event_graph)
  app.websocket("/run_live")(adk_dev_server.agent_live_run)

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
# LINT.ThenChange(//depot/google3/learning/agents/orcas/devtools/fast_api_1p.py)
