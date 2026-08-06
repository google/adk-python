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

"""A FastAPI client for interacting with ADK remote agents and handling GCP authentication."""

import asyncio
import base64
import importlib
import json
import os
import sys
import traceback
from typing import Optional
import uuid

from fastapi import FastAPI
from fastapi import Request
from fastapi import Response
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk.auth import AuthConfig
from google.adk.runners import InMemoryRunner
from google.api_core.client_options import ClientOptions
import google.auth
import google.auth.transport.requests
from google.cloud.agentidentitycredentials_v1 import AuthProviderCredentialsServiceClient
from google.cloud.agentidentitycredentials_v1 import FinalizeCredentialsRequest
from google.genai import types
from pydantic import BaseModel
import uvicorn
import vertexai

# Add agent project directory to path to allow importing local agents
AGENT_PROJECT_DIR = os.environ.get("AGENT_PROJECT_DIR") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if AGENT_PROJECT_DIR not in sys.path:
  sys.path.append(AGENT_PROJECT_DIR)

app = FastAPI()

# Global cache for local runners to persist session history
local_runners = {}

# Mount static files
try:
  app.mount("/static", StaticFiles(directory="static"), name="static")
  print("Successfully mounted /static")
except Exception as e:
  print(f"Error mounting /static: {e}")


# Serve the index page for the root path
@app.get("/")
async def get_index():
  try:
    return FileResponse("static/index.html")
  except Exception as e:
    print(f"Error serving static/index.html: {e}")
    return {"error": str(e)}, 500


# Helper function to stream SSE error messages
def stream_error(msg: str, tb: str = None):
  async def err_gen():
    payload = {"error": msg}
    if tb:
      payload["traceback"] = tb
    yield f"data: {json.dumps(payload)}\n\n"

  return StreamingResponse(err_gen(), media_type="text/event-stream")


# List local agents in the agent project directory (e.g. agent.py)
@app.get("/list_local_agents")
async def list_local_agents():
  try:
    agents = [
        {
            "id": f[:-3],
            "name": f[:-3].replace("_", " ").title(),
            "import_path": f[:-3],
        }
        for f in os.listdir(AGENT_PROJECT_DIR)
        if f.endswith(".py") and not f.startswith(".") and f != "__init__.py"
    ]
    return {"agents": agents}
  except Exception as e:
    return {"error": str(e)}


# List remote agents in the given project and location.
@app.get("/list_agents")
async def list_remote_agents(project_id: str, location: str):
  try:
    client = vertexai.Client(project=project_id, location=location)
    return {
        "agents": [
            {
                "id": a.api_resource.name.split("/")[-1],
                "name": a.api_resource.display_name,
                "full_name": a.api_resource.name,
            }
            for a in client.agent_engines.list()
        ]
    }
  except Exception as e:
    return {"error": str(e)}


# Helper function to extract the auth URI and nonce from the auth config
def handle_adk_request_credential(auth_config):
  if (
      auth_config.exchanged_auth_credential
      and auth_config.exchanged_auth_credential.oauth2
  ):
    oauth2 = auth_config.exchanged_auth_credential.oauth2
    return oauth2.auth_uri, oauth2.nonce
  return None, None


try:
  _, default_project = google.auth.default()
except Exception:
  default_project = ""


class ChatRequest(BaseModel):
  message: str = ""
  agent_type: str = "remote"
  local_agent: str = ""
  project_id: Optional[str] = os.environ.get(
      "GOOGLE_CLOUD_PROJECT", default_project or ""
  )
  location: Optional[str] = os.environ.get("GOOGLE_CLOUD_LOCATION", "")
  agent_id: Optional[str] = os.environ.get("AGENT_ID", "")
  user_id: str = "default_user_id"
  session_id: Optional[str] = None
  is_auth_resume: Optional[bool] = False
  auth_config: Optional[dict] = None
  auth_request_function_call_id: Optional[str] = None


# Endpoint for querying the agent.
@app.post("/chat")
async def chat(request: ChatRequest, response: Response):
  session_id = request.session_id or str(uuid.uuid4())
  current_agent = None
  client = None
  local_runner = None

  if request.agent_type == "local":
    if not request.local_agent:
      return stream_error("No local agent specified.")

    # Validate that the local agent exists in the project directory
    agent_file = os.path.join(AGENT_PROJECT_DIR, f"{request.local_agent}.py")
    if not os.path.exists(agent_file):
      return stream_error(
          f"Local agent module {request.local_agent} not found in"
          f" {AGENT_PROJECT_DIR}."
      )

    try:
      # Use cached runner if available to persist session history
      if request.local_agent in local_runners:
        local_runner = local_runners[request.local_agent]
      else:
        module = importlib.import_module(request.local_agent)
        app_obj = getattr(module, "app", None)
        if not app_obj:
          return stream_error(
              f"Local agent module {request.local_agent} has no app attribute."
          )
        local_runner = InMemoryRunner(app=app_obj)
        local_runner.auto_create_session = True
        local_runners[request.local_agent] = local_runner
    except Exception as e:
      return stream_error(
          f"Failed to load local agent {request.local_agent}: {e}",
          traceback.format_exc(),
      )
  else:
    client = vertexai.Client(
        project=request.project_id, location=request.location
    )
    remote_name = (
        f"projects/{request.project_id}/locations/{request.location}"
        f"/reasoningEngines/{request.agent_id}"
    )
    try:
      current_agent = client.agent_engines.get(name=remote_name)
    except Exception as e:
      return stream_error(
          f"Failed to load remote agent: {e}", traceback.format_exc()
      )

    if not request.session_id and current_agent:
      try:
        session_obj = (
            await current_agent.async_create_session(user_id=request.user_id)
            if hasattr(current_agent, "async_create_session")
            else current_agent.create_session(user_id=request.user_id)
        )
        session_id = (
            getattr(session_obj, "id", None)
            or (
                session_obj.get("id") if isinstance(session_obj, dict) else None
            )
            or session_id
        )
        client = vertexai.Client(
            project=request.project_id, location=request.location
        )
        current_agent = client.agent_engines.get(name=remote_name)
      except Exception as e:
        return stream_error(
            f"Failed to create session: {e}", traceback.format_exc()
        )

  response.set_cookie(
      key="session_id", value=session_id, httponly=True, samesite="lax"
  )

  def process_agent_event(event):
    if hasattr(event, "model_dump"):
      event_data = (
          event.model_dump(mode="json")
          if "mode" in event.model_dump.__code__.co_varnames
          else event.model_dump()
      )
    elif hasattr(event, "dict"):
      event_data = event.dict()
    elif hasattr(event, "to_dict"):
      event_data = event.to_dict()
    elif isinstance(event, dict):
      event_data = event
    else:
      try:
        event_data = json.loads(json.dumps(event, default=lambda o: o.__dict__))
      except Exception:
        event_data = {"text": str(event)}

    content = event_data.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    long_running = event_data.get("long_running_tool_ids") or event_data.get(
        "longRunningToolIds", []
    )

    for part in parts:
      fc = (
          (part.get("function_call") or part.get("functionCall"))
          if isinstance(part, dict)
          else None
      )
      if fc and fc.get("name") == "adk_request_credential":
        fc_id = fc.get("id")
        if not long_running or fc_id in long_running:
          try:
            args = fc.get("args", {})
            cfg_data = args.get("authConfig") or args.get("auth_config")
            if cfg_data:
              auth_config = (
                  AuthConfig.model_validate(cfg_data)
                  if isinstance(cfg_data, dict)
                  else cfg_data
              )
              auth_uri, consent_nonce = handle_adk_request_credential(
                  auth_config
              )
              if auth_uri:
                event_data.update({
                    "popup_auth_uri": auth_uri,
                    "auth_request_function_call_id": fc_id,
                    "auth_config": (
                        auth_config.model_dump()
                        if hasattr(auth_config, "model_dump")
                        else (
                            auth_config.dict()
                            if hasattr(auth_config, "dict")
                            else auth_config
                        )
                    ),
                    "consent_nonce": consent_nonce,
                })
          except Exception as e:
            print(f"Error processing auth wrapper: {e}")
          break

    return event_data

  async def event_generator():
    _ = client
    yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    if (
        request.is_auth_resume
        and request.auth_request_function_call_id
        and request.auth_config
    ):
      message_to_send = types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id=request.auth_request_function_call_id,
                      name="adk_request_credential",
                      response=request.auth_config,
                  )
              )
          ],
      )
    else:
      message_to_send = types.Content(
          role="user", parts=[types.Part(text=request.message)]
      )

    try:
      if request.agent_type == "local" and local_runner:
        async for event in local_runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=message_to_send,
        ):
          yield f"data: {json.dumps(process_agent_event(event))}\n\n"
      elif current_agent:
        dumped_msg = (
            message_to_send.model_dump(exclude_none=True)
            if hasattr(message_to_send, "model_dump")
            else message_to_send.dict(exclude_none=True)
        )
        async for event in current_agent.async_stream_query(
            user_id=request.user_id,
            message=dumped_msg,
            session_id=session_id,
        ):
          yield f"data: {json.dumps(process_agent_event(event))}\n\n"
    except Exception as e:
      yield (
          "data:"
          f" {json.dumps({'error': str(e), 'traceback': traceback.format_exc()})}\n\n"
      )

  return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/validateUserId")
@app.get("/commit")
async def validate_user_id(request: Request):
  # Session data stored in cookies
  user_id = request.cookies.get("consent_user_id") or request.cookies.get(
      "user_id"
  )
  consent_nonce = request.cookies.get("consent_nonce")
  session_id = request.cookies.get("session_id")
  # Query params
  user_id_validation_state = request.query_params.get(
      "user_id_validation_state"
  )
  auth_provider_name = request.query_params.get(
      "connector_name"
  ) or request.query_params.get("auth_provider_name")
  if auth_provider_name:
    auth_provider_name = auth_provider_name.replace(
        "/connectors/", "/authProviders/"
    )

  print(
      f"Callback received: user_id_validation_state={user_id_validation_state},"
      f" auth_provider_name={auth_provider_name}, user_id={user_id}"
  )
  # Note: In production, you should probably throw an if the below checks fail.
  # For this example, we'll just return an error message to the user and 200 OK.
  if not user_id:
    return {
        "status": "error",
        "message": (
            "user_id cookie not found. Please ensure cookies are enabled."
        ),
    }
  if not consent_nonce:
    return {
        "status": "error",
        "message": (
            "consent_nonce cookie not found. Please ensure cookies are enabled."
        ),
    }
  if not user_id_validation_state:
    return {
        "status": "error",
        "message": "user_id_validation_state query param not found",
    }
  if not auth_provider_name:
    return {
        "status": "error",
        "message": "connector_name or auth_provider_name query param not found",
    }

  try:
    state_bytes = base64.urlsafe_b64decode(
        user_id_validation_state + "=" * (-len(user_id_validation_state) % 4)
    )

    client_options = None
    if host := os.environ.get("AGENT_IDENTITY_CREDENTIALS_TARGET_HOST"):
      client_options = ClientOptions(api_endpoint=host)

    client = AuthProviderCredentialsServiceClient(
        client_options=client_options, transport="rest"
    )

    finalize_request = FinalizeCredentialsRequest(
        auth_provider=auth_provider_name,
        user_id=user_id,
        user_id_validation_state=state_bytes,
        consent_nonce=consent_nonce,
    )

    print(
        "Calling FinalizeCredentials via AuthProviderCredentialsServiceClient"
        f" for auth_provider: {auth_provider_name}"
    )
    await asyncio.to_thread(client.finalize_credentials, finalize_request)

    # Return a simple HTML page to indicate OAuth success
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Authorization Successful</title>
    </head>
    <body>
        <p>Authorization successful! You can close this window.</p>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

  except Exception as e:
    print(f"Error finalizing credentials: {e}")
    return {
        "status": "error",
        "message": f"Failed to finalize credentials: {str(e)}",
    }


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8080)
