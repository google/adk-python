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
import google.auth
import google.auth.transport.requests
from google.genai import types
import httpx
from pydantic import BaseModel
import uvicorn
import vertexai

TARGET_HOST = (
    os.environ.get("IAM_CONNECTOR_CREDENTIALS_TARGET_HOST")
    or "iamconnectorcredentials.googleapis.com"
)

app = FastAPI()

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


# List remote agents in the given project and location.
@app.get("/list_agents")
async def list_remote_agents(project_id: str, location: str):
  try:
    client = vertexai.Client(
        project=project_id,
        location=location,
    )
    agents = client.agent_engines.list()
    agent_list = []
    for agent in agents:
      name_parts = agent.api_resource.name.split("/")
      agent_id = name_parts[-1] if len(name_parts) > 0 else ""

      agent_list.append({
          "id": agent_id,
          "name": agent.api_resource.display_name,
          "full_name": agent.api_resource.name,
      })
    return {"agents": agent_list}
  except Exception as e:
    print(f"Error listing agents: {e}")
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
  project_id: str = os.environ.get(
      "GOOGLE_CLOUD_PROJECT", default_project or ""
  )
  location: str = os.environ.get("GOOGLE_CLOUD_LOCATION", "")
  agent_id: str = os.environ.get("AGENT_ID", "")
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

  client = vertexai.Client(
      project=request.project_id,
      location=request.location,
  )
  remote_agent_name = (
      f"projects/{request.project_id}/locations/{request.location}"
      f"/reasoningEngines/{request.agent_id}"
  )
  try:
    current_agent = client.agent_engines.get(name=remote_agent_name)
  except Exception as e:
    import traceback

    tb_str = traceback.format_exc()
    err_str = str(e)

    async def error_generator():
      err_data = {
          "error": f"Failed to load remote agent: {err_str}",
          "traceback": tb_str,
      }
      yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(error_generator(), media_type="text/event-stream")

  if not request.session_id and current_agent:
    try:
      if hasattr(current_agent, "async_create_session"):
        print(f"DEBUG: Creating async session for {request.user_id}")
        session_obj = await current_agent.async_create_session(
            user_id=request.user_id
        )
      else:
        session_obj = current_agent.create_session(user_id=request.user_id)
      session_id = (
          session_obj.id
          if hasattr(session_obj, "id")
          else session_obj.get("id")
      )

      client = vertexai.Client(
          project=request.project_id,
          location=request.location,
      )
      current_agent = client.agent_engines.get(name=remote_agent_name)
    except Exception as e:
      import traceback

      print(f"Failed to create session: {e}")
      tb_str = traceback.format_exc()
      err_str = str(e)

      async def error_generator():
        err_data = {
            "error": f"Failed to create session: {err_str}",
            "traceback": tb_str,
        }
        yield f"data: {json.dumps(err_data)}\n\n"

      return StreamingResponse(
          error_generator(), media_type="text/event-stream"
      )

  response.set_cookie(
      key="session_id", value=session_id, httponly=True, samesite="lax"
  )
  print(f"Set session_id cookie: {session_id}")

  def process_agent_event(event):
    # 1. Normalize the event object into a standard Python dictionary
    # representation.
    if hasattr(event, "model_dump"):
      if "mode" in event.model_dump.__code__.co_varnames:
        event_data = event.model_dump(mode="json")
      else:
        event_data = event.model_dump()
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

    # 2. Extract message content and check for long-running tool calls.
    print(f"DEBUG: event_data: {event_data}")
    content = event_data.get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    long_running = event_data.get("long_running_tool_ids") or event_data.get(
        "longRunningToolIds", []
    )

    # 3. Scan tool calls for the special 'adk_request_credential' wrapper tool.
    for part in parts:
      fc = (
          (part.get("function_call") or part.get("functionCall"))
          if isinstance(part, dict)
          else None
      )
      if fc and fc.get("name") == "adk_request_credential":
        fc_id = fc.get("id")
        if not long_running or fc_id in long_running:
          print("--> Authentication required by agent.")
          try:
            args = fc.get("args", {})
            cfg_data = args.get("authConfig") or args.get("auth_config")
            if cfg_data:
              # Parse auth configuration and extract OAuth URI/nonce for popup.
              if isinstance(cfg_data, dict):
                auth_config = AuthConfig.model_validate(cfg_data)
              else:
                auth_config = cfg_data
              auth_uri, consent_nonce = handle_adk_request_credential(
                  auth_config
              )
              if auth_uri:
                event_data["popup_auth_uri"] = auth_uri
                event_data["auth_request_function_call_id"] = fc_id
                if hasattr(auth_config, "model_dump"):
                  event_data["auth_config"] = auth_config.model_dump()
                elif hasattr(auth_config, "dict"):
                  event_data["auth_config"] = auth_config.dict()
                else:
                  event_data["auth_config"] = auth_config
                event_data["consent_nonce"] = consent_nonce
          except Exception as e:
            print(f"Error processing auth wrapper: {e}")
          break

    return event_data

  async def event_generator():
    # Keep vertexai Client alive during async streaming to prevent httpx client
    # from being closed by GC
    _ = client
    yield f"data: {json.dumps({'session_id': session_id})}\n\n"

    message_to_send = request.message
    if (
        request.is_auth_resume
        and request.auth_request_function_call_id
        and request.auth_config
    ):
      auth_content = types.Content(
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
      message_to_send = auth_content
    else:
      message_to_send = types.Content(
          role="user", parts=[types.Part(text=request.message)]
      )

    try:
      if hasattr(message_to_send, "model_dump"):
        dumped_msg = message_to_send.model_dump(exclude_none=True)
      else:
        dumped_msg = message_to_send.dict(exclude_none=True)

      async for event in current_agent.async_stream_query(
          user_id=request.user_id,
          message=dumped_msg,
          session_id=session_id,
      ):
        event_data = process_agent_event(event)
        yield f"data: {json.dumps(event_data)}\n\n"
    except Exception as e:
      import traceback

      tb_str = traceback.format_exc()
      err_data = {"error": str(e), "traceback": tb_str}
      yield f"data: {json.dumps(err_data)}\n\n"

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
    url = (
        f"https://{TARGET_HOST}/v1alpha/{auth_provider_name}"
        "/credentials:finalize"
    )
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "userId": user_id,
        "userIdValidationState": user_id_validation_state,
        "consentNonce": consent_nonce,
    }

    print(f"Calling FinalizeCredentials via HTTP POST to: {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {payload}")

    async with httpx.AsyncClient() as client:
      response = await client.post(url, json=payload, headers=headers)

    print(f"HTTP Response Status: {response.status_code}")
    print(f"HTTP Response Body: {response.text}")

    if response.status_code == 200:
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
    else:
      return {
          "status": "error",
          "message": f"HTTP Error {response.status_code}: {response.text}",
      }

  except Exception as e:
    print(f"Error calling FinalizeCredentials via HTTP: {e}")
    return {
        "status": "error",
        "message": f"Failed to finalize credentials: {str(e)}",
    }


if __name__ == "__main__":
  uvicorn.run(app, host="127.0.0.1", port=8080)
