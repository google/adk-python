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

import json
import os
import traceback
from typing import Optional
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk.auth import AuthConfig
import google.auth
from google.auth.transport.requests import Request as AuthRequest
from google.genai import types
import httpx
from pydantic import BaseModel
import uvicorn

# Configured for your deployed Reasoning Engine
REASONING_ENGINE_ID = os.environ.get("AGENT_ID", "1421963203830611968")

TARGET_HOST = (
    os.environ.get("AGENT_IDENTITY_CREDENTIALS_TARGET_HOST")
    or "agentidentitycredentials.googleapis.com"
)

# In-memory cache for consent nonces to bypass third-party cookie restrictions
NONCE_CACHE = {}

app = FastAPI()

try:
  app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
  print(f"Static mount notice: {e}")


@app.get("/")
async def get_index():
  return FileResponse("static/index.html")


def handle_adk_request_credential(auth_config):
  if (
      auth_config.exchanged_auth_credential
      and auth_config.exchanged_auth_credential.oauth2
  ):
    oauth2 = auth_config.exchanged_auth_credential.oauth2
    return oauth2.auth_uri, oauth2.nonce
  return None, None


try:
  credentials, default_project = google.auth.default()
except Exception:
  credentials, default_project = None, "qwiklabs-gcp-03-d061ae25a425"


@app.get("/list_agents")
async def list_remote_agents(project_id: str, location: str):
  default_agent_entry = {
      "id": REASONING_ENGINE_ID,
      "name": "secure-agent-demo",
      "full_name": (
          f"projects/{project_id}/locations/{location}/reasoningEngines/{REASONING_ENGINE_ID}"
      ),
  }

  try:
    auth_req = AuthRequest()
    credentials.refresh(auth_req)
    token = credentials.token

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/reasoningEngines"
    )
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
      resp = await client.get(url, headers=headers)
      data = resp.json()

    agent_list = []
    reasoning_engines = data.get("reasoningEngines", [])
    for re_obj in reasoning_engines:
      full_name = re_obj.get("name", "")
      disp_name = re_obj.get("displayName", "")
      agent_id = full_name.split("/")[-1] if full_name else ""
      agent_list.append({
          "id": agent_id,
          "name": disp_name,
          "full_name": full_name,
      })

    if not agent_list:
      agent_list.append(default_agent_entry)

    return {"agents": agent_list}
  except Exception:
    return {"agents": [default_agent_entry]}


class ChatRequest(BaseModel):
  message: str = ""
  agent_type: str = "remote"
  local_agent: str = ""
  project_id: str = os.environ.get(
      "GOOGLE_CLOUD_PROJECT", default_project or "qwiklabs-gcp-03-d061ae25a425"
  )
  location: str = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
  agent_id: str = REASONING_ENGINE_ID
  user_id: str = "default_user_id"
  session_id: Optional[str] = None
  is_auth_resume: Optional[bool] = False
  auth_config: Optional[dict] = None
  auth_request_function_call_id: Optional[str] = None


@app.post("/chat")
async def chat(request: ChatRequest, response: Response):
  try:
    session_id = request.session_id if request.session_id else None

    raw_user_id = request.user_id or "default_user_id"
    clean_user_id = (
        raw_user_id.split("@")[0] if "@" in raw_user_id else raw_user_id
    )

    response.set_cookie(
        key="consent_user_id",
        value=clean_user_id,
        httponly=True,
        samesite="none",
        secure=True,
    )

    def process_agent_event(event_data):
      if not isinstance(event_data, dict):
        return {"text": str(event_data)}

      content = event_data.get("content", {})
      parts = content.get("parts", []) if isinstance(content, dict) else []
      long_running = event_data.get("long_running_tool_ids") or event_data.get(
          "longRunningToolIds", []
      )

      text_parts = [
          p.get("text", "")
          for p in parts
          if isinstance(p, dict) and "text" in p
      ]
      if text_parts:
        event_data["text"] = "".join(text_parts)

      for part in parts:
        if not isinstance(part, dict):
          continue
        fc = part.get("function_call") or part.get("functionCall")
        if fc:
          fc_name = fc.get("name", "")
          if (
              "credential" in fc_name
              or "adk_request_credential" in fc_name
              or "request_credential" in fc_name
          ):
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
                    if "scope=" not in auth_uri:
                      auth_uri += "&scope=repo"
                    event_data["popup_auth_uri"] = auth_uri
                    event_data["auth_request_function_call_id"] = fc_id
                    event_data["auth_config"] = (
                        auth_config.model_dump()
                        if hasattr(auth_config, "model_dump")
                        else auth_config
                    )
                    event_data["consent_nonce"] = consent_nonce
                    if consent_nonce:
                      NONCE_CACHE["last_nonce"] = consent_nonce
                      NONCE_CACHE[clean_user_id] = consent_nonce
              except Exception as e:
                print(f"Auth wrapper processing notice: {e}")
              break

      return event_data

    async def event_generator():
      if (
          request.is_auth_resume
          and request.auth_request_function_call_id
          and request.auth_config
      ):
        message = types.Content(
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
        message = types.Content(
            role="user", parts=[types.Part(text=request.message)]
        )

      dumped_message = message.model_dump(exclude_none=True)

      payload_input = {
          "class_method": "stream_query",
          "user_id": clean_user_id,
          "message": dumped_message,
      }

      if session_id:
        payload_input["session_id"] = session_id

      payload = {"input": payload_input}

      auth_req = AuthRequest()
      credentials.refresh(auth_req)
      token = credentials.token

      url = (
          f"https://{request.location}-aiplatform.googleapis.com/v1/"
          f"projects/{request.project_id}/locations/{request.location}/"
          f"reasoningEngines/{request.agent_id}:streamQuery"
      )

      headers = {
          "Authorization": f"Bearer {token}",
          "Content-Type": "application/json",
      }

      async with httpx.AsyncClient(timeout=120.0) as http_client:
        async with http_client.stream(
            "POST", url, headers=headers, json=payload
        ) as resp:
          async for line in resp.aiter_lines():
            if not line or not line.strip():
              continue

            clean_line = line.strip()
            if clean_line.startswith("data:"):
              json_text = clean_line[5:].strip()
            else:
              json_text = clean_line

            try:
              raw_json = json.loads(json_text)
              processed = process_agent_event(raw_json)
              yield f"data: {json.dumps(processed)}\n\n"
            except Exception:
              pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

  except Exception as e:
    tb_str = traceback.format_exc()

    async def fail_generator():
      err_data = {"error": f"Server error: {str(e)}", "traceback": tb_str}
      yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(fail_generator(), media_type="text/event-stream")


@app.get("/validateUserId")
@app.get("/commit")
async def validate_user_id(request: Request):
  user_id = (
      request.cookies.get("consent_user_id")
      or request.cookies.get("user_id")
      or "default_user_id"
  )
  consent_nonce = (
      request.cookies.get("consent_nonce")
      or NONCE_CACHE.get(user_id)
      or NONCE_CACHE.get("last_nonce")
  )
  user_id_validation_state = request.query_params.get(
      "user_id_validation_state"
  )
  auth_provider_name = request.query_params.get(
      "auth_provider_name"
  ) or request.query_params.get("connector_name")

  if not user_id or not consent_nonce or not user_id_validation_state:
    return {"status": "error", "message": "Missing required cookies or state"}

  auth_req = AuthRequest()
  credentials.refresh(auth_req)
  token = credentials.token

  url = (
      f"https://{TARGET_HOST}/v1alpha/{auth_provider_name}/credentials:finalize"
  )
  payload = {
      "userId": user_id,
      "userIdValidationState": user_id_validation_state,
      "consentNonce": consent_nonce,
  }
  headers = {
      "Authorization": f"Bearer {token}",
      "Content-Type": "application/json",
  }
  async with httpx.AsyncClient() as client:
    resp = await client.post(url, json=payload, headers=headers)

  if resp.status_code == 200:
    return HTMLResponse(
        "<p>Authorization successful! You can close this window.</p>"
    )
  return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text}"}


if __name__ == "__main__":
  uvicorn.run("main:app", host="127.0.0.1", port=8501, reload=True)
