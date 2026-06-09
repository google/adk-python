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

import json
import logging
import os
import sys

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
import httpx

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout, format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("google_adk." + __name__)

app = FastAPI()

AGENT_URL = os.environ.get("AGENT_BACKEND_URL", "http://localhost:8000")


@app.get("/")
def ui():
  with open("index.html", "r") as f:
    return HTMLResponse(content=f.read())


@app.post("/chat")
async def chat(request: Request):
  data = await request.json()
  message = data.get("message")
  function_response = data.get("function_response")

  app_name = "gcp_auth"
  user_id = data.get("user_id", "test_user")
  session_id = data.get("session_id", "default_session_id")

  payload = {
      "appName": app_name,
      "userId": user_id,
      "sessionId": session_id,
      "streaming": True,
  }

  if message:
    payload["newMessage"] = {
        "role": "user",
        "parts": [{"text": message}],
    }
  elif function_response:
    payload["newMessage"] = {
        "role": "user",
        "parts": [{"functionResponse": function_response}],
    }

  # Ensure the session exists before we try to continue it via /run_sse
  async def proxy_stream():
    async with httpx.AsyncClient(timeout=120.0) as client:
      # Attempt to create the session (ignoring if it already exists or fails quietly)
      await client.post(
          f"{AGENT_URL}/apps/{app_name}/users/{user_id}/sessions/{session_id}"
      )

      async with client.stream(
          "POST", f"{AGENT_URL}/run_sse", json=payload
      ) as r:
        if r.status_code != 200:
          err = await r.aread()
          yield f"data: {json.dumps({'error': err.decode()})}\n\n"
          return

        async for line in r.aiter_lines():
          if line:
            yield f"data: {line}\n\n" if line.startswith("{") else f"{line}\n\n"

  return StreamingResponse(proxy_stream(), media_type="text/event-stream")


@app.api_route("/commit", methods=["GET"])
async def commit(request: Request):
  connector = request.query_params.get("connector_name")
  payload = {
      "userId": request.cookies.get("user_id"),
      "userIdValidationState": request.query_params.get(
          "user_id_validation_state"
      ),
      "consentNonce": request.cookies.get("consent_nonce"),
  }

  url = f"https://iamconnectorcredentials.googleapis.com/v1alpha/{connector}/credentials:finalize"
  try:
    async with httpx.AsyncClient(timeout=30.0) as client:
      resp = await client.post(url, json=payload)
      resp.raise_for_status()
  except httpx.HTTPError as e:
    err_text = e.response.text if hasattr(e, "response") else str(e)
    status = e.response.status_code if hasattr(e, "response") else 500
    logger.error(f"Commit failed: {err_text}")
    return HTMLResponse(err_text, status_code=status)

  return HTMLResponse("""
      <script>
          window.close();
      </script>
      <p>Success. You can close this window.</p>
  """)
