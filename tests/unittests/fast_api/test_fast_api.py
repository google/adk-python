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

import asyncio
import json
import sys
import threading
import time
import types as ptypes
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import LiveRequest
from google.adk.agents.run_config import RunConfig
from google.adk.cli.fast_api import AgentRunRequest
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils import envs
from google.adk.events import Event
from google.adk.runners import Runner
from google.genai import types
import httpx
import pytest
from uvicorn.main import run as uvicorn_run
import websockets


# Here we “fake” the agent module that get_fast_api_app expects.
# The server code does: `agent_module = importlib.import_module(agent_name)`
# and then accesses: agent_module.agent.root_agent.
class DummyAgent(BaseAgent):
  pass


dummy_module = ptypes.ModuleType("test_agent")
dummy_module.agent = ptypes.SimpleNamespace(
    root_agent=DummyAgent(name="dummy_agent")
)
sys.modules["test_app"] = dummy_module
envs.load_dotenv_for_agent("test_app", ".")

event1 = Event(
    author="dummy agent",
    invocation_id="invocation_id",
    content=types.Content(
        role="model", parts=[types.Part(text="LLM reply", inline_data=None)]
    ),
)

event2 = Event(
    author="dummy agent",
    invocation_id="invocation_id",
    content=types.Content(
        role="model",
        parts=[
            types.Part(
                text=None,
                inline_data=types.Blob(
                    mime_type="audio/pcm;rate=24000", data=b"\x00\xFF"
                ),
            )
        ],
    ),
)

event3 = Event(
    author="dummy agent", invocation_id="invocation_id", interrupted=True
)


# For simplicity, we patch Runner.run_live to yield dummy events.
# We use SimpleNamespace to mimic attribute-access (i.e. event.content.parts).
async def dummy_run_live(
    self, session, live_request_queue
) -> AsyncGenerator[Event, None]:
  # Immediately yield a dummy event with a text reply.
  yield event1
  await asyncio.sleep(0)

  yield event2
  await asyncio.sleep(0)

  yield event3

  raise Exception()


async def dummy_run_async(
    self,
    user_id,
    session_id,
    new_message,
    run_config: RunConfig = RunConfig(),
) -> AsyncGenerator[Event, None]:
  # Immediately yield a dummy event with a text reply.
  yield event1
  await asyncio.sleep(0)

  yield event2
  await asyncio.sleep(0)

  yield event3

  return


###############################################################################
# Pytest fixtures to patch methods and start the server
###############################################################################


@pytest.fixture(autouse=True)
def patch_runner(monkeypatch):
  # Patch the Runner methods to use our dummy implementations.
  monkeypatch.setattr(Runner, "run_live", dummy_run_live)
  monkeypatch.setattr(Runner, "run_async", dummy_run_async)


@pytest.fixture(scope="module", autouse=True)
def start_server():
  """Start the FastAPI server in a background thread."""

  def run_server():
    uvicorn_run(
        get_fast_api_app(agent_dir=".", web=True),
        host="0.0.0.0",
        log_config=None,
    )

  server_thread = threading.Thread(target=run_server, daemon=True)
  server_thread.start()
  # Wait a moment to ensure the server is up.
  time.sleep(2)
  yield
  # The daemon thread will be terminated when tests complete.


@pytest.mark.asyncio
async def test_sse_endpoint():
  base_http_url = "http://127.0.0.1:8000"
  user_id = "test_user"
  session_id = "test_session"

  # Ensure that the session exists (create if necessary).
  url_create = (
      f"{base_http_url}/apps/test_app/users/{user_id}/sessions/{session_id}"
  )
  httpx.post(url_create, json={"state": {}})

  async with httpx.AsyncClient() as client:
    # Make a POST request to the SSE endpoint.
    async with client.stream(
        "POST",
        f"{base_http_url}/run_sse",
        json=json.loads(
            AgentRunRequest(
                app_name="test_app",
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    parts=[types.Part(text="Hello via SSE", inline_data=None)]
                ),
                streaming=False,
            ).model_dump_json(exclude_none=True)
        ),
    ) as response:
      # Ensure the status code and header are as expected.
      assert response.status_code == 200
      assert (
          response.headers.get("content-type")
          == "text/event-stream; charset=utf-8"
      )

      # Iterate over events from the stream.
      event_count = 0
      event_buffer = ""

      async for line in response.aiter_lines():
        event_buffer += line + "\n"

        # An SSE event is terminated by an empty line (double newline)
        if line == "" and event_buffer.strip():
          # Process the complete event
          event_data = None
          for event_line in event_buffer.split("\n"):
            if event_line.startswith("data: "):
              event_data = event_line[6:]  # Remove "data: " prefix

          if event_data:
            event_count += 1
            if event_count == 1:
              assert event_data == event1.model_dump_json(
                  exclude_none=True, by_alias=True
              )
            elif event_count == 2:
              assert event_data == event2.model_dump_json(
                  exclude_none=True, by_alias=True
              )
            elif event_count == 3:
              assert event_data == event3.model_dump_json(
                  exclude_none=True, by_alias=True
              )
            else:
              pass

          # Reset buffer for next event
          event_buffer = ""

      assert event_count == 3  # Expecting 3 events from dummy_run_async


@pytest.mark.asyncio
async def test_websocket_endpoint():
  base_http_url = "http://127.0.0.1:8000"
  base_ws_url = "ws://127.0.0.1:8000"
  user_id = "test_user"
  session_id = "test_session"

  # Ensure that the session exists (create if necessary).
  url_create = (
      f"{base_http_url}/apps/test_app/users/{user_id}/sessions/{session_id}"
  )
  httpx.post(url_create, json={"state": {}})

  ws_url = f"{base_ws_url}/run_live?app_name=test_app&user_id={user_id}&session_id={session_id}"
  async with websockets.connect(ws_url) as ws:
    # --- Test sending text data ---
    text_payload = LiveRequest(
        content=types.Content(
            parts=[types.Part(text="Hello via WebSocket", inline_data=None)]
        )
    )
    await ws.send(text_payload.model_dump_json())
    # Wait for a reply from our dummy_run_live.
    reply = await ws.recv()
    event = Event.model_validate_json(reply)
    assert event.content.parts[0].text == "LLM reply"

    # --- Test sending binary data (allowed mime type "audio/pcm") ---
    sample_audio = b"\x00\xFF"
    binary_payload = LiveRequest(
        blob=types.Blob(
            mime_type="audio/pcm",
            data=sample_audio,
        )
    )
    await ws.send(binary_payload.model_dump_json())
    # Wait for a reply.
    reply = await ws.recv()
    event = Event.model_validate_json(reply)
    assert (
        event.content.parts[0].inline_data.mime_type == "audio/pcm;rate=24000"
    )
    assert event.content.parts[0].inline_data.data == b"\x00\xFF"

    reply = await ws.recv()
    event = Event.model_validate_json(reply)
    assert event.interrupted is True
    assert event.content is None
