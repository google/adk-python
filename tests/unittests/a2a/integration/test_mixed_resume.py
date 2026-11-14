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

"""Exercise mixed long-running tool responses through real local A2A HTTP."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import socket

from a2a.client.client_factory import ClientFactory
from a2a.server.context import ServerCallContext
from a2a.server.tasks import InMemoryTaskStore
from google.adk.a2a import _compat
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.agents import LlmAgent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.utils._agent_mode import AgentMode
from google.genai import types
import httpx
import pytest
from starlette.applications import Starlette
import uvicorn

from ... import testing_utils


@asynccontextmanager
async def _local_http(
    app: Starlette, sock: socket.socket
) -> AsyncIterator[None]:
  server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
  task = asyncio.create_task(server.serve(sockets=[sock]))
  try:
    async with asyncio.timeout(10):
      while not server.started:
        if task.done():
          await task
          raise RuntimeError("HTTP server exited before startup")
        await asyncio.sleep(0)
    yield
  finally:
    server.should_exit = True
    try:
      await asyncio.wait_for(task, 10)
    finally:
      if not task.done():
        task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      sock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_first", [True, False])
async def test_mixed_long_running_responses_resume_original_task(
    peer_first: bool,
) -> None:
  """Mixed replies resume the original remote task regardless of part order.

  Setup: a coordinator runs a local long-running tool and delegates to a peer
    Runner over local HTTP; the peer pauses on its own long-running tool.
  Act: answer both tools in one user message, using their actual call IDs.
  Assert: only the peer response crosses the wire, the original task completes,
    and both responses remain in the coordinator's session.
  """
  calls = {"local": 0, "peer": 0}

  def local_approval() -> dict[str, str]:
    """Request approval for the local operation."""
    calls["local"] += 1
    return {"status": "pending"}

  def peer_approval() -> dict[str, str]:
    """Request approval for the peer operation."""
    calls["peer"] += 1
    return {"status": "pending"}

  peer_model = testing_utils.MockModel.create(
      responses=[
          types.Part.from_function_call(name="peer_approval", args={}),
          types.Part.from_text(text="Waiting for peer approval"),
          types.Part.from_function_call(
              name="finish_task", args={"result": "peer completed"}
          ),
      ]
  )
  peer = LlmAgent(
      name="peer",
      mode=AgentMode.TASK,
      model=peer_model,
      tools=[LongRunningFunctionTool(peer_approval)],
  )
  peer_sessions = InMemorySessionService()
  peer_runner = Runner(
      app_name="peer_app",
      agent=peer,
      session_service=peer_sessions,
      auto_create_session=True,
  )
  store = InMemoryTaskStore()
  sock = socket.socket()
  sock.bind(("127.0.0.1", 0))
  sock.listen()
  url = f"http://127.0.0.1:{sock.getsockname()[1]}"
  card_data = {
      "name": "remote_agent",
      "description": "Local peer requiring approval",
      "capabilities": {"streaming": True},
      "version": "0.0.1",
      "defaultInputModes": ["text/plain"],
      "defaultOutputModes": ["text/plain"],
      "skills": [{
          "id": "approval",
          "name": "Approval",
          "description": "Request approval for the peer operation",
          "tags": ["approval"],
      }],
  }
  executor = A2aAgentExecutor(runner=peer_runner, force_new_version=True)
  if _compat.IS_A2A_V1:
    card_data["supportedInterfaces"] = [
        {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}
    ]
    card = _compat.parse_agent_card(card_data)
    app = Starlette()
    _compat.attach_a2a_routes_to_app(
        app, agent_card=card, agent_executor=executor, task_store=store
    )
  else:
    from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
    from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler

    card_data["url"] = url
    card = _compat.parse_agent_card(card_data)
    app = A2AFastAPIApplication(
        agent_card=card,
        http_handler=DefaultRequestHandler(
            agent_executor=executor, task_store=store
        ),
    ).build()
  sent = []

  async def record_request(request: httpx.Request) -> None:
    if request.method == "POST":
      body = json.loads(request.content)
      if "message" in body.get("params", {}):
        sent.append(body["params"]["message"])

  async with _local_http(app, sock):
    async with httpx.AsyncClient(
        timeout=10, event_hooks={"request": [record_request]}
    ) as client:
      factory = ClientFactory(
          config=_compat.make_client_config(
              httpx_client=client, streaming=False, polling=False
          )
      )
      remote = RemoteA2aAgent(
          name="remote_agent",
          agent_card=card,
          a2a_client_factory=factory,
          use_legacy=False,
          mode=AgentMode.TASK,
      )
      coordinator_model = testing_utils.MockModel.create(
          responses=[
              [
                  types.Part.from_function_call(name="local_approval", args={}),
                  types.Part.from_function_call(
                      name="remote_agent", args={"request": "peer operation"}
                  ),
              ],
              types.Part.from_text(text="All operations completed"),
          ]
      )
      coordinator = LlmAgent(
          name="coordinator",
          model=coordinator_model,
          tools=[LongRunningFunctionTool(local_approval)],
          sub_agents=[remote],
      )
      sessions = InMemorySessionService()
      await sessions.create_session(
          app_name="client_app", user_id="user", session_id="session"
      )
      runner = Runner(
          app_name="client_app", agent=coordinator, session_service=sessions
      )
      first_events = [
          event
          async for event in runner.run_async(
              user_id="user",
              session_id="session",
              new_message=types.Content(
                  role="user", parts=[types.Part.from_text(text="Run both")]
              ),
          )
      ]
      assert calls == {"local": 1, "peer": 1}
      by_name = {
          call.name: call
          for event in first_events
          for call in event.get_function_calls()
      }
      remote_event = next(
          event
          for event in first_events
          if any(
              call.name == "peer_approval"
              for call in event.get_function_calls()
          )
      )
      task_id = remote_event.custom_metadata["a2a:task_id"]
      context_id = remote_event.custom_metadata["a2a:context_id"]
      original = await store.get(task_id, context=ServerCallContext())
      assert original.status.state == _compat.TS_INPUT_REQUIRED
      responses = [
          types.Part(
              function_response=types.FunctionResponse(
                  id=by_name[name].id, name=name, response={"approved": True}
              )
          )
          for name in (
              ("peer_approval", "local_approval")
              if peer_first
              else ("local_approval", "peer_approval")
          )
      ]
      second_events = [
          event
          async for event in runner.run_async(
              user_id="user",
              session_id="session",
              new_message=types.Content(role="user", parts=responses),
          )
      ]
      original = await store.get(task_id, context=ServerCallContext())
      assert len(sent) == 2
      assert sent[1].get("taskId") == task_id
      assert sent[1].get("contextId") == context_id
      assert len(sent[1]["parts"]) == 1
      assert sent[1]["parts"][0]["data"]["id"] == by_name["peer_approval"].id
      assert original.status.state == _compat.TS_COMPLETED
      assert not any(event.error_code for event in second_events)
      assert calls == {"local": 1, "peer": 1}
      session = await sessions.get_session(
          app_name="client_app", user_id="user", session_id="session"
      )
      # Filtering the wire resume must not remove the coordinator's local
      # result from the original mixed user event.
      assert any(
          event.author == "user"
          and event.get_function_responses()
          == [part.function_response for part in responses]
          for event in session.events
      )
