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

"""Integration tests for A2A client-server interaction."""

import json
from pathlib import Path
import textwrap
import warnings

from a2a.client.client import ClientConfig as A2AClientConfig
from a2a.client.client_factory import ClientFactory as A2AClientFactory
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import Message as A2AMessage
from a2a.types import Part as A2APart
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TextPart
from a2a.types import TransportProtocol as A2ATransport
from google.adk.agents.remote_a2a_agent import A2A_METADATA_PREFIX
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.events.event import Event
from google.adk.platform import uuid as platform_uuid
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import httpx
import pytest

from .client import create_a2a_client
from .client import create_client
from .server import create_server_app


def _write_fast_api_a2a_agent(
    agents_dir: Path, *, app_name: str, response_text: str
) -> AgentCard:
  """Writes a minimal A2A-enabled agent package for FastAPI tests."""
  agent_dir = agents_dir / app_name
  agent_dir.mkdir(parents=True)

  (agent_dir / "__init__.py").write_text(
      "from . import agent\n",
      encoding="utf-8",
  )
  (agent_dir / "agent.py").write_text(
      textwrap.dedent(
          f"""\
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

          from typing import AsyncGenerator

          from google.adk.agents.base_agent import BaseAgent
          from google.adk.events.event import Event
          from google.genai import types


          class TestA2AAgent(BaseAgent):

            async def _run_async_impl(
                self, ctx
            ) -> AsyncGenerator[Event, None]:
              yield Event(
                  invocation_id=ctx.invocation_id,
                  author=self.name,
                  content=types.Content(
                      role="model",
                      parts=[types.Part(text={response_text!r})],
                  ),
              )


          root_agent = TestA2AAgent(name={app_name!r})
          """
      ),
      encoding="utf-8",
  )

  agent_card = AgentCard(
      name=app_name,
      url=f"http://test/a2a/{app_name}",
      description="Test A2A agent",
      capabilities=AgentCapabilities(
          streaming=True,
          extensions=[
              {"uri": "https://a2a-adk/a2a-extension/new-integration"}
          ],
      ),
      version="0.0.1",
      default_input_modes=["text/plain"],
      default_output_modes=["text/plain"],
      skills=[],
  )
  (agent_dir / "agent.json").write_text(
      json.dumps(agent_card.model_dump(by_alias=True, exclude_none=True)),
      encoding="utf-8",
  )
  return agent_card


def _create_fast_api_remote_client(
    app, agent_card: AgentCard
) -> tuple[RemoteA2aAgent, httpx.AsyncClient]:
  """Creates a RemoteA2aAgent backed by an in-process FastAPI app."""
  httpx_client = httpx.AsyncClient(
      transport=httpx.ASGITransport(app=app),
      base_url="http://test",
  )
  client_config = A2AClientConfig(
      httpx_client=httpx_client,
      streaming=False,
      polling=False,
      supported_transports=[A2ATransport.jsonrpc],
  )
  factory = A2AClientFactory(config=client_config)
  agent = RemoteA2aAgent(
      name="remote_agent",
      agent_card=agent_card,
      a2a_client_factory=factory,
      use_legacy=False,
  )
  return agent, httpx_client


async def _run_fast_api_a2a_round_trip(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app_name: str,
    response_text: str,
    suppress_warnings: bool,
) -> tuple[list[str], list[str]]:
  """Runs a real A2A request through get_fast_api_app()."""
  agents_dir = tmp_path / f"{app_name}_workspace"
  agents_dir.mkdir()
  agent_card = _write_fast_api_a2a_agent(
      agents_dir,
      app_name=app_name,
      response_text=response_text,
  )

  monkeypatch.delenv(
      "ADK_SUPPRESS_A2A_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
  )
  if suppress_warnings:
    monkeypatch.setenv("ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", "1")
  else:
    monkeypatch.delenv(
        "ADK_SUPPRESS_EXPERIMENTAL_FEATURE_WARNINGS", raising=False
    )

  with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    app = get_fast_api_app(agents_dir=str(agents_dir), web=True, a2a=True)
    agent, httpx_client = _create_fast_api_remote_client(app, agent_card)
    try:
      session_service = InMemorySessionService()
      await session_service.create_session(
          app_name="ClientApp",
          user_id="test_user",
          session_id="test_session",
      )
      client_runner = Runner(
          app_name="ClientApp",
          agent=agent,
          session_service=session_service,
      )
      new_message = types.Content(
          parts=[types.Part(text="Hi")], role="user"
      )

      texts = []
      async for event in client_runner.run_async(
          user_id="test_user",
          session_id="test_session",
          new_message=new_message,
      ):
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.text:
              texts.append(part.text)
    finally:
      await httpx_client.aclose()

  experimental_warnings = [
      str(warning.message)
      for warning in caught
      if "[EXPERIMENTAL]" in str(warning.message)
  ]
  return texts, experimental_warnings


@pytest.mark.asyncio
async def test_fast_api_a2a_request_warns_by_default(tmp_path, monkeypatch):
  """A real FastAPI A2A request still emits the request-converter warning."""
  response_text = "Hello from unsuppressed FastAPI A2A"
  texts, experimental_warnings = await _run_fast_api_a2a_round_trip(
      tmp_path=tmp_path,
      monkeypatch=monkeypatch,
      app_name="unsuppressed_a2a_agent",
      response_text=response_text,
      suppress_warnings=False,
  )

  assert texts == [response_text]
  assert any(
      "convert_a2a_request_to_agent_run_request" in warning
      for warning in experimental_warnings
  )


@pytest.mark.asyncio
async def test_fast_api_a2a_request_suppresses_experimental_warnings(
    tmp_path, monkeypatch
):
  """The general suppression env var silences the full FastAPI A2A path."""
  response_text = "Hello from suppressed FastAPI A2A"
  texts, experimental_warnings = await _run_fast_api_a2a_round_trip(
      tmp_path=tmp_path,
      monkeypatch=monkeypatch,
      app_name="suppressed_a2a_agent",
      response_text=response_text,
      suppress_warnings=True,
  )

  assert texts == [response_text]
  assert not experimental_warnings


def create_streaming_mock_run_async(received_requests: list):
  """Creates a mock_run_async that streams multiple chunks."""

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="Hello")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text=" world")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="Hello world")]),
        partial=False,
    )

  return mock_run_async


def create_non_streaming_mock_run_async(received_requests: list):
  """Creates a mock_run_async that returns a single non-streaming event."""

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="Hello world")]),
        partial=False,
    )

  return mock_run_async


@pytest.mark.asyncio
async def test_streaming_adk_to_streaming_a2a():
  """Test streaming of normal text chunks."""
  received_requests = []
  mock_run_async = create_streaming_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=True)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp",
      agent=agent,
      session_service=session_service,
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  texts = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert len(received_requests) == 1
  assert received_requests[0]["session_id"] is not None

  assert texts == ["Hello", " world", "Hello world"]


@pytest.mark.asyncio
async def test_streaming_adk_to_non_streaming_a2a():
  """Test ADK streaming into A2A Non-Streaming."""
  received_requests = []
  mock_run_async = create_streaming_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=False)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp", agent=agent, session_service=session_service
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  texts = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert len(received_requests) == 1
  assert texts == ["Hello world"]


@pytest.mark.asyncio
async def test_non_streaming_adk_to_streaming_a2a():
  """Test ADK Non-Streaming into A2A Streaming."""
  received_requests = []
  mock_run_async = create_non_streaming_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=True)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp", agent=agent, session_service=session_service
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  texts = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert len(received_requests) == 1
  assert texts == ["Hello world"]


@pytest.mark.asyncio
async def test_non_streaming_adk_to_non_streaming_a2a():
  """Test ADK Non-Streaming into A2A Non-Streaming."""
  received_requests = []
  mock_run_async = create_non_streaming_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=False)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp", agent=agent, session_service=session_service
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  texts = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert len(received_requests) == 1
  assert texts == ["Hello world"]


def create_streaming_mock_run_async_with_multiple_agents(
    received_requests: list,
):
  """Creates a mock_run_async that streams multiple chunks."""

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent1",
        content=types.Content(parts=[types.Part(text="Hello")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent2",
        content=types.Content(parts=[types.Part(text=" Hi")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent1",
        content=types.Content(parts=[types.Part(text=" world")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent2",
        content=types.Content(parts=[types.Part(text=" human")]),
        partial=True,
    )
    yield Event(
        author="FakeAgent1",
        content=types.Content(parts=[types.Part(text="Hello world")]),
        partial=False,
    )
    yield Event(
        author="FakeAgent2",
        content=types.Content(parts=[types.Part(text="Hi human")]),
        partial=False,
    )

  return mock_run_async


@pytest.mark.asyncio
async def test_multiple_agents_streaming_adk_to_streaming_a2a():
  """Test streaming multiple agents chunks into A2A Streaming."""
  received_requests = []
  mock_run_async = create_streaming_mock_run_async_with_multiple_agents(
      received_requests
  )

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=True)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp", agent=agent, session_service=session_service
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  texts = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert len(received_requests) == 1
  assert texts == [
      "Hello",
      " Hi",
      " world",
      " human",
      "Hello world",
      "Hi human",
  ]


@pytest.mark.asyncio
async def test_function_calls():
  """Test function call execution from agent."""
  received_requests = []

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_weather",
                        args={"location": "San Francisco"},
                        id="call_1",
                    )
                ),
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_weather",
                        response={"temperature": "22C"},
                        id="call_1",
                    )
                ),
            ],
            role="model",
        ),
    )

  app = create_server_app(mock_run_async)
  agent = create_client(app)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp",
      agent=agent,
      session_service=session_service,
  )

  new_message = types.Content(parts=[types.Part(text="Hi")], role="user")

  func_calls = []
  func_responses = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    func_calls.extend(event.get_function_calls())
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.function_response:
          func_responses.append(p.function_response)

  assert len(func_calls) == 1
  assert func_calls[0].name == "get_weather"
  assert func_calls[0].args == {"location": "San Francisco"}

  assert len(func_responses) == 1
  assert func_responses[0].name == "get_weather"
  assert func_responses[0].response == {"temperature": "22C"}


def create_long_running_mock_run_async(received_requests: list):
  """Creates a mock_run_async for long running function tests."""

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    if len(received_requests) == 1:
      yield Event(
          author="FakeAgent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="long_task", args={}, id="call_long"
                      )
                  )
              ],
              role="model",
          ),
          long_running_tool_ids={"call_long"},
      )
      yield Event(
          author="FakeAgent",
          content=types.Content(
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          name="long_task",
                          response={"status": "pending"},
                          id="call_long",
                      )
                  )
              ],
              role="model",
          ),
      )
    else:
      yield Event(
          author="FakeAgent",
          content=types.Content(
              parts=[types.Part(text="Task completed well")], role="model"
          ),
      )

  return mock_run_async


@pytest.mark.asyncio
async def test_long_running_function_calls_success():
  """Test long running function calls flow success with user response."""
  received_requests = []
  mock_run_async = create_long_running_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  agent = create_client(app, streaming=True)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp",
      agent=agent,
      session_service=session_service,
  )

  new_message_1 = types.Content(parts=[types.Part(text="Hi")], role="user")

  func_calls_1 = []
  func_responses_1 = []
  task_id_1 = ""
  has_long_running_id = False
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message_1
  ):
    if event.custom_metadata:
      task_id_1 = event.custom_metadata.get(
          A2A_METADATA_PREFIX + "task_id", task_id_1
      )
    if (
        event.long_running_tool_ids
        and "call_long" in event.long_running_tool_ids
    ):
      has_long_running_id = True

    func_calls_1.extend(event.get_function_calls())
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.function_response:
          func_responses_1.append(p.function_response)

  assert has_long_running_id
  assert len(func_calls_1) == 1
  assert func_calls_1[0].name == "long_task"

  assert len(func_responses_1) == 1
  assert func_responses_1[0].name == "long_task"
  assert func_responses_1[0].response == {"status": "pending"}

  new_message_2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name="long_task", response={"result": "done"}, id="call_long"
              )
          )
      ],
      role="user",
  )

  texts = []
  task_id_2 = ""
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message_2
  ):
    if event.custom_metadata:
      task_id_2 = event.custom_metadata.get(
          A2A_METADATA_PREFIX + "task_id", task_id_2
      )
    if event.content and event.content.parts:
      for p in event.content.parts:
        if p.text:
          texts.append(p.text)

  assert task_id_1 == task_id_2
  assert "Task completed well" in texts


@pytest.mark.asyncio
async def test_long_running_function_calls_error():
  """Test long running function calls returns error on missing response."""
  received_requests = []
  mock_run_async = create_long_running_mock_run_async(received_requests)

  app = create_server_app(mock_run_async)
  a2a_client = create_a2a_client(app, streaming=False)

  request_1 = A2AMessage(
      message_id=platform_uuid.new_uuid(),
      parts=[A2APart(root=TextPart(text="Hi"))],
      role="user",
  )
  response_1_events = []
  async for event in a2a_client.send_message(request=request_1):
    response_1_events.append(event)

  assert len(response_1_events) == 1
  # Extract task_id from Turn 1 responses
  assert response_1_events[0][1] is None
  task = response_1_events[0][0]
  assert isinstance(task, Task)
  assert task.status.state == TaskState.input_required
  extracted_task_id = task.id
  assert extracted_task_id is not None

  request_2 = A2AMessage(
      message_id=platform_uuid.new_uuid(),
      parts=[A2APart(root=TextPart(text="Any update?"))],
      role="user",
      task_id=extracted_task_id,
      context_id=task.context_id if hasattr(task, "context_id") else None,
  )
  response_2_events = []
  async for event in a2a_client.send_message(request=request_2):
    response_2_events.append(event)

  # Verify that we get an error response for the second request due to missing function response
  assert len(response_2_events) == 1
  assert response_2_events[0][1] is None
  error_response = response_2_events[0][0]
  assert isinstance(error_response, Task)
  assert error_response.status.message.parts[0].root.text == (
      "It was not provided a function response for the function call."
  )


@pytest.mark.asyncio
async def test_user_follow_up():
  """Test multi-turn interaction or follow up with state."""
  received_requests = []

  async def mock_run_async(**kwargs):
    received_requests.append(kwargs)
    # Yield response with custom metadata to test passing back
    yield Event(
        author="FakeAgent",
        content=types.Content(
            parts=[types.Part(text="Follow up response")], role="model"
        ),
        custom_metadata={"server_state": "active"},
    )

  app = create_server_app(mock_run_async)
  agent = create_client(app)

  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp",
      agent=agent,
      session_service=session_service,
  )

  # First Turn
  new_message_1 = types.Content(parts=[types.Part(text="Turn 1")], role="user")
  async for _ in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message_1
  ):
    pass

  # Second Turn
  new_message_2 = types.Content(parts=[types.Part(text="Turn 2")], role="user")
  last_event = None
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message_2
  ):
    last_event = event

  assert len(received_requests) == 2
  # The second request should carry the same session ID as the first
  assert (
      received_requests[1]["session_id"] == received_requests[0]["session_id"]
  )

  assert last_event is not None
