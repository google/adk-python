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

"""End-to-end client-server integration tests for a2a-sdk 1.x."""

from __future__ import annotations

from google.adk.a2a import _compat
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.a2a.executor.interceptors.include_artifacts_in_a2a_event import include_artifacts_in_a2a_event_interceptor
from google.adk.agents.remote_a2a_agent import A2A_METADATA_PREFIX
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.platform import uuid as platform_uuid
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

pytestmark = pytest.mark.skipif(
    not _compat.IS_A2A_V1,
    reason="1.x-only client-server tests (0.3.x covered by test_client_server)",
)

from .client import create_a2a_client
from .client import create_client
from .server import agent_card
from .server import create_server_app_v1


# -----------------------------------------------------------------------------
# Mock server-side agents
# -----------------------------------------------------------------------------
def _streaming_run_async(received_requests: list):
  """A mock run_async that streams partial chunks (text + artifact) then final."""

  async def run_async(**kwargs):
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
        partial=True,
        actions=EventActions(artifact_delta={"file1": 1}),
    )
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="Hello world")]),
        partial=False,
    )

  return run_async


def _non_streaming_run_async(received_requests: list):
  """A mock run_async that yields a single non-partial event."""

  async def run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="Hello world")]),
        partial=False,
    )

  return run_async


def _multi_agent_run_async(received_requests: list):
  """A mock run_async that interleaves two agents' streamed chunks."""

  async def run_async(**kwargs):
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

  return run_async


def _function_call_run_async(received_requests: list):
  """A mock run_async that yields a function call + response pair."""

  async def run_async(**kwargs):
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

  return run_async


def _long_running_run_async(received_requests: list):
  """A mock run_async modeling a long-running tool needing a user response."""

  async def run_async(**kwargs):
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

  return run_async


# -----------------------------------------------------------------------------
# Client driver helpers
# -----------------------------------------------------------------------------
async def _run_client(agent, *, message_text: str = "Hi"):
  """Drives the agent through a client-side Runner and collects results."""
  session_service = InMemorySessionService()
  await session_service.create_session(
      app_name="ClientApp", user_id="test_user", session_id="test_session"
  )
  client_runner = Runner(
      app_name="ClientApp", agent=agent, session_service=session_service
  )
  new_message = types.Content(
      parts=[types.Part(text=message_text)], role="user"
  )

  texts = []
  artifact_deltas = []
  func_calls = []
  func_responses = []
  async for event in client_runner.run_async(
      user_id="test_user", session_id="test_session", new_message=new_message
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text:
          texts.append(part.text)
        if part.function_response:
          func_responses.append(part.function_response)
    func_calls.extend(event.get_function_calls())
    if event.actions and event.actions.artifact_delta:
      artifact_deltas.append(event.actions.artifact_delta)
  return {
      "texts": texts,
      "artifact_deltas": artifact_deltas,
      "func_calls": func_calls,
      "func_responses": func_responses,
  }


# -----------------------------------------------------------------------------
# RemoteA2aAgent round-trip tests (streaming variants)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_non_streaming_round_trip():
  """A non-streaming agent response round-trips to the client on 1.x."""
  received_requests = []
  app = create_server_app_v1(_non_streaming_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=False)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  assert received_requests[0]["session_id"] is not None
  assert "Hello world" in result["texts"]


@pytest.mark.asyncio
async def test_non_streaming_adk_to_streaming_a2a():
  """A non-streaming agent response round-trips over a streaming client."""
  received_requests = []
  app = create_server_app_v1(_non_streaming_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  assert "Hello world" in result["texts"]


@pytest.mark.asyncio
async def test_streaming_round_trip():
  """A streaming agent response delivers its aggregate text on 1.x."""
  received_requests = []
  app = create_server_app_v1(_streaming_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  assert "Hello world" in result["texts"]


@pytest.mark.asyncio
async def test_streaming_adk_to_non_streaming_a2a():
  """A streaming agent response collapses to its final text on a non-streaming client."""
  received_requests = []
  app = create_server_app_v1(_streaming_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=False)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  assert "Hello world" in result["texts"]


@pytest.mark.asyncio
async def test_multiple_agents_streaming_round_trip():
  """Interleaved chunks from multiple server-side agents stream on 1.x."""
  received_requests = []
  app = create_server_app_v1(_multi_agent_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  # The request reached the server and the last author's final text arrives.
  assert "Hi human" in result["texts"]


@pytest.mark.asyncio
async def test_artifact_producing_agent_round_trip():
  """An agent that records an artifact delta still round-trips its content."""
  received_requests = []

  async def run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(parts=[types.Part(text="with artifact")]),
        actions=EventActions(artifact_delta={"file1": 1}),
        partial=False,
    )

  app = create_server_app_v1(run_async)

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  assert "with artifact" in result["texts"]


# -----------------------------------------------------------------------------
# Function-call round trip
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_function_calls():
  """Function call + response pairs round-trip to the client on 1.x."""
  received_requests = []
  app = create_server_app_v1(_function_call_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app)
    result = await _run_client(agent)

  assert len(result["func_calls"]) == 1
  assert result["func_calls"][0].name == "get_weather"
  assert result["func_calls"][0].args == {"location": "San Francisco"}

  assert len(result["func_responses"]) == 1
  assert result["func_responses"][0].name == "get_weather"
  assert result["func_responses"][0].response == {"temperature": "22C"}


# -----------------------------------------------------------------------------
# Long-running flows (multi-turn)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_long_running_function_calls_success():
  """A long-running tool flow completes across two turns, preserving task_id."""
  received_requests = []
  app = create_server_app_v1(_long_running_run_async(received_requests))

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="ClientApp", user_id="test_user", session_id="test_session"
    )
    client_runner = Runner(
        app_name="ClientApp", agent=agent, session_service=session_service
    )

    # Turn 1: triggers the long-running tool, surfaces it to the client.
    new_message_1 = types.Content(parts=[types.Part(text="Hi")], role="user")
    func_calls_1 = []
    task_id_1 = ""
    has_long_running_id = False
    async for event in client_runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=new_message_1,
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

    assert has_long_running_id
    assert any(c.name == "long_task" for c in func_calls_1)

    # Turn 2: provide the function response; task should complete.
    new_message_2 = types.Content(
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="long_task",
                    response={"result": "done"},
                    id="call_long",
                )
            )
        ],
        role="user",
    )
    texts = []
    task_id_2 = ""
    async for event in client_runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=new_message_2,
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
  """A follow-up without a function response yields an INPUT_REQUIRED error."""
  received_requests = []
  app = create_server_app_v1(_long_running_run_async(received_requests))

  async with app.router.lifespan_context(app):
    a2a_client = create_a2a_client(app, streaming=False)

    request_1 = _compat.make_message(
        message_id=platform_uuid.new_uuid(),
        role=_compat.ROLE_USER,
        parts=[_compat.make_text_part("Hi")],
    )
    response_1_events = []
    normalize = _compat.make_stream_normalizer()
    async for item in _compat.send_message(a2a_client, request=request_1):
      response_1_events.append(normalize(item))

    assert len(response_1_events) == 1
    task, update = response_1_events[0]
    assert update is None
    assert task.status.state == _compat.TS_INPUT_REQUIRED
    extracted_task_id = task.id
    assert extracted_task_id

    request_2 = _compat.make_message(
        message_id=platform_uuid.new_uuid(),
        role=_compat.ROLE_USER,
        parts=[_compat.make_text_part("Any update?")],
        task_id=extracted_task_id,
        context_id=task.context_id,
    )
    response_2_events = []
    normalize = _compat.make_stream_normalizer()
    async for item in _compat.send_message(a2a_client, request=request_2):
      response_2_events.append(normalize(item))

  assert len(response_2_events) == 1
  error_task, error_update = response_2_events[0]
  assert error_update is None
  status_message = _compat.normalize_message(error_task.status.message)
  assert status_message is not None
  assert _compat.part_text(status_message.parts[0]) == (
      "It was not provided a function response for the function call."
  )


# -----------------------------------------------------------------------------
# Multi-turn session continuity
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_user_follow_up():
  """Two turns on the same session reuse the same server-side session id."""
  received_requests = []

  async def run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        content=types.Content(
            parts=[types.Part(text="Follow up response")], role="model"
        ),
        custom_metadata={"server_state": "active"},
    )

  app = create_server_app_v1(run_async)

  async with app.router.lifespan_context(app):
    agent = create_client(app)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="ClientApp", user_id="test_user", session_id="test_session"
    )
    client_runner = Runner(
        app_name="ClientApp", agent=agent, session_service=session_service
    )

    new_message_1 = types.Content(
        parts=[types.Part(text="Turn 1")], role="user"
    )
    async for _ in client_runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=new_message_1,
    ):
      pass

    new_message_2 = types.Content(
        parts=[types.Part(text="Turn 2")], role="user"
    )
    last_event = None
    async for event in client_runner.run_async(
        user_id="test_user",
        session_id="test_session",
        new_message=new_message_2,
    ):
      last_event = event

  assert len(received_requests) == 2
  assert (
      received_requests[1]["session_id"] == received_requests[0]["session_id"]
  )
  assert last_event is not None


# -----------------------------------------------------------------------------
# Artifact interceptor
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_include_artifacts_in_a2a_event():
  """The artifact interceptor surfaces recorded artifacts on the task on 1.x."""

  async def run_async(**kwargs):
    yield Event(
        actions=EventActions(artifact_delta={"artifact1": 1, "artifact2": 1}),
        author="agent",
        content=types.Content(
            parts=[types.Part(text="Here are the artifacts")]
        ),
    )

  config = A2aAgentExecutorConfig(
      execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
  )
  app = create_server_app_v1(run_async, config=config)

  async with app.router.lifespan_context(app):
    a2a_client = create_a2a_client(app, streaming=False)

    request = _compat.make_message(
        message_id="test_message_id",
        role=_compat.ROLE_USER,
        parts=[_compat.make_text_part("Hi")],
    )
    events = []
    normalize = _compat.make_stream_normalizer()
    async for item in _compat.send_message(a2a_client, request=request):
      events.append(normalize(item))

  assert len(events) == 1
  task, update = events[0]
  assert update is None
  assert task.artifacts is not None
  # The text part plus the two recorded artifacts.
  assert len(task.artifacts) == 3
  assert _compat.part_text(task.artifacts[0].parts[0]) == (
      "Here are the artifacts"
  )
  artifact_names = {a.name for a in task.artifacts[1:]}
  assert artifact_names == {"artifact1", "artifact2"}
  for art in task.artifacts[1:]:
    assert _compat.part_text(art.parts[0]) == "artifact content"


def _multi_artifact_streaming_run_async(received_requests: list):
  """A streaming agent that records two distinct artifacts across chunks."""

  async def run_async(**kwargs):
    received_requests.append(kwargs)
    yield Event(
        author="FakeAgent",
        partial=True,
        content=types.Content(parts=[types.Part(text="chunk one")]),
        actions=EventActions(artifact_delta={"file1": 1}),
    )
    yield Event(
        author="FakeAgent",
        partial=True,
        content=types.Content(parts=[types.Part(text="chunk two")]),
        actions=EventActions(artifact_delta={"file2": 1}),
    )
    yield Event(
        author="FakeAgent",
        partial=False,
        content=types.Content(parts=[types.Part(text="done")]),
    )

  return run_async


@pytest.mark.asyncio
async def test_streaming_artifacts_are_aggregated_into_single_task():
  """Streaming multi-artifact responses arrive aggregated as one ``task`` item."""
  received_requests = []
  config = A2aAgentExecutorConfig(
      execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
  )
  app = create_server_app_v1(
      _multi_artifact_streaming_run_async(received_requests), config=config
  )

  async with app.router.lifespan_context(app):
    a2a_client = create_a2a_client(app, streaming=True)

    request = _compat.make_message(
        message_id="test_message_id",
        role=_compat.ROLE_USER,
        parts=[_compat.make_text_part("Hi")],
    )
    events = []
    normalize = _compat.make_stream_normalizer()
    async for item in _compat.send_message(a2a_client, request=request):
      events.append(normalize(item))

  # The whole stream collapses to a single aggregated task carrier.
  assert len(events) == 1
  task, update = events[0]
  assert update is None
  assert task.artifacts is not None
  # Both recorded artifacts are present (plus any text-carrying artifact).
  artifact_names = {a.name for a in task.artifacts}
  assert {"file1", "file2"}.issubset(artifact_names)


@pytest.mark.asyncio
async def test_streaming_artifact_run_completes_through_remote_agent():
  """A streaming multi-artifact run round-trips through RemoteA2aAgent"""
  received_requests = []
  config = A2aAgentExecutorConfig(
      execute_interceptors=[include_artifacts_in_a2a_event_interceptor]
  )
  app = create_server_app_v1(
      _multi_artifact_streaming_run_async(received_requests), config=config
  )

  async with app.router.lifespan_context(app):
    agent = create_client(app, streaming=True)
    result = await _run_client(agent)

  assert len(received_requests) == 1
  # The aggregated final response reaches the client (no mid-stream failure).
  assert "done" in result["texts"]


@pytest.mark.asyncio
async def test_make_stream_normalizer_aggregates_incremental_artifacts():
  """The stateful normalizer accumulates artifacts across incremental updates."""
  from a2a.types import a2a_pb2 as pb

  def _artifact_update(name: str) -> pb.StreamResponse:
    item = pb.StreamResponse()
    item.artifact_update.task_id = "task-1"
    item.artifact_update.context_id = "ctx-1"
    item.artifact_update.append = False
    item.artifact_update.last_chunk = True
    artifact = item.artifact_update.artifact
    artifact.artifact_id = name
    artifact.name = name
    artifact.parts.add().text = f"content-{name}"
    return item

  initial = pb.StreamResponse()
  initial.task.id = "task-1"
  initial.task.context_id = "ctx-1"

  stream = [initial, _artifact_update("file1"), _artifact_update("file2")]

  normalize = _compat.make_stream_normalizer()
  results = [normalize(item) for item in stream]

  # The running task accumulates artifacts across updates.
  names_per_step = [{a.name for a in task.artifacts} for task, _ in results]
  assert names_per_step[0] == set()
  assert names_per_step[1] == {"file1"}
  assert names_per_step[2] == {"file1", "file2"}


@pytest.mark.asyncio
async def test_make_stream_normalizer_accumulates_status_history():
  """Status update messages accumulate into task.history; status is applied.

  Mirrors the 0.3.x ClientTaskManager, which appended each status message to
  task.history before overwriting the task status.
  """
  from a2a.types import a2a_pb2 as pb

  def _status_update(text: str, state: int) -> pb.StreamResponse:
    item = pb.StreamResponse()
    item.status_update.task_id = "task-1"
    item.status_update.context_id = "ctx-1"
    item.status_update.status.state = state
    item.status_update.status.message.message_id = text
    item.status_update.status.message.parts.add().text = text
    return item

  initial = pb.StreamResponse()
  initial.task.id = "task-1"
  initial.task.context_id = "ctx-1"

  stream = [
      initial,
      _status_update("working", pb.TASK_STATE_WORKING),
      _status_update("done", pb.TASK_STATE_COMPLETED),
  ]

  normalize = _compat.make_stream_normalizer()
  results = [normalize(item) for item in stream]

  # history accumulates one message per status update carrying a message.
  history_ids_per_step = [
      [m.message_id for m in task.history] for task, _ in results
  ]
  assert history_ids_per_step[0] == []
  assert history_ids_per_step[1] == ["working"]
  assert history_ids_per_step[2] == ["working", "done"]
  # the latest status is applied to the running task.
  final_task, _ = results[2]
  assert final_task.status.state == pb.TASK_STATE_COMPLETED
