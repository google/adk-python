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

import asyncio
import base64
import gc
from types import SimpleNamespace
from typing import Any
from typing import AsyncGenerator
from typing import Optional
import weakref

import anyio
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.dependencies._mcp import ClientSession
from google.adk.dependencies._mcp import IS_MCP_SDK_V2
from google.adk.dependencies._mcp import streamable_http_client
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.mcp_tool._agent_to_mcp import _connection_key
from google.adk.tools.mcp_tool._agent_to_mcp import _MAX_REAP_ATTEMPTS
from google.adk.tools.mcp_tool._agent_to_mcp import _reap_orphaned_sessions
from google.adk.tools.mcp_tool._agent_to_mcp import _run_agent
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server
from google.genai import types
import pytest

from ._in_memory_session import _lowlevel_server
from ._in_memory_session import connected_client_session
from ._sdk_compat import field


class _EchoAgent(BaseAgent):
  """Minimal agent that emits a single final text event."""

  reply: str = "hello from the agent"

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(
        author=self.name,
        content=types.Content(
            role="model", parts=[types.Part(text=self.reply)]
        ),
    )


class _ResourceToolset(BaseToolset):
  """Records the lifetime of an asynchronously released tool resource."""

  def __init__(self):
    super().__init__()
    self.resource_open = False
    self.close_count = 0
    self.close_started = asyncio.Event()
    self.release_close: asyncio.Event | None = None

  async def get_tools(self, readonly_context=None) -> list[BaseTool]:
    self.resource_open = True
    return []

  async def close(self) -> None:
    self.close_started.set()
    if self.release_close is not None:
      await self.release_close.wait()
    # Exercise cleanup under the SDK's cancelled AnyIO scope.
    await anyio.sleep(0)
    self.resource_open = False
    self.close_count += 1


class _ToolsetAgent(_EchoAgent):
  tools: list[BaseToolset]

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    for toolset in self.tools:
      await toolset.get_tools()
    async for event in super()._run_async_impl(ctx):
      yield event


def _text_event(text: str, *, partial: bool = False) -> Event:
  return Event(
      author="a",
      partial=partial,
      content=types.Content(role="model", parts=[types.Part(text=text)]),
  )


def _image_event(data: bytes, mime_type: str) -> Event:
  return Event(
      author="a",
      content=types.Content(
          role="model",
          parts=[
              types.Part(inline_data=types.Blob(data=data, mime_type=mime_type))
          ],
      ),
  )


class _FakeSessionService(InMemorySessionService):

  def __init__(self, runner: _FakeRunner):
    super().__init__()
    self._runner = runner

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Any:
    del state, session_id
    return await self._runner._create_session(
        app_name=app_name, user_id=user_id
    )

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    await self._runner._delete_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )


class _FakeRunner:
  """Runner stub that yields a fixed event sequence."""

  app_name = "fake"

  def __init__(self, events: list[Event]):
    self._events = events
    self.create_session_calls = 0
    self.session_ids: list[str] = []
    self.deleted_session_ids: list[str] = []
    self.failing_deletes = 0
    self.create_session_delay = 0.0
    self.session_service = _FakeSessionService(self)

  async def _create_session(self, *, app_name: str, user_id: str):
    if self.create_session_delay > 0:
      await asyncio.sleep(self.create_session_delay)
    self.create_session_calls += 1
    return SimpleNamespace(id=f"session-{self.create_session_calls}")

  async def _delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ):
    if self.failing_deletes > 0:
      self.failing_deletes -= 1
      raise ConnectionError("session service unavailable")
    self.deleted_session_ids.append(session_id)

  async def run_async(
      self, *, user_id: str, session_id: str, new_message: types.Content
  ) -> AsyncGenerator[Event, None]:
    self.session_ids.append(session_id)
    for event in self._events:
      yield event


class _ConnCtx:
  """Fake MCP Context carrying a per-connection session object."""

  def __init__(self, session: object):
    self.session = session

  async def report_progress(self, *, progress, total=None, message=None):
    pass


class _Connection:
  """Stand-in for an MCP connection object (weak-referenceable)."""


class _RequestScopedSession:
  """Stand-in for an MCP SDK 2.x ServerSession.

  The 2.x server builds one of these per inbound request and holds the
  connection on the private ``_connection``. It is hashable, like the real
  class, so a stale per-request key silently starts a new conversation rather
  than raising.
  """

  def __init__(self, connection: object):
    self._connection = connection


class _RequestScopedCtx:
  """Fake MCP Context shaped like MCP SDK 2.x."""

  def __init__(self, connection: object):
    self.session = _RequestScopedSession(connection)

  async def report_progress(self, *, progress, total=None, message=None):
    pass


@pytest.mark.asyncio
async def test_to_mcp_server_registers_agent_as_single_tool():
  agent = _EchoAgent(name="my_agent", description="does useful things")

  server = to_mcp_server(agent)
  tools = await server.list_tools()

  assert len(tools) == 1
  assert tools[0].name == "my_agent"
  assert tools[0].description == "does useful things"
  assert "request" in field(tools[0], "inputSchema")["properties"]


@pytest.mark.asyncio
async def test_to_mcp_server_name_override():
  agent = _EchoAgent(name="my_agent")

  server = to_mcp_server(agent, name="custom")
  tools = await server.list_tools()

  assert tools[0].name == "custom"


@pytest.mark.asyncio
async def test_call_tool_runs_agent_end_to_end():
  agent = _EchoAgent(name="assistant")
  server = to_mcp_server(agent)

  async with connected_client_session(server) as client:
    result = await client.call_tool("assistant", {"request": "hi"})

  assert not field(result, "isError")
  assert "hello from the agent" in result.content[0].text


@pytest.mark.asyncio
async def test_server_shutdown_closes_owned_runner_toolsets():
  """An MCP server releases the resources its default runner opened."""
  toolset = _ResourceToolset()
  server = to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "hi"})
    assert toolset.resource_open

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
async def test_streamable_http_shutdown_closes_owned_resources():
  """Streamable HTTP releases resources at application lifecycle shutdown."""
  toolset = _ResourceToolset()
  server = to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))
  if IS_MCP_SDK_V2:
    import httpx2 as httpx

    app = server.streamable_http_app(json_response=True)
  else:
    import httpx

    server.settings.json_response = True
    app = server.streamable_http_app()

  async with app.router.lifespan_context(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app)
    ) as http:
      async with streamable_http_client(
          "http://127.0.0.1:8000/mcp",
          http_client=http,
          terminate_on_close=False,
      ) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
          await client.initialize()
          result = await client.call_tool("assistant", {"request": "hi"})
          assert not field(result, "isError")
          assert result.content[0].text == "hello from the agent"

    # Leave the HTTP session alive so application shutdown owns its cleanup.
    assert toolset.resource_open
    assert toolset.close_count == 0

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
async def test_server_shutdown_leaves_borrowed_runner_open():
  """A caller retains ownership of an explicitly supplied runner."""
  toolset = _ResourceToolset()
  agent = _ToolsetAgent(name="assistant", tools=[toolset])
  runner = Runner(
      app_name=agent.name,
      agent=agent,
      session_service=InMemorySessionService(),
  )
  server = to_mcp_server(agent, runner=runner)

  try:
    async with connected_client_session(server) as client:
      await client.call_tool("assistant", {"request": "hi"})

    assert toolset.resource_open
    assert toolset.close_count == 0
  finally:
    await runner.close()

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
async def test_connection_shutdown_keeps_other_connections_usable():
  """Closing one connection cannot release another connection's tools.

  Setup: two SDK connections share the server and its default runner.
  Act: close one connection, then call the agent on the surviving connection.
  Assert: tool resources remain open until the final connection exits.
  """
  toolset = _ResourceToolset()
  server = to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))

  async with connected_client_session(server) as first:
    await first.call_tool("assistant", {"request": "first"})
    async with connected_client_session(server) as second:
      await second.call_tool("assistant", {"request": "second"})

    assert toolset.resource_open
    assert toolset.close_count == 0
    result = await first.call_tool("assistant", {"request": "still here"})
    assert not field(result, "isError")
    assert result.content[0].text == "hello from the agent"

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
async def test_server_can_reopen_resources_after_shutdown():
  """A later server lifecycle can open and close its resources again."""
  toolset = _ResourceToolset()
  server = to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))

  for expected_closes in (1, 2):
    async with connected_client_session(server) as client:
      result = await client.call_tool("assistant", {"request": "hi"})
      assert not field(result, "isError")
      assert toolset.resource_open

    assert not toolset.resource_open
    assert toolset.close_count == expected_closes


@pytest.mark.asyncio
async def test_independent_servers_close_only_their_own_resources():
  """An active server does not keep another server's resources open."""
  first_tools = _ResourceToolset()
  second_tools = _ResourceToolset()
  first = to_mcp_server(_ToolsetAgent(name="first", tools=[first_tools]))
  second = to_mcp_server(_ToolsetAgent(name="second", tools=[second_tools]))

  async with connected_client_session(first) as first_client:
    await first_client.call_tool("first", {"request": "hi"})
    async with connected_client_session(second) as second_client:
      await second_client.call_tool("second", {"request": "hi"})

    assert not second_tools.resource_open
    assert second_tools.close_count == 1
    assert first_tools.resource_open
    assert first_tools.close_count == 0

  assert not first_tools.resource_open
  assert first_tools.close_count == 1


@pytest.mark.asyncio
async def test_server_lifespan_error_still_closes_resources():
  """A failure inside the server lifecycle still releases owned resources."""
  toolset = _ResourceToolset()
  server = _lowlevel_server(
      to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))
  )

  with pytest.raises(RuntimeError, match="server failed"):
    async with server.lifespan(server):
      await toolset.get_tools()
      raise RuntimeError("server failed")

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
async def test_cancelled_server_scope_still_closes_resources():
  """An AnyIO cancellation cannot skip asynchronous runner cleanup."""
  toolset = _ResourceToolset()
  server = _lowlevel_server(
      to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))
  )

  with anyio.CancelScope() as scope:
    async with server.lifespan(server):
      await toolset.get_tools()
      scope.cancel()
      await anyio.sleep(0)

  assert not toolset.resource_open
  assert toolset.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_phase", [None, "shutdown", "startup"])
async def test_new_lifespan_waits_for_previous_cleanup(cancel_phase):
  """A new connection cannot race with the previous lifecycle's cleanup.

  Setup: hold the final connection's asynchronous resource cleanup open.
  Act: begin a new lifecycle, optionally cancelling shutdown or a waiting start.
  Assert: the new lifecycle enters only after cleanup and remains usable.
  """
  toolset = _ResourceToolset()
  toolset.release_close = asyncio.Event()
  server = _lowlevel_server(
      to_mcp_server(_ToolsetAgent(name="assistant", tools=[toolset]))
  )
  first_ready = asyncio.Event()
  finish_first = asyncio.Event()
  second_ready = asyncio.Event()
  finish_second = asyncio.Event()

  async def first_lifecycle():
    async with server.lifespan(server):
      await toolset.get_tools()
      first_ready.set()
      await finish_first.wait()

  async def second_lifecycle():
    async with server.lifespan(server):
      await toolset.get_tools()
      second_ready.set()
      await finish_second.wait()

  first = asyncio.create_task(first_lifecycle())
  second = None
  try:
    await asyncio.wait_for(first_ready.wait(), timeout=2)
    finish_first.set()
    await asyncio.wait_for(toolset.close_started.wait(), timeout=2)
    if cancel_phase == "shutdown":
      first.cancel()

    second = asyncio.create_task(second_lifecycle())
    await asyncio.sleep(0)
    assert not second_ready.is_set()
    assert toolset.close_count == 0
    if cancel_phase == "startup":
      second.cancel()
      with pytest.raises(asyncio.CancelledError):
        await second
      second = None

    toolset.release_close.set()
    if cancel_phase == "shutdown":
      with pytest.raises(asyncio.CancelledError):
        await first
    else:
      await first
    if second is None:
      second = asyncio.create_task(second_lifecycle())
    await asyncio.wait_for(second_ready.wait(), timeout=2)
    assert toolset.resource_open
    assert toolset.close_count == 1
    finish_second.set()
    await second
  finally:
    finish_first.set()
    finish_second.set()
    toolset.release_close.set()
    await asyncio.gather(
        first, *([second] if second is not None else []), return_exceptions=True
    )

  assert not toolset.resource_open
  assert toolset.close_count == 2


@pytest.mark.asyncio
async def test_run_agent_returns_only_final_text():
  runner = _FakeRunner([_text_event("answer")])

  result = await _run_agent(runner, "hi")

  assert [block.type for block in result] == ["text"]
  assert result[0].text == "answer"


@pytest.mark.asyncio
async def test_run_agent_reports_intermediate_events_as_progress():
  reported: list[str] = []

  class _Ctx:

    async def report_progress(self, *, progress, total=None, message=None):
      reported.append(message)

  runner = _FakeRunner(
      [_text_event("thinking", partial=True), _text_event("done")]
  )

  result = await _run_agent(runner, "hi", _Ctx())

  assert result[0].text == "done"
  assert reported == ["thinking"]


def _event_with_thought(
    thought: str, text: str, *, partial: bool = False
) -> Event:
  return Event(
      author="a",
      partial=partial,
      content=types.Content(
          role="model",
          parts=[types.Part(text=thought, thought=True), types.Part(text=text)],
      ),
  )


@pytest.mark.asyncio
async def test_run_agent_leaves_thoughts_out_of_the_result():
  runner = _FakeRunner([_event_with_thought("private reasoning", "answer")])

  result = await _run_agent(runner, "hi")

  # The model's thoughts are not its answer; MCP clients only get the answer.
  assert [block.type for block in result] == ["text"]
  assert result[0].text == "answer"


@pytest.mark.asyncio
async def test_run_agent_leaves_thoughts_out_of_progress():
  reported: list[str] = []

  class _Ctx:

    async def report_progress(self, *, progress, total=None, message=None):
      reported.append(message)

  thought_only = Event(
      author="a",
      partial=True,
      content=types.Content(
          role="model", parts=[types.Part(text="planning", thought=True)]
      ),
  )
  runner = _FakeRunner([
      thought_only,
      _event_with_thought("private reasoning", "working", partial=True),
      _text_event("done"),
  ])

  result = await _run_agent(runner, "hi", _Ctx())

  assert result[0].text == "done"
  assert reported == ["working"]


@pytest.mark.asyncio
async def test_run_agent_maps_image_output_to_image_content():
  png = b"\x89PNG\r\n\x1a\n"
  runner = _FakeRunner([_image_event(png, "image/png")])

  result = await _run_agent(runner, "draw a logo")

  assert len(result) == 1
  assert result[0].type == "image"
  assert field(result[0], "mimeType") == "image/png"
  assert base64.b64decode(result[0].data) == png


@pytest.mark.asyncio
async def test_run_agent_reuses_one_session_per_connection():
  runner = _FakeRunner([_text_event("ok")])
  sessions: dict[object, str] = {}
  ctx = _ConnCtx(_Connection())

  await _run_agent(runner, "first", ctx, sessions)
  await _run_agent(runner, "second", ctx, sessions)

  assert runner.create_session_calls == 1
  assert runner.session_ids == ["session-1", "session-1"]


@pytest.mark.asyncio
async def test_run_agent_uses_separate_sessions_across_connections():
  runner = _FakeRunner([_text_event("ok")])
  sessions: dict[object, str] = {}

  await _run_agent(runner, "a", _ConnCtx(_Connection()), sessions)
  await _run_agent(runner, "b", _ConnCtx(_Connection()), sessions)

  assert runner.create_session_calls == 2
  assert runner.session_ids == ["session-1", "session-2"]


def test_connection_key_uses_the_session_when_it_has_no_connection():
  """MCP SDK 1.x: the session is already one object per connection."""
  session = _Connection()

  assert _connection_key(_ConnCtx(session)) is session


def test_connection_key_prefers_the_connection_behind_the_session():
  """MCP SDK 2.x: the session is per request, the connection is not."""
  connection = _Connection()
  ctx = _RequestScopedCtx(connection)

  assert _connection_key(ctx) is connection


@pytest.mark.asyncio
async def test_run_agent_reuses_one_session_when_sessions_are_per_request():
  """Two requests on one connection stay in one conversation under SDK 2.x."""
  runner = _FakeRunner([_text_event("ok")])
  sessions: dict[object, str] = {}
  connection = _Connection()

  await _run_agent(runner, "first", _RequestScopedCtx(connection), sessions)
  await _run_agent(runner, "second", _RequestScopedCtx(connection), sessions)

  assert runner.create_session_calls == 1
  assert runner.session_ids == ["session-1", "session-1"]


@pytest.mark.asyncio
async def test_run_agent_separates_connections_when_sessions_are_per_request():
  """Separate clients never share a conversation under SDK 2.x."""
  runner = _FakeRunner([_text_event("ok")])
  sessions: dict[object, str] = {}

  await _run_agent(runner, "a", _RequestScopedCtx(_Connection()), sessions)
  await _run_agent(runner, "b", _RequestScopedCtx(_Connection()), sessions)

  assert runner.create_session_calls == 2
  assert runner.session_ids == ["session-1", "session-2"]


@pytest.mark.asyncio
async def test_reap_deletes_only_sessions_no_longer_reachable():
  runner = _FakeRunner([_text_event("ok")])
  connection = _Connection()
  sessions: dict[object, str] = {connection: "session-live"}
  created = {"session-live", "session-dead"}

  await _reap_orphaned_sessions(runner, sessions, created)

  assert runner.deleted_session_ids == ["session-dead"]
  assert created == {"session-live"}


@pytest.mark.asyncio
async def test_session_of_a_collected_connection_is_reaped():
  """A conversation must not outlive its connection in the session service."""
  runner = _FakeRunner([_text_event("ok")])
  sessions: weakref.WeakKeyDictionary[object, str] = (
      # pylint: disable-next=abstract-class-instantiated
      weakref.WeakKeyDictionary()
  )
  created: set[str] = set()
  ctx = _ConnCtx(_Connection())

  await _run_agent(runner, "hi", ctx, sessions, created)
  del ctx
  gc.collect()
  await _reap_orphaned_sessions(runner, sessions, created)

  assert runner.deleted_session_ids == ["session-1"]
  assert not created


@pytest.mark.asyncio
async def test_per_request_connections_do_not_accumulate_sessions():
  """Stateless streamable HTTP builds a fresh connection per request; each
  request's session must be reclaimed instead of leaking one per tool call."""
  runner = _FakeRunner([_text_event("ok")])
  sessions: weakref.WeakKeyDictionary[object, str] = (
      # pylint: disable-next=abstract-class-instantiated
      weakref.WeakKeyDictionary()
  )
  created: set[str] = set()

  for request in ("a", "b", "c"):
    await _reap_orphaned_sessions(runner, sessions, created)
    ctx = _RequestScopedCtx(_Connection())
    await _run_agent(runner, request, ctx, sessions, created)
    del ctx
    gc.collect()
  await _reap_orphaned_sessions(runner, sessions, created)

  assert runner.deleted_session_ids == ["session-1", "session-2", "session-3"]
  assert not created


@pytest.mark.asyncio
async def test_reap_failure_does_not_raise_and_is_retried():
  """A session service outage must not fail the live tool call, and the
  orphaned session must be deleted once the service recovers."""
  runner = _FakeRunner([_text_event("ok")])
  runner.failing_deletes = 1
  connection = _Connection()
  sessions: dict[object, str] = {connection: "session-live"}
  created = {"session-live", "session-dead"}
  reap_failures: dict[str, int] = {}

  await _reap_orphaned_sessions(
      runner, sessions, created, reap_failures=reap_failures
  )

  assert runner.deleted_session_ids == []
  assert created == {"session-live", "session-dead"}
  assert reap_failures == {"session-dead": 1}

  await _reap_orphaned_sessions(
      runner, sessions, created, reap_failures=reap_failures
  )

  assert runner.deleted_session_ids == ["session-dead"]
  assert created == {"session-live"}
  assert not reap_failures


@pytest.mark.asyncio
async def test_reap_failure_dropped_after_max_attempts():
  """A session that repeatedly fails deletion is dropped after max attempts."""
  runner = _FakeRunner([_text_event("ok")])
  runner.failing_deletes = _MAX_REAP_ATTEMPTS + 2
  connection = _Connection()
  sessions: dict[object, str] = {connection: "session-live"}
  created = {"session-live", "session-dead"}
  reap_failures: dict[str, int] = {}

  for _ in range(_MAX_REAP_ATTEMPTS):
    await _reap_orphaned_sessions(
        runner, sessions, created, reap_failures=reap_failures
    )

  assert runner.deleted_session_ids == []
  assert created == {"session-live"}
  assert not reap_failures


@pytest.mark.asyncio
async def test_reap_cancellation_restores_session_and_reraises():
  """A task cancellation during delete_session must restore the session id."""
  runner = _FakeRunner([_text_event("ok")])

  async def _cancelled_delete(*, app_name: str, user_id: str, session_id: str):
    del app_name, user_id, session_id
    raise asyncio.CancelledError()

  runner.session_service.delete_session = _cancelled_delete
  connection = _Connection()
  sessions: dict[object, str] = {connection: "session-live"}
  created = {"session-live", "session-dead"}

  with pytest.raises(asyncio.CancelledError):
    await _reap_orphaned_sessions(runner, sessions, created)

  assert created == {"session-live", "session-dead"}


@pytest.mark.asyncio
async def test_call_tool_reaps_conversation_of_closed_connection():
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
  gc.collect()
  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "second"})

  assert runner.session_ids == ["session-1", "session-2"]
  assert runner.deleted_session_ids == ["session-1"]


@pytest.mark.asyncio
async def test_call_tool_reuses_session_across_calls_on_one_connection():
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
    await client.call_tool("assistant", {"request": "second"})

  assert runner.create_session_calls == 1
  assert runner.session_ids == ["session-1", "session-1"]


@pytest.mark.asyncio
async def test_call_tool_concurrent_first_calls_reuse_session():
  """Two concurrent first calls on one connection must share one session."""
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  runner.create_session_delay = 0.01
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await asyncio.gather(
        client.call_tool("assistant", {"request": "first"}),
        client.call_tool("assistant", {"request": "second"}),
    )

  assert runner.create_session_calls == 1
  assert runner.session_ids == ["session-1", "session-1"]
  assert runner.deleted_session_ids == []


@pytest.mark.asyncio
async def test_call_tool_does_not_reap_persistent_session():
  """A caller-supplied persistent session service must not have its sessions deleted on connection teardown."""
  from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
  from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
  from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
  from google.adk.runners import Runner
  from google.adk.sessions.database_session_service import DatabaseSessionService

  agent = _EchoAgent(name="assistant")
  session_service = DatabaseSessionService(
      db_url="sqlite+aiosqlite:///:memory:"
  )
  runner = Runner(
      app_name="assistant",
      agent=agent,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      memory_service=InMemoryMemoryService(),
      credential_service=InMemoryCredentialService(),
  )
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
  gc.collect()
  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "second"})

  sessions_response = await session_service.list_sessions(
      app_name="assistant", user_id="mcp_user"
  )
  assert len(sessions_response.sessions) == 2
