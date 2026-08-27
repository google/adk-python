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

import base64
from types import SimpleNamespace
from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools.mcp_tool._agent_to_mcp import _connection_key
from google.adk.tools.mcp_tool._agent_to_mcp import _run_agent
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server
from google.genai import types
import pytest

from ._in_memory_session import connected_client_session


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


class _FakeRunner:
  """Runner stub that yields a fixed event sequence."""

  app_name = "fake"

  def __init__(self, events: list[Event]):
    self._events = events
    self.create_session_calls = 0
    self.session_ids: list[str] = []
    self.session_service = SimpleNamespace(create_session=self._create_session)

  async def _create_session(self, *, app_name: str, user_id: str):
    self.create_session_calls += 1
    return SimpleNamespace(id=f"session-{self.create_session_calls}")

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
  assert "request" in tools[0].inputSchema["properties"]


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

  assert not result.isError
  assert "hello from the agent" in result.content[0].text


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


@pytest.mark.asyncio
async def test_run_agent_maps_image_output_to_image_content():
  png = b"\x89PNG\r\n\x1a\n"
  runner = _FakeRunner([_image_event(png, "image/png")])

  result = await _run_agent(runner, "draw a logo")

  assert len(result) == 1
  assert result[0].type == "image"
  assert result[0].mimeType == "image/png"
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
async def test_call_tool_reuses_session_across_calls_on_one_connection():
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
    await client.call_tool("assistant", {"request": "second"})

  assert runner.create_session_calls == 1
  assert runner.session_ids == ["session-1", "session-1"]
