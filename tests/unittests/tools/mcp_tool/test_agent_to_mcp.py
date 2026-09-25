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
import contextvars
import gc
import logging
from types import SimpleNamespace
from typing import AsyncGenerator
import weakref

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.tools.mcp_tool._agent_to_mcp import _connection_key
from google.adk.tools.mcp_tool._agent_to_mcp import _MAX_CONCURRENT_DELETES
from google.adk.tools.mcp_tool._agent_to_mcp import _reap_orphaned_sessions
from google.adk.tools.mcp_tool._agent_to_mcp import _run_agent
from google.adk.tools.mcp_tool._agent_to_mcp import to_mcp_server
from google.genai import types
import pytest

from ._in_memory_session import connected_client_session
from ._sdk_compat import field

_CALLER_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("_CALLER_VAR")


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
    self.deleted_session_ids: list[str] = []
    self.failing_deletes = 0
    self.session_service = SimpleNamespace(
        create_session=self._create_session,
        delete_session=self._delete_session,
    )

  async def _create_session(self, *, app_name: str, user_id: str):
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

  await _reap_orphaned_sessions(runner, sessions, created)

  assert runner.deleted_session_ids == []
  assert created == {"session-live", "session-dead"}

  await _reap_orphaned_sessions(runner, sessions, created)

  assert runner.deleted_session_ids == ["session-dead"]
  assert created == {"session-live"}


@pytest.mark.asyncio
async def test_reap_partial_failure_requeues_only_the_failed_ids():
  """In a batch, a failed delete is retried later without undoing the
  deletes that were already in flight and succeeded."""
  runner = _FakeRunner([_text_event("ok")])
  runner.failing_deletes = 1
  record_delete = runner.session_service.delete_session

  async def io_bound_delete(**kwargs):
    # Yield like real I/O, so all three deletes are in flight together.
    await asyncio.sleep(0)
    await record_delete(**kwargs)

  runner.session_service.delete_session = io_bound_delete
  created = {"session-a", "session-b", "session-c"}

  await _reap_orphaned_sessions(runner, {}, created)

  assert len(runner.deleted_session_ids) == 2
  assert len(created) == 1
  assert created.isdisjoint(runner.deleted_session_ids)

  await _reap_orphaned_sessions(runner, {}, created)

  assert sorted(runner.deleted_session_ids) == [
      "session-a",
      "session-b",
      "session-c",
  ]
  assert not created


@pytest.mark.asyncio
async def test_reap_stops_after_a_failure_instead_of_hammering_the_service(
    caplog,
):
  """While the session service is down, a reap makes a number of delete
  attempts bounded by the pool size instead of one per waiting session, and
  keeps every id for a later retry."""
  attempts = 0

  async def unavailable_delete(**kwargs):
    nonlocal attempts
    attempts += 1
    await asyncio.sleep(0.01)
    raise ConnectionError("session service unavailable")

  runner = _FakeRunner([_text_event("ok")])
  runner.session_service.delete_session = unavailable_delete
  backlog = {f"session-{i}" for i in range(10 * _MAX_CONCURRENT_DELETES)}
  created = set(backlog)

  with caplog.at_level(logging.WARNING):
    await _reap_orphaned_sessions(runner, {}, created)

  # Workers whose delete failed early may each start one more before the
  # pool's last failure lands, hence fewer than twice the pool size.
  assert _MAX_CONCURRENT_DELETES <= attempts < 2 * _MAX_CONCURRENT_DELETES
  assert created == backlog
  # One summary per pass, not one record per failed delete.
  assert caplog.text.count("Failed to delete") == 1
  assert f"Failed to delete {attempts} orphaned" in caplog.text


@pytest.mark.asyncio
async def test_one_failing_session_does_not_hold_up_the_rest():
  """A session whose delete keeps failing, e.g. one a custom session service
  rejects, must not stop the rest of the batch from being deleted."""
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session

  async def delete_rejecting_one(**kwargs):
    if kwargs["session_id"] == "session-rejected":
      raise ValueError("rejected by the session service")
    await asyncio.sleep(0)
    await record_delete(**kwargs)

  runner.session_service.delete_session = delete_rejecting_one
  healthy = {f"session-{i}" for i in range(100)}
  created = healthy | {"session-rejected"}

  await _reap_orphaned_sessions(runner, {}, created)

  assert set(runner.deleted_session_ids) == healthy
  assert created == {"session-rejected"}


@pytest.mark.asyncio
async def test_hung_delete_times_out_instead_of_stopping_reaping(
    monkeypatch, caplog
):
  """Only one reap runs at a time, so a delete that never returns must time
  out and be retried later rather than block every future reap."""
  monkeypatch.setattr(
      "google.adk.tools.mcp_tool._agent_to_mcp._DELETE_TIMEOUT_SECONDS", 0.05
  )
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session

  async def delete_hanging_once(**kwargs):
    if kwargs["session_id"] == "session-hung":
      await asyncio.Event().wait()
    await record_delete(**kwargs)

  runner.session_service.delete_session = delete_hanging_once
  created = {"session-hung", "session-ok"}

  with caplog.at_level(logging.WARNING):
    await asyncio.wait_for(
        _reap_orphaned_sessions(runner, {}, created), timeout=5
    )

  assert runner.deleted_session_ids == ["session-ok"]
  assert created == {"session-hung"}
  assert "TimeoutError" in caplog.text


@pytest.mark.asyncio
async def test_reap_task_count_stays_bounded_for_a_large_backlog():
  """After an outage the backlog can hold many thousands of ids; draining it
  must not create a task per waiting session."""
  peak_tasks = 0
  deleted = 0

  async def counting_delete(**kwargs):
    nonlocal peak_tasks, deleted
    peak_tasks = max(peak_tasks, len(asyncio.all_tasks()))
    await asyncio.sleep(0)
    deleted += 1

  runner = _FakeRunner([_text_event("ok")])
  runner.session_service.delete_session = counting_delete
  created = {f"session-{i}" for i in range(1000)}

  await _reap_orphaned_sessions(runner, {}, created)

  assert deleted == 1000
  assert not created
  # Bounded by the pool size, not the backlog: the workers, the task running
  # this test, and on Python < 3.12 one wait_for task per in-flight delete.
  assert peak_tasks <= 2 * _MAX_CONCURRENT_DELETES + 1


@pytest.mark.asyncio
async def test_slow_deletes_run_concurrently_up_to_the_cap():
  """With a slow session service, one reap deletes a backlog concurrently
  but never has more than the cap in flight. Deleting one at a time caps
  throughput at one session per delete round trip, so under frequent short
  connections the backlog grows faster than it drains."""
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session
  in_flight = 0
  max_in_flight = 0

  async def slow_delete(**kwargs):
    nonlocal in_flight, max_in_flight
    in_flight += 1
    max_in_flight = max(max_in_flight, in_flight)
    await asyncio.sleep(0.01)
    in_flight -= 1
    await record_delete(**kwargs)

  runner.session_service.delete_session = slow_delete
  backlog = {f"session-{i}" for i in range(3 * _MAX_CONCURRENT_DELETES)}
  created = set(backlog)

  await _reap_orphaned_sessions(runner, {}, created)

  assert sorted(runner.deleted_session_ids) == sorted(backlog)
  assert max_in_flight == _MAX_CONCURRENT_DELETES
  assert not created


@pytest.mark.asyncio
async def test_reap_drains_orphans_that_appear_while_it_runs():
  """Sessions orphaned while a reap is deleting are picked up by that same
  reap, so a backlog drains without waiting for another tool call."""
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session
  release = asyncio.Event()
  created = {"session-early"}

  async def gated_delete(**kwargs):
    if kwargs["session_id"] == "session-early":
      await release.wait()
    await record_delete(**kwargs)

  runner.session_service.delete_session = gated_delete
  reap = asyncio.create_task(_reap_orphaned_sessions(runner, {}, created))
  await asyncio.sleep(0)
  # Orphaned while the first delete is still in flight.
  created.add("session-late")
  release.set()
  await asyncio.wait_for(reap, timeout=5)

  assert runner.deleted_session_ids == ["session-early", "session-late"]
  assert not created


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
async def test_call_tool_does_not_wait_on_slow_session_deletes():
  """Reaping runs in the background, so a slow session service (e.g. a
  database) never adds delete latency to the live tool call."""
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  release = asyncio.Event()
  record_delete = runner.session_service.delete_session

  async def blocked_delete(**kwargs):
    await release.wait()
    await record_delete(**kwargs)

  runner.session_service.delete_session = blocked_delete
  server = to_mcp_server(agent, runner=runner)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
  gc.collect()
  async with connected_client_session(server) as client:
    # Completes while the delete of session-1 is still blocked.
    await asyncio.wait_for(
        client.call_tool("assistant", {"request": "second"}), timeout=5
    )
    assert runner.deleted_session_ids == []

    release.set()
    for _ in range(100):
      if runner.deleted_session_ids:
        break
      await asyncio.sleep(0.01)

  assert runner.deleted_session_ids == ["session-1"]


@pytest.mark.asyncio
async def test_background_reap_failure_is_logged_not_raised(
    monkeypatch, caplog
):
  """An unexpected reap error is logged and never fails the live call."""

  async def failing_reap(*args, **kwargs):
    raise RuntimeError("reap exploded")

  monkeypatch.setattr(
      "google.adk.tools.mcp_tool._agent_to_mcp._reap_orphaned_sessions",
      failing_reap,
  )
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  server = to_mcp_server(agent, runner=runner)

  with caplog.at_level(logging.WARNING):
    async with connected_client_session(server) as client:
      result = await client.call_tool("assistant", {"request": "first"})
      await asyncio.sleep(0)

  assert not field(result, "isError")
  assert "Background reap of orphaned MCP agent sessions failed" in caplog.text
  assert "reap exploded" in caplog.text


@pytest.mark.asyncio
async def test_background_reap_does_not_inherit_the_callers_context():
  """The reap task starts from an empty context. Inheriting the tool call's
  would keep that request, and on MCP SDK 1.x its connection, alive for as
  long as the task runs."""
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session
  seen_in_reap: list[object] = []

  async def delete_recording_context(**kwargs):
    seen_in_reap.append(_CALLER_VAR.get(None))
    await record_delete(**kwargs)

  runner.session_service.delete_session = delete_recording_context
  server = to_mcp_server(agent, runner=runner)

  token = _CALLER_VAR.set("request-scoped value")
  try:
    async with connected_client_session(server) as client:
      await client.call_tool("assistant", {"request": "first"})
    gc.collect()
    async with connected_client_session(server) as client:
      await client.call_tool("assistant", {"request": "second"})
  finally:
    _CALLER_VAR.reset(token)

  assert runner.deleted_session_ids == ["session-1"]
  assert seen_in_reap == [None]


def test_reap_is_not_blocked_by_a_task_from_another_event_loop():
  """A reap stuck in an event loop that stopped without cancelling it must not
  stop reaping in the loop that serves later calls."""
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  record_delete = runner.session_service.delete_session

  async def delete_blocking_first(**kwargs):
    if kwargs["session_id"] == "session-1":
      await asyncio.Event().wait()
    await record_delete(**kwargs)

  runner.session_service.delete_session = delete_blocking_first
  server = to_mcp_server(agent, runner=runner)

  async def call_on_new_connection(request: str) -> None:
    async with connected_client_session(server) as client:
      await client.call_tool("assistant", {"request": request})
    gc.collect()

  async def first_loop_calls() -> None:
    await call_on_new_connection("a")
    # Starts the reap of session-1, which never finishes in this loop.
    await call_on_new_connection("b")

  first_loop = asyncio.new_event_loop()
  try:
    first_loop.run_until_complete(first_loop_calls())

    async def second_loop_calls() -> None:
      await call_on_new_connection("c")
      for _ in range(100):
        if "session-2" in runner.deleted_session_ids:
          break
        await asyncio.sleep(0.01)

    asyncio.run(second_loop_calls())

    assert "session-2" in runner.deleted_session_ids
  finally:

    async def cancel_stuck_tasks() -> None:
      pending = asyncio.all_tasks() - {asyncio.current_task()}
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)

    first_loop.run_until_complete(cancel_stuck_tasks())
    first_loop.close()


@pytest.mark.asyncio
async def test_call_tool_retains_sessions_when_deletion_is_opted_out():
  """delete_orphaned_sessions=False keeps finished conversations in the
  session service, for persistent services whose records are read later."""
  agent = _EchoAgent(name="assistant")
  runner = _FakeRunner([_text_event("ok")])
  server = to_mcp_server(agent, runner=runner, delete_orphaned_sessions=False)

  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "first"})
  gc.collect()
  async with connected_client_session(server) as client:
    await client.call_tool("assistant", {"request": "second"})

  assert runner.session_ids == ["session-1", "session-2"]
  assert runner.deleted_session_ids == []


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
