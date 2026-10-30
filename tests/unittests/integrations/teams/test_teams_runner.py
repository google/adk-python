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

"""Tests for TeamsRunner.

Drives the runner's aiohttp endpoint with Bot Framework activities and records
the replies on a fake Teams connector service. Only the LLM is mocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Any
from typing import AsyncIterator

import pytest

pytest.importorskip("microsoft_agents.hosting.aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.teams import TeamsRunner
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from microsoft_agents.hosting.aiohttp import CloudAdapter
from microsoft_agents.hosting.core import AgentApplication
from microsoft_agents.hosting.core import AgentAuthConfiguration
from microsoft_agents.hosting.core import AnonymousTokenProvider
from microsoft_agents.hosting.core import ConnectionManager
from microsoft_agents.hosting.core import MemoryStorage
from microsoft_agents.hosting.core import TurnState

from tests.unittests import testing_utils

_APP_NAME = "teams_test_app"
_CONVERSATION_ID = "19:conversation@thread.v2"
_USER_AAD_ID = "00000000-0000-0000-0000-000000000001"


class _FakeConnector:
  """Records the activities a bot sends back through the Teams connector."""

  def __init__(self) -> None:
    self.activities: list[dict[str, Any]] = []
    self.server: TestServer | None = None
    self._received = asyncio.Condition()

  async def __aenter__(self) -> _FakeConnector:
    app = web.Application()
    app.router.add_post(
        "/v3/conversations/{conversation_id}/activities/{activity_id}",
        self._record,
    )
    app.router.add_post(
        "/v3/conversations/{conversation_id}/activities", self._record
    )
    self.server = TestServer(app)
    await self.server.start_server()
    return self

  async def __aexit__(self, *exc_info: Any) -> None:
    await self.server.close()

  async def _record(self, request: web.Request) -> web.Response:
    activity = await request.json()
    async with self._received:
      self.activities.append(activity)
      self._received.notify_all()
    return web.json_response({"id": f"reply-{len(self.activities)}"})

  async def wait_for_replies(self, count: int) -> list[str]:
    async with self._received:
      await asyncio.wait_for(
          self._received.wait_for(lambda: len(self.replies) >= count),
          timeout=10,
      )
    return self.replies

  @property
  def url(self) -> str:
    return str(self.server.make_url("/"))

  @property
  def replies(self) -> list[str]:
    return [a["text"] for a in self.activities if a.get("type") == "message"]


def _make_runner(model: testing_utils.MockModel) -> Runner:
  return Runner(
      app_name=_APP_NAME,
      agent=LlmAgent(name="teams_agent", model=model),
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )


def _make_teams_app(
    runner: Runner, *, anonymous: bool = True
) -> web.Application:
  auth_configuration = AgentAuthConfiguration(anonymous_allowed=anonymous)
  connection_manager = ConnectionManager(
      provider_factory=lambda _: AnonymousTokenProvider(),
      connections_configurations={"SERVICE_CONNECTION": auth_configuration},
  )
  agent_app = AgentApplication[TurnState](
      storage=MemoryStorage(),
      adapter=CloudAdapter(connection_manager=connection_manager),
      connection_manager=connection_manager,
  )
  return TeamsRunner(runner=runner, agent_app=agent_app).create_app(
      auth_configuration
  )


def _message_activity(
    text: str, service_url: str, **overrides: Any
) -> dict[str, Any]:
  activity = {
      "type": "message",
      "id": "activity-1",
      "channelId": "msteams",
      "serviceUrl": service_url,
      "text": text,
      "from": {"id": "29:user", "name": "User", "aadObjectId": _USER_AAD_ID},
      "recipient": {"id": "28:bot", "name": "Bot"},
      "conversation": {"id": _CONVERSATION_ID},
  }
  activity.update(overrides)
  return activity


@contextlib.asynccontextmanager
async def _teams_client(
    runner: Runner, *, anonymous: bool = True
) -> AsyncIterator[TestClient]:
  """Serves the Teams endpoint for the duration of a test.

  The server drains background replies on shutdown, so it must close before
  the fake connector does.
  """
  async with TestClient(
      TestServer(_make_teams_app(runner, anonymous=anonymous))
  ) as client:
    yield client


async def _post(client: TestClient, activity: dict[str, Any]) -> int:
  response = await client.post("/api/messages", json=activity)
  return response.status


async def _only_session(runner: Runner) -> Session:
  """Returns the single session the test user has, with its events."""
  listed = await runner.session_service.list_sessions(
      app_name=_APP_NAME, user_id=_USER_AAD_ID
  )
  assert len(listed.sessions) == 1
  return await runner.session_service.get_session(
      app_name=_APP_NAME,
      user_id=_USER_AAD_ID,
      session_id=listed.sessions[0].id,
  )


def _blocked_model(
    release: asyncio.Event, responses: list[str]
) -> testing_utils.MockModel:
  """Returns a mock model that waits for `release` before each response."""

  class _BlockedModel(testing_utils.MockModel):

    async def generate_content_async(self, llm_request, stream=False):
      await release.wait()
      async for response in super().generate_content_async(llm_request, stream):
        yield response

  return _BlockedModel.create(responses)


async def test_message_gets_agent_reply_in_same_conversation():
  """A Teams message is answered with the agent's text response."""
  runner = _make_runner(testing_utils.MockModel.create(["Hi from ADK!"]))

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("Hello", connector.url))
    replies = await connector.wait_for_replies(1)

  assert replies == ["Hi from ADK!"]
  reply = next(a for a in connector.activities if a.get("type") == "message")
  assert reply["conversation"]["id"] == _CONVERSATION_ID


async def test_request_is_acknowledged_before_agent_finishes():
  """Teams gets its HTTP response without waiting for the agent's reply.

  Teams retries requests that take longer than 15 seconds, so a slow agent
  must not hold the request open.
  """
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["done"]))

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    status = await _post(client, _message_activity("Hello", connector.url))
    replies_while_blocked = list(connector.replies)
    release.set()
    replies = await connector.wait_for_replies(1)

  assert 200 <= status < 300
  assert replies_while_blocked == []
  assert replies == ["done"]


async def test_messages_in_one_conversation_share_a_session():
  """Messages in one conversation build up a single ADK session."""
  runner = _make_runner(testing_utils.MockModel.create(["one", "two"]))

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("first", connector.url))
    await connector.wait_for_replies(1)
    await _post(client, _message_activity("second", connector.url))
    await connector.wait_for_replies(2)

  session = await _only_session(runner)
  user_texts = [
      e.content.parts[0].text for e in session.events if e.author == "user"
  ]
  assert user_texts == ["first", "second"]


async def test_session_id_uses_only_url_safe_characters():
  """Session ids avoid the ':' and ';' in Teams conversation ids.

  Some session services, such as VertexAiSessionService, only accept letters,
  digits, '_' and '-'.
  """
  runner = _make_runner(testing_utils.MockModel.create(["ok"]))
  conversation = {"id": "19:abc@thread.tacv2;messageid=1700000000000"}

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(
        client,
        _message_activity("hi", connector.url, conversation=conversation),
    )
    await connector.wait_for_replies(1)

  session = await _only_session(runner)
  assert re.fullmatch(r"[A-Za-z0-9_-]+", session.id)


async def test_messages_in_same_session_are_handled_in_order():
  """A second message waits for the agent to finish answering the first."""
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["one", "two"]))

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("first", connector.url))
    await _post(client, _message_activity("second", connector.url, id="a2"))
    release.set()
    replies = await connector.wait_for_replies(2)

  session = await _only_session(runner)
  assert replies == ["one", "two"]
  assert [e.author for e in session.events] == [
      "user",
      "teams_agent",
      "user",
      "teams_agent",
  ]


async def test_bot_mention_is_removed_before_agent_sees_message():
  """The bot's @mention in a channel message is not passed to the agent."""
  model = testing_utils.MockModel.create(["ok"])
  runner = _make_runner(model)
  mention = {
      "type": "mention",
      "text": "<at>Bot</at>",
      "mentioned": {"id": "28:bot", "name": "Bot"},
  }

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(
        client,
        _message_activity(
            "<at>Bot</at> what is new?", connector.url, entities=[mention]
        ),
    )
    await connector.wait_for_replies(1)

  last_user_content = model.requests[-1].contents[-1]
  assert last_user_content.parts[0].text == "what is new?"


async def test_agent_error_sends_generic_reply_without_details():
  """A failing agent run replies with a generic message, not the exception."""
  model = testing_utils.MockModel.create(
      [], error=RuntimeError("secret internal detail")
  )
  runner = _make_runner(model)

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("Hello", connector.url))
    replies = await connector.wait_for_replies(1)

  assert len(replies) == 1
  assert "secret internal detail" not in replies[0]


async def test_error_event_without_content_sends_generic_reply():
  """A response blocked by the model, with no content, still gets a reply."""
  blocked = LlmResponse(error_code="SAFETY", error_message="Blocked by safety")
  runner = _make_runner(testing_utils.MockModel.create([blocked]))

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("Hello", connector.url))
    replies = await connector.wait_for_replies(1)

  assert len(replies) == 1
  assert "Blocked by safety" not in replies[0]


async def test_failed_reply_closes_agent_run_before_error_reply():
  """If posting a reply fails, the agent run is closed right away."""

  class _RecordingRunner(Runner):
    closed = False

    async def run_async(self, **kwargs):
      try:
        async for event in super().run_async(**kwargs):
          yield event
      finally:
        self.closed = True

  class _FailFirstConnector(_FakeConnector):
    closed_at_error_reply: bool | None = None

    async def _record(self, request: web.Request) -> web.Response:
      if not self.activities:
        self.activities.append({"type": "failed"})
        return web.Response(status=500)
      self.closed_at_error_reply = runner.closed
      return await super()._record(request)

  runner = _RecordingRunner(
      app_name=_APP_NAME,
      agent=LlmAgent(
          name="teams_agent",
          model=testing_utils.MockModel.create(["first"]),
      ),
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )

  async with (
      _FailFirstConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("Hello", connector.url))
    await connector.wait_for_replies(1)

  assert connector.closed_at_error_reply is True


async def test_empty_message_is_ignored():
  """A message with no text, such as an attachment only, gets no reply."""
  model = testing_utils.MockModel.create(["ok"])
  runner = _make_runner(model)

  async with (
      _FakeConnector() as connector,
      _teams_client(runner) as client,
  ):
    await _post(client, _message_activity("", connector.url))
    await _post(client, _message_activity("Hello", connector.url, id="a2"))
    replies = await connector.wait_for_replies(1)

  assert replies == ["ok"]
  assert len(model.requests) == 1


async def test_shutdown_waits_for_replies_in_progress():
  """Stopping the server lets a reply that is already running finish."""
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["done"]))

  async with _FakeConnector() as connector:
    async with _teams_client(runner) as client:
      await _post(client, _message_activity("Hello", connector.url))
      closing = asyncio.create_task(client.close())
      await asyncio.sleep(0.1)
      assert not closing.done()
      release.set()
      await closing

    assert connector.replies == ["done"]


async def test_unauthenticated_request_is_rejected():
  """Without anonymous access, a request with no bearer token gets a 401."""
  model = testing_utils.MockModel.create(["unused"])
  runner = _make_runner(model)

  async with (
      _FakeConnector() as connector,
      _teams_client(runner, anonymous=False) as client,
  ):
    status = await _post(client, _message_activity("Hello", connector.url))

  assert status == 401
  assert connector.activities == []
