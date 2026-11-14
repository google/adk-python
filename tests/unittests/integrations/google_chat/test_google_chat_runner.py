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

"""Tests for GoogleChatRunner.

Drives the runner's aiohttp endpoint with Google Chat interaction events and
records the replies on a fake Chat API. Only the LLM and token verification
are faked.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Any
from typing import AsyncIterator

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient
from aiohttp.test_utils import TestServer
from google.adk.agents.llm_agent import LlmAgent
from google.adk.integrations.google_chat import _google_chat_runner
from google.adk.integrations.google_chat import GoogleChatRunner
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.auth import credentials as auth_credentials
from google.genai import types

from tests.unittests import testing_utils

_APP_NAME = "google_chat_test_app"
_USER = "users/1234567890"
_DM_SPACE = "spaces/DM_SPACE"
_ROOM_SPACE = "spaces/ROOM_SPACE"
_ENDPOINT_URL = "https://example.com/chat"
_PROJECT_NUMBER = "1234567890"


class _FakeCredentials(auth_credentials.Credentials):
  """Credentials that hand out a fixed token and count refreshes."""

  def __init__(self) -> None:
    super().__init__()
    self.refresh_count = 0

  def refresh(self, request: Any) -> None:
    del request
    self.refresh_count += 1
    self.token = "app-token"
    self.expiry = None


class _FakeChatApi:
  """Records the messages a Chat app posts through spaces.messages.create."""

  def __init__(self) -> None:
    self.requests: list[dict[str, Any]] = []
    self.server: TestServer | None = None
    self._received = asyncio.Condition()

  async def __aenter__(self) -> _FakeChatApi:
    app = web.Application()
    app.router.add_post(
        "/v1/spaces/{space}/messages",
        self._record,
    )
    self.server = TestServer(app)
    await self.server.start_server()
    return self

  async def __aexit__(self, *exc_info: Any) -> None:
    await self.server.close()

  async def _record(self, request: web.Request) -> web.Response:
    record = {
        "space": f"spaces/{request.match_info['space']}",
        "params": dict(request.query),
        "authorization": request.headers.get("Authorization"),
        "body": await request.json(),
    }
    async with self._received:
      self.requests.append(record)
      self._received.notify_all()
    return web.json_response({"name": f"{record['space']}/messages/m"})

  async def wait_for_replies(self, count: int) -> list[str]:
    async with self._received:
      await asyncio.wait_for(
          self._received.wait_for(lambda: len(self.replies) >= count),
          timeout=10,
      )
    return self.replies

  @property
  def url(self) -> str:
    return str(self.server.make_url("/")).rstrip("/")

  @property
  def replies(self) -> list[str]:
    return [r["body"]["text"] for r in self.requests]


def _make_runner(model: testing_utils.MockModel) -> Runner:
  return Runner(
      app_name=_APP_NAME,
      agent=LlmAgent(name="chat_agent", model=model),
      session_service=InMemorySessionService(),
      auto_create_session=True,
  )


def _dm_space() -> dict[str, Any]:
  return {
      "name": _DM_SPACE,
      "type": "DM",
      "spaceType": "DIRECT_MESSAGE",
      "singleUserBotDm": True,
      "spaceThreadingState": "UNTHREADED_MESSAGES",
  }


def _room_space() -> dict[str, Any]:
  return {
      "name": _ROOM_SPACE,
      "type": "ROOM",
      "spaceType": "SPACE",
      "spaceThreadingState": "THREADED_MESSAGES",
  }


def _message_event(
    text: str,
    *,
    space: dict[str, Any] | None = None,
    thread: str = "threads/T1",
    argument_text: str | None = None,
    sender: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Builds a MESSAGE event as sent to a Chat app that isn't an add-on."""
  space = space or _dm_space()
  sender = sender or {"name": _USER, "displayName": "User", "type": "HUMAN"}
  message = {
      "name": f"{space['name']}/messages/M1",
      "sender": sender,
      "text": text,
      "argumentText": text if argument_text is None else argument_text,
      "thread": {"name": f"{space['name']}/{thread}"},
      "space": space,
  }
  return {
      "type": "MESSAGE",
      "eventTime": "2026-11-14T10:00:00Z",
      "message": message,
      "space": space,
      "user": sender,
  }


def _add_on_message_event(
    text: str, *, space: dict[str, Any] | None = None
) -> dict[str, Any]:
  """Builds the same message as sent to a Google Workspace add-on."""
  legacy = _message_event(text, space=space)
  return {
      "commonEventObject": {"hostApp": "CHAT", "platform": "WEB"},
      "chat": {
          "user": legacy["user"],
          "eventTime": legacy["eventTime"],
          "messagePayload": {
              "space": legacy["space"],
              "message": legacy["message"],
          },
      },
  }


@contextlib.asynccontextmanager
async def _chat_client(
    runner: Runner,
    chat_api: _FakeChatApi,
    *,
    audience: str | None = None,
    service_account_email: str = "chat@system.gserviceaccount.com",
    credentials: auth_credentials.Credentials | None = None,
) -> AsyncIterator[TestClient]:
  """Serves the Chat endpoint for the duration of a test.

  The server drains background replies on shutdown, so it must close before
  the fake Chat API does.
  """
  chat_runner = GoogleChatRunner(
      runner=runner,
      credentials=credentials or _FakeCredentials(),
      api_endpoint=chat_api.url,
  )
  app = chat_runner.create_app(
      audience=audience, service_account_email=service_account_email
  )
  async with TestClient(TestServer(app)) as client:
    yield client


async def _post(
    client: TestClient,
    event: dict[str, Any],
    *,
    token: str | None = None,
) -> tuple[int, Any]:
  headers = {"Authorization": f"Bearer {token}"} if token else {}
  response = await client.post("/", json=event, headers=headers)
  body = (
      await response.json()
      if response.content_type == "application/json"
      else None
  )
  return response.status, body


async def _sessions(runner: Runner) -> list[Session]:
  listed = await runner.session_service.list_sessions(
      app_name=_APP_NAME, user_id=_USER
  )
  return [
      await runner.session_service.get_session(
          app_name=_APP_NAME, user_id=_USER, session_id=s.id
      )
      for s in listed.sessions
  ]


def _user_texts(session: Session) -> list[str]:
  return [e.content.parts[0].text for e in session.events if e.author == "user"]


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


async def test_direct_message_gets_agent_reply_in_same_space():
  """A direct message is answered in the DM with the agent's text."""
  runner = _make_runner(testing_utils.MockModel.create(["Hi from ADK!"]))
  credentials = _FakeCredentials()

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api, credentials=credentials) as client,
  ):
    status, body = await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert status == 200
  assert body == {}
  assert replies == ["Hi from ADK!"]
  request = chat_api.requests[0]
  assert request["space"] == _DM_SPACE
  assert request["authorization"] == "Bearer app-token"
  # Unthreaded conversations get a plain message, not a thread reply.
  assert "thread" not in request["body"]
  assert request["params"] == {}
  assert credentials.refresh_count == 1


async def test_request_is_acknowledged_before_agent_finishes():
  """Chat gets its HTTP response without waiting for the agent's reply.

  Chat shows an error if the endpoint takes longer than 30 seconds, so a slow
  agent must not hold the request open.
  """
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["done"]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    status, _ = await _post(client, _message_event("Hello"))
    replies_while_blocked = list(chat_api.replies)
    release.set()
    replies = await chat_api.wait_for_replies(1)

  assert status == 200
  assert replies_while_blocked == []
  assert replies == ["done"]


async def test_direct_messages_share_one_session():
  """Messages in an unthreaded DM build up a single session.

  Each top-level DM message carries its own thread name, so keying on the
  thread would start a new conversation for every message.
  """
  runner = _make_runner(testing_utils.MockModel.create(["one", "two"]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("first", thread="threads/T1"))
    await chat_api.wait_for_replies(1)
    await _post(client, _message_event("second", thread="threads/T2"))
    await chat_api.wait_for_replies(2)

  sessions = await _sessions(runner)
  assert len(sessions) == 1
  assert _user_texts(sessions[0]) == ["first", "second"]


async def test_space_threads_get_their_own_sessions_and_replies():
  """In a threaded space, each thread is its own conversation."""
  runner = _make_runner(testing_utils.MockModel.create(["a", "b", "c"]))
  space = _room_space()

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("t1 first", space=space))
    await chat_api.wait_for_replies(1)
    await _post(
        client, _message_event("t2 first", space=space, thread="threads/T2")
    )
    await chat_api.wait_for_replies(2)
    await _post(client, _message_event("t1 second", space=space))
    await chat_api.wait_for_replies(3)

  sessions = await _sessions(runner)
  assert sorted(_user_texts(s) for s in sessions) == [
      ["t1 first", "t1 second"],
      ["t2 first"],
  ]
  first = chat_api.requests[0]
  assert first["space"] == _ROOM_SPACE
  assert first["body"]["thread"] == {"name": f"{_ROOM_SPACE}/threads/T1"}
  assert first["params"] == {
      "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"
  }


async def test_direct_message_sessions_follow_threads():
  """A DM with conversation sessions turned on keys each session on a thread."""
  runner = _make_runner(testing_utils.MockModel.create(["a", "b"]))
  space = _dm_space() | {"sessionState": "THREAD"}

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("one", space=space))
    await chat_api.wait_for_replies(1)
    await _post(client, _message_event("two", space=space, thread="threads/T2"))
    await chat_api.wait_for_replies(2)

  assert len(await _sessions(runner)) == 2
  assert chat_api.requests[1]["body"]["thread"] == {
      "name": f"{_DM_SPACE}/threads/T2"
  }


async def test_session_id_uses_only_url_safe_characters():
  """Session ids avoid the '/' in Chat resource names.

  Some session services, such as VertexAiSessionService, only accept letters,
  digits, '_' and '-'.
  """
  runner = _make_runner(testing_utils.MockModel.create(["ok"]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("hi", space=_room_space()))
    await chat_api.wait_for_replies(1)

  (session,) = await _sessions(runner)
  assert re.fullmatch(r"[A-Za-z0-9_-]+", session.id)


async def test_add_on_event_is_handled_like_a_chat_app_event():
  """Chat apps built as Workspace add-ons receive the same handling."""
  model = testing_utils.MockModel.create(["from add-on"])
  runner = _make_runner(model)

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    status, body = await _post(client, _add_on_message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert status == 200
  assert body == {}
  assert replies == ["from add-on"]
  assert model.requests[-1].contents[-1].parts[0].text == "Hello"


async def test_app_mention_is_removed_before_agent_sees_message():
  """The agent receives argumentText, which omits the app's @mention."""
  model = testing_utils.MockModel.create(["ok"])
  runner = _make_runner(model)

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(
        client,
        _message_event(
            "@Agent what is new?",
            space=_room_space(),
            argument_text=" what is new?",
        ),
    )
    await chat_api.wait_for_replies(1)

  assert model.requests[-1].contents[-1].parts[0].text == "what is new?"


async def test_messages_in_same_session_are_handled_in_order():
  """A second message waits for the agent to finish answering the first."""
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["one", "two"]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("first"))
    await _post(client, _message_event("second"))
    release.set()
    replies = await chat_api.wait_for_replies(2)

  (session,) = await _sessions(runner)
  assert replies == ["one", "two"]
  assert [e.author for e in session.events] == [
      "user",
      "chat_agent",
      "user",
      "chat_agent",
  ]


async def test_events_without_user_text_are_ignored():
  """Other event types, bot senders and empty messages get no reply."""
  model = testing_utils.MockModel.create(["ok"])
  runner = _make_runner(model)
  bot = {"name": "users/bot", "type": "BOT"}

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    added = {"type": "ADDED_TO_SPACE", "space": _room_space()}
    statuses = [
        (await _post(client, added))[0],
        (await _post(client, _message_event("hi", sender=bot)))[0],
        (await _post(client, _message_event("", argument_text="")))[0],
    ]
    await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert statuses == [200, 200, 200]
  assert replies == ["ok"]
  assert len(model.requests) == 1


async def test_thoughts_are_not_posted():
  """Only the agent's answer is posted, not its thought parts."""
  thought = LlmResponse(
      content=types.Content(
          role="model",
          parts=[
              types.Part(text="thinking...", thought=True),
              types.Part(text="answer"),
          ],
      )
  )
  runner = _make_runner(testing_utils.MockModel.create([thought]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert replies == ["answer"]


async def test_agent_error_sends_generic_reply_without_details():
  """A failing agent run replies with a generic message, not the exception."""
  model = testing_utils.MockModel.create(
      [], error=RuntimeError("secret internal detail")
  )
  runner = _make_runner(model)

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert len(replies) == 1
  assert "secret internal detail" not in replies[0]


async def test_error_event_without_content_sends_generic_reply():
  """A response blocked by the model, with no content, still gets a reply."""
  blocked = LlmResponse(error_code="SAFETY", error_message="Blocked by safety")
  runner = _make_runner(testing_utils.MockModel.create([blocked]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert len(replies) == 1
  assert "Blocked by safety" not in replies[0]


async def test_failed_post_sends_generic_reply():
  """If the Chat API rejects a reply, the user still gets an error reply."""

  class _FailFirstChatApi(_FakeChatApi):
    failed = False

    async def _record(self, request: web.Request) -> web.Response:
      if not self.failed:
        self.failed = True
        return web.Response(status=500)
      return await super()._record(request)

  runner = _make_runner(testing_utils.MockModel.create(["first"]))

  async with (
      _FailFirstChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    await _post(client, _message_event("Hello"))
    replies = await chat_api.wait_for_replies(1)

  assert replies == [_google_chat_runner._ERROR_REPLY]


async def test_shutdown_waits_for_replies_in_progress():
  """Stopping the server lets a reply that is already running finish."""
  release = asyncio.Event()
  runner = _make_runner(_blocked_model(release, ["done"]))

  async with _FakeChatApi() as chat_api:
    async with _chat_client(runner, chat_api) as client:
      await _post(client, _message_event("Hello"))
      closing = asyncio.create_task(client.close())
      await asyncio.sleep(0.1)
      assert not closing.done()
      release.set()
      await closing

    assert chat_api.replies == ["done"]


async def test_invalid_json_is_rejected():
  runner = _make_runner(testing_utils.MockModel.create(["unused"]))

  async with (
      _FakeChatApi() as chat_api,
      _chat_client(runner, chat_api) as client,
  ):
    response = await client.post("/", data="not json")

  assert response.status == 400
  assert chat_api.requests == []


class TestRequestVerification:
  """Requests must carry a bearer token signed by Google Chat."""

  @pytest.fixture
  def id_tokens(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replaces Google's token verification with a lookup table."""
    tokens: dict[str, Any] = {}
    calls: list[tuple[str, str, str | None]] = []

    def verify_oauth2_token(token, request, audience):
      del request
      calls.append(("id_token", audience, None))
      if token not in tokens:
        raise ValueError("bad token")
      return tokens[token]

    def verify_token(token, request, audience, certs_url):
      del request
      calls.append(("jwt", audience, certs_url))
      if token not in tokens:
        raise ValueError("bad token")
      return tokens[token]

    monkeypatch.setattr(
        _google_chat_runner.id_token,
        "verify_oauth2_token",
        verify_oauth2_token,
    )
    monkeypatch.setattr(
        _google_chat_runner.id_token, "verify_token", verify_token
    )
    return {"tokens": tokens, "calls": calls}

  async def test_request_without_token_is_rejected(self, id_tokens):
    runner = _make_runner(testing_utils.MockModel.create(["unused"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(runner, chat_api, audience=_ENDPOINT_URL) as client,
    ):
      status, _ = await _post(client, _message_event("Hello"))

    assert status == 401
    assert chat_api.requests == []

  async def test_invalid_token_is_rejected(self, id_tokens):
    runner = _make_runner(testing_utils.MockModel.create(["unused"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(runner, chat_api, audience=_ENDPOINT_URL) as client,
    ):
      status, _ = await _post(client, _message_event("Hello"), token="forged")

    assert status == 401
    assert chat_api.requests == []

  async def test_id_token_from_chat_is_accepted(self, id_tokens):
    id_tokens["tokens"]["good"] = {
        "email": "chat@system.gserviceaccount.com",
        "email_verified": True,
    }
    runner = _make_runner(testing_utils.MockModel.create(["ok"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(runner, chat_api, audience=_ENDPOINT_URL) as client,
    ):
      status, _ = await _post(client, _message_event("Hello"), token="good")
      replies = await chat_api.wait_for_replies(1)

    assert status == 200
    assert replies == ["ok"]
    assert id_tokens["calls"] == [("id_token", _ENDPOINT_URL, None)]

  async def test_id_token_from_another_account_is_rejected(self, id_tokens):
    """A valid Google ID token for the endpoint is not enough on its own."""
    id_tokens["tokens"]["other"] = {
        "email": "attacker@example.iam.gserviceaccount.com",
        "email_verified": True,
    }
    runner = _make_runner(testing_utils.MockModel.create(["unused"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(runner, chat_api, audience=_ENDPOINT_URL) as client,
    ):
      status, _ = await _post(client, _message_event("Hello"), token="other")

    assert status == 401

  async def test_add_on_service_account_is_accepted_when_configured(
      self, id_tokens
  ):
    add_on_account = (
        f"service-{_PROJECT_NUMBER}@gcp-sa-gsuiteaddons.iam.gserviceaccount.com"
    )
    id_tokens["tokens"]["add-on"] = {
        "email": add_on_account,
        "email_verified": True,
    }
    runner = _make_runner(testing_utils.MockModel.create(["ok"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(
            runner,
            chat_api,
            audience=_ENDPOINT_URL,
            service_account_email=add_on_account,
        ) as client,
    ):
      status, _ = await _post(
          client, _add_on_message_event("Hello"), token="add-on"
      )
      await chat_api.wait_for_replies(1)

    assert status == 200

  async def test_project_number_audience_verifies_chat_signed_jwt(
      self, id_tokens
  ):
    id_tokens["tokens"]["jwt"] = {"iss": "chat@system.gserviceaccount.com"}
    runner = _make_runner(testing_utils.MockModel.create(["ok"]))

    async with (
        _FakeChatApi() as chat_api,
        _chat_client(runner, chat_api, audience=_PROJECT_NUMBER) as client,
    ):
      status, _ = await _post(client, _message_event("Hello"), token="jwt")
      await chat_api.wait_for_replies(1)

    assert status == 200
    assert id_tokens["calls"] == [(
        "jwt",
        _PROJECT_NUMBER,
        _google_chat_runner._CERTS_URL + "chat@system.gserviceaccount.com",
    )]


@pytest.mark.parametrize(
    ("use_mtls_endpoint", "expected_host"),
    [("never", "chat.googleapis.com"), ("always", "chat.mtls.googleapis.com")],
)
def test_default_api_endpoint_follows_mtls_setting(
    monkeypatch: pytest.MonkeyPatch, use_mtls_endpoint: str, expected_host: str
):
  """The Chat API host switches to its mTLS variant when mTLS is in use."""
  monkeypatch.setenv("GOOGLE_API_USE_MTLS_ENDPOINT", use_mtls_endpoint)
  runner = _make_runner(testing_utils.MockModel.create([]))

  chat_runner = GoogleChatRunner(runner=runner, credentials=_FakeCredentials())

  assert chat_runner._api_endpoint == f"https://{expected_host}"
