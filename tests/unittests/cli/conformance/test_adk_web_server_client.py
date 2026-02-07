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

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.cli.adk_web_server import RunAgentRequest
from google.adk.cli.conformance.adk_web_server_client import AdkWebServerClient
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest


def test_init_default_values():
  client = AdkWebServerClient()
  assert client.base_url == "http://127.0.0.1:8000"
  assert client.timeout == 30.0


def test_init_custom_values():
  client = AdkWebServerClient(
      base_url="https://custom.example.com/", timeout=60.0
  )
  assert client.base_url == "https://custom.example.com"
  assert client.timeout == 60.0


def test_init_strips_trailing_slash():
  client = AdkWebServerClient(base_url="http://test.com/")
  assert client.base_url == "http://test.com"


@pytest.mark.asyncio
async def test_get_session():
  client = AdkWebServerClient()

  # Mock the HTTP response
  mock_response = MagicMock()
  mock_response.json.return_value = {
      "id": "test_session",
      "app_name": "test_app",
      "user_id": "test_user",
      "events": [],
      "state": {},
  }

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client_class.return_value = mock_client

    session = await client.get_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    assert isinstance(session, Session)
    assert session.id == "test_session"
    mock_client.get.assert_called_once_with(
        "/apps/test_app/users/test_user/sessions/test_session"
    )


@pytest.mark.asyncio
async def test_create_session():
  client = AdkWebServerClient()

  # Mock the HTTP response
  mock_response = MagicMock()
  mock_response.json.return_value = {
      "id": "new_session",
      "app_name": "test_app",
      "user_id": "test_user",
      "events": [],
      "state": {"key": "value"},
  }

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client_class.return_value = mock_client

    session = await client.create_session(
        app_name="test_app", user_id="test_user", state={"key": "value"}
    )

    assert isinstance(session, Session)
    assert session.id == "new_session"
    mock_client.post.assert_called_once_with(
        "/apps/test_app/users/test_user/sessions",
        json={"state": {"key": "value"}},
    )


@pytest.mark.asyncio
async def test_delete_session():
  client = AdkWebServerClient()

  # Mock the HTTP response
  mock_response = MagicMock()

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.delete.return_value = mock_response
    mock_client_class.return_value = mock_client

    await client.delete_session(
        app_name="test_app", user_id="test_user", session_id="test_session"
    )

    mock_client.delete.assert_called_once_with(
        "/apps/test_app/users/test_user/sessions/test_session"
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_update_session():
  client = AdkWebServerClient()

  # Mock the HTTP response
  mock_response = MagicMock()
  mock_response.json.return_value = {
      "id": "test_session",
      "app_name": "test_app",
      "user_id": "test_user",
      "events": [],
      "state": {"key": "updated_value", "new_key": "new_value"},
  }

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.patch.return_value = mock_response
    mock_client_class.return_value = mock_client

    state_delta = {"key": "updated_value", "new_key": "new_value"}
    session = await client.update_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state_delta=state_delta,
    )

    assert isinstance(session, Session)
    assert session.id == "test_session"
    assert session.state == {"key": "updated_value", "new_key": "new_value"}
    mock_client.patch.assert_called_once_with(
        "/apps/test_app/users/test_user/sessions/test_session",
        json={"state_delta": state_delta},
    )
    mock_response.raise_for_status.assert_called_once()


@pytest.mark.asyncio
async def test_run_agent():
  client = AdkWebServerClient()

  # Create sample events
  event1 = Event(
      author="test_agent",
      invocation_id="test_invocation_1",
      content=types.Content(role="model", parts=[types.Part(text="Hello")]),
  )
  event2 = Event(
      author="test_agent",
      invocation_id="test_invocation_2",
      content=types.Content(role="model", parts=[types.Part(text="World")]),
  )

  # Mock streaming response
  class MockStreamResponse:

    def raise_for_status(self):
      pass

    async def aiter_lines(self):
      yield f"data:{json.dumps(event1.model_dump())}"
      yield "data:"  # Empty line should be ignored
      yield f"data:{json.dumps(event2.model_dump())}"

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
      pass

  def mock_stream(*_args, **_kwargs):
    return MockStreamResponse()

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.stream = mock_stream
    mock_client_class.return_value = mock_client

    request = RunAgentRequest(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(
            role="user", parts=[types.Part(text="Hello")]
        ),
    )

    events = []
    async for event in client.run_agent(request):
      events.append(event)

    assert len(events) == 2
    assert all(isinstance(event, Event) for event in events)
    assert events[0].invocation_id == "test_invocation_1"
    assert events[1].invocation_id == "test_invocation_2"


@pytest.mark.asyncio
async def test_run_agent_raises_on_streamed_error():
  client = AdkWebServerClient()

  class MockStreamResponse:

    def raise_for_status(self):
      pass

    async def aiter_lines(self):
      yield 'data: {"error": "boom"}'

    async def __aenter__(self):
      return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
      pass

  def mock_stream(*_args, **_kwargs):
    return MockStreamResponse()

  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client.stream = mock_stream
    mock_client_class.return_value = mock_client

    request = RunAgentRequest(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        new_message=types.Content(role="user", parts=[types.Part(text="Hi")]),
    )

    with pytest.raises(RuntimeError, match="boom"):
      async for _ in client.run_agent(request):
        pass


@pytest.mark.asyncio
async def test_close():
  client = AdkWebServerClient()

  # Create a mock client to close
  with patch("httpx.AsyncClient") as mock_client_class:
    mock_client = AsyncMock()
    mock_client_class.return_value = mock_client

    # Force client creation
    async with client._get_client():
      pass

    # Now close should work
    await client.close()
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager():
  async with AdkWebServerClient() as client:
    assert isinstance(client, AdkWebServerClient)
