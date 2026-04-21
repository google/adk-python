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

"""Integration tests for /trigger/* endpoints.

Tests exercise the full FastAPI request → TriggerRouter → Runner pipeline
using the same TestClient pattern as test_fast_api.py, with a mocked Runner
that returns deterministic events.
"""

import asyncio
import base64
import json
import signal
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.cli import fast_api as fast_api_module
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.trigger_routes import _is_transient_error
from google.adk.cli.trigger_routes import TransientError
from google.adk.cli.trigger_routes import TriggerRouter
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

# ---------------------------------------------------------------------------
# Dummy agent & mocked runner (same pattern as test_fast_api.py)
# ---------------------------------------------------------------------------


class DummyAgent(BaseAgent):

  def __init__(self, name):
    super().__init__(name=name)
    self.sub_agents = []


root_agent = DummyAgent(name="trigger_test_agent")


def _model_event(text: str = "Agent reply") -> Event:
  return Event(
      author="trigger_test_agent",
      invocation_id="inv-trigger",
      content=types.Content(
          role="model",
          parts=[types.Part(text=text)],
      ),
  )


async def dummy_run_async(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
):
  """Mocked Runner.run_async that echoes input text back."""
  # Extract the input text to echo it back — proves the pipeline works e2e
  input_text = ""
  if new_message and new_message.parts:
    input_text = new_message.parts[0].text or ""

  yield _model_event(f"Processed: {input_text}")
  await asyncio.sleep(0)


async def dummy_run_async_error(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
):
  """Mocked Runner.run_async that raises an exception."""
  raise RuntimeError("Agent crashed")
  yield  # make it an async generator  # noqa: E305


def _make_rate_limit_runner(fail_count: int):
  """Create a runner that fails with 429 `fail_count` times, then succeeds."""
  call_count = {"value": 0}

  async def dummy_run_async_rate_limit(
      self,
      user_id,
      session_id,
      new_message,
      state_delta=None,
      run_config: Optional[RunConfig] = None,
      invocation_id: Optional[str] = None,
  ):
    call_count["value"] += 1
    if call_count["value"] <= fail_count:
      raise RuntimeError("429 Resource has been exhausted")
    input_text = ""
    if new_message and new_message.parts:
      input_text = new_message.parts[0].text or ""
    yield _model_event(f"Processed: {input_text}")
    await asyncio.sleep(0)

  return dummy_run_async_rate_limit


async def dummy_run_async_always_429(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
):
  """Mocked Runner.run_async that always raises a 429 error."""
  raise RuntimeError("RESOURCE_EXHAUSTED: 429 quota exceeded")
  yield  # noqa: E305


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def patch_runner(monkeypatch):
  monkeypatch.setattr(Runner, "run_async", dummy_run_async)


@pytest.fixture
def mock_agent_loader():

  class MockAgentLoader:

    def __init__(self, agents_dir: str):
      pass

    def load_agent(self, app_name):
      return root_agent

    def list_agents(self):
      return ["test_app"]

    def list_agents_detailed(self):
      return [{
          "name": "test_app",
          "root_agent_name": "trigger_test_agent",
          "description": "Test agent for triggers",
          "language": "python",
          "is_computer_use": False,
      }]

  return MockAgentLoader(".")


@pytest.fixture
def mock_session_service():
  return InMemorySessionService()


@pytest.fixture
def mock_artifact_service():
  service = AsyncMock()
  service.list_artifact_keys = AsyncMock(return_value=[])
  return service


@pytest.fixture
def mock_memory_service():
  return AsyncMock()


def _make_test_client(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    trigger_sources: Optional[list[str]] = None,
) -> TestClient:
  """Build a TestClient with the given trigger setting."""
  with (
      patch.object(signal, "signal", autospec=True, return_value=None),
      patch.object(
          fast_api_module,
          "create_session_service_from_options",
          autospec=True,
          return_value=mock_session_service,
      ),
      patch.object(
          fast_api_module,
          "create_artifact_service_from_options",
          autospec=True,
          return_value=mock_artifact_service,
      ),
      patch.object(
          fast_api_module,
          "create_memory_service_from_options",
          autospec=True,
          return_value=mock_memory_service,
      ),
      patch.object(
          fast_api_module,
          "AgentLoader",
          autospec=True,
          return_value=mock_agent_loader,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetsManager",
          autospec=True,
          return_value=AsyncMock(),
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=AsyncMock(),
      ),
  ):
    app = get_fast_api_app(
        agents_dir=".",
        web=False,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        trigger_sources=trigger_sources,
    )
    return TestClient(app)


@pytest.fixture
def client(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
):
  """TestClient with all triggers enabled."""
  return _make_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      trigger_sources=["pubsub", "eventarc"],
  )


@pytest.fixture
def client_no_triggers(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
):
  """TestClient with triggers disabled (default)."""
  return _make_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      trigger_sources=None,
  )


# ===================================================================
# /apps/test_app/trigger/pubsub — Pub/Sub Push Subscription
# ===================================================================


class TestTriggerPubSub:
  """Integration tests for the Pub/Sub push subscription trigger."""

  def test_success(self, client, monkeypatch):
    """Valid Pub/Sub message is processed and returns success."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    message_data = base64.b64encode(b"Hello from Pub/Sub").decode("utf-8")
    payload = {
        "message": {
            "data": message_data,
            "messageId": "msg-001",
        },
        "subscription": "projects/my-project/subscriptions/my-sub",
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == "Hello from Pub/Sub"
    assert parsed_msg["attributes"] == {}

  def test_message_with_attributes(self, client, monkeypatch):
    """Pub/Sub message with attributes (no data) is processed."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "message": {
            "attributes": {"key": "value", "action": "process"},
            "messageId": "msg-002",
        },
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] is None
    assert parsed_msg["attributes"] == {"key": "value", "action": "process"}

  def test_json_payload_in_data(self, client, monkeypatch):
    """JSON-encoded data in Pub/Sub message is decoded properly."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    inner_json = json.dumps({"order_id": 42, "amount": 99.99})
    message_data = base64.b64encode(inner_json.encode("utf-8")).decode("utf-8")
    payload = {
        "message": {
            "data": message_data,
            "messageId": "msg-003",
        },
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == {"order_id": 42, "amount": 99.99}
    assert parsed_msg["attributes"] == {}

  def test_invalid_base64_returns_400(self, client):
    """Invalid base64 data returns 400."""
    payload = {
        "message": {
            "data": "!!!not-valid-base64!!!",
            "messageId": "msg-bad",
        },
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 400
    assert "base64" in resp.json()["detail"].lower()

  def test_agent_error_returns_500(self, client, monkeypatch):
    """Agent failure returns 500, allowing Pub/Sub to retry."""
    monkeypatch.setattr(Runner, "run_async", dummy_run_async_error)

    message_data = base64.b64encode(b"trigger error").decode("utf-8")
    payload = {
        "message": {"data": message_data},
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 500
    assert "Agent processing failed" in resp.json()["detail"]

  def test_with_subscription_metadata(self, client):
    """Subscription field is used for user_id derivation."""
    message_data = base64.b64encode(b"test").decode("utf-8")
    payload = {
        "message": {"data": message_data},
        "subscription": "projects/p/subscriptions/orders-sub",
    }
    resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 200

  def test_unknown_app_fails_early(
      self, client, mock_agent_loader, mock_session_service
  ):
    """Unknown app fails early and does NOT create a session."""

    def load_agent_raising(app_name):
      if app_name == "unknown_app":
        raise Exception("App not found")
      return root_agent

    mock_agent_loader.load_agent = load_agent_raising

    message_data = base64.b64encode(b"test").decode("utf-8")
    payload = {
        "message": {"data": message_data},
    }
    resp = client.post("/apps/unknown_app/trigger/pubsub", json=payload)

    assert resp.status_code == 500
    assert "unknown_app" not in mock_session_service.sessions


# ===================================================================
# /apps/test_app/trigger/eventarc — Eventarc / CloudEvents
# ===================================================================


class TestTriggerEventarc:
  """Integration tests for the Eventarc / CloudEvents trigger."""

  def test_success(self, client, monkeypatch):
    """Valid CloudEvent payload is processed and returns success."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "data": {
            "bucket": "my-bucket",
            "name": "path/to/file.pdf",
            "contentType": "application/pdf",
        },
        "source": "storage.googleapis.com",
        "type": "google.cloud.storage.object.v1.finalized",
        "id": "evt-001",
        "specversion": "1.0",
    }
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == payload["data"]
    assert parsed_msg["attributes"]["ce-id"] == "evt-001"
    assert (
        parsed_msg["attributes"]["ce-type"]
        == "google.cloud.storage.object.v1.finalized"
    )

  def test_source_derived_from_body(self, client):
    """Source from body is used for user_id."""
    payload = {
        "data": {"key": "value"},
        "source": "my-custom-source",
    }
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 200

  def test_source_from_ce_header(self, client):
    """ce-source header is used when body source is absent."""
    payload = {
        "data": {"key": "value"},
    }
    resp = client.post(
        "/apps/test_app/trigger/eventarc",
        json=payload,
        headers={"ce-source": "header-source"},
    )
    assert resp.status_code == 200

  def test_complex_event_data(self, client, monkeypatch):
    """Complex nested event data is serialized as JSON for the agent."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "data": {
            "resource": {
                "name": "projects/p/topics/t",
                "labels": {"env": "prod"},
            },
            "insertId": "abc123",
            "timestamp": "2026-01-01T00:00:00Z",
        },
    }
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == payload["data"]

  def test_agent_error_returns_500(self, client, monkeypatch):
    """Agent failure returns 500, allowing Eventarc to retry."""
    monkeypatch.setattr(Runner, "run_async", dummy_run_async_error)

    payload = {
        "data": {"trigger": "error"},
    }
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 500
    assert "Agent processing failed" in resp.json()["detail"]

  def test_minimal_payload(self, client, monkeypatch):
    """Minimal payload with just data field works."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {"data": {}}
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)
    assert resp.status_code == 200

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == {}

  def test_structured_mode_pubsub_wrapper(self, client, monkeypatch):
    """Eventarc structured mode with Pub/Sub envelope is base64-decoded."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    inner_message = "Hello from structured Eventarc"
    encoded_message = base64.b64encode(inner_message.encode("utf-8")).decode(
        "utf-8"
    )
    payload = {
        "data": {
            "message": {
                "data": encoded_message,
            }
        },
        "source": "my-source",
    }
    resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == "Hello from structured Eventarc"
    assert parsed_msg["attributes"] == {}

  def test_binary_content_mode_pubsub_wrapper(self, client, monkeypatch):
    """Binary content mode: Pub/Sub message wrapper in body, CE attrs in headers."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "message": {
            "data": base64.b64encode(b"hello from eventarc").decode(),
            "messageId": "evt-msg-001",
        },
        "subscription": "projects/p/subscriptions/eventarc-sub",
    }
    resp = client.post(
        "/apps/test_app/trigger/eventarc",
        json=payload,
        headers={
            "ce-source": "//pubsub.googleapis.com/projects/p/topics/t",
            "ce-type": "google.cloud.pubsub.topic.v1.messagePublished",
            "ce-id": "binary-test-1",
            "ce-specversion": "1.0",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] == "hello from eventarc"
    assert parsed_msg["attributes"] == {}

  def test_binary_content_mode_attributes_only(self, client, monkeypatch):
    """Binary content mode with attributes only (no data)."""
    captured_messages = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_messages.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "message": {
            "attributes": {"key": "value"},
            "messageId": "evt-msg-002",
        },
    }
    resp = client.post(
        "/apps/test_app/trigger/eventarc",
        json=payload,
        headers={"ce-source": "//pubsub.googleapis.com/test"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    assert len(captured_messages) == 1
    parsed_msg = json.loads(captured_messages[0])
    assert parsed_msg["data"] is None
    assert parsed_msg["attributes"] == {"key": "value"}

  def test_binary_content_mode_arbitrary_payload(self, client, monkeypatch):
    """Binary content mode with arbitrary JSON payload (not Pub/Sub)."""
    captured_message = []

    async def dummy_run_async_capture(
        self, user_id, session_id, new_message, **kwargs
    ):
      captured_message.append(new_message.parts[0].text)
      yield _model_event("Success")
      await asyncio.sleep(0)

    monkeypatch.setattr(Runner, "run_async", dummy_run_async_capture)

    payload = {
        "bucket": "my-bucket",
        "name": "file.txt",
        "contentType": "application/json",
    }
    resp = client.post(
        "/apps/test_app/trigger/eventarc",
        json=payload,
        headers={
            "ce-source": (
                "//storage.googleapis.com/projects/_/buckets/my-bucket"
            ),
            "ce-type": "google.cloud.storage.object.v1.finalized",
            "ce-id": "12345",
            "ce-specversion": "1.0",
        },
    )
    assert resp.status_code == 200
    assert len(captured_message) == 1
    received_data = json.loads(captured_message[0])
    assert received_data["data"]["bucket"] == "my-bucket"
    assert received_data["data"]["name"] == "file.txt"
    assert received_data["attributes"]["ce-id"] == "12345"


# ===================================================================
# Triggers disabled (default behavior)
# ===================================================================


class TestTriggersDisabled:
  """Verify trigger endpoints return 404 when not enabled."""

  def test_pubsub_returns_404(self, client_no_triggers):
    resp = client_no_triggers.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"x").decode()}},
    )
    assert resp.status_code == 404

  def test_eventarc_returns_404(self, client_no_triggers):
    resp = client_no_triggers.post(
        "/apps/test_app/trigger/eventarc", json={"data": {}}
    )
    assert resp.status_code == 404


# ===================================================================
# Transient error detection
# ===================================================================


class TestTransientErrorDetection:
  """Unit tests for the _is_transient_error helper."""

  def test_429_in_message(self):
    assert _is_transient_error(RuntimeError("HTTP 429 Too Many Requests"))

  def test_resource_exhausted(self):
    assert _is_transient_error(RuntimeError("RESOURCE_EXHAUSTED"))

  def test_rate_limit(self):
    assert _is_transient_error(RuntimeError("rate limit exceeded"))

  def test_quota(self):
    assert _is_transient_error(RuntimeError("quota exceeded for project"))

  def test_non_transient(self):
    assert not _is_transient_error(RuntimeError("Agent crashed"))

  def test_permission_denied(self):
    assert not _is_transient_error(RuntimeError("PERMISSION_DENIED"))


# ===================================================================
# Retry with exponential backoff
# ===================================================================


class TestRetryLogic:
  """Integration tests for retry with exponential backoff on 429 errors."""

  def test_pubsub_retry_exhausted_returns_500(self, client, monkeypatch):
    """Pub/Sub trigger returns 500 when retries are exhausted."""
    monkeypatch.setattr(Runner, "run_async", dummy_run_async_always_429)

    with patch(
        "google.adk.cli.trigger_routes.asyncio.sleep", new_callable=AsyncMock
    ):
      message_data = base64.b64encode(b"429 test").decode("utf-8")
      payload = {"message": {"data": message_data}}
      resp = client.post("/apps/test_app/trigger/pubsub", json=payload)

    assert resp.status_code == 500
    assert "Rate limit" in resp.json()["detail"]

  def test_eventarc_retry_exhausted_returns_500(self, client, monkeypatch):
    """Eventarc trigger returns 500 when retries are exhausted."""
    monkeypatch.setattr(Runner, "run_async", dummy_run_async_always_429)

    with patch(
        "google.adk.cli.trigger_routes.asyncio.sleep", new_callable=AsyncMock
    ):
      payload = {"data": {"test": "429"}}
      resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 500
    assert "Rate limit" in resp.json()["detail"]

  def test_non_transient_error_not_retried(self, client, monkeypatch):
    """Non-429 errors are NOT retried — they fail immediately."""
    call_count = 0

    async def counting_error_runner(
        self, user_id, session_id, new_message, **kwargs
    ):
      nonlocal call_count
      call_count += 1
      raise RuntimeError("PERMISSION_DENIED: no access")
      yield  # noqa: E305

    monkeypatch.setattr(Runner, "run_async", counting_error_runner)

    with patch(
        "google.adk.cli.trigger_routes.asyncio.sleep", new_callable=AsyncMock
    ):
      payload = {"data": {"test": True}}
      resp = client.post("/apps/test_app/trigger/eventarc", json=payload)

    assert resp.status_code == 500
    # Non-transient errors should NOT be retried — only 1 call
    assert call_count == 1


# ===================================================================
# Semaphore / concurrency control
# ===================================================================


class TestConcurrencyControl:
  """Tests for semaphore-based concurrency limiting."""

  def test_concurrent_pubsub_and_eventarc(self, client):
    """Multiple trigger types can be called without semaphore starvation."""
    # Pub/Sub
    ps_resp = client.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"ps").decode()}},
    )
    assert ps_resp.status_code == 200

    # Eventarc
    ea_resp = client.post(
        "/apps/test_app/trigger/eventarc",
        json={"data": {"key": "value"}},
    )
    assert ea_resp.status_code == 200


# ===================================================================
# Selective trigger registration
# ===================================================================


class TestSelectiveRegistration:
  """Tests that only requested trigger sources are registered."""

  def test_only_pubsub(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    """When trigger_sources=['pubsub'], only Pub/Sub is available."""
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=["pubsub"],
    )
    # Pub/Sub should work
    ps_resp = client.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"test").decode()}},
    )
    assert ps_resp.status_code == 200

    # Eventarc should NOT be available
    ea_resp = client.post("/apps/test_app/trigger/eventarc", json={"data": {}})
    assert ea_resp.status_code == 404

  def test_only_eventarc(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    """When trigger_sources=['eventarc'], only Eventarc is available."""
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=["eventarc"],
    )
    # Eventarc should work
    ea_resp = client.post(
        "/apps/test_app/trigger/eventarc", json={"data": {"k": "v"}}
    )
    assert ea_resp.status_code == 200

    # Pub/Sub should NOT be available
    ps_resp = client.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"x").decode()}},
    )
    assert ps_resp.status_code == 404


class TestUnknownTriggerSources:
  """Verify unknown trigger sources are filtered and warned about."""

  def test_unknown_source_ignored(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    """Unknown source is silently dropped; valid sources still work."""
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=["unknown_source", "pubsub"],
    )
    # "pubsub" should still be registered
    ps_resp = client.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"test").decode()}},
    )
    assert ps_resp.status_code == 200

    # "unknown_source" should NOT be registered
    unknown_resp = client.post(
        "/apps/test_app/trigger/unknown_source", json={"calls": [["test"]]}
    )
    assert unknown_resp.status_code == 404

  def test_all_unknown_sources_results_in_no_endpoints(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    """All invalid sources means no trigger endpoints registered."""
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=["foo", "bar"],
    )
    unknown_resp = client.post(
        "/apps/test_app/trigger/unknown_source", json={"calls": [["test"]]}
    )
    assert unknown_resp.status_code == 404


class TestTriggersDisabled:
  """Verify trigger endpoints return 404 when not enabled."""

  def test_pubsub_returns_404(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=[],
    )
    resp = client.post(
        "/apps/test_app/trigger/pubsub",
        json={"message": {"data": base64.b64encode(b"x").decode()}},
    )
    assert resp.status_code == 404

  def test_eventarc_returns_404(
      self,
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
  ):
    client = _make_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        trigger_sources=[],
    )
    resp = client.post("/apps/test_app/trigger/eventarc", json={"data": {}})
    assert resp.status_code == 404
