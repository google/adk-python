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
import logging
from typing import Optional
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.cli import fast_api as fast_api_module
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest

logger = logging.getLogger("google_adk." + __name__)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

# Shared mutable flag so the mocked runner can signal that cancellation
# was actually detected (CancelledError caught).
_cancellation_signal: list[bool] = []


def _make_text_event(text: str) -> Event:
  return Event(
      author="test_agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="model", parts=[types.Part(text=text)]
      ),
  )


async def _cancellable_run_async(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
):
  """Yields one event, then blocks until cancelled via task.cancel().

  Sets ``_cancellation_signal[0] = True`` when CancelledError is caught,
  so the test can verify the cancellation propagated to the runner.
  """
  _cancellation_signal.clear()
  yield _make_text_event("starting run...")
  try:
    await asyncio.sleep(3600)  # cancelled by the /cancel endpoint
  except asyncio.CancelledError:
    _cancellation_signal.append(True)
    yield _make_text_event("run was cancelled")
    raise


@pytest.fixture(autouse=True)
def _clear_cancellation_signal():
  """Reset the shared cancellation signal before each test."""
  _cancellation_signal.clear()


@pytest.fixture
def test_session_info():
  return {
      "app_name": "test_app",
      "user_id": "test_user",
  }


@pytest.fixture
def mock_agent_loader():
  """Minimal agent loader that returns a single LlmAgent."""

  class Loader:
    def load_agent(self, app_name):
      agent = LlmAgent(name=app_name, model="gemini-2.5-flash")
      return agent

    def list_apps(self):
      return ["test_app"]

    def list_app_info(self):
      return [{"name": "test_app", "description": "Test app"}]

  return Loader()


@pytest.fixture
def client(monkeypatch, mock_agent_loader):
  """Create a TestClient for the FastAPI app with a cancellable runner."""
  monkeypatch.setattr(Runner, "run_async", _cancellable_run_async)
  session_service = InMemorySessionService()

  app = fast_api_module.get_fast_api_app(
      agent_loader=mock_agent_loader,
      session_service=session_service,
  )
  return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCancelSessionEndpoint:
  """Integration tests for POST /apps/.../sessions/...:cancel."""

  def test_cancel_active_run_interrupts_runner(
      self, client, test_session_info
  ):
    """Start a blocking run, cancel it, and verify the runner was interrupted."""
    app_name = test_session_info["app_name"]
    user_id = test_session_info["user_id"]

    # 1. Create a session
    create_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions",
        json={"app_name": app_name, "user_id": user_id},
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # 2. Start a blocking run in a background thread.
    #    Use the TestClient (not raw requests) so the call reaches
    #    the in-memory FastAPI app.  TestClient.post() is synchronous
    #    and will block until the server responds — which only happens
    #    after we cancel the run in step 4.
    import threading

    run_result = {"status": None, "error": None}
    run_started = threading.Event()

    def do_run(test_client):
      try:
        resp = test_client.post(
            f"/apps/{app_name}/users/{user_id}"
            f"/sessions/{session_id}/run",
            json={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": "hello"}],
                },
            },
        )
        run_result["status"] = resp.status_code
        run_result["body"] = resp.json() if resp.text else None
      except Exception as e:
        run_result["error"] = str(e)

    run_thread = threading.Thread(
        target=do_run, args=(client,), daemon=True
    )
    run_thread.start()

    # 3. Wait for the runner to start processing (signal from the
    #    mocked runner that it entered the cancellation-sensitive block).
    #    The runner yields one event before blocking, so the thread
    #    will have sent the request and be waiting on the response.
    import time
    time.sleep(0.5)

    # 4. Cancel the run via the new endpoint
    cancel_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions/{session_id}:cancel",
    )
    assert cancel_resp.status_code == 200
    data = cancel_resp.json()
    assert data["status"] == "cancelled"
    assert data["session_id"] == session_id

    # 5. Wait for the background run to finish (should happen quickly
    #    after cancellation)
    run_thread.join(timeout=5.0)
    assert not run_thread.is_alive(), (
        "Background run thread should have completed after cancellation"
    )

    # 6. Verify the runner actually detected cancellation.
    #    The _cancellable_run_async sets this flag when CancelledError
    #    is caught inside the runner coroutine.
    assert len(_cancellation_signal) > 0, (
        "CancelledError was NOT raised inside the runner — "
        "the task.cancel() did not propagate to the agent coroutine"
    )
    logger.info("Run result after cancellation: %s", run_result)

  def test_cancel_nonexistent_session_returns_404(self, client):
    """Cancelling a session with no active run returns 404."""
    resp = client.post(
        "/apps/test_app/users/test_user/sessions/nonexistent:cancel",
    )
    assert resp.status_code == 404
    assert "no active run" in resp.json()["detail"].lower()

  def test_cancel_idempotent_returns_404_on_second_call(self, client):
    """Double-cancelling the same session returns 404 on the second call."""
    url = "/apps/test_app/users/test_user/sessions/nonexistent:cancel"
    assert client.post(url).status_code == 404
    assert client.post(url).status_code == 404


class TestTaskRegistry:
  """Tests for the active_tasks registry lifecycle."""

  def test_registry_cleanup_after_run_completion(
      self, client, test_session_info, monkeypatch
  ):
    """After a run completes normally, /cancel returns 404 (task cleaned up)."""
    async def fast_run(self, **kwargs):
      yield _make_text_event("done")

    monkeypatch.setattr(Runner, "run_async", fast_run)

    app_name = test_session_info["app_name"]
    user_id = test_session_info["user_id"]

    create_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions",
        json={"app_name": app_name, "user_id": user_id},
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # Run to completion
    run_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions/{session_id}/run",
        json={
            "app_name": app_name,
            "user_id": user_id,
            "session_id": session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": "hello"}],
            },
        },
    )
    assert run_resp.status_code == 200

    # Task should already be popped from registry
    cancel_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions/{session_id}:cancel",
    )
    assert cancel_resp.status_code == 404
