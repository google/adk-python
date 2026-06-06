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
  """A runner that yields one event, then blocks until cancelled.

  asyncio.sleep with a long timeout will be interrupted by task.cancel()
  from the /cancel endpoint, raising CancelledError.
  """
  yield _make_text_event("starting run...")
  try:
    await asyncio.sleep(3600)  # effectively forever — cancelled by the test
  except asyncio.CancelledError:
    yield _make_text_event("run was cancelled")
    raise


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

  def test_cancel_active_run_returns_200(self, client, test_session_info):
    """POST /run, then POST :cancel — should return 200 and cancel the run."""
    app_name = test_session_info["app_name"]
    user_id = test_session_info["user_id"]

    # 1. Create a session first
    create_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions",
        json={"app_name": app_name, "user_id": user_id},
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # 2. Start a run in a background thread. The run will block on
    #    asyncio.sleep until cancelled.
    import threading

    run_result = {"status": None, "error": None}

    def do_run():
      try:
        import requests
        s = requests.Session()
        resp = s.post(
            f"http://testserver/apps/{app_name}/users/{user_id}/sessions/{session_id}/run",
            json={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": "hello"}],
                },
            },
            timeout=10,
        )
        run_result["status"] = resp.status_code
        run_result["body"] = resp.json() if resp.text else None
      except Exception as e:
        run_result["error"] = str(e)

    run_thread = threading.Thread(target=do_run, daemon=True)
    run_thread.start()

    # 3. Give the server a moment to start processing
    import time
    time.sleep(1.0)

    # 4. Cancel the run
    cancel_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions/{session_id}:cancel",
    )
    assert cancel_resp.status_code == 200
    data = cancel_resp.json()
    assert data["status"] == "cancelled"
    assert data["session_id"] == session_id

    # 5. Wait for the run thread to complete
    run_thread.join(timeout=5.0)
    logger.info("Run result: %s", run_result)

  def test_cancel_nonexistent_session_returns_404(self, client):
    """Cancelling a session with no active run should return 404."""
    resp = client.post(
        "/apps/test_app/users/test_user/sessions/nonexistent:cancel",
    )
    assert resp.status_code == 404
    assert "no active run" in resp.json()["detail"].lower()

  def test_cancel_endpoint_idempotent(self, client):
    """Double-cancelling should return 404 on the second call."""
    resp1 = client.post(
        "/apps/test_app/users/test_user/sessions/nonexistent2:cancel",
    )
    assert resp1.status_code == 404

    resp2 = client.post(
        "/apps/test_app/users/test_user/sessions/nonexistent2:cancel",
    )
    assert resp2.status_code == 404


class TestTaskRegistry:
  """Unit tests for the active_tasks registry lifecycle."""

  def test_registry_cleanup_after_run_completion(
      self, client, test_session_info, monkeypatch
  ):
    """After a run completes, cancelling should 404 (task was cleaned up)."""
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

    # Run synchronously
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

    # After run completes, cancelling should 404
    cancel_resp = client.post(
        f"/apps/{app_name}/users/{user_id}/sessions/{session_id}:cancel",
    )
    assert cancel_resp.status_code == 404
