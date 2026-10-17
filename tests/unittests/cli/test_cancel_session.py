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
import threading

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.credential_service.in_memory_credential_service import InMemoryCredentialService
from google.adk.cli.api_server import ApiServer
from google.adk.cli.utils.base_agent_loader import BaseAgentLoader
from google.adk.evaluation.local_eval_set_results_manager import LocalEvalSetResultsManager
from google.adk.evaluation.local_eval_sets_manager import LocalEvalSetsManager
from google.adk.events.event import Event
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
import pytest


@pytest.fixture
def running_agent():
  """A real agent exposes when its cancellable wait begins and ends."""
  started = threading.Event()
  cancelled = threading.Event()

  class TestAgent(BaseAgent):
    blocking: bool = True

    async def _run_async_impl(self, ctx):
      yield Event(
          author=self.name,
          invocation_id=ctx.invocation_id,
          content=types.Content(
              role="model", parts=[types.Part(text="started")]
          ),
      )
      if self.blocking:
        started.set()
        try:
          await asyncio.sleep(30)
        except asyncio.CancelledError:
          cancelled.set()
          raise

    async def _run_live_impl(self, ctx):
      async for event in self._run_async_impl(ctx):
        yield event

  return TestAgent(name="test_agent"), started, cancelled


@pytest.fixture
def client(tmp_path, running_agent):
  """Use the real API server, runner, and in-memory services."""
  agent, _, _ = running_agent

  class Loader(BaseAgentLoader):

    def load_agent(self, agent_name):
      return agent

    def list_agents(self):
      return ["test_app"]

  server = ApiServer(
      agent_loader=Loader(),
      session_service=InMemorySessionService(),
      memory_service=InMemoryMemoryService(),
      artifact_service=InMemoryArtifactService(),
      credential_service=InMemoryCredentialService(),
      eval_sets_manager=LocalEvalSetsManager(str(tmp_path)),
      eval_set_results_manager=LocalEvalSetResultsManager(str(tmp_path)),
      agents_dir=str(tmp_path),
  )
  with TestClient(server.get_fast_api_app()) as test_client:
    yield test_client


@pytest.fixture
def session_id(client):
  response = client.post("/apps/test_app/users/test_user/sessions", json={})
  assert response.status_code == 200
  return response.json()["id"]


def _run_request(session_id):
  return {
      "app_name": "test_app",
      "user_id": "test_user",
      "session_id": session_id,
      "new_message": {"role": "user", "parts": [{"text": "hello"}]},
  }


def _cancel_url(session_id):
  return f"/apps/test_app/users/test_user/sessions/{session_id}:cancel"


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_cancel_active_run_interrupts_agent(
    client, session_id, running_agent, endpoint
):
  """The cancel endpoint interrupts an agent reached through the real runner."""
  _, started, cancelled = running_agent
  run_result = {}

  def do_run():
    try:
      run_result["response"] = client.post(
          endpoint, json=_run_request(session_id)
      )
    except Exception as error:
      run_result["error"] = error

  run_thread = threading.Thread(target=do_run, daemon=True)
  run_thread.start()
  try:
    assert started.wait(5), "The request never reached the agent"
    response = client.post(_cancel_url(session_id))
    assert response.status_code == 200
    assert response.json() == {"status": "cancelled", "session_id": session_id}
    assert cancelled.wait(5), "CancelledError did not reach the agent"
    run_thread.join(5)
    assert not run_thread.is_alive()
    if endpoint == "/run_sse":
      assert run_result["response"].status_code == 200
      assert "started" in run_result["response"].text
    assert client.post(_cancel_url(session_id)).status_code == 404
  finally:
    client.post(_cancel_url(session_id))
    run_thread.join(5)


def test_cancel_live_run_interrupts_agent(client, session_id, running_agent):
  """The same endpoint cancels the agent behind a live websocket run."""
  _, started, cancelled = running_agent
  with client.websocket_connect(
      f"/run_live?app_name=test_app&user_id=test_user&session_id={session_id}"
  ) as websocket:
    assert websocket.receive_json()["content"]["parts"][0]["text"] == "started"
    assert started.wait(5)
    assert client.post(_cancel_url(session_id)).status_code == 200
    assert cancelled.wait(5)


def test_cancel_nonexistent_session_returns_404(client):
  """A session with no active run has nothing to cancel."""
  response = client.post(_cancel_url("nonexistent"))
  assert response.status_code == 404
  assert "no active run" in response.json()["detail"].lower()


def test_cancel_repeated_missing_session_returns_404(client):
  """Repeated cancellation of a missing task does not create a task."""
  url = _cancel_url("nonexistent")
  assert client.post(url).status_code == 404
  assert client.post(url).status_code == 404


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_registry_cleanup_after_run_completion(
    client, session_id, running_agent, endpoint
):
  """Completed runs are no longer cancellable."""
  agent, _, _ = running_agent
  agent.blocking = False
  response = client.post(endpoint, json=_run_request(session_id))
  assert response.status_code == 200
  assert client.post(_cancel_url(session_id)).status_code == 404
