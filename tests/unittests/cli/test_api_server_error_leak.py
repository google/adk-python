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

"""Tests that 500 responses from the API server do not echo exception text."""

import tempfile
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from google.adk.cli.dev_server import DevServer
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.utils._error_reporting import ENV_DEBUG_ERRORS

_SECRET = "postgresql://adk:hunter2@10.0.3.7/sessions"


class _BrokenSessionService(InMemorySessionService):

  async def create_session(self, **kwargs):
    raise RuntimeError(f"could not connect to {_SECRET}")


def _client(session_service):
  server = DevServer(
      agent_loader=MagicMock(),
      session_service=session_service,
      memory_service=MagicMock(),
      artifact_service=MagicMock(),
      credential_service=MagicMock(),
      eval_sets_manager=MagicMock(),
      eval_set_results_manager=None,
      agents_dir=tempfile.mkdtemp(),
  )
  return TestClient(server.get_fast_api_app(), base_url="http://localhost")


def _create_session(client):
  return client.post("/apps/a/users/u/sessions/s1", json={"state": {}})


def test_session_create_500_does_not_echo_exception_text():
  response = _create_session(_client(_BrokenSessionService()))

  assert response.status_code == 500
  assert _SECRET not in response.text
  assert "error_id" in response.json()["detail"]


def test_debug_env_restores_exception_text(monkeypatch):
  monkeypatch.setenv(ENV_DEBUG_ERRORS, "1")

  response = _create_session(_client(_BrokenSessionService()))

  assert response.status_code == 500
  assert _SECRET in response.json()["detail"]
