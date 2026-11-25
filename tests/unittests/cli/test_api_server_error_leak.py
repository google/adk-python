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

"""Tests that error responses from the API server do not echo exception text."""

import sys
import tempfile
import types
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from google.adk.cli.dev_server import DevServer
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.utils._error_reporting import ENV_DEBUG_ERRORS
from google.api_core import exceptions as api_exceptions
import pytest

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


_ENGINE_ID = "projects/123456/locations/us-central1/reasoningEngines/987"


class _MismatchedSessionService(InMemorySessionService):
  """Raises like VertexAiSessionService does for another engine's session id."""

  async def create_session(self, **kwargs):
    raise ValueError(f"service is configured for {_ENGINE_ID}")


class _CorruptStateSessionService(InMemorySessionService):

  async def create_session(self, **kwargs):
    raise ValueError("Persisted session state must be a JSON object.")


@pytest.mark.parametrize(
    ("service", "leaked"),
    [
        (_MismatchedSessionService(), _ENGINE_ID),
        (_CorruptStateSessionService(), "Persisted session state"),
    ],
)
def test_session_create_400_does_not_echo_exception_text(service, leaked):
  response = _create_session(_client(service))

  assert response.status_code == 400
  assert leaked not in response.text
  assert "error_id" in response.json()["detail"]


def test_session_create_400_debug_env_restores_exception_text(monkeypatch):
  monkeypatch.setenv(ENV_DEBUG_ERRORS, "1")

  response = _create_session(_client(_MismatchedSessionService()))

  assert response.status_code == 400
  assert _ENGINE_ID in response.json()["detail"]


def _install_fake_agent_identity_client(monkeypatch, error):
  """Fakes the optional agent-identity package so finalize raises `error`."""

  class _Client:

    def __init__(self, **kwargs):
      pass

    def finalize_credentials(self, request):
      raise error

  module = types.ModuleType("google.cloud.agentidentitycredentials_v1")
  module.AuthProviderCredentialsServiceClient = _Client
  module.FinalizeCredentialsRequest = lambda **kwargs: kwargs
  monkeypatch.setitem(
      sys.modules, "google.cloud.agentidentitycredentials_v1", module
  )


def _finalize(client):
  return client.post(
      "/agent-identity/finalize",
      json={
          "connectorName": "projects/p/locations/l/connectors/github",
          "userId": "u",
          "userIdValidationState": "c3RhdGU",
          "consentNonce": "nonce",
      },
  )


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (api_exceptions.InvalidArgument(f"bad nonce for {_SECRET}"), 400),
        (api_exceptions.PermissionDenied(f"no access to {_SECRET}"), 403),
        (api_exceptions.InternalServerError(f"backend {_SECRET} down"), 500),
        (RuntimeError(f"could not reach {_SECRET}"), 500),
    ],
)
def test_finalize_credentials_errors_do_not_echo_exception_text(
    monkeypatch, error, status_code
):
  _install_fake_agent_identity_client(monkeypatch, error)

  response = _finalize(_client(InMemorySessionService()))

  assert response.status_code == status_code
  assert _SECRET not in response.text
  assert "error_id" in response.json()["detail"]
