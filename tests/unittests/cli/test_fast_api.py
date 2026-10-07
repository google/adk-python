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

import asyncio
import json
import logging
import os
from pathlib import Path
import signal
import sys
import tempfile
from typing import Any
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.testclient import TestClient
from google.adk.a2a import _compat
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.auth.auth_credential import _redact_credential_secrets
from google.adk.cli import api_server as api_server_module
from google.adk.cli import fast_api as fast_api_module
from google.adk.cli import service_registry as service_registry_module
from google.adk.cli.api_server import RunAgentRequest
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils.base_agent_loader import _AgentLoadError
from google.adk.errors.input_validation_error import InputValidationError
from google.adk.errors.session_not_found_error import SessionNotFoundError
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_case import SessionInput
from google.adk.evaluation.eval_metrics import EvalStatus
from google.adk.evaluation.eval_result import EvalCaseResult
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.events._internal_metadata import INTERNAL_METADATA_PREFIX
from google.adk.events._internal_metadata import RESTORED_EVENT_KEY
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.bigquery_agent_analytics_plugin import BigQueryAgentAnalyticsPlugin
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.api_core.exceptions import GoogleAPICallError
from google.api_core.exceptions import InvalidArgument
from google.genai import types
from pydantic import BaseModel
import pytest
from starlette.applications import Starlette
import starlette.requests
from starlette.routing import Mount

# Configure logging to help diagnose server startup issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("google_adk." + __name__)

# An app told it binds 127.0.0.1 rejects requests addressed to any other host,
# so its client cannot use TestClient's default "http://testserver".
_LOOPBACK_BASE_URL = "http://127.0.0.1:8000"


# Here we create a dummy agent module that get_fast_api_app expects
class DummyAgent(BaseAgent):

  def __init__(self, name):
    super().__init__(name=name)
    self.sub_agents = []


root_agent = DummyAgent(name="dummy_agent")


# Create sample events that our mocked runner will return
def _event_1():
  return Event(
      author="dummy agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="model", parts=[types.Part(text="LLM reply", inline_data=None)]
      ),
  )


def _event_2():
  return Event(
      author="dummy agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  text=None,
                  inline_data=types.Blob(
                      mime_type="audio/pcm;rate=24000", data=b"\x00\xFF"
                  ),
              )
          ],
      ),
  )


def _event_3():
  return Event(
      author="dummy agent", invocation_id="invocation_id", interrupted=True
  )


def _event_state_delta(state_delta: dict[str, Any]):
  return Event(
      author="dummy agent",
      invocation_id="invocation_id",
      actions=EventActions(state_delta=state_delta),
  )


# Define mocked async generator functions for the Runner
async def dummy_run_live(self, session, live_request_queue, **kwargs):
  yield _event_1()
  await asyncio.sleep(0)

  yield _event_2()
  await asyncio.sleep(0)

  yield _event_3()


_ORIGINAL_RUNNER_RUN_ASYNC = Runner.run_async


async def dummy_run_async(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
    invocation_id: Optional[str] = None,
    abort_signal: Optional[asyncio.Event] = None,
    **kwargs,
):
  run_config = run_config or RunConfig()
  yield _event_1()
  await asyncio.sleep(0)

  yield _event_2()
  await asyncio.sleep(0)

  yield _event_3()
  await asyncio.sleep(0)

  if state_delta is not None:
    yield _event_state_delta(state_delta)


# Define a local mock for EvalCaseResult specific to fast_api tests
class _MockEvalCaseResult(BaseModel):
  eval_set_id: str
  eval_id: str
  final_eval_status: Any
  user_id: str
  session_id: str
  eval_set_file: str
  eval_metric_results: list = {}
  overall_eval_metric_results: list = ({},)
  eval_metric_result_per_invocation: list = {}


#################################################
# Test Fixtures
#################################################


@pytest.fixture(autouse=True)
def patch_runner(monkeypatch):
  """Patch the Runner methods to use our dummy implementations."""
  monkeypatch.setattr(Runner, "run_live", dummy_run_live)
  monkeypatch.setattr(Runner, "run_async", dummy_run_async)


@pytest.fixture
def test_session_info():
  """Return test user and session IDs for testing."""
  return {
      "app_name": "test_app",
      "user_id": "test_user",
      "session_id": "test_session",
  }


@pytest.fixture
def mock_agent_loader():

  class MockAgentLoader:

    def __init__(self, agents_dir: str):
      pass

    def load_agent(self, app_name):
      if app_name == "yaml_app" or app_name == "bq_app":
        agent = DummyAgent(name="yaml_agent")
        agent._config = MagicMock(logging=None)
        return agent
      return root_agent

    def list_agents(self):
      return ["test_app", "yaml_app", "bq_app"]

    def list_agents_detailed(self):
      return [
          {
              "name": "test_app",
              "root_agent_name": "test_agent",
              "description": "A test agent for unit testing",
              "language": "python",
              "is_computer_use": False,
          },
          {
              "name": "yaml_app",
              "root_agent_name": "yaml_agent",
              "description": "A yaml agent for unit testing",
              "language": "yaml",
              "is_computer_use": False,
          },
          {
              "name": "bq_app",
              "root_agent_name": "yaml_agent",
              "description": "A bq agent for unit testing",
              "language": "yaml",
              "is_computer_use": False,
          },
      ]

  return MockAgentLoader(".")


@pytest.fixture
def mock_session_service():
  """Create an in-memory session service instance for testing."""
  return InMemorySessionService()


@pytest.fixture
def mock_artifact_service():
  """Create a mock artifact service."""

  artifacts: dict[str, list[dict[str, Any]]] = {}

  def _artifact_key(
      app_name: str, user_id: str, session_id: Optional[str], filename: str
  ) -> str:
    if session_id is None:
      return f"{app_name}:{user_id}:user:{filename}"
    return f"{app_name}:{user_id}:{session_id}:{filename}"

  def _canonical_uri(
      app_name: str,
      user_id: str,
      session_id: Optional[str],
      filename: str,
      version: int,
  ) -> str:
    if session_id is None:
      return (
          f"artifact://apps/{app_name}/users/{user_id}/artifacts/"
          f"{filename}/versions/{version}"
      )
    return (
        f"artifact://apps/{app_name}/users/{user_id}/sessions/{session_id}/"
        f"artifacts/{filename}/versions/{version}"
    )

  class MockArtifactService:

    def __init__(self):
      self._artifacts = artifacts
      self.save_artifact_side_effect: Optional[BaseException] = None

    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        artifact: types.Part,
        session_id: Optional[str] = None,
        custom_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
      if self.save_artifact_side_effect is not None:
        effect = self.save_artifact_side_effect
        if isinstance(effect, BaseException):
          raise effect
        raise TypeError(
            "save_artifact_side_effect must be an exception instance."
        )
      key = _artifact_key(app_name, user_id, session_id, filename)
      entries = artifacts.setdefault(key, [])
      version = len(entries)
      artifact_version = ArtifactVersion(
          version=version,
          canonical_uri=_canonical_uri(
              app_name, user_id, session_id, filename, version
          ),
          custom_metadata=custom_metadata or {},
      )
      if artifact.inline_data is not None:
        artifact_version.mime_type = artifact.inline_data.mime_type
      elif artifact.text is not None:
        artifact_version.mime_type = "text/plain"
      elif artifact.file_data is not None:
        artifact_version.mime_type = artifact.file_data.mime_type

      entries.append({
          "version": version,
          "artifact": artifact,
          "metadata": artifact_version,
      })
      return version

    def add_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: types.Part,
        custom_metadata: Optional[dict[str, Any]] = None,
        canonical_uri: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> int:
      """Synchronous helper for tests to add artifacts."""
      key = _artifact_key(app_name, user_id, session_id, filename)
      entries = artifacts.setdefault(key, [])
      version = len(entries)
      artifact_version = ArtifactVersion(
          version=version,
          canonical_uri=(
              canonical_uri
              or _canonical_uri(
                  app_name, user_id, session_id, filename, version
              )
          ),
          custom_metadata=custom_metadata or {},
      )
      if mime_type:
        artifact_version.mime_type = mime_type
      elif artifact.inline_data is not None:
        artifact_version.mime_type = artifact.inline_data.mime_type
      elif artifact.text is not None:
        artifact_version.mime_type = "text/plain"
      elif artifact.file_data is not None:
        artifact_version.mime_type = artifact.file_data.mime_type

      entries.append({
          "version": version,
          "artifact": artifact,
          "metadata": artifact_version,
      })
      return version

    async def load_artifact(
        self, app_name, user_id, session_id, filename, version=None
    ):
      """Load an artifact by filename."""
      key = _artifact_key(app_name, user_id, session_id, filename)
      if key not in artifacts:
        return None

      if version is not None:
        for entry in artifacts[key]:
          if entry["version"] == version:
            return entry["artifact"]
        return None

      return artifacts[key][-1]["artifact"]

    async def list_artifact_keys(self, app_name, user_id, session_id):
      """List artifact names for a session."""
      prefix = f"{app_name}:{user_id}:{session_id}:"
      return [
          key.split(":")[-1]
          for key in artifacts.keys()
          if key.startswith(prefix)
      ]

    async def list_versions(self, app_name, user_id, session_id, filename):
      """List versions of an artifact."""
      key = _artifact_key(app_name, user_id, session_id, filename)
      if key not in artifacts:
        return []
      return [entry["version"] for entry in artifacts[key]]

    async def list_artifact_versions(
        self, app_name, user_id, session_id, filename
    ):
      """List all artifact versions with metadata."""
      key = _artifact_key(app_name, user_id, session_id, filename)
      if key not in artifacts:
        return []
      return [entry["metadata"] for entry in artifacts[key]]

    async def delete_artifact(self, app_name, user_id, session_id, filename):
      """Delete an artifact."""
      key = _artifact_key(app_name, user_id, session_id, filename)
      artifacts.pop(key, None)

    async def get_artifact_version(
        self,
        *,
        app_name: str,
        user_id: str,
        filename: str,
        session_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[ArtifactVersion]:
      key = _artifact_key(app_name, user_id, session_id, filename)
      entries = artifacts.get(key)
      if not entries:
        return None
      if version is None:
        return entries[-1]["metadata"]
      for entry in entries:
        if entry["version"] == version:
          return entry["metadata"]
      return None

  return MockArtifactService()


@pytest.fixture
def mock_memory_service():
  """Create a mock memory service."""
  return AsyncMock()


@pytest.fixture
def mock_eval_sets_manager():
  """Create a mock eval sets manager."""
  return InMemoryEvalSetsManager()


@pytest.fixture
def mock_eval_set_results_manager():
  """Create a mock local eval set results manager."""

  # Storage for eval set results.
  eval_set_results = {}

  class MockEvalSetResultsManager:
    """Mock eval set results manager."""

    def save_eval_set_result(self, app_name, eval_set_id, eval_case_results):
      if app_name not in eval_set_results:
        eval_set_results[app_name] = {}
      eval_set_result_id = f"{app_name}_{eval_set_id}_eval_result"
      eval_set_result = EvalSetResult(
          eval_set_result_id=eval_set_result_id,
          eval_set_result_name=eval_set_result_id,
          eval_set_id=eval_set_id,
          eval_case_results=eval_case_results,
      )
      if eval_set_result_id not in eval_set_results[app_name]:
        eval_set_results[app_name][eval_set_result_id] = eval_set_result
      else:
        eval_set_results[app_name][eval_set_result_id].append(eval_set_result)

    def get_eval_set_result(self, app_name, eval_set_result_id):
      if app_name not in eval_set_results:
        raise ValueError(f"App {app_name} not found.")
      if eval_set_result_id not in eval_set_results[app_name]:
        raise ValueError(
            f"Eval set result {eval_set_result_id} not found in app {app_name}."
        )
      return eval_set_results[app_name][eval_set_result_id]

    def list_eval_set_results(self, app_name):
      """List eval set results."""
      if app_name not in eval_set_results:
        raise ValueError(f"App {app_name} not found.")
      return list(eval_set_results[app_name].keys())

  return MockEvalSetResultsManager()


def _create_test_client(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    **app_kwargs,
):
  """Helper to create a TestClient with the given get_fast_api_app overrides."""
  defaults = dict(
      agents_dir=".",
      web=True,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )
  defaults.update(app_kwargs)
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
          "NestedAgentLoader",
          autospec=True,
          return_value=mock_agent_loader,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
  ):
    app = get_fast_api_app(**defaults)
    return TestClient(app, client=("127.0.0.1", 51234))


@pytest.mark.parametrize(
    "bind_host, expect_warning",
    [
        (None, False),
        ("127.0.0.1", False),
        ("localhost", False),
        ("::1", False),
        ("0.0.0.0", True),
        ("::", True),
        ("192.168.1.10", True),
    ],
)
def test_no_auth_warning_on_non_loopback_bind(
    bind_host,
    expect_warning,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    caplog,
):
  """Warns about missing auth only when bound to a reachable (non-loopback) address."""
  with caplog.at_level(logging.WARNING):
    _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        bind_host=bind_host,
    )
  warned = any(
      "has no authentication" in record.getMessage()
      for record in caplog.records
  )
  assert warned is expect_warning


def test_agent_with_bigquery_analytics_plugin(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Verify that plugins.yaml is correctly read to attach BigQueryAgentAnalyticsPlugin."""
  app_name = "bq_app"
  app_dir = tmp_path / app_name
  app_dir.mkdir(parents=True)

  plugins_yaml_content = """\
bigquery_agent_analytics:
  project_id: test-project
  dataset_id: test-dataset
  table_id: test-table
  dataset_location: US
"""
  (app_dir / "plugins.yaml").write_text(plugins_yaml_content)

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
          "NestedAgentLoader",
          autospec=True,
          return_value=mock_agent_loader,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
      patch.object(
          os.path,
          "exists",
          autospec=True,
          side_effect=lambda p: str(p).endswith("plugins.yaml")
          or str(p).endswith("root_agent.yaml"),
      ),
  ):
    from google.adk.cli.adk_web_server import AdkWebServer

    adk_web_server = AdkWebServer(
        agent_loader=mock_agent_loader,
        session_service=mock_session_service,
        memory_service=mock_memory_service,
        artifact_service=mock_artifact_service,
        credential_service=MagicMock(),
        eval_sets_manager=mock_eval_sets_manager,
        eval_set_results_manager=mock_eval_set_results_manager,
        agents_dir=str(tmp_path),
    )

    runner = asyncio.run(adk_web_server.get_runner_async(app_name))

    # Assert that the plugin was attached
    assert any(
        isinstance(p, BigQueryAgentAnalyticsPlugin) for p in runner.app.plugins
    )

    # Check the configuration of the plugin
    bq_plugin = next(
        p
        for p in runner.app.plugins
        if isinstance(p, BigQueryAgentAnalyticsPlugin)
    )
    assert bq_plugin.project_id == "test-project"
    assert bq_plugin.dataset_id == "test-dataset"
    assert bq_plugin.table_id == "test-table"
    assert bq_plugin.location == "US"

    # Assert that the internal visual builder flag is set on the app
    assert getattr(runner.app, "_is_visual_builder_app", False) is True


def test_get_runner_async_accepts_internal_special_agent_name(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  from google.adk.cli.adk_web_server import AdkWebServer

  special_app_name = "__adk_agent_builder_assistant"
  special_agent = DummyAgent(name="agent_builder_assistant")
  mock_agent_loader.load_agent = MagicMock(return_value=special_agent)

  adk_web_server = AdkWebServer(
      agent_loader=mock_agent_loader,
      session_service=mock_session_service,
      memory_service=mock_memory_service,
      artifact_service=mock_artifact_service,
      credential_service=MagicMock(),
      eval_sets_manager=mock_eval_sets_manager,
      eval_set_results_manager=mock_eval_set_results_manager,
      agents_dir=str(tmp_path),
  )

  runner = asyncio.run(adk_web_server.get_runner_async(special_app_name))

  assert runner.app.name == special_app_name
  assert runner.app.root_agent is special_agent
  mock_agent_loader.load_agent.assert_called_once_with(special_app_name)


def test_api_server_get_runner_async_rejects_internal_special_agent_name(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  from google.adk.cli.api_server import ApiServer

  special_app_name = "__adk_agent_builder_assistant"
  special_agent = DummyAgent(name="agent_builder_assistant")
  mock_agent_loader.load_agent = MagicMock(return_value=special_agent)

  api_server = ApiServer(
      agent_loader=mock_agent_loader,
      session_service=mock_session_service,
      memory_service=mock_memory_service,
      artifact_service=mock_artifact_service,
      credential_service=MagicMock(),
      eval_sets_manager=mock_eval_sets_manager,
      eval_set_results_manager=mock_eval_set_results_manager,
      agents_dir=str(tmp_path),
  )

  with pytest.raises(HTTPException) as exc_info:
    asyncio.run(api_server.get_runner_async(special_app_name))

  assert exc_info.value.status_code == 403
  assert (
      "Access to internal special agents is disabled in API server mode"
      in exc_info.value.detail
  )


@pytest.mark.parametrize(
    ("web", "bind_host", "expected"),
    [
        (True, "127.0.0.1", True),
        (True, "localhost", True),
        (True, "::1", True),
        (True, "0.0.0.0", False),
        (True, "::", False),
        (True, "192.168.1.10", False),
        (True, None, False),
        (False, "127.0.0.1", False),
    ],
)
def test_special_agents_allowed_only_on_loopback_web_server(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    web,
    bind_host,
    expected,
):
  # The agent builder assistant writes files the server imports, and the dev
  # server is unauthenticated, so it must not be reachable off the machine.
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=web,
      bind_host=bind_host,
  )

  assert mock_agent_loader._allow_special_agents is expected
  if not expected:
    # Refused by the server itself, not by a 500 from the loader.
    response = client.get(
        "/apps/__adk_agent_builder_assistant/app-info",
        headers={"host": "127.0.0.1:8000"},
    )
    assert response.status_code == 403
    assert "internal special agents" in response.json()["detail"]


@pytest.fixture
def test_app(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient for the FastAPI app without starting a server."""
  return _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


@pytest.fixture
def builder_test_app(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Return a dev-server app rooted in a temporary agents directory."""
  with (
      patch.object(signal, "signal", autospec=True, return_value=None),
      # Building the app adds tmp_path to sys.path; undo it for later tests.
      patch.object(sys, "path", list(sys.path)),
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
          "NestedAgentLoader",
          autospec=True,
          return_value=mock_agent_loader,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
  ):
    app = get_fast_api_app(
        agents_dir=str(tmp_path),
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=None,
        a2a=False,
        host="127.0.0.1",
        bind_host="127.0.0.1",
        port=8000,
    )
    return app


@pytest.fixture
def builder_test_client(builder_test_app):
  """A client that reaches the server from the machine it runs on."""
  return TestClient(
      builder_test_app,
      base_url=_LOOPBACK_BASE_URL,
      client=("127.0.0.1", 51234),
  )


@pytest.fixture
def remote_builder_test_client(builder_test_app):
  """A client that reaches the server from somewhere else on the network."""
  return TestClient(
      builder_test_app,
      base_url=_LOOPBACK_BASE_URL,
      client=("203.0.113.7", 51234),
  )


@pytest.fixture
async def create_test_session(
    test_app, test_session_info, mock_session_service
):
  """Create a test session using the mocked session service."""

  # Create the session directly through the mock service
  session = await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id=test_session_info["session_id"],
      state={},
  )

  logger.info(f"Created test session: {session.id}")
  return test_session_info


@pytest.fixture
async def create_test_eval_set(
    test_app, test_session_info, mock_eval_sets_manager
):
  """Create a test eval set using the mocked eval sets manager."""
  _ = mock_eval_sets_manager.create_eval_set(
      app_name=test_session_info["app_name"],
      eval_set_id="test_eval_set_id",
  )
  test_eval_case = EvalCase(
      eval_id="test_eval_case_id",
      conversation=[
          Invocation(
              invocation_id="test_invocation_id",
              user_content=types.Content(
                  parts=[types.Part(text="test_user_content")],
                  role="user",
              ),
          )
      ],
  )
  _ = mock_eval_sets_manager.add_eval_case(
      app_name=test_session_info["app_name"],
      eval_set_id="test_eval_set_id",
      eval_case=test_eval_case,
  )
  return test_session_info


@pytest.fixture
def temp_agents_dir_with_a2a():
  """Create a temporary agents directory with A2A agent configurations for testing."""
  with tempfile.TemporaryDirectory() as temp_dir:
    # Create test agent directory
    agent_dir = Path(temp_dir) / "test_a2a_agent"
    agent_dir.mkdir()

    # Create agent.json file
    agent_card = {
        "name": "test_a2a_agent",
        "description": "Test A2A agent",
        "version": "1.0.0",
        "url": "http://localhost:8000/a2a/test_a2a_agent",
        "capabilities": {},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [],
    }

    with open(agent_dir / "agent.json", "w") as f:
      json.dump(agent_card, f)

    # Create a simple agent.py file
    agent_py_content = """
from google.adk.agents.base_agent import BaseAgent

class TestA2AAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="test_a2a_agent")
"""

    with open(agent_dir / "agent.py", "w") as f:
      f.write(agent_py_content)

    yield temp_dir


@pytest.fixture
def test_app_with_a2a(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
  """Create a TestClient for the FastAPI app with A2A enabled."""
  # Mock A2A related classes
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=mock_memory_service,
      ),
      patch(
          "google.adk.cli.fast_api.AgentLoader",
          return_value=mock_agent_loader,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=mock_eval_sets_manager,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=mock_eval_set_results_manager,
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
          return_value=MagicMock(),
      ),
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    # Configure mocks
    mock_executor.return_value = MagicMock()
    mock_handler.return_value = MagicMock()

    # Mock A2AStarletteApplication
    mock_app_instance = MagicMock()
    mock_app_instance.routes.return_value = (
        []
    )  # Return empty routes for testing
    mock_a2a_app.return_value = mock_app_instance

    # Change to temp directory
    monkeypatch.chdir(temp_agents_dir_with_a2a)

    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=True,
        host="127.0.0.1",
        port=8000,
    )

    client = TestClient(app)
    yield client


@pytest.fixture
def test_app_with_gemini_enterprise(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Create a TestClient with gemini_enterprise_app_name set."""
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
  mock_agent_loader.list_agents = MagicMock(
      return_value=["test_app", "gemini_app"]
  )

  mock_adk_app_instance = MagicMock()
  mock_adk_app_instance._tmpl_attrs = {}

  async def get_session_impl(**kwargs):
    return {"result": "success", "kwargs": kwargs}

  mock_adk_app_instance.get_session = get_session_impl

  async def stream_query_impl(**kwargs):
    yield {"chunk": 1, "kwargs": kwargs}
    await asyncio.sleep(0)
    yield {"chunk": 2, "kwargs": kwargs}

  mock_adk_app_instance.stream_query = stream_query_impl

  with (
      patch("google.auth.default", return_value=(MagicMock(), "test-project")),
      patch("vertexai.init", new_callable=MagicMock) as mock_vertexai_init,
      patch(
          "vertexai.agent_engines.AdkApp", return_value=mock_adk_app_instance
      ) as mock_adk_app_cls,
      patch("google.adk.agents.Agent", new_callable=MagicMock),
      patch(
          "google.adk.telemetry._agent_engine.TopSpanProcessor",
          new_callable=MagicMock,
      ),
      patch(
          "google.adk.telemetry._agent_engine.get_propagated_context",
          new_callable=MagicMock,
      ),
  ):
    client = _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        gemini_enterprise_app_name="gemini_app",
    )
    client.mock_vertexai_init = mock_vertexai_init
    client.mock_adk_app_cls = mock_adk_app_cls
    client.mock_adk_app_instance = mock_adk_app_instance
    yield client


@pytest.fixture
def test_app_with_gemini_enterprise_sync_stream(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Like test_app_with_gemini_enterprise but stream_query is a sync generator.

  This exercises the inspect.isgenerator() branch in stream_reasoning_engine,
  where the sync iterator is adapted to an async iterator via a threadpool.
  """
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  mock_agent_loader.list_agents = MagicMock(
      return_value=["test_app", "gemini_app"]
  )

  mock_adk_app_instance = MagicMock()
  mock_adk_app_instance._tmpl_attrs = {}

  def stream_query_impl(**kwargs):
    yield {"chunk": 1, "kwargs": kwargs}
    yield {"chunk": 2, "kwargs": kwargs}

  mock_adk_app_instance.stream_query = stream_query_impl

  with (
      patch("google.auth.default", return_value=(MagicMock(), "test-project")),
      patch("vertexai.init", new_callable=MagicMock),
      patch(
          "vertexai.agent_engines.AdkApp", return_value=mock_adk_app_instance
      ),
      patch("google.adk.agents.Agent", new_callable=MagicMock),
      patch(
          "google.adk.telemetry._agent_engine.TopSpanProcessor",
          new_callable=MagicMock,
      ),
      patch(
          "google.adk.telemetry._agent_engine.get_propagated_context",
          new_callable=MagicMock,
      ),
  ):
    client = _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        gemini_enterprise_app_name="gemini_app",
    )
    yield client


#################################################
# Test Cases
#################################################


def test_list_apps(test_app):
  """Test listing available applications."""
  # Use the TestClient to make a request
  response = test_app.get("/list-apps")

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  logger.info(f"Listed apps: {data}")


def test_list_apps_detailed(test_app):
  """Test listing available applications with detailed metadata."""
  response = test_app.get("/list-apps?detailed=true")

  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, dict)
  assert "apps" in data
  assert isinstance(data["apps"], list)

  for app in data["apps"]:
    assert "name" in app
    assert "rootAgentName" in app
    assert "description" in app
    assert "language" in app
    assert app["language"] in ["yaml", "python"]
    assert "isComputerUse" in app
    assert not app["isComputerUse"]

  logger.info(f"Listed apps: {data}")


def test_get_adk_app_info_llm_agent(test_app, mock_agent_loader):
  """Test retrieving app info when root agent is an LlmAgent."""
  agent = LlmAgent(
      name="test_llm_agent", description="test description", model="test_model"
  )
  with patch.object(mock_agent_loader, "load_agent", return_value=agent):
    response = test_app.get("/apps/test_app/app-info")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_app"
    assert data["rootAgentName"] == "test_llm_agent"
    assert data["description"] == "test description"
    assert data["language"] == "python"
    assert "agents" in data
    assert "test_llm_agent" in data["agents"]


def test_get_adk_app_info_llm_agent_with_subagents(test_app, mock_agent_loader):
  """Test retrieving app info when root agent is an LlmAgent with sub_agents and tools."""

  def sub_tool1(a: int) -> str:
    """Sub tool 1."""
    return str(a)

  def sub_tool2(b: str) -> str:
    """Sub tool 2."""
    return b

  sub_agent1 = LlmAgent(
      name="sub_agent1",
      description="sub description 1",
      model="test_model",
      tools=[sub_tool1],
  )
  sub_agent2 = LlmAgent(
      name="sub_agent2",
      description="sub description 2",
      model="test_model",
      tools=[sub_tool2],
  )
  agent = LlmAgent(
      name="test_llm_agent",
      description="test description",
      model="test_model",
      sub_agents=[sub_agent1, sub_agent2],
  )
  with patch.object(mock_agent_loader, "load_agent", return_value=agent):
    response = test_app.get("/apps/test_app/app-info")
    assert response.status_code == 200
    data = response.json()
    assert data["rootAgentName"] == "test_llm_agent"
    assert "test_llm_agent" in data["agents"]
    assert "sub_agent1" in data["agents"]
    assert "sub_agent2" in data["agents"]

    # Verify tools for sub_agent1
    agent1_info = data["agents"]["sub_agent1"]
    assert "tools" in agent1_info
    assert len(agent1_info["tools"]) == 1
    tool1 = agent1_info["tools"][0]
    field_name1 = (
        "functionDeclarations"
        if "functionDeclarations" in tool1
        else "function_declarations"
    )
    assert field_name1 in tool1
    assert tool1[field_name1][0]["name"] == "sub_tool1"

    # Verify tools for sub_agent2
    agent2_info = data["agents"]["sub_agent2"]
    assert "tools" in agent2_info
    assert len(agent2_info["tools"]) == 1
    tool2 = agent2_info["tools"][0]
    field_name2 = (
        "functionDeclarations"
        if "functionDeclarations" in tool2
        else "function_declarations"
    )
    assert field_name2 in tool2
    assert tool2[field_name2][0]["name"] == "sub_tool2"


def test_get_adk_app_info_triple_nested_agents_with_tools(
    test_app, mock_agent_loader
):
  """Test retrieving app info when there are triple nested agents with tools."""

  def tool1(a: int) -> str:
    """Tool 1."""
    return str(a)

  def tool2(b: str) -> str:
    """Tool 2."""
    return b

  def tool3(c: float) -> str:
    """Tool 3."""
    return str(c)

  # Level 3 (deepest)
  agent3 = LlmAgent(
      name="agent3",
      description="Level 3 agent",
      model="test_model",
      tools=[tool3],
  )

  # Level 2
  agent2 = LlmAgent(
      name="agent2",
      description="Level 2 agent",
      model="test_model",
      tools=[tool2],
      sub_agents=[agent3],
  )

  # Level 1 (root)
  root_agent = LlmAgent(
      name="root_agent",
      description="Level 1 agent",
      model="test_model",
      tools=[tool1],
      sub_agents=[agent2],
  )

  with patch.object(mock_agent_loader, "load_agent", return_value=root_agent):
    response = test_app.get("/apps/test_app/app-info")
    assert response.status_code == 200
    data = response.json()
    assert data["rootAgentName"] == "root_agent"
    assert "root_agent" in data["agents"]
    assert "agent2" in data["agents"]
    assert "agent3" in data["agents"]

    # Verify each has its tools
    for agent_name, exp_tool_name in [
        ("root_agent", "tool1"),
        ("agent2", "tool2"),
        ("agent3", "tool3"),
    ]:
      ai = data["agents"][agent_name]
      assert len(ai["tools"]) == 1
      tool = ai["tools"][0]
      field_name = (
          "functionDeclarations"
          if "functionDeclarations" in tool
          else "function_declarations"
      )
      assert tool[field_name][0]["name"] == exp_tool_name


def test_get_adk_app_info_llm_agent_with_function_tool(
    test_app, mock_agent_loader
):
  """Test retrieving app info when root agent has tools."""

  def my_tool(a: int, b: str) -> str:
    """A dummy tool function."""
    return f"{a} {b}"

  agent = LlmAgent(
      name="test_llm_agent",
      description="test description",
      model="test_model",
      tools=[my_tool],
  )
  with patch.object(mock_agent_loader, "load_agent", return_value=agent):
    response = test_app.get("/apps/test_app/app-info")
    assert response.status_code == 200
    data = response.json()
    assert data["rootAgentName"] == "test_llm_agent"
    assert "test_llm_agent" in data["agents"]
    agent_info = data["agents"]["test_llm_agent"]
    assert "tools" in agent_info
    assert len(agent_info["tools"]) == 1

    # Verify tool serialization
    tool = agent_info["tools"][0]
    func_decls = tool["functionDeclarations"]
    assert len(func_decls) == 1
    assert func_decls[0]["name"] == "my_tool"


def test_get_adk_app_info_non_llm_agent(test_app, mock_agent_loader):
  """Test retrieving app info when root agent is not an LlmAgent raises 400."""
  agent = DummyAgent("dummy_agent")
  with patch.object(mock_agent_loader, "load_agent", return_value=agent):
    response = test_app.get("/apps/test_app/app-info")
    assert response.status_code == 400
    assert "Root agent is not an LlmAgent" in response.json()["detail"]


def test_get_adk_app_info_unknown_app_returns_404(test_app, mock_agent_loader):
  """Test app-info returns 404 when the app_name matches no agent."""
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=ValueError("Agent not found: unknown_app"),
  ):
    response = test_app.get("/apps/unknown_app/app-info")
    assert response.status_code == 404
    assert "Agent not found: unknown_app" in response.json()["detail"]


def test_agent_run_unknown_app_returns_404(test_app, mock_agent_loader):
  """Test /run returns 404 instead of 500 when the app_name matches no agent."""
  payload = {
      "app_name": "unknown_app",
      "user_id": "test_user",
      "session_id": "test_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": False,
  }
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=ValueError("Agent not found: unknown_app"),
  ):
    response = test_app.post("/run", json=payload)
    assert response.status_code == 404
    assert "Agent not found: unknown_app" in response.json()["detail"]


def test_agent_run_sse_unknown_app_returns_404(test_app, mock_agent_loader):
  """Test /run_sse returns 404 instead of 500 when the app_name matches no agent."""
  payload = {
      "app_name": "unknown_app",
      "user_id": "test_user",
      "session_id": "test_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
  }
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=ValueError("Agent not found: unknown_app"),
  ):
    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 404
    assert "Agent not found: unknown_app" in response.json()["detail"]


def test_get_adk_app_info_load_failure_returns_500(test_app, mock_agent_loader):
  """Test app-info returns 500, not 404, when the agent fails to load."""
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=_AgentLoadError(
          "Fail to load 'broken_app' module. ToolConfig"
      ),
  ):
    response = test_app.get("/apps/broken_app/app-info")
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to load agent"


def test_get_adk_app_info_loader_http_error_is_preserved(
    test_app, mock_agent_loader
):
  """Test a loader's own HTTPException keeps its status."""
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=HTTPException(status_code=403, detail="Not your agent"),
  ):
    response = test_app.get("/apps/forbidden_app/app-info")
    assert response.status_code == 403
    assert response.json()["detail"] == "Not your agent"


def test_agent_run_sse_load_failure_returns_500(test_app, mock_agent_loader):
  """Test /run_sse returns 500, not 404, when the agent fails to load."""
  payload = {
      "app_name": "broken_app",
      "user_id": "test_user",
      "session_id": "test_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
  }
  with patch.object(
      mock_agent_loader,
      "load_agent",
      side_effect=_AgentLoadError(
          "Fail to load 'broken_app' module. ToolConfig"
      ),
  ):
    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to load agent"


def test_create_session_with_id(test_app, test_session_info):
  """Test creating a session with a specific ID."""
  new_session_id = "new_session_id"
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions/{new_session_id}"
  response = test_app.post(url, json={"state": {}})

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert data["id"] == new_session_id
  assert data["appName"] == test_session_info["app_name"]
  assert data["userId"] == test_session_info["user_id"]
  logger.info(f"Created session with ID: {data['id']}")


def test_create_session_with_id_already_exists(test_app, test_session_info):
  """Test creating a session with an ID that already exists."""
  session_id = "existing_session_id"
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions/{session_id}"

  # Create the session for the first time
  response = test_app.post(url, json={"state": {}})
  assert response.status_code == 200

  # Attempt to create it again
  response = test_app.post(url, json={"state": {}})
  assert response.status_code == 409
  assert "Session already exists" in response.json()["detail"]
  logger.info("Verified 409 on duplicate session creation.")


def test_create_session_without_id(test_app, test_session_info):
  """Test creating a session with a generated ID."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.post(url, json={"state": {}})

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert "id" in data
  assert data["appName"] == test_session_info["app_name"]
  assert data["userId"] == test_session_info["user_id"]
  logger.info(f"Created session with generated ID: {data['id']}")


def test_create_session_accepts_initial_text_events(
    test_app, test_session_info
):
  """Test initializing a session with text-only history."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  event = Event(
      author="user",
      invocation_id="init-invocation",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
  )
  response = test_app.post(
      url,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ]
      },
  )

  assert response.status_code == 200
  data = response.json()
  assert data["events"][0]["content"]["parts"][0]["text"] == "hello"


def test_create_session_accepts_initial_tool_events(
    test_app, test_session_info
):
  """Test restoring history from a conversation that used tools."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  function_call = types.FunctionCall(
      id="tool-call-id", name="write_files", args={"files": {"x": "y"}}
  )
  events = [
      Event(
          author="agent",
          invocation_id="init-invocation",
          content=types.Content(
              role="model", parts=[types.Part(function_call=function_call)]
          ),
      ),
      Event(
          author="agent",
          invocation_id="init-invocation",
          content=types.Content(
              role="user",
              parts=[
                  types.Part(
                      function_response=types.FunctionResponse(
                          id="tool-call-id",
                          name="write_files",
                          response={"status": "ok"},
                      )
                  )
              ],
          ),
      ),
  ]
  response = test_app.post(
      url,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
              for event in events
          ]
      },
  )

  assert response.status_code == 200
  stored = response.json()["events"]
  assert stored[0]["content"]["parts"][0]["functionCall"]["name"] == (
      "write_files"
  )
  assert stored[1]["content"]["parts"][0]["functionResponse"]["name"] == (
      "write_files"
  )


def test_create_session_strips_internal_metadata_and_marks_events_restored(
    test_app, test_session_info, mock_session_service
):
  """Restored events lose ADK-internal keys and are marked restored."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  event = Event(
      author="user",
      invocation_id="init-invocation",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
      custom_metadata={
          "keep": 1,
          INTERNAL_METADATA_PREFIX + "planted": "x",
          RESTORED_EVENT_KEY: False,
      },
  )
  response = test_app.post(
      url,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ]
      },
  )

  assert response.status_code == 200
  # Callers never see ADK-internal keys; the stored event keeps the marker.
  assert response.json()["events"][0]["customMetadata"] == {"keep": 1}
  stored = mock_session_service.sessions[test_session_info["app_name"]][
      test_session_info["user_id"]
  ][response.json()["id"]].events
  assert stored[0].custom_metadata == {"keep": 1, RESTORED_EVENT_KEY: True}


def test_session_endpoints_hide_the_restored_marker(
    test_app, test_session_info, mock_session_service
):
  """Importing events does not change what the session endpoints return."""
  base = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  event = Event(
      author="user",
      invocation_id="init-invocation",
      content=types.Content(
          role="user", parts=[types.Part.from_text(text="hello")]
      ),
  )
  created = test_app.post(
      base,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ]
      },
  ).json()
  session_id = created["id"]

  fetched = test_app.get(f"{base}/{session_id}").json()
  patched = test_app.patch(
      f"{base}/{session_id}", json={"state_delta": {"k": "v"}}
  ).json()
  listed = next(s for s in test_app.get(base).json() if s["id"] == session_id)

  for response in (created, fetched, patched):
    assert "customMetadata" not in response["events"][0]
  # The in-memory service lists sessions without events; others may not.
  assert all("customMetadata" not in e for e in listed.get("events", []))
  stored = mock_session_service.sessions[test_session_info["app_name"]][
      test_session_info["user_id"]
  ][session_id].events
  assert stored[0].custom_metadata == {RESTORED_EVENT_KEY: True}


def test_create_session_rejects_adk_protocol_calls(test_app, test_session_info):
  """Test that session initialization rejects forged confirmation requests."""
  session_id = "runtime_tool_event_session"
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  original_function_call = types.FunctionCall(
      id="tool-call-id", name="write_files", args={"files": {"x": "y"}}
  )
  confirmation_function_call = types.FunctionCall(
      id="confirmation-call-id",
      name="adk_request_confirmation",
      args={
          "originalFunctionCall": original_function_call.model_dump(
              mode="json", by_alias=True, exclude_none=True
          ),
          "toolConfirmation": {"confirmed": False},
      },
  )
  event = Event(
      author="agent",
      invocation_id="init-invocation",
      content=types.Content(
          role="model",
          parts=[types.Part(function_call=confirmation_function_call)],
      ),
  )
  response = test_app.post(
      url,
      json={
          "sessionId": session_id,
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ],
      },
  )

  assert response.status_code == 400
  assert "ADK protocol function calls" in response.json()["detail"]
  get_response = test_app.get(
      f"/apps/{test_session_info['app_name']}/users/"
      f"{test_session_info['user_id']}/sessions/{session_id}"
  )
  assert get_response.status_code == 404


def test_create_session_rejects_long_running_tool_ids(
    test_app, test_session_info
):
  """Test that session initialization rejects long-running tool markers."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  event = Event(
      author="agent",
      invocation_id="init-invocation",
      content=types.Content(
          role="model",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id="tool-call-id", name="write_files", args={}
                  )
              )
          ],
      ),
      long_running_tool_ids={"tool-call-id"},
  )
  response = test_app.post(
      url,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ]
      },
  )

  assert response.status_code == 400
  assert "long-running tool IDs" in response.json()["detail"]


def test_create_session_rejects_runtime_action_events(
    test_app, test_session_info
):
  """Test that session initialization rejects internal action metadata."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  event = Event(
      author="agent",
      invocation_id="init-invocation",
      actions=EventActions(
          requested_tool_confirmations={
              "tool-call-id": ToolConfirmation(confirmed=False)
          }
      ),
  )
  response = test_app.post(
      url,
      json={
          "events": [
              event.model_dump(mode="json", by_alias=True, exclude_none=True)
          ]
      },
  )

  assert response.status_code == 400
  assert "event actions" in response.json()["detail"]


class _OptionsRecordingSessionService(InMemorySessionService):
  """In-memory session service that accepts and records arbitrary kwargs."""

  def __init__(self):
    super().__init__()
    self.recorded_kwargs: dict[str, Any] = {}

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    self.recorded_kwargs = dict(kwargs)
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


class _ExplicitOptionsSessionService(InMemorySessionService):
  """In-memory session service that explicitly accepts custom parameters."""

  def __init__(self):
    super().__init__()
    self.recorded_kwargs: dict[str, Any] = {}

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      custom_option: Optional[str] = None,
  ) -> Session:
    self.recorded_kwargs = {"custom_option": custom_option}
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


class _ValidatingOptionsSessionService(InMemorySessionService):
  """In-memory session service that validates options and raises ValueError."""

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
      **kwargs: Any,
  ) -> Session:
    if kwargs.get("ttl") is not None and kwargs.get("expire_time") is not None:
      raise ValueError(
          "Cannot specify both 'ttl' and 'expire_time' simultaneously."
      )
    return await super().create_session(
        app_name=app_name,
        user_id=user_id,
        state=state,
        session_id=session_id,
    )


@pytest.fixture
def explicit_options_session_service():
  """Create a session service with explicit custom parameters."""
  return _ExplicitOptionsSessionService()


@pytest.fixture
def explicit_options_test_app(
    explicit_options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by an explicit options session service."""
  return _create_test_client(
      explicit_options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


@pytest.fixture
def options_session_service():
  """Create a session service whose create_session accepts kwargs."""
  return _OptionsRecordingSessionService()


@pytest.fixture
def options_test_app(
    options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by a kwargs-capable session service."""
  return _create_test_client(
      options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


@pytest.fixture
def validating_options_session_service():
  """Create a session service that validates kwargs."""
  return _ValidatingOptionsSessionService()


@pytest.fixture
def validating_options_test_app(
    validating_options_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient backed by a validating session service."""
  return _create_test_client(
      validating_options_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )


def test_create_session_forwards_options(
    options_test_app, options_session_service, test_session_info
):
  """Test that options dict is forwarded to a supporting session service."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = options_test_app.post(
      url,
      json={
          "options": {
              "ttl": "7200s",
              "expire_time": "2026-08-01T00:00:00Z",
              "custom_param": "foo",
          }
      },
  )

  assert response.status_code == 200
  assert options_session_service.recorded_kwargs == {
      "ttl": "7200s",
      "expire_time": "2026-08-01T00:00:00Z",
      "custom_param": "foo",
  }


def test_create_session_explicit_options_forwards_kwargs(
    explicit_options_test_app,
    explicit_options_session_service,
    test_session_info,
):
  """Test that options are forwarded to service with explicit parameter."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = explicit_options_test_app.post(
      url, json={"options": {"custom_option": "bar"}}
  )

  assert response.status_code == 200
  assert (
      explicit_options_session_service.recorded_kwargs.get("custom_option")
      == "bar"
  )


def test_create_session_options_with_unsupported_service(
    test_app, test_session_info
):
  """Test 400 when options are requested but the service cannot honor them."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.post(url, json={"options": {"ttl": "7200s"}})

  assert response.status_code == 400
  assert "not supported" in response.json()["detail"]


def test_create_session_options_validation_error_returns_400(
    validating_options_test_app, test_session_info
):
  """Test 400 when session service raises ValueError on invalid options."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = validating_options_test_app.post(
      url,
      json={
          "options": {
              "ttl": "7200s",
              "expire_time": "2026-08-01T00:00:00Z",
          }
      },
  )

  assert response.status_code == 400
  assert (
      "Cannot specify both 'ttl' and 'expire_time'" in response.json()["detail"]
  )


def test_create_session_options_conflicting_key_returns_400(
    test_app, test_session_info
):
  """Test 400 when options contains a key already bound by the endpoint."""
  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.post(url, json={"options": {"app_name": "other_app"}})

  assert response.status_code == 400


def test_accepts_kwargs_rejects_var_positional_parameter():
  """_accepts_kwargs should return False for variadic positional parameters."""
  from google.adk.cli.api_server import _accepts_kwargs

  def func(*args: Any) -> None:
    pass

  assert not _accepts_kwargs(func, {"args": "value"})


def test_get_session(test_app, create_test_session):
  """Test retrieving a session by ID."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"
  response = test_app.get(url)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert data["id"] == info["session_id"]
  assert data["appName"] == info["app_name"]
  assert data["userId"] == info["user_id"]
  logger.info(f"Retrieved session: {data['id']}")


def test_list_sessions(test_app, create_test_session):
  """Test listing all sessions for a user."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions"
  response = test_app.get(url)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  # At least our test session should be present
  assert any(session["id"] == info["session_id"] for session in data)
  logger.info(f"Listed {len(data)} sessions")


async def test_list_sessions_filters_eval_sessions(
    test_app, test_session_info, mock_session_service
):
  """Test that eval sessions (both old and new prefixes) are filtered from list."""
  # Create a normal session
  await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id="normal-session",
      state={},
  )
  # Create a new style eval session
  await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id="adk-eval-session-new-style",
      state={},
  )
  # Create an old style eval session
  await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id="___eval___session___old-style",
      state={},
  )

  url = f"/apps/{test_session_info['app_name']}/users/{test_session_info['user_id']}/sessions"
  response = test_app.get(url)

  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)

  session_ids = [session["id"] for session in data]
  assert "normal-session" in session_ids
  assert "adk-eval-session-new-style" not in session_ids
  assert "___eval___session___old-style" not in session_ids


def test_delete_session(test_app, create_test_session):
  """Test deleting a session."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"
  response = test_app.delete(url)

  # Verify the response
  assert response.status_code == 200

  # Verify the session is deleted
  response = test_app.get(url)
  assert response.status_code == 404
  logger.info("Session deleted successfully")


def test_update_session(test_app, create_test_session):
  """Test patching a session state."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}"

  # Get the original session
  response = test_app.get(url)
  assert response.status_code == 200
  original_session = response.json()
  original_state = original_session.get("state", {})

  # Prepare state delta
  state_delta = {"test_key": "test_value", "counter": 42}

  # Patch the session
  response = test_app.patch(url, json={"state_delta": state_delta})
  assert response.status_code == 200

  # Verify the response
  patched_session = response.json()
  assert patched_session["id"] == info["session_id"]

  # Verify state was updated correctly
  expected_state = {**original_state, **state_delta}
  assert patched_session["state"] == expected_state

  # Verify the session was actually updated in storage
  response = test_app.get(url)
  assert response.status_code == 200
  retrieved_session = response.json()
  assert retrieved_session["state"] == expected_state

  # Verify an event was created for the state change
  events = retrieved_session.get("events", [])
  assert len(events) > len(original_session.get("events", []))

  # Find the state patch event (looking for "p-" prefix pattern)
  state_patch_events = [
      event
      for event in events
      if event.get("invocationId", "").startswith("p-")
  ]

  assert len(state_patch_events) == 1, (
      f"Expected 1 state_patch event, found {len(state_patch_events)}. Events:"
      f" {events}"
  )
  state_patch_event = state_patch_events[0]
  assert state_patch_event["author"] == "user"

  # Check for actions in both camelCase and snake_case
  actions = state_patch_event.get("actions")
  assert actions is not None, f"No actions found in event: {state_patch_event}"
  state_delta_in_event = actions.get("stateDelta")
  assert state_delta_in_event == state_delta

  logger.info("Session state patched successfully")


def test_patch_session_not_found(test_app, test_session_info):
  """Test patching a nonexistent session."""
  info = test_session_info
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/nonexistent"

  state_delta = {"test_key": "test_value"}
  response = test_app.patch(url, json={"state_delta": state_delta})

  assert response.status_code == 404
  assert "Session not found" in response.json()["detail"]
  logger.info("Patch session not found test passed")


def test_agent_run(test_app, create_test_session):
  """Test running an agent with a message."""
  info = create_test_session
  url = "/run"
  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": False,
  }

  response = test_app.post(url, json=payload)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  assert len(data) == 3  # We expect 3 events from our dummy_run_async

  # Verify we got the expected events
  assert data[0]["author"] == "dummy agent"
  assert data[0]["content"]["parts"][0]["text"] == "LLM reply"

  # Second event should have binary data
  assert (
      data[1]["content"]["parts"][0]["inlineData"]["mimeType"]
      == "audio/pcm;rate=24000"
  )

  # Third event should have interrupted flag
  assert data[2]["interrupted"] is True

  logger.info("Agent run test completed successfully")


async def _run_async_with_internal_metadata(
    self,
    *,
    user_id: str,
    session_id: str,
    invocation_id: Optional[str] = None,
    new_message: Optional[types.Content] = None,
    state_delta: Optional[dict[str, Any]] = None,
    run_config: Optional[RunConfig] = None,
):
  del user_id, session_id, invocation_id, new_message, state_delta, run_config
  yield Event(
      author="dummy agent",
      invocation_id="invocation_id",
      content=types.Content(role="model", parts=[types.Part(text="reply")]),
      custom_metadata={"keep": 1, INTERNAL_METADATA_PREFIX + "stamp": "x"},
  )


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_agent_run_hides_internal_metadata(
    test_app, create_test_session, monkeypatch, endpoint
):
  """Run endpoints stream events without ADK-internal custom_metadata."""
  info = create_test_session
  monkeypatch.setattr(Runner, "run_async", _run_async_with_internal_metadata)
  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
      "streaming": False,
  }

  response = test_app.post(endpoint, json=payload)

  assert response.status_code == 200
  if endpoint == "/run":
    events = response.json()
  else:
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
  assert [e["customMetadata"] for e in events] == [{"keep": 1}]


def test_agent_run_passes_state_delta(test_app, create_test_session):
  """Test /run forwards state_delta and surfaces it in events."""
  info = create_test_session
  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
      "streaming": False,
      "state_delta": {"k": "v", "count": 1},
  }

  # Verify the response
  response = test_app.post("/run", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  assert len(data) == 4

  # Verify we got the expected event
  assert data[3]["actions"]["stateDelta"] == payload["state_delta"]


def test_agent_run_passes_invocation_id(
    test_app, create_test_session, monkeypatch
):
  """Test /run forwards invocation_id for resumable invocations."""
  info = create_test_session
  captured_invocation_id: dict[str, Optional[str]] = {"invocation_id": None}

  async def run_async_capture(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del self, user_id, session_id, new_message, state_delta, run_config
    captured_invocation_id["invocation_id"] = invocation_id
    yield _event_1()

  monkeypatch.setattr(Runner, "run_async", run_async_capture)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Resume run"}]},
      "streaming": False,
      "invocation_id": "resume-invocation-id",
  }

  response = test_app.post("/run", json=payload)

  assert response.status_code == 200
  assert captured_invocation_id["invocation_id"] == payload["invocation_id"]


def test_agent_run_passes_custom_metadata(
    test_app, create_test_session, monkeypatch
):
  """Test /run forwards custom_metadata via the run config."""
  info = create_test_session
  captured: dict[str, Optional[RunConfig]] = {"run_config": None}

  async def run_async_capture(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del self, user_id, session_id, invocation_id, new_message, state_delta
    captured["run_config"] = run_config
    yield _event_1()

  monkeypatch.setattr(Runner, "run_async", run_async_capture)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
      "streaming": False,
      "custom_metadata": {"tenant": "acme", "trace": "abc123"},
  }

  response = test_app.post("/run", json=payload)

  assert response.status_code == 200
  assert captured["run_config"] is not None
  assert captured["run_config"].custom_metadata == payload["custom_metadata"]


def test_agent_run_passes_max_llm_calls(
    create_test_session,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test /run forwards the server's max_llm_calls via the run config."""
  info = create_test_session
  captured: dict[str, Optional[RunConfig]] = {"run_config": None}

  async def run_async_capture(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, object]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del self, user_id, session_id, invocation_id, new_message, state_delta
    captured["run_config"] = run_config
    yield _event_1()

  monkeypatch.setattr(Runner, "run_async", run_async_capture)
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      max_llm_calls=37,
  )

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
      "streaming": False,
  }

  response = client.post("/run", json=payload)

  assert response.status_code == 200
  assert captured["run_config"] is not None
  assert captured["run_config"].max_llm_calls == 37


def test_agent_run_sse_passes_max_llm_calls(
    create_test_session,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test /run_sse forwards the server's max_llm_calls via the run config."""
  info = create_test_session
  captured: dict[str, Optional[RunConfig]] = {"run_config": None}

  async def run_async_capture(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, object]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del self, user_id, session_id, invocation_id, new_message, state_delta
    captured["run_config"] = run_config
    yield _event_1()

  monkeypatch.setattr(Runner, "run_async", run_async_capture)
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      max_llm_calls=37,
  )

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
      "streaming": True,
  }

  response = client.post("/run_sse", json=payload)

  assert response.status_code == 200
  assert captured["run_config"] is not None
  assert captured["run_config"].max_llm_calls == 37


def test_agent_run_sse_splits_artifact_delta(
    test_app, create_test_session, monkeypatch
):
  """Test /run_sse splits artifact deltas to avoid double-rendering in web."""
  info = create_test_session

  async def run_async_with_artifact_delta(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
      **kwargs,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta, run_config
    yield Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="model", parts=[types.Part(text="LLM reply")]
        ),
        actions=EventActions(artifact_delta={"artifact.txt": 0}),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_with_artifact_delta)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
  }

  response = test_app.post("/run_sse", json=payload)
  assert response.status_code == 200

  sse_events = [
      json.loads(line.removeprefix("data: "))
      for line in response.text.splitlines()
      if line.startswith("data: ")
  ]

  assert len(sse_events) == 2

  # First event: content but artifactDelta cleared.
  assert sse_events[0]["content"]["parts"][0]["text"] == "LLM reply"
  assert sse_events[0]["actions"]["artifactDelta"] == {}

  # Second event: artifactDelta but no content.
  assert "content" not in sse_events[1]
  assert sse_events[1]["actions"]["artifactDelta"] == {"artifact.txt": 0}


@pytest.fixture
def oauth2_auth_config_dict():
  return {
      "authScheme": {
          "type": "oauth2",
          "flows": {
              "authorizationCode": {
                  "scopes": {"read": "read"},
                  "authorizationUrl": "https://idp.example.com/oauth2/auth",
                  "tokenUrl": "https://idp.example.com/oauth2/token",
              }
          },
      },
      "rawAuthCredential": {
          "authType": "oauth2",
          "oauth2": {
              "clientId": "public-client-id",
              "clientSecret": "should-never-reach-the-client",
          },
      },
      "exchangedAuthCredential": {
          "authType": "oauth2",
          "oauth2": {
              "clientId": "public-client-id",
              "clientSecret": "should-never-reach-the-client",
              "authUri": (
                  "https://idp.example.com/oauth2/auth?client_id="
                  "public-client-id&state=xyz"
              ),
              "state": "xyz",
              "codeVerifier": "pkce-verifier-should-not-leak-either",
          },
      },
      "credentialKey": "my_tool:oauth2:abcd1234",
  }


def test_agent_run_redacts_oauth2_client_secret(
    test_app, create_test_session, monkeypatch, oauth2_auth_config_dict
):
  """/run must not leak OAuth2 secrets embedded in response payloads."""
  info = create_test_session

  async def run_async_with_auth_request(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta, run_config
    yield Event(
        author="agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="adk_request_credential",
                        id="adk-req-cred-id",
                        args={
                            "functionCallId": "adk-original-fc-id",
                            "authConfig": oauth2_auth_config_dict,
                        },
                    )
                )
            ],
        ),
        actions=EventActions(
            requested_auth_configs={
                "adk-original-fc-id": oauth2_auth_config_dict
            }
        ),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_with_auth_request)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
  }

  response = test_app.post("/run", json=payload)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  events = response.json()
  assert len(events) == 1
  args = events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"

  action_auth = events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]
  assert "clientSecret" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert "codeVerifier" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert (
      action_auth["rawAuthCredential"]["oauth2"]["clientId"]
      == "public-client-id"
  )


def test_agent_run_sse_redacts_oauth2_client_secret(
    test_app, create_test_session, monkeypatch, oauth2_auth_config_dict
):
  """/run_sse must not leak OAuth2 secrets embedded in a function call or actions.

  When a tool needs OAuth, ADK attaches the credential -- including the
  app's `client_secret` -- to an `adk_request_credential` function call's
  `args` and the event's `actions.requested_auth_configs`. That `args` value is
  an opaque dict, not a nested pydantic model, so it is not covered by
  `Event.model_dump(exclude=...)`. This asserts the streamed event has the secret
  fields stripped across both carriers while the fields the client actually
  needs to complete the OAuth redirect (client_id, the authorization URL, the
  credential key) are preserved.
  """
  info = create_test_session

  async def run_async_with_auth_request(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta, run_config
    yield Event(
        author="agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="adk_request_credential",
                        id="adk-req-cred-id",
                        args={
                            "functionCallId": "adk-original-fc-id",
                            "authConfig": oauth2_auth_config_dict,
                        },
                    )
                )
            ],
        ),
        actions=EventActions(
            requested_auth_configs={
                "adk-original-fc-id": oauth2_auth_config_dict
            }
        ),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_with_auth_request)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
  }

  response = test_app.post("/run_sse", json=payload)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  sse_events = [
      json.loads(line.removeprefix("data: "))
      for line in response.text.splitlines()
      if line.startswith("data: ")
  ]
  assert len(sse_events) == 1
  args = sse_events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  # Fields the client actually needs to complete the OAuth redirect must
  # survive the redaction.
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"

  # Event actions requestedAuthConfigs must also be redacted.
  action_auth = sse_events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]
  assert "clientSecret" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert "codeVerifier" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert (
      action_auth["rawAuthCredential"]["oauth2"]["clientId"]
      == "public-client-id"
  )


def test_agent_run_live_redacts_oauth2_client_secret(
    test_app, create_test_session, monkeypatch, oauth2_auth_config_dict
):
  """/run_live websocket must not leak OAuth2 secrets in event frames."""
  info = create_test_session

  async def run_live_with_auth_request(
      self,
      *,
      session,
      live_request_queue,
      run_config=None,
  ):
    del self, session, live_request_queue, run_config
    yield Event(
        author="agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="adk_request_credential",
                        id="adk-req-cred-id",
                        args={
                            "functionCallId": "adk-original-fc-id",
                            "authConfig": oauth2_auth_config_dict,
                        },
                    )
                )
            ],
        ),
        actions=EventActions(
            requested_auth_configs={
                "adk-original-fc-id": oauth2_auth_config_dict
            }
        ),
    )

  monkeypatch.setattr(Runner, "run_live", run_live_with_auth_request)

  url = f"/run_live?app_name={info['app_name']}&user_id={info['user_id']}&session_id={info['session_id']}&modalities=AUDIO"

  with test_app.websocket_connect(url) as ws:
    text_data = ws.receive_text()
    assert "should-never-reach-the-client" not in text_data
    assert "pkce-verifier-should-not-leak-either" not in text_data

    event_data = json.loads(text_data)
    args = event_data["content"]["parts"][0]["functionCall"]["args"]
    raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
    exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
    assert "clientSecret" not in raw_oauth2
    assert "clientSecret" not in exchanged_oauth2
    assert "codeVerifier" not in exchanged_oauth2
    assert raw_oauth2["clientId"] == "public-client-id"
    assert exchanged_oauth2["authUri"].startswith(
        "https://idp.example.com/oauth2/auth"
    )
    assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"

    action_auth = event_data["actions"]["requestedAuthConfigs"][
        "adk-original-fc-id"
    ]
    assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]
    assert (
        "clientSecret" not in action_auth["exchangedAuthCredential"]["oauth2"]
    )
    assert (
        "codeVerifier" not in action_auth["exchangedAuthCredential"]["oauth2"]
    )
    assert (
        action_auth["rawAuthCredential"]["oauth2"]["clientId"]
        == "public-client-id"
    )


async def test_get_session_redacts_oauth2_client_secret(
    test_app, test_session_info, mock_session_service, oauth2_auth_config_dict
):
  """GET /apps/{app_name}/users/{user_id}/sessions/{session_id} redacts secrets across all carriers."""
  session = await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id=test_session_info["session_id"],
      state={
          "oauth_cred_snake": {
              "auth_type": "oauth2",
              "oauth2": {
                  "client_id": "public-client-id-snake",
                  "client_secret": "snake-secret-should-never-reach-client",
                  "access_token": "snake-token-should-never-reach-client",
              },
          }
      },
  )
  event1 = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  response_auth_config_dict = {
      "authScheme": {
          "type": "oauth2",
          "flows": {
              "authorizationCode": {
                  "scopes": {"read": "read"},
                  "authorizationUrl": "https://idp.example.com/oauth2/auth",
                  "tokenUrl": "https://idp.example.com/oauth2/token",
              }
          },
      },
      "exchangedAuthCredential": {
          "authType": "oauth2",
          "oauth2": {
              "clientId": "public-client-id",
              "clientSecret": "should-never-reach-the-client",
              "authResponseUri": (
                  "https://idp.example.com/oauth2/callback?code=secret-auth-code"
              ),
          },
      },
  }
  event2 = Event(
      author="user",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      response=response_auth_config_dict,
                  )
              )
          ],
      ),
  )
  await mock_session_service.append_event(session=session, event=event1)
  await mock_session_service.append_event(session=session, event=event2)

  url = (
      f"/apps/{test_session_info['app_name']}/users/"
      f"{test_session_info['user_id']}/sessions/{test_session_info['session_id']}"
  )
  response = test_app.get(url)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text
  assert "secret-auth-code" not in response.text
  assert "snake-secret-should-never-reach-client" not in response.text
  assert "snake-token-should-never-reach-client" not in response.text

  data = response.json()
  assert data["id"] == test_session_info["session_id"]
  assert len(data["events"]) == 2

  snake_oauth2 = data["state"]["oauth_cred_snake"]["oauth2"]
  assert "client_secret" not in snake_oauth2
  assert "access_token" not in snake_oauth2
  assert snake_oauth2["client_id"] == "public-client-id-snake"

  # Carrier 1: function call args.authConfig
  args = data["events"][0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"

  # Carrier 2: actions requestedAuthConfigs
  action_auth = data["events"][0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]
  assert "clientSecret" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert "codeVerifier" not in action_auth["exchangedAuthCredential"]["oauth2"]
  assert (
      action_auth["rawAuthCredential"]["oauth2"]["clientId"]
      == "public-client-id"
  )

  # Carrier 3: function response response (authResponseUri)
  resp = data["events"][1]["content"]["parts"][0]["functionResponse"][
      "response"
  ]
  resp_oauth2 = resp["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in resp_oauth2
  assert "authResponseUri" not in resp_oauth2
  assert resp_oauth2["clientId"] == "public-client-id"


async def test_list_sessions_redacts_oauth2_client_secret(
    test_app,
    test_session_info,
    mock_session_service,
    monkeypatch,
    oauth2_auth_config_dict,
):
  """GET /apps/{app_name}/users/{user_id}/sessions redacts secrets."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  session = Session(
      id=test_session_info["session_id"],
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      state={},
      events=[event],
  )
  monkeypatch.setattr(
      mock_session_service,
      "list_sessions",
      AsyncMock(return_value=ListSessionsResponse(sessions=[session])),
  )

  url = (
      f"/apps/{test_session_info['app_name']}/users/"
      f"{test_session_info['user_id']}/sessions"
  )
  response = test_app.get(url)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  data = response.json()
  assert isinstance(data, list)
  matching = [s for s in data if s["id"] == test_session_info["session_id"]]
  assert len(matching) == 1
  matched_session = matching[0]
  assert len(matched_session["events"]) == 1
  args = matched_session["events"][0]["content"]["parts"][0]["functionCall"][
      "args"
  ]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"
  action_auth = matched_session["events"][0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]


async def test_update_session_redacts_oauth2_client_secret(
    test_app, test_session_info, mock_session_service, oauth2_auth_config_dict
):
  """PATCH /apps/{app_name}/users/{user_id}/sessions/{session_id} redacts secrets."""
  session = await mock_session_service.create_session(
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      session_id=test_session_info["session_id"],
      state={},
  )
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  await mock_session_service.append_event(session=session, event=event)

  url = (
      f"/apps/{test_session_info['app_name']}/users/"
      f"{test_session_info['user_id']}/sessions/{test_session_info['session_id']}"
  )
  response = test_app.patch(url, json={"state_delta": {"key": "val"}})
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  data = response.json()
  assert data["id"] == test_session_info["session_id"]
  assert len(data["events"]) >= 1
  matching_events = [
      e
      for e in data["events"]
      if e.get("content")
      and e["content"].get("parts")
      and e["content"]["parts"][0].get("functionCall", {}).get("name")
      == "adk_request_credential"
  ]
  assert len(matching_events) == 1
  args = matching_events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"
  action_auth = matching_events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]


def test_get_eval_redacts_oauth2_client_secret(
    test_app, test_session_info, mock_eval_sets_manager, oauth2_auth_config_dict
):
  """GET /dev/apps/{app_name}/eval-sets/{eval_set_id}/eval-cases/{eval_case_id} redacts secrets."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  eval_case = EvalCase(
      eval_id="test_eval_case_id",
      conversation=[],
      session_input=SessionInput(
          app_name=test_session_info["app_name"],
          user_id=test_session_info["user_id"],
          events=[event],
      ),
  )
  mock_eval_sets_manager.create_eval_set(
      app_name=test_session_info["app_name"],
      eval_set_id="test_eval_set_id",
  )
  mock_eval_sets_manager.add_eval_case(
      app_name=test_session_info["app_name"],
      eval_set_id="test_eval_set_id",
      eval_case=eval_case,
  )

  url = f"/dev/apps/{test_session_info['app_name']}/eval-sets/test_eval_set_id/eval-cases/test_eval_case_id"
  response = test_app.get(url)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  data = response.json()
  assert data["evalId"] == "test_eval_case_id"
  events = data["sessionInput"]["events"]
  assert len(events) == 1
  args = events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"
  action_auth = events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]


def test_get_eval_result_redacts_oauth2_client_secret(
    test_app,
    test_session_info,
    mock_eval_set_results_manager,
    oauth2_auth_config_dict,
):
  """GET /dev/apps/{app_name}/eval-results/{eval_result_id} redacts secrets."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  session = Session(
      id=test_session_info["session_id"],
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      state={},
      events=[event],
  )
  eval_case_result = EvalCaseResult(
      eval_set_id="test_eval_set_id",
      eval_id="test_eval_case_id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id=test_session_info["session_id"],
      session_details=session,
      user_id=test_session_info["user_id"],
  )
  mock_eval_set_results_manager.save_eval_set_result(
      test_session_info["app_name"],
      "test_eval_set_id",
      [eval_case_result],
  )

  url = (
      f"/dev/apps/{test_session_info['app_name']}/eval-results/"
      f"{test_session_info['app_name']}_test_eval_set_id_eval_result"
  )
  response = test_app.get(url)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  data = response.json()
  assert (
      data["evalSetResultId"]
      == f"{test_session_info['app_name']}_test_eval_set_id_eval_result"
  )
  assert len(data["evalCaseResults"]) == 1
  case_result = data["evalCaseResults"][0]
  events = case_result["sessionDetails"]["events"]
  assert len(events) == 1
  args = events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"
  action_auth = events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]


def test_get_eval_result_legacy_redacts_oauth2_client_secret(
    test_app,
    test_session_info,
    mock_eval_set_results_manager,
    oauth2_auth_config_dict,
):
  """GET /dev/apps/{app_name}/eval_results/{eval_result_id} redacts secrets."""
  event = Event(
      author="agent",
      invocation_id="invocation_id",
      content=types.Content(
          role="user",
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      name="adk_request_credential",
                      id="adk-req-cred-id",
                      args={
                          "functionCallId": "adk-original-fc-id",
                          "authConfig": oauth2_auth_config_dict,
                      },
                  )
              )
          ],
      ),
      actions=EventActions(
          requested_auth_configs={"adk-original-fc-id": oauth2_auth_config_dict}
      ),
  )
  session = Session(
      id=test_session_info["session_id"],
      app_name=test_session_info["app_name"],
      user_id=test_session_info["user_id"],
      state={},
      events=[event],
  )
  eval_case_result = EvalCaseResult(
      eval_set_id="test_eval_set_id",
      eval_id="test_eval_case_id",
      final_eval_status=EvalStatus.PASSED,
      overall_eval_metric_results=[],
      eval_metric_result_per_invocation=[],
      session_id=test_session_info["session_id"],
      session_details=session,
      user_id=test_session_info["user_id"],
  )
  mock_eval_set_results_manager.save_eval_set_result(
      test_session_info["app_name"],
      "test_eval_set_id",
      [eval_case_result],
  )

  url = (
      f"/dev/apps/{test_session_info['app_name']}/eval_results/"
      f"{test_session_info['app_name']}_test_eval_set_id_eval_result"
  )
  response = test_app.get(url)
  assert response.status_code == 200
  assert "should-never-reach-the-client" not in response.text
  assert "pkce-verifier-should-not-leak-either" not in response.text

  data = response.json()
  assert (
      data["evalSetResultId"]
      == f"{test_session_info['app_name']}_test_eval_set_id_eval_result"
  )
  assert len(data["evalCaseResults"]) == 1
  case_result = data["evalCaseResults"][0]
  events = case_result["sessionDetails"]["events"]
  assert len(events) == 1
  args = events[0]["content"]["parts"][0]["functionCall"]["args"]
  raw_oauth2 = args["authConfig"]["rawAuthCredential"]["oauth2"]
  exchanged_oauth2 = args["authConfig"]["exchangedAuthCredential"]["oauth2"]
  assert "clientSecret" not in raw_oauth2
  assert "clientSecret" not in exchanged_oauth2
  assert "codeVerifier" not in exchanged_oauth2
  assert raw_oauth2["clientId"] == "public-client-id"
  assert exchanged_oauth2["authUri"].startswith(
      "https://idp.example.com/oauth2/auth"
  )
  assert args["authConfig"]["credentialKey"] == "my_tool:oauth2:abcd1234"
  action_auth = events[0]["actions"]["requestedAuthConfigs"][
      "adk-original-fc-id"
  ]
  assert "clientSecret" not in action_auth["rawAuthCredential"]["oauth2"]


def test_redaction_does_not_drop_non_auth_config_keys():
  """Fields named token, password, apiKey outside authConfig must not be dropped."""
  payload = {
      "session_state": {
          "apiKey": "my-api-key",
          "token": "pagination-token",
          "user_credential": {
              "authType": "oauth2",
              "oauth2": {
                  "clientId": "cid-state",
                  "accessToken": "secret-access-token",
                  "refreshToken": "secret-refresh-token",
                  "clientSecret": "drop-me-state",
              },
          },
          "user_credential_snake": {
              "auth_type": "oauth2",
              "oauth2": {
                  "client_id": "cid-state-snake",
                  "access_token": "secret-access-token-snake",
                  "refresh_token": "secret-refresh-token-snake",
                  "client_secret": "drop-me-state-snake",
              },
          },
      },
      "actions": {
          "stateDelta": {
              "password": "secret-pw",
              "auth_update": {
                  "authType": "oauth2",
                  "oauth2": {
                      "accessToken": "secret-delta-at",
                      "clientId": "cid-delta",
                  },
              },
              "auth_update_snake": {
                  "auth_type": "oauth2",
                  "oauth2": {
                      "access_token": "secret-delta-at-snake",
                      "client_id": "cid-delta-snake",
                  },
              },
          },
          "requestedAuthConfigs": {
              "fc-1": {
                  "rawAuthCredential": {
                      "oauth2": {
                          "clientId": "cid-action",
                          "clientSecret": "drop-me-action",
                      }
                  }
              },
              "fc-snake": {
                  "raw_auth_credential": {
                      "oauth2": {
                          "client_id": "cid-action-snake",
                          "client_secret": "drop-me-action-snake",
                      }
                  }
              },
          },
      },
      "events": [{
          "content": {
              "parts": [
                  {
                      "functionCall": {
                          "name": "custom_search",
                          "args": {"apiKey": "key123", "token": "tok456"},
                      }
                  },
                  {
                      "functionResponse": {
                          "name": "custom_search",
                          "response": {"token": "next_page_token"},
                      }
                  },
                  {
                      "functionCall": {
                          "name": "adk_request_credential",
                          "args": {
                              "functionCallId": "fc-1",
                              "authConfig": {
                                  "rawAuthCredential": {
                                      "oauth2": {
                                          "clientId": "cid",
                                          "clientSecret": "drop-me",
                                      }
                                  }
                              },
                          },
                      }
                  },
                  {
                      "functionResponse": {
                          "name": "adk_request_credential",
                          "response": {
                              "exchangedAuthCredential": {
                                  "oauth2": {
                                      "clientId": "cid-resp",
                                      "authResponseUri": (
                                          "https://idp.example.com/cb?code=secret_code"
                                      ),
                                      "clientSecret": "drop-me-resp",
                                  }
                              }
                          },
                      }
                  },
                  {
                      "functionCall": {
                          "name": "adk_request_credential",
                          "args": {
                              "functionCallId": "fc-snake",
                              "auth_config": {
                                  "raw_auth_credential": {
                                      "oauth2": {
                                          "client_id": "cid-snake",
                                          "client_secret": "drop-me-snake",
                                      }
                                  }
                              },
                          },
                      }
                  },
                  {
                      "functionResponse": {
                          "name": "adk_request_credential",
                          "response": {
                              "exchanged_auth_credential": {
                                  "oauth2": {
                                      "client_id": "cid-resp-snake",
                                      "auth_response_uri": (
                                          "https://idp.example.com/cb?code=secret_code_snake"
                                      ),
                                      "client_secret": "drop-me-resp-snake",
                                  }
                              }
                          },
                      }
                  },
              ]
          }
      }],
  }
  redacted = _redact_credential_secrets(payload)
  assert redacted["session_state"]["apiKey"] == "my-api-key"
  assert redacted["session_state"]["token"] == "pagination-token"
  assert (
      "accessToken"
      not in redacted["session_state"]["user_credential"]["oauth2"]
  )
  assert (
      "refreshToken"
      not in redacted["session_state"]["user_credential"]["oauth2"]
  )
  assert (
      "clientSecret"
      not in redacted["session_state"]["user_credential"]["oauth2"]
  )
  assert (
      redacted["session_state"]["user_credential"]["oauth2"]["clientId"]
      == "cid-state"
  )
  assert (
      "access_token"
      not in redacted["session_state"]["user_credential_snake"]["oauth2"]
  )
  assert (
      "refresh_token"
      not in redacted["session_state"]["user_credential_snake"]["oauth2"]
  )
  assert (
      "client_secret"
      not in redacted["session_state"]["user_credential_snake"]["oauth2"]
  )
  assert (
      redacted["session_state"]["user_credential_snake"]["oauth2"]["client_id"]
      == "cid-state-snake"
  )
  assert redacted["actions"]["stateDelta"]["password"] == "secret-pw"
  assert (
      "accessToken"
      not in redacted["actions"]["stateDelta"]["auth_update"]["oauth2"]
  )
  assert (
      redacted["actions"]["stateDelta"]["auth_update"]["oauth2"]["clientId"]
      == "cid-delta"
  )
  assert (
      "access_token"
      not in redacted["actions"]["stateDelta"]["auth_update_snake"]["oauth2"]
  )
  assert (
      redacted["actions"]["stateDelta"]["auth_update_snake"]["oauth2"][
          "client_id"
      ]
      == "cid-delta-snake"
  )
  assert redacted["actions"]["requestedAuthConfigs"]["fc-1"][
      "rawAuthCredential"
  ]["oauth2"] == {"clientId": "cid-action"}
  assert redacted["actions"]["requestedAuthConfigs"]["fc-snake"][
      "raw_auth_credential"
  ]["oauth2"] == {"client_id": "cid-action-snake"}
  assert redacted["events"][0]["content"]["parts"][0]["functionCall"][
      "args"
  ] == {
      "apiKey": "key123",
      "token": "tok456",
  }
  assert redacted["events"][0]["content"]["parts"][1]["functionResponse"][
      "response"
  ] == {"token": "next_page_token"}
  fc2 = redacted["events"][0]["content"]["parts"][2]["functionCall"]
  assert fc2["args"]["authConfig"]["rawAuthCredential"]["oauth2"] == {
      "clientId": "cid"
  }
  fr2 = redacted["events"][0]["content"]["parts"][3]["functionResponse"]
  assert fr2["response"]["exchangedAuthCredential"]["oauth2"] == {
      "clientId": "cid-resp"
  }
  fc_snake = redacted["events"][0]["content"]["parts"][4]["functionCall"]
  assert fc_snake["args"]["auth_config"]["raw_auth_credential"]["oauth2"] == {
      "client_id": "cid-snake"
  }
  fr_snake = redacted["events"][0]["content"]["parts"][5]["functionResponse"]
  assert fr_snake["response"]["exchanged_auth_credential"]["oauth2"] == {
      "client_id": "cid-resp-snake"
  }


def test_agent_run_sse_does_not_split_artifact_delta_for_function_resume(
    test_app, create_test_session, monkeypatch
):
  """Test /run_sse keeps artifactDelta with content for function resume flow."""
  info = create_test_session

  async def run_async_with_artifact_delta(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
      **kwargs,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta, run_config
    yield Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(
            role="model", parts=[types.Part(text="LLM reply")]
        ),
        actions=EventActions(artifact_delta={"artifact.txt": 0}),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_with_artifact_delta)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
      "functionCallEventId": "function-call-event-id",
  }

  response = test_app.post("/run_sse", json=payload)
  assert response.status_code == 200

  sse_events = [
      json.loads(line.removeprefix("data: "))
      for line in response.text.splitlines()
      if line.startswith("data: ")
  ]

  assert len(sse_events) == 1
  assert sse_events[0]["content"]["parts"][0]["text"] == "LLM reply"
  assert sse_events[0]["actions"]["artifactDelta"] == {"artifact.txt": 0}


def test_agent_run_sse_yields_error_object_on_exception(
    test_app, create_test_session, monkeypatch
):
  """Test /run_sse streams structured error details on exception."""
  info = create_test_session

  async def run_async_raises(self, **kwargs):
    raise ValueError("boom")
    yield  # make it an async generator  # pylint: disable=unreachable

  monkeypatch.setattr(Runner, "run_async", run_async_raises)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
  }

  # 1. Test without DEBUG enabled
  with patch(
      "google.adk.cli.api_server.logger.isEnabledFor", return_value=False
  ):
    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 200
    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(sse_events) == 1
    error_event = sse_events[0]
    assert error_event["error"] == "ValueError: boom"
    assert "error_details" in error_event
    assert error_event["error_details"]["error_type"] == "ValueError"
    assert error_event["error_details"]["error_message"] == "boom"
    assert "stacktrace" not in error_event["error_details"]
    assert "timestamp" in error_event["error_details"]

  # 2. Test with DEBUG enabled
  with patch(
      "google.adk.cli.api_server.logger.isEnabledFor", return_value=True
  ):
    response = test_app.post("/run_sse", json=payload)
    assert response.status_code == 200
    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(sse_events) == 1
    error_event = sse_events[0]
    assert error_event["error"] == "ValueError: boom"
    assert "stacktrace" in error_event["error_details"]
    assert "ValueError: boom" in error_event["error_details"]["stacktrace"]


async def test_agent_run_sse_disconnect_with_cleanup_exception(
    test_app, create_test_session, monkeypatch
):
  """Test that exception during aclose() of runner is caught in /run_sse."""
  from google.adk.cli.api_server import RunAgentRequest

  info = create_test_session

  class MockAsyncGenerator:

    def __init__(self):
      self.yielded = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      if not self.yielded:
        self.yielded = True
        return Event(
            author="dummy agent",
            invocation_id="invocation_id",
            content=types.Content(
                role="model", parts=[types.Part(text="LLM reply")]
            ),
        )
      raise StopAsyncIteration

    async def aclose(self):
      raise ValueError("cleanup failed")

  def run_async_mock(self, **kwargs):
    return MockAsyncGenerator()

  monkeypatch.setattr(Runner, "run_async", run_async_mock)

  # Get the app and handler
  app = test_app.app
  handler = None
  for route in app.routes:
    if route.path == "/run_sse":
      handler = route.endpoint
      break
  assert handler is not None

  # Prepare request
  req = RunAgentRequest(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      new_message={"role": "user", "parts": [{"text": "Hello agent"}]},
      streaming=True,
  )

  # Call handler
  response = await handler(req)
  assert response.status_code == 200

  # Iterate generator and close it early
  generator = response.body_iterator

  event = await generator.__anext__()
  assert "LLM reply" in event

  # Close the generator early (simulating disconnect)
  try:
    await generator.aclose()
  except Exception as e:
    pytest.fail(f"generator.aclose() raised exception: {e}")


async def test_agent_run_sse_disconnect_with_cleanup_exception_and_cancellation(
    test_app, create_test_session, monkeypatch
):
  """Test that CancelledError is propagated during /run_sse even if cleanup fails."""
  from google.adk.cli.api_server import RunAgentRequest

  info = create_test_session

  class MockAsyncGenerator:

    def __init__(self):
      self.yielded = False

    def __aiter__(self):
      return self

    async def __anext__(self):
      if not self.yielded:
        self.yielded = True
        return Event(
            author="dummy agent",
            invocation_id="invocation_id",
            content=types.Content(
                role="model", parts=[types.Part(text="LLM reply")]
            ),
        )
      # Block indefinitely to allow cancellation simulation
      await asyncio.sleep(10)
      raise StopAsyncIteration

    async def aclose(self):
      raise ValueError("cleanup failed")

  def run_async_mock(self, **kwargs):
    return MockAsyncGenerator()

  monkeypatch.setattr(Runner, "run_async", run_async_mock)

  # Get the app and handler
  app = test_app.app
  handler = None
  for route in app.routes:
    if route.path == "/run_sse":
      handler = route.endpoint
      break
  assert handler is not None

  # Prepare request
  req = RunAgentRequest(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      new_message={"role": "user", "parts": [{"text": "Hello agent"}]},
      streaming=True,
  )

  # Call handler
  response = await handler(req)
  assert response.status_code == 200

  # Iterate generator
  generator = response.body_iterator

  # Read first event (this enters the generator and yields)
  event = await generator.__anext__()
  assert "LLM reply" in event

  # Now the generator is blocked on the next __anext__ (which is sleeping)
  # Run the next __anext__ in a task so we can cancel it
  task = asyncio.create_task(generator.__anext__())

  # Yield control to let the task start and block on sleep
  await asyncio.sleep(0.1)

  # Cancel the task
  task.cancel()

  # Verify that the task raises CancelledError, and NOT ValueError (cleanup failed)
  with pytest.raises(asyncio.CancelledError):
    await task


def _slow_tool_app(
    info,
    captured_contexts,
    tool_in_flight: asyncio.Event,
    after_run_flag: asyncio.Event,
    *,
    call_id: str,
) -> App:
  """Builds an App with a slow-tool agent and an after_run recorder plugin."""

  class _AfterRunPlugin(BasePlugin):

    async def after_run_callback(self, *, invocation_context):
      del invocation_context
      after_run_flag.set()

  class SlowToolAgent(BaseAgent):

    def __init__(self, name: str):
      super().__init__(name=name, sub_agents=[])

    async def _run_async_impl(self, invocation_context):
      captured_contexts.append(invocation_context)
      fc = types.Part.from_function_call(name="slow_tool", args={"q": "test"})
      fc.function_call.id = call_id
      yield Event(
          invocation_id=invocation_context.invocation_id,
          author=self.name,
          content=types.Content(role="model", parts=[fc]),
      )
      tool_in_flight.set()
      await asyncio.sleep(5.0)

  return App(
      name=info["app_name"],
      root_agent=SlowToolAgent("slow_tool_agent"),
      plugins=[_AfterRunPlugin(name="after_run")],
  )


async def test_agent_run_sse_disconnect_seals_dangling_function_call(
    create_test_session,
    mock_session_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test /run_sse disconnect aborts, seals FunctionCall, and runs after_run."""
  info = create_test_session
  captured_contexts = []
  tool_in_flight = asyncio.Event()
  after_run_flag = asyncio.Event()

  # Restore real Runner.run_async instead of the autouse dummy_run_async mock
  monkeypatch.setattr(Runner, "run_async", _ORIGINAL_RUNNER_RUN_ASYNC)
  loaded_app = _slow_tool_app(
      info,
      captured_contexts,
      tool_in_flight,
      after_run_flag,
      call_id="call_sse_1",
  )
  monkeypatch.setattr(
      mock_agent_loader, "load_agent", lambda app_name: loaded_app
  )

  client = _create_test_client(
      mock_session_service,
      InMemoryArtifactService(),
      InMemoryMemoryService(),
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )
  app = client.app
  handler = None
  for route in app.routes:
    if route.path == "/run_sse":
      handler = route.endpoint
      break
  assert handler is not None

  req = RunAgentRequest(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      new_message={"role": "user", "parts": [{"text": "Run slow tool"}]},
      streaming=True,
  )

  response = await handler(req)
  assert response.status_code == 200

  sent_chunks: list[str] = []

  async def receive():
    await tool_in_flight.wait()
    return {"type": "http.disconnect"}

  async def send(message):
    if message["type"] == "http.response.body" and message.get("body"):
      sent_chunks.append(message["body"].decode("utf-8"))

  await response(
      {"type": "http", "asgi": {"spec_version": "2.1"}},
      receive,
      send,
  )

  assert any("slow_tool" in chunk for chunk in sent_chunks)
  assert len(captured_contexts) == 1
  assert captured_contexts[0].is_aborted is True
  assert after_run_flag.is_set()

  # Verify the dangling FunctionCall was sealed with a synthetic FunctionResponse in session
  session = await mock_session_service.get_session(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
  )
  abort_events = [
      e for e in session.events if e.error_code == "INVOCATION_ABORTED"
  ]
  assert len(abort_events) == 1
  frs = abort_events[0].get_function_responses()
  assert len(frs) == 1
  assert frs[0].id == "call_sse_1"
  assert frs[0].name == "slow_tool"


async def test_agent_run_sse_consumer_exception_surfaces_error_event(
    create_test_session,
    mock_session_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test that an exception during SSE consumer formatting surfaces an SSE error payload."""
  info = create_test_session

  client = _create_test_client(
      mock_session_service,
      InMemoryArtifactService(),
      InMemoryMemoryService(),
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )
  app = client.app
  handler = None
  for route in app.routes:
    if route.path == "/run_sse":
      handler = route.endpoint
      break
  assert handler is not None

  req = RunAgentRequest(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      new_message={"role": "user", "parts": [{"text": "Hello"}]},
      streaming=True,
  )

  def _failing_model_dump_json(*args, **kwargs):
    raise ValueError("Simulated JSON serialization error in consumer")

  monkeypatch.setattr(Event, "model_dump", _failing_model_dump_json)
  monkeypatch.setattr(Event, "model_dump_json", _failing_model_dump_json)

  response = await handler(req)
  assert response.status_code == 200

  sent_chunks: list[str] = []

  async def receive():
    # Client stays connected; StreamingResponse cancels this when streaming ends.
    await asyncio.Event().wait()

  async def send(message):
    if message["type"] == "http.response.body" and message.get("body"):
      sent_chunks.append(message["body"].decode("utf-8"))

  await response(
      {"type": "http", "asgi": {"spec_version": "2.1"}},
      receive,
      send,
  )

  full_response = "".join(sent_chunks)
  assert "Simulated JSON serialization error in consumer" in full_response
  assert "ValueError" in full_response


def test_list_artifact_names(test_app, create_test_session):
  """Test listing artifact names for a session."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/{info['session_id']}/artifacts"
  response = test_app.get(url)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  logger.info(f"Listed {len(data)} artifacts")


def test_save_artifact(test_app, create_test_session, mock_artifact_service):
  """Test saving an artifact through the FastAPI endpoint."""
  info = create_test_session
  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts"
  )
  artifact_part = types.Part(text="hello world")
  payload = {
      "filename": "greeting.txt",
      "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
  }

  response = test_app.post(url, json=payload)
  assert response.status_code == 200
  data = response.json()
  assert data["version"] == 0
  assert data["customMetadata"] == {}
  assert data["mimeType"] in (None, "text/plain")
  assert data["canonicalUri"].endswith(
      f"/sessions/{info['session_id']}/artifacts/"
      f"{payload['filename']}/versions/0"
  )
  assert isinstance(data["createTime"], float)

  key = (
      f"{info['app_name']}:{info['user_id']}:{info['session_id']}:"
      f"{payload['filename']}"
  )
  stored = mock_artifact_service._artifacts[key][0]
  assert stored["artifact"].text == "hello world"


def test_save_artifact_reference(
    test_app, create_test_session, mock_artifact_service
):
  """Test saving an artifact reference through the FastAPI endpoint."""
  info = create_test_session
  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts"
  )
  payload = {
      "filename": "reference.txt",
      "artifact": {
          "fileData": {
              "fileUri": (
                  f"artifact://apps/{info['app_name']}/users/{info['user_id']}/"
                  f"sessions/{info['session_id']}/artifacts/target_file/versions/0"
              ),
              "mimeType": "text/plain",
          }
      },
  }

  response = test_app.post(url, json=payload)
  assert response.status_code == 200
  data = response.json()
  assert data["version"] == 0
  assert data["customMetadata"] == {}
  assert data["mimeType"] in (None, "text/plain")
  assert data["canonicalUri"].endswith(
      f"/sessions/{info['session_id']}/artifacts/"
      f"{payload['filename']}/versions/0"
  )
  assert isinstance(data["createTime"], float)

  key = (
      f"{info['app_name']}:{info['user_id']}:{info['session_id']}:"
      f"{payload['filename']}"
  )
  stored = mock_artifact_service._artifacts[key][0]
  assert stored["artifact"].file_data is not None
  assert (
      stored["artifact"].file_data.file_uri
      == payload["artifact"]["fileData"]["fileUri"]
  )
  assert stored["artifact"].file_data.mime_type == "text/plain"


def test_artifact_endpoints_support_nested_names(
    test_app, create_test_session, mock_artifact_service
):
  """Test artifact endpoints support names containing path separators."""
  info = create_test_session
  filename = "reports/summary.txt"
  encoded_filename = quote(filename, safe="")
  base_url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts"
  )

  mock_artifact_service.add_artifact(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      filename=filename,
      artifact=types.Part(text="v0"),
  )
  mock_artifact_service.add_artifact(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      filename=filename,
      artifact=types.Part(text="v1"),
      custom_metadata={"rev": "one"},
      mime_type="text/plain",
  )

  response = test_app.get(base_url)
  assert response.status_code == 200
  assert filename in response.json()

  for artifact_path in (filename, encoded_filename):
    response = test_app.get(f"{base_url}/{artifact_path}")
    assert response.status_code == 200
    assert response.json()["text"] == "v1"

  response = test_app.get(f"{base_url}/{encoded_filename}?version=0")
  assert response.status_code == 200
  assert response.json()["text"] == "v0"

  response = test_app.get(f"{base_url}/{filename}/versions/0")
  assert response.status_code == 200
  assert response.json()["text"] == "v0"

  response = test_app.get(f"{base_url}/{encoded_filename}/versions/1")
  assert response.status_code == 200
  assert response.json()["text"] == "v1"

  response = test_app.get(f"{base_url}/{filename}/versions")
  assert response.status_code == 200
  assert response.json() == [0, 1]

  response = test_app.get(f"{base_url}/{encoded_filename}/versions/metadata")
  assert response.status_code == 200
  versions_metadata = response.json()
  assert len(versions_metadata) == 2
  assert versions_metadata[1]["customMetadata"] == {"rev": "one"}

  response = test_app.get(f"{base_url}/{filename}/versions/1/metadata")
  assert response.status_code == 200
  version_metadata = response.json()
  assert version_metadata["version"] == 1
  assert version_metadata["customMetadata"] == {"rev": "one"}

  # Test loading latest version via path
  for path in (filename, encoded_filename):
    response = test_app.get(f"{base_url}/{path}/versions/latest")
    assert response.status_code == 200
    assert response.json()["text"] == "v1"

    response = test_app.get(f"{base_url}/{path}/versions/latest/metadata")
    assert response.status_code == 200
    assert response.json()["version"] == 1

  # Test invalid version ID
  response = test_app.get(f"{base_url}/{filename}/versions/invalid")
  assert response.status_code == 422
  assert "Invalid version ID" in response.json()["detail"]

  response = test_app.get(f"{base_url}/{filename}/versions/invalid/metadata")
  assert response.status_code == 422
  assert "Invalid version ID" in response.json()["detail"]

  response = test_app.delete(f"{base_url}/{encoded_filename}")
  assert response.status_code == 200

  response = test_app.get(f"{base_url}/{encoded_filename}")
  assert response.status_code == 404


def test_save_artifact_returns_400_on_validation_error(
    test_app, create_test_session, mock_artifact_service
):
  """Test save artifact endpoint surfaces validation errors as HTTP 400."""
  info = create_test_session
  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts"
  )
  artifact_part = types.Part(text="bad data")
  payload = {
      "filename": "invalid.txt",
      "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
  }

  mock_artifact_service.save_artifact_side_effect = InputValidationError(
      "invalid artifact"
  )

  response = test_app.post(url, json=payload)
  assert response.status_code == 400
  assert response.json()["detail"] == "invalid artifact"


def test_save_artifact_returns_500_on_unexpected_error(
    test_app, create_test_session, mock_artifact_service
):
  """Test save artifact endpoint surfaces unexpected errors as HTTP 500."""
  info = create_test_session
  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts"
  )
  artifact_part = types.Part(text="bad data")
  payload = {
      "filename": "invalid.txt",
      "artifact": artifact_part.model_dump(by_alias=True, exclude_none=True),
  }

  mock_artifact_service.save_artifact_side_effect = RuntimeError(
      "unexpected failure"
  )

  response = test_app.post(url, json=payload)
  assert response.status_code == 500
  assert response.json()["detail"] == "unexpected failure"


def test_get_artifact_version_metadata(
    test_app, create_test_session, mock_artifact_service
):
  """Test retrieving metadata for a specific artifact version."""
  info = create_test_session
  mock_artifact_service.add_artifact(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      filename="report.txt",
      artifact=types.Part(text="hello"),
      custom_metadata={"foo": "bar"},
      mime_type="text/plain",
  )

  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts/report.txt/versions/0/metadata"
  )
  response = test_app.get(url)

  assert response.status_code == 200
  data = response.json()
  assert data["version"] == 0
  assert data["customMetadata"] == {"foo": "bar"}
  assert data["mimeType"] == "text/plain"


def test_list_artifact_versions_metadata(
    test_app, create_test_session, mock_artifact_service
):
  """Test listing metadata for all versions of an artifact."""
  info = create_test_session
  mock_artifact_service.add_artifact(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      filename="report.txt",
      artifact=types.Part(text="v0"),
  )
  mock_artifact_service.add_artifact(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      filename="report.txt",
      artifact=types.Part(text="v1"),
      custom_metadata={"foo": "bar"},
  )

  url = (
      f"/apps/{info['app_name']}/users/{info['user_id']}/sessions/"
      f"{info['session_id']}/artifacts/report.txt/versions/metadata"
  )
  response = test_app.get(url)

  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  assert len(data) == 2
  assert data[1]["version"] == 1
  assert data[1]["customMetadata"] == {"foo": "bar"}


def test_get_eval_set_result_not_found(test_app):
  """Test getting an eval set result that doesn't exist."""
  url = "/apps/test_app_name/eval_results/test_eval_result_id_not_found"
  response = test_app.get(url)
  assert response.status_code == 404


def test_list_metrics_info(builder_test_client):
  """Test listing metrics info."""
  url = "/dev/apps/test_app/metrics-info"
  response = builder_test_client.get(url)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  metrics_info_key = "metricsInfo"
  assert metrics_info_key in data
  assert isinstance(data[metrics_info_key], list)
  # Add more assertions based on the expected metrics
  assert len(data[metrics_info_key]) > 0
  for metric in data[metrics_info_key]:
    assert "metricName" in metric
    assert "description" in metric
    assert "metricValueInfo" in metric


def test_list_metrics_info_includes_metrics_that_need_no_threshold(
    builder_test_client,
):
  """Informational metrics are listed too, flagged as needing no threshold.

  A caller that asks the user to pick metrics and set a threshold for each
  filters on `requiresThreshold`; a caller that only describes metrics, such
  as the Dev UI's result tooltips, needs every registered metric present.
  """
  response = builder_test_client.get("/dev/apps/test_app/metrics-info")

  assert response.status_code == 200
  by_name = {
      metric["metricName"]: metric for metric in response.json()["metricsInfo"]
  }
  assert by_name["tool_trajectory_avg_score"]["requiresThreshold"] is True
  for informational in (
      "tool_call_count_v1",
      "inference_call_count_v1",
      "token_usage_v1",
      "invocation_duration_v1",
  ):
    assert by_name[informational]["requiresThreshold"] is False
    # Nothing bounds an informational value, so a threshold control has no
    # interval to size itself by. That is why the caller filters instead.
    assert "interval" not in by_name[informational]["metricValueInfo"]


def test_debug_trace(test_app):
  """Test the debug trace endpoint."""
  # This test will likely return 404 since we haven't set up trace data,
  # but it tests that the endpoint exists and handles missing traces correctly.
  url = "/dev/apps/test_app/debug/trace/nonexistent-event"
  response = test_app.get(url)

  # Verify we get a 404 for a nonexistent trace
  assert response.status_code == 404
  logger.info("Debug trace test completed successfully")


def test_openapi_json_schema_accessible(test_app):
  """Test that the OpenAPI /openapi.json endpoint is accessible."""
  response = test_app.get("/openapi.json")
  assert response.status_code == 200
  logger.info("OpenAPI /openapi.json endpoint is accessible")


@pytest.mark.skipif(
    _compat.IS_A2A_V1,
    reason=(
        "0.3.x-only: mocks server.apps.A2AStarletteApplication (gone in 1.x)"
    ),
)
def test_a2a_agent_discovery(test_app_with_a2a):
  """Test that A2A agents are properly discovered and configured."""
  # This test mainly verifies that the A2A setup doesn't break the app
  response = test_app_with_a2a.get("/list-apps")
  assert response.status_code == 200
  logger.info("A2A agent discovery test passed")


@pytest.mark.skipif(
    _compat.IS_A2A_V1,
    reason=(
        "0.3.x-only: mocks server.apps.A2AStarletteApplication (gone in 1.x)"
    ),
)
def test_a2a_request_handler_uses_push_config_store(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
  """Test A2A request handler gets push config store when supported."""
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=mock_memory_service,
      ),
      patch(
          "google.adk.cli.fast_api.AgentLoader",
          return_value=mock_agent_loader,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=mock_eval_sets_manager,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=mock_eval_set_results_manager,
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
      ) as mock_create_task_store,
      patch(
          "a2a.server.tasks.InMemoryPushNotificationConfigStore"
      ) as mock_push_config_store_class,
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    mock_task_store_instance = MagicMock()
    mock_create_task_store.return_value = mock_task_store_instance
    mock_push_config_store = MagicMock()
    mock_push_config_store_class.return_value = mock_push_config_store
    mock_executor_instance = MagicMock()
    mock_executor.return_value = mock_executor_instance
    mock_handler.return_value = MagicMock()
    mock_a2a_app_instance = MagicMock()
    mock_a2a_app_instance.routes.return_value = []
    mock_a2a_app.return_value = mock_a2a_app_instance

    monkeypatch.chdir(temp_agents_dir_with_a2a)
    _ = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=True,
        host="127.0.0.1",
        port=8000,
    )

    mock_handler.assert_called_once_with(
        agent_executor=mock_executor_instance,
        push_config_store=mock_push_config_store,
        task_store=mock_task_store_instance,
    )


@pytest.mark.skipif(
    _compat.IS_A2A_V1,
    reason=(
        "0.3.x-only: mocks server.apps.A2AStarletteApplication (gone in 1.x)"
    ),
)
def test_a2a_request_handler_uses_task_store_uri(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
  """Test A2A request handler uses task store created from URI."""
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=mock_memory_service,
      ),
      patch(
          "google.adk.cli.fast_api.AgentLoader",
          return_value=mock_agent_loader,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=mock_eval_sets_manager,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=mock_eval_set_results_manager,
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
      ) as mock_create_task_store,
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    custom_task_store = MagicMock()
    mock_create_task_store.return_value = custom_task_store
    mock_executor_instance = MagicMock()
    mock_executor.return_value = mock_executor_instance
    mock_handler.return_value = MagicMock()
    mock_a2a_app_instance = MagicMock()
    mock_a2a_app_instance.routes.return_value = []
    mock_a2a_app.return_value = mock_a2a_app_instance

    test_uri = "postgresql+asyncpg://user:pass@host/db"
    monkeypatch.chdir(temp_agents_dir_with_a2a)
    _ = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=True,
        task_store_uri=test_uri,
        host="127.0.0.1",
        port=8000,
    )

    mock_create_task_store.assert_called_once_with(
        task_store_uri=test_uri,
    )
    mock_handler.assert_called_once()
    call_kwargs = mock_handler.call_args[1]
    assert call_kwargs["task_store"] is custom_task_store


@pytest.mark.skipif(
    _compat.IS_A2A_V1,
    reason=(
        "0.3.x-only: mocks server.apps.A2AStarletteApplication (gone in 1.x)"
    ),
)
def test_a2a_task_store_engine_disposed_on_shutdown(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
  """Test that the A2A task store engine is disposed on app shutdown."""
  mock_engine = AsyncMock()
  custom_task_store = MagicMock()
  custom_task_store.engine = mock_engine

  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=mock_memory_service,
      ),
      patch(
          "google.adk.cli.fast_api.AgentLoader",
          return_value=mock_agent_loader,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=mock_eval_sets_manager,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=mock_eval_set_results_manager,
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
          return_value=custom_task_store,
      ),
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    mock_executor.return_value = MagicMock()
    mock_handler.return_value = MagicMock()
    mock_a2a_app_instance = MagicMock()
    mock_a2a_app_instance.routes.return_value = []
    mock_a2a_app.return_value = mock_a2a_app_instance

    monkeypatch.chdir(temp_agents_dir_with_a2a)
    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=True,
        task_store_uri="postgresql+asyncpg://user:pass@host/db",
        host="127.0.0.1",
        port=8000,
    )

    # Exercise the lifespan to trigger shutdown cleanup.
    # TestClient enters/exits the lifespan context on __enter__/__exit__.
    with TestClient(app):
      pass

    mock_engine.dispose.assert_awaited_once()


@pytest.mark.skipif(
    _compat.IS_A2A_V1,
    reason=(
        "0.3.x-only: mocks server.apps.A2AStarletteApplication (gone in 1.x)"
    ),
)
def test_a2a_in_memory_task_store_no_engine_dispose(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
    monkeypatch,
):
  """Test that in-memory task stores (no engine attr) skip disposal."""
  custom_task_store = MagicMock(spec=[])  # no attributes at all

  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.create_session_service_from_options",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_artifact_service_from_options",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.create_memory_service_from_options",
          return_value=mock_memory_service,
      ),
      patch(
          "google.adk.cli.fast_api.AgentLoader",
          return_value=mock_agent_loader,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetsManager",
          return_value=mock_eval_sets_manager,
      ),
      patch(
          "google.adk.cli.fast_api.LocalEvalSetResultsManager",
          return_value=mock_eval_set_results_manager,
      ),
      patch(
          "google.adk.cli.fast_api._create_task_store_from_options",
          return_value=custom_task_store,
      ),
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    mock_executor.return_value = MagicMock()
    mock_handler.return_value = MagicMock()
    mock_a2a_app_instance = MagicMock()
    mock_a2a_app_instance.routes.return_value = []
    mock_a2a_app.return_value = mock_a2a_app_instance

    monkeypatch.chdir(temp_agents_dir_with_a2a)
    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=True,
        host="127.0.0.1",
        port=8000,
    )

    # Lifespan should complete without errors even with no engine.
    with TestClient(app):
      pass


def test_a2a_disabled_by_default(test_app):
  """Test that A2A functionality is disabled by default."""
  # The regular test_app fixture has a2a=False
  # This test ensures no A2A routes are added
  response = test_app.get("/list-apps")
  assert response.status_code == 200
  logger.info("A2A disabled by default test passed")


def test_patch_memory(test_app, create_test_session, mock_memory_service):
  """Test adding a session to memory."""
  info = create_test_session
  url = f"/apps/{info['app_name']}/users/{info['user_id']}/memory"
  payload = {"session_id": info["session_id"]}
  response = test_app.patch(url, json=payload)

  # Verify the response
  assert response.status_code == 200
  mock_memory_service.add_session_to_memory.assert_called_once()
  logger.info("Add session to memory test completed successfully")


def test_builder_final_save_preserves_files_and_cleans_tmp(
    builder_test_client, tmp_path
):
  files = [
      (
          "files",
          ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
      ),
      (
          "files",
          ("app/sub_agent.yaml", b"name: sub\n", "application/x-yaml"),
      ),
  ]
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true", files=files
  )
  assert response.status_code == 200
  assert response.json() is True

  response = builder_test_client.post(
      "/dev/apps/app/builder/save",
      files=[(
          "files",
          (
              "app/root_agent.yaml",
              b"name: app_updated\n",
              "application/x-yaml",
          ),
      )],
  )
  assert response.status_code == 200
  assert response.json() is True

  assert (tmp_path / "app" / "sub_agent.yaml").is_file()
  assert not (tmp_path / "app" / "tmp" / "app").exists()
  tmp_dir = tmp_path / "app" / "tmp"
  assert not tmp_dir.exists() or not any(tmp_dir.iterdir())


def test_builder_save_rejects_cross_origin_post(builder_test_client, tmp_path):
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      headers={"origin": "https://evil.com"},
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
      )],
  )

  assert response.status_code == 403
  assert response.text == "Forbidden: origin not allowed"
  assert not (tmp_path / "app" / "tmp" / "app").exists()


def test_builder_save_allows_same_origin_post(builder_test_client, tmp_path):
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      headers={"origin": _LOOPBACK_BASE_URL},
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
      )],
  )

  assert response.status_code == 200
  assert response.json() is True
  assert (tmp_path / "app" / "tmp" / "app" / "root_agent.yaml").is_file()


def test_builder_get_rejects_cross_origin_get(builder_test_client):
  """Reads expose agent config and session data, so they are guarded too."""
  response = builder_test_client.get(
      "/dev/apps/missing/builder?tmp=true",
      headers={"origin": "https://evil.com"},
  )

  assert response.status_code == 403
  assert response.text == "Forbidden: origin not allowed"


def test_builder_get_allows_same_origin_get(builder_test_client):
  """The dev UI reads its own agent config from the same origin."""
  response = builder_test_client.get(
      "/dev/apps/missing/builder?tmp=true",
      headers={"origin": _LOOPBACK_BASE_URL},
  )

  assert response.status_code == 200
  assert not response.text


def test_builder_get_allows_request_without_origin(builder_test_client):
  """Browsers omit Origin on same-origin reads, and CLI clients never send it."""
  response = builder_test_client.get("/dev/apps/missing/builder?tmp=true")

  assert response.status_code == 200
  assert not response.text


def test_builder_save_rejects_remote_client(
    remote_builder_test_client, tmp_path
):
  """A non-browser client off-machine must not be able to write agent YAML."""
  # Omitting the Origin header skips _OriginCheckMiddleware entirely, so the
  # loopback check is the only thing between the network and agents_dir.
  response = remote_builder_test_client.post(
      "/dev/apps/app/builder/save",
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: pwned\n", "application/x-yaml"),
      )],
  )

  assert response.status_code == 403
  assert not (tmp_path / "app" / "root_agent.yaml").exists()


def test_builder_get_rejects_remote_client(remote_builder_test_client):
  """The YAML readback is a disclosure too, so it is gated the same way."""
  response = remote_builder_test_client.get("/dev/apps/app/builder")

  assert response.status_code == 403


def test_builder_cancel_rejects_remote_client(remote_builder_test_client):
  """Discarding another developer's draft is a remote write as well."""
  response = remote_builder_test_client.post("/dev/apps/app/builder/cancel")

  assert response.status_code == 403


def test_builder_save_rejects_forwarded_loopback_client(
    builder_test_client, tmp_path
):
  """Behind a proxy the peer is loopback but the caller is still remote."""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save",
      headers={"x-forwarded-for": "203.0.113.7"},
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: pwned\n", "application/x-yaml"),
      )],
  )

  assert response.status_code == 403
  assert not (tmp_path / "app" / "root_agent.yaml").exists()


def test_builder_save_allows_remote_client_when_opted_in(
    remote_builder_test_client, tmp_path, monkeypatch
):
  """Serving the builder off-machine stays possible, but has to be chosen."""
  monkeypatch.setenv("ADK_ALLOW_REMOTE_AGENT_BUILDER", "1")

  response = remote_builder_test_client.post(
      "/dev/apps/app/builder/save",
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
      )],
  )

  assert response.status_code == 200
  assert (tmp_path / "app" / "root_agent.yaml").is_file()


def test_remote_client_can_still_reach_non_builder_endpoints(
    remote_builder_test_client,
):
  """The gate is scoped to mutating /dev routes and builder readback."""
  assert remote_builder_test_client.get("/list-apps").status_code == 200
  assert (
      remote_builder_test_client.get("/dev/apps/app/tests").status_code == 200
  )
  assert (
      remote_builder_test_client.get("/dev/apps/app/eval-sets").status_code
      == 200
  )


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("POST", "/dev/apps/app/tests/rebuild", None),
        ("POST", "/dev/apps/app/tests/run", {}),
        ("PUT", "/dev/apps/app/tests/test_smoke", {"content": "x = 1\n"}),
        ("DELETE", "/dev/apps/app/tests/test_smoke", None),
        ("POST", "/dev/apps/app/eval-sets", {"eval_set": {"eval_set_id": "s"}}),
        ("POST", "/dev/apps/app/eval_sets/s", None),
        (
            "POST",
            "/dev/apps/app/eval_sets/s/run_eval",
            {"eval_ids": [], "eval_metrics": []},
        ),
        (
            "POST",
            "/dev/apps/app/eval-sets/s/add-session",
            {"eval_id": "e1", "session_id": "s1", "user_id": "u1"},
        ),
        (
            "POST",
            "/dev/apps/app/eval_sets/s/add_session",
            {"eval_id": "e1", "session_id": "s1", "user_id": "u1"},
        ),
        (
            "PUT",
            "/dev/apps/app/eval-sets/s/eval-cases/c",
            {"eval_id": "c", "conversation": []},
        ),
        (
            "PUT",
            "/dev/apps/app/eval_sets/s/evals/c",
            {"eval_id": "c", "conversation": []},
        ),
        ("DELETE", "/dev/apps/app/eval-sets/s/eval-cases/c", None),
        ("DELETE", "/dev/apps/app/eval_sets/s/evals/c", None),
        (
            "POST",
            "/dev/apps/app/eval-sets/s/run",
            {"eval_ids": [], "eval_metrics": []},
        ),
        ("POST", "/dev/apps/app/deploy/agent_engine", {}),
        ("POST", "/dev/apps/app/deploy/cloud_run", {"project": "p"}),
        (
            "POST",
            "/dev/apps/app/deploy/gke",
            {"project": "p", "region": "r", "cluster_name": "c"},
        ),
    ],
)
def test_mutating_dev_routes_reject_remote_client(
    remote_builder_test_client, builder_test_client, method, path, json_body
):
  """All mutating /dev endpoints reject non-loopback and proxied callers."""
  kwargs = {"json": json_body} if json_body is not None else {}
  remote_response = remote_builder_test_client.request(method, path, **kwargs)
  assert remote_response.status_code == 403

  forwarded_response = builder_test_client.request(
      method,
      path,
      headers={"x-forwarded-for": "203.0.113.7"},
      **kwargs,
  )
  assert forwarded_response.status_code == 403


def test_builder_cancel_deletes_tmp_idempotent(builder_test_client, tmp_path):
  tmp_agent_root = tmp_path / "app" / "tmp" / "app"
  tmp_agent_root.mkdir(parents=True, exist_ok=True)
  (tmp_agent_root / "root_agent.yaml").write_text("name: app\n")

  response = builder_test_client.post("/dev/apps/app/builder/cancel")
  assert response.status_code == 200
  assert response.json() is True
  assert not (tmp_path / "app" / "tmp").exists()

  response = builder_test_client.post("/dev/apps/app/builder/cancel")
  assert response.status_code == 200
  assert response.json() is True
  assert not (tmp_path / "app" / "tmp").exists()


def test_builder_get_tmp_true_recreates_tmp(builder_test_client, tmp_path):
  app_root = tmp_path / "app"
  app_root.mkdir(parents=True, exist_ok=True)
  (app_root / "root_agent.yaml").write_bytes(b"name: app\n")
  nested_dir = app_root / "nested"
  nested_dir.mkdir(parents=True, exist_ok=True)
  (nested_dir / "nested.yaml").write_bytes(b"nested: true\n")

  assert not (app_root / "tmp").exists()
  response = builder_test_client.get("/dev/apps/app/builder?tmp=true")
  assert response.status_code == 200
  assert response.text == "name: app\n"

  tmp_agent_root = app_root / "tmp" / "app"
  assert (tmp_agent_root / "root_agent.yaml").is_file()
  assert (tmp_agent_root / "nested" / "nested.yaml").is_file()

  response = builder_test_client.get(
      "/dev/apps/app/builder?tmp=true&file_path=nested/nested.yaml"
  )
  assert response.status_code == 200
  assert response.text == "nested: true\n"


def test_builder_get_tmp_true_missing_app_returns_empty(
    builder_test_client, tmp_path
):
  response = builder_test_client.get("/dev/apps/missing/builder?tmp=true")
  assert response.status_code == 200
  assert response.text == ""
  assert not (tmp_path / "missing").exists()


def test_builder_save_rejects_traversal(builder_test_client, tmp_path):
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/../escape.yaml", b"nope\n", "application/x-yaml"),
      )],
  )
  assert response.status_code == 400
  assert not (tmp_path / "escape.yaml").exists()
  assert not (tmp_path / "app" / "tmp" / "escape.yaml").exists()


def test_builder_save_rejects_py_files(builder_test_client, tmp_path):
  """Uploading .py files via /builder/save is rejected."""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/agent.py", b"import os\nos.system('id')\n", "text/plain"),
      )],
  )
  assert response.status_code == 400
  assert not (tmp_path / "app" / "tmp" / "app" / "agent.py").exists()


def test_builder_save_rejects_non_yaml_extensions(
    builder_test_client, tmp_path
):
  """Uploading non-YAML files (.json, .txt, .sh, etc.) is rejected."""
  for ext, content in [
      (".py", b"print('hi')"),
      (".json", b"{}"),
      (".txt", b"hello"),
      (".sh", b"#!/bin/bash"),
      (".pth", b"import os"),
  ]:
    response = builder_test_client.post(
        "/dev/apps/app/builder/save?tmp=true",
        files=[(
            "files",
            (f"app/file{ext}", content, "application/octet-stream"),
        )],
    )
    assert response.status_code == 400, f"Expected 400 for {ext}"


def test_builder_save_allows_yaml_files(builder_test_client, tmp_path):
  """Uploading .yaml and .yml files is allowed."""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/root_agent.yaml", b"name: app\n", "application/x-yaml"),
      )],
  )
  assert response.status_code == 200
  assert response.json() is True

  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/sub_agent.yml", b"name: sub\n", "application/x-yaml"),
      )],
  )
  assert response.status_code == 200
  assert response.json() is True


def test_builder_save_rejects_args_key(builder_test_client, tmp_path):
  """Uploading YAML with an `args` key is rejected (RCE prevention)."""
  yaml_with_args = b"""\
name: my_tool
args:
  key: value
"""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/root_agent.yaml", yaml_with_args, "application/x-yaml"),
      )],
  )
  assert response.status_code == 400
  assert "args" in response.json()["detail"]
  assert not (tmp_path / "app" / "tmp" / "app" / "root_agent.yaml").exists()


def test_builder_save_rejects_nested_args_key(builder_test_client, tmp_path):
  """Uploading YAML with a nested `args` key is rejected."""
  yaml_with_nested_args = b"""\
tools:
  - name: some_tool
    args:
      param: value
"""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[(
          "files",
          ("app/root_agent.yaml", yaml_with_nested_args, "application/x-yaml"),
      )],
  )
  assert response.status_code == 400
  assert "args" in response.json()["detail"]


def _save_builder_yaml(client, content, *, app_name="app"):
  """POST YAML to the builder save endpoint for the given app."""
  return client.post(
      f"/dev/apps/{app_name}/builder/save?tmp=true",
      files=[(
          "files",
          (f"{app_name}/root_agent.yaml", content, "application/x-yaml"),
      )],
  )


def test_builder_save_rejects_external_tool_reference(
    builder_test_client, tmp_path
):
  """A tool naming code outside the app is rejected."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ntools:\n  - name: os.system\n",
  )
  assert response.status_code == 400
  assert "os.system" in response.json()["detail"]
  assert not (tmp_path / "app" / "tmp" / "app" / "root_agent.yaml").exists()


def test_builder_save_allows_project_tool_reference(builder_test_client):
  """A tool under the app being edited is allowed."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ntools:\n  - name: app.tools.search\n",
  )
  assert response.status_code == 200


def test_builder_save_allows_built_in_tool_short_name(builder_test_client):
  """An undotted tool name still resolves against ADK's own built-ins."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ntools:\n  - name: google_search\n",
  )
  assert response.status_code == 200


def test_builder_save_allows_built_in_agent_class(builder_test_client):
  """A qualified ADK agent class is allowed."""
  response = _save_builder_yaml(
      builder_test_client,
      b"agent_class: google.adk.agents.LlmAgent\nname: my_agent\n",
  )
  assert response.status_code == 200


def test_builder_save_rejects_adk_submodule_reference(builder_test_client):
  """An ADK path reaching past the exported built-ins is rejected."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ntools:\n"
      b"  - name: google.adk.tools.bash_tool.BashTool\n",
  )
  assert response.status_code == 400
  assert "BashTool" in response.json()["detail"]


def test_builder_save_rejects_external_callback_reference(builder_test_client):
  """A callback naming code outside the app is rejected."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\nbefore_agent_callbacks:\n  - name: os.system\n",
  )
  assert response.status_code == 400
  assert "before_agent_callbacks" in response.json()["detail"]


def test_builder_save_rejects_external_sub_agent_code(builder_test_client):
  """A sub-agent naming code outside the app is rejected."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\nsub_agents:\n  - code: other_package.agent\n",
  )
  assert response.status_code == 400
  assert "other_package.agent" in response.json()["detail"]


def test_builder_save_rejects_external_schema_reference(builder_test_client):
  """A schema given as a bare string is validated like any other reference."""
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ninput_schema: os.path\n",
  )
  assert response.status_code == 400
  assert "input_schema" in response.json()["detail"]


@pytest.mark.parametrize(
    ("app_name", "reference"),
    [
        ("os", "os.system"),
        ("sys", "sys.exit"),
        ("google", "google.genai.Client"),
        ("dotenv", "dotenv.cli.run_command"),
    ],
)
def test_builder_save_rejects_reference_when_app_name_shadows_module(
    builder_test_client, app_name, reference
):
  """An app named after a real module cannot vouch for its own references."""
  response = _save_builder_yaml(
      builder_test_client,
      f"name: my_agent\ntools:\n  - name: {reference}\n".encode(),
      app_name=app_name,
  )
  assert response.status_code == 400
  assert "shadows" in response.json()["detail"]


def test_builder_save_allows_reference_when_app_imports_from_its_directory(
    builder_test_client, tmp_path, monkeypatch
):
  """An app that is importable passes when it imports from its own folder."""
  (tmp_path / "importable_app").mkdir()
  (tmp_path / "importable_app" / "__init__.py").touch()
  monkeypatch.syspath_prepend(str(tmp_path))
  response = _save_builder_yaml(
      builder_test_client,
      b"name: my_agent\ntools:\n  - name: importable_app.tools.search\n",
      app_name="importable_app",
  )
  assert response.status_code == 200


def test_builder_save_covers_every_code_config_field(builder_test_client):
  """Every config field holding a CodeConfig is checked on upload."""
  code_config_fields = set()
  for agent in (BaseAgent, LlmAgent):
    for name, field in agent.config_type.model_fields.items():
      if "CodeConfig" in str(field.annotation):
        code_config_fields.add(name)
  assert code_config_fields, "expected agent configs to declare CodeConfig"

  for field_name in sorted(code_config_fields):
    content = f"name: my_agent\n{field_name}:\n  name: os.system\n"
    response = _save_builder_yaml(builder_test_client, content.encode())
    assert response.status_code == 400, field_name
    assert field_name in response.json()["detail"]


def test_builder_get_rejects_non_yaml_file_paths(builder_test_client, tmp_path):
  """GET /dev/apps/{app_name}/builder?file_path=...

  rejects non-YAML extensions.
  """
  app_root = tmp_path / "app"
  app_root.mkdir(parents=True, exist_ok=True)
  (app_root / ".env").write_text("SECRET=supersecret\n")
  (app_root / "agent.py").write_text("root_agent = None\n")
  (app_root / "config.json").write_text("{}\n")

  for file_path in [".env", "agent.py", "config.json"]:
    response = builder_test_client.get(
        f"/dev/apps/app/builder?file_path={file_path}"
    )
    assert response.status_code == 200, f"Expected 200 for {file_path}"
    assert response.text == "", f"Expected empty response for {file_path}"


def test_builder_get_allows_yaml_file_paths(builder_test_client, tmp_path):
  """GET /dev/apps/{app_name}/builder?file_path=... allows YAML extensions."""
  app_root = tmp_path / "app"
  app_root.mkdir(parents=True, exist_ok=True)
  (app_root / "sub_agent.yaml").write_bytes(b"name: sub\n")
  (app_root / "tool.yml").write_bytes(b"name: tool\n")

  response = builder_test_client.get(
      "/dev/apps/app/builder?file_path=sub_agent.yaml"
  )
  assert response.status_code == 200
  assert response.text == "name: sub\n"

  response = builder_test_client.get("/dev/apps/app/builder?file_path=tool.yml")
  assert response.status_code == 200
  assert response.text == "name: tool\n"


def test_builder_endpoints_not_registered_without_web(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Builder endpoints must not be registered when web=False (e.g. deploy)."""
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
  )
  # /dev/apps/app/builder/save should return 404/405, not 200
  response = client.post(
      "/dev/apps/app/builder/save",
      files=[
          ("files", ("app/agent.yaml", b"name: test\n", "application/x-yaml"))
      ],
  )
  assert response.status_code in (404, 405)

  # /dev/apps/{name}/builder/cancel should also be absent
  response = client.post("/dev/apps/app/builder/cancel")
  assert response.status_code in (404, 405)

  # /dev/apps/{name}/builder GET should also be absent
  response = client.get("/dev/apps/app/builder")
  assert response.status_code in (404, 405)


def test_builder_endpoints_registered_with_web(builder_test_client):
  """Builder endpoints are available when web=True."""
  response = builder_test_client.post(
      "/dev/apps/app/builder/save?tmp=true",
      files=[
          ("files", ("app/agent.yaml", b"name: test\n", "application/x-yaml"))
      ],
  )
  assert response.status_code == 200


def test_agent_run_resume_without_message_success(
    test_app, create_test_session
):
  """Test that /run allows resuming a session with only an invocation_id, without a new message."""
  info = create_test_session
  url = "/run"
  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "invocation_id": "test_invocation_id",
      "streaming": False,
  }
  response = test_app.post(url, json=payload)
  assert response.status_code == 200


def test_health_endpoint(test_app):
  """Test the health endpoint."""
  response = test_app.get("/health")
  assert response.status_code == 200
  assert response.json() == {"status": "ok"}


def test_version_endpoint(test_app):
  """Test the version endpoint."""
  response = test_app.get("/version")
  assert response.status_code == 200
  data = response.json()
  assert "version" in data
  assert "language" in data
  assert data["language"] == "python"
  assert "language_version" in data


def test_telemetry_get_endpoint(test_app):
  """Test the GET telemetry consent endpoint."""
  with patch(
      "google.adk.cli.dev_server.read_telemetry_consent", return_value=True
  ):
    response = test_app.get("/config/telemetry")
    assert response.status_code == 200
    assert response.json() == {"telemetry": True}


def test_telemetry_post_endpoint_success(test_app):
  """Test the POST telemetry consent endpoint with required header."""
  with patch("google.adk.cli.dev_server.write_telemetry_consent") as mock_write:
    headers = {"x-adk-telemetry-request": "true"}
    response = test_app.post(
        "/config/telemetry", json={"telemetry": True}, headers=headers
    )
    assert response.status_code == 200
    assert response.json() == {"telemetry": True}
    mock_write.assert_called_once_with(True)


def test_telemetry_post_endpoint_missing_header(test_app):
  """Test the POST telemetry consent endpoint without required header."""
  response = test_app.post("/config/telemetry", json={"telemetry": True})
  assert response.status_code == 400
  assert "Forbidden: missing required security header" in response.text


def test_setup_gcp_telemetry_requests_cloud_platform_scope(monkeypatch):
  """A service-account key file ADC has requires_scopes=True and no scopes,

  so google.auth.default() must be asked for the cloud-platform scope or the
  OTLP exporters' token refresh fails with invalid_scope.
  """
  from google.adk.cli.api_server import _setup_gcp_telemetry
  from google.adk.telemetry.google_cloud import _CLOUD_PLATFORM_SCOPE

  auth_default = MagicMock(return_value=("creds", "project-id"))
  monkeypatch.setattr("google.auth.default", auth_default)
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud.get_gcp_exporters",
      lambda **kwargs: MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.google_cloud.get_gcp_resource",
      lambda project_id: MagicMock(),
  )
  monkeypatch.setattr(
      "google.adk.telemetry.setup.maybe_set_otel_providers",
      lambda **kwargs: None,
  )
  monkeypatch.setattr(
      "google.adk.cli.api_server._setup_instrumentation_lib_if_installed",
      lambda: None,
  )

  _setup_gcp_telemetry()

  auth_default.assert_called_once_with(scopes=[_CLOUD_PLATFORM_SCOPE])


@pytest.fixture
def test_app_auto_session(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Create a TestClient with auto_create_session=True."""
  return _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
      auto_create_session=True,
  )


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_auto_creates_session(
    test_app_auto_session, test_session_info, endpoint
):
  """Test /run and /run_sse auto-create sessions when auto_create_session=True."""
  payload = {
      "app_name": test_session_info["app_name"],
      "user_id": test_session_info["user_id"],
      "session_id": "nonexistent_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
  }

  response = test_app_auto_session.post(endpoint, json=payload)
  assert response.status_code == 200

  if endpoint == "/run":
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
  else:
    sse_events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(sse_events) > 0
    assert not any("error" in e for e in sse_events)


@pytest.mark.parametrize("endpoint", ["/run", "/run_sse"])
def test_returns_404_without_auto_create(
    test_app, test_session_info, monkeypatch, endpoint
):
  """Test /run and /run_sse return 404 for missing sessions without auto_create."""

  async def run_async_session_not_found(self, **kwargs):
    raise SessionNotFoundError(f"Session not found: {kwargs['session_id']}")
    yield  # make it an async generator  # pylint: disable=unreachable

  monkeypatch.setattr(Runner, "run_async", run_async_session_not_found)

  payload = {
      "app_name": test_session_info["app_name"],
      "user_id": test_session_info["user_id"],
      "session_id": "nonexistent_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
  }

  response = test_app.post(endpoint, json=payload)
  assert response.status_code == 404
  assert "Session not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_independent_telemetry_context(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test that two agents have independent is_visual_builder context variables."""
  from google.adk.utils._telemetry_context import _is_visual_builder
  import httpx

  # We use httpx.AsyncClient to send concurrent requests to the FastAPI app.
  # This proves that is_visual_builder doesn't leak across concurrent requests.
  captured_visual_builder_values = {}

  async def run_async_capture(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    # Capture the value of is_visual_builder inside the request context
    captured_visual_builder_values[self.app.name] = _is_visual_builder.get()

    # Sleep to ensure both requests overlap in time
    await asyncio.sleep(0.1)

    # Read again to ensure it wasn't overwritten by the other concurrent request
    captured_visual_builder_values[self.app.name + "_after_sleep"] = (
        _is_visual_builder.get()
    )

    yield _event_1()

  monkeypatch.setattr(Runner, "run_async", run_async_capture)

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
          "NestedAgentLoader",
          autospec=True,
          return_value=mock_agent_loader,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
      patch.object(
          os.path,
          "exists",
          autospec=True,
          side_effect=lambda p: "yaml_app" in str(p)
          and str(p).endswith("root_agent.yaml"),
      ),
  ):
    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=False,
        host="127.0.0.1",
        port=8000,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
      # Send concurrent requests
      req1 = client.post(
          "/run",
          json={
              "app_name": "test_app",
              "user_id": "test_user",
              "session_id": "test_session",
              "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
          },
      )
      req2 = client.post(
          "/run",
          json={
              "app_name": "yaml_app",
              "user_id": "test_user",
              "session_id": "test_session",
              "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
          },
      )

      await asyncio.gather(req1, req2)

  assert captured_visual_builder_values.get("test_app") == False
  assert captured_visual_builder_values.get("test_app_after_sleep") == False

  assert captured_visual_builder_values.get("yaml_app") == True
  assert captured_visual_builder_values.get("yaml_app_after_sleep") == True


def test_default_app_name_middleware_and_resolution(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test that when ADK_DEFAULT_APP_NAME is set, path rewriting works for get_session and run."""
  # Set environment variable
  monkeypatch.setenv("ADK_DEFAULT_APP_NAME", "test_app")

  test_app = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )

  # Create session for test_app
  async def setup_session():
    await mock_session_service.create_session(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        state={},
    )

  asyncio.run(setup_session())

  # 1. Test path rewriting for GET /users/{user_id}/sessions/{session_id}
  response = test_app.get("/users/test_user/sessions/test_session")
  assert response.status_code == 200
  assert response.json()["id"] == "test_session"

  # 2. Test app_name omission in /run request payload
  payload = {
      "user_id": "test_user",
      "session_id": "test_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
  }
  response = test_app.post("/run", json=payload)
  assert response.status_code == 200
  assert isinstance(response.json(), list)


def test_default_app_name_not_set_raises_error(test_app, monkeypatch):
  """Test that omitting app_name when ADK_DEFAULT_APP_NAME is not set raises 400/404."""
  # Make sure environment variable is NOT set
  monkeypatch.delenv("ADK_DEFAULT_APP_NAME", raising=False)

  # 1. Accessing /users/{user_id}/sessions/{session_id} should return 404 because no rewrite happened
  response = test_app.get("/users/test_user/sessions/test_session")
  assert response.status_code == 404

  # 2. Accessing /run with omitted app_name should return 400
  payload = {
      "user_id": "test_user",
      "session_id": "test_session",
      "new_message": {"role": "user", "parts": [{"text": "Hello"}]},
  }
  response = test_app.post("/run", json=payload)
  assert response.status_code == 400
  assert "app_name is required" in response.json()["detail"]


def test_run_live_websocket_default_app_name(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test that /run_live websocket endpoint resolves app_name using ADK_DEFAULT_APP_NAME."""
  monkeypatch.setenv("ADK_DEFAULT_APP_NAME", "test_app")

  test_app = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )

  async def setup_session():
    await mock_session_service.create_session(
        app_name="test_app",
        user_id="user",
        session_id="session",
        state={},
    )

  asyncio.run(setup_session())

  url = "/run_live?user_id=user&session_id=session&modalities=AUDIO"

  with test_app.websocket_connect(url) as ws:
    data = ws.receive_json()
    assert data["author"] == "dummy agent"


def test_run_live_hides_internal_metadata(
    test_app, mock_session_service, monkeypatch
):
  """/run_live sends events without ADK-internal custom_metadata."""

  async def run_live_with_internal_metadata(
      self, session, live_request_queue, **kwargs
  ):
    del session, live_request_queue, kwargs
    yield Event(
        author="dummy agent",
        invocation_id="invocation_id",
        custom_metadata={"keep": 1, INTERNAL_METADATA_PREFIX + "stamp": "x"},
    )

  monkeypatch.setattr(Runner, "run_live", run_live_with_internal_metadata)

  async def setup_session():
    await mock_session_service.create_session(
        app_name="test_app", user_id="user", session_id="session", state={}
    )

  asyncio.run(setup_session())

  url = "/run_live?app_name=test_app&user_id=user&session_id=session&modalities=AUDIO"
  with test_app.websocket_connect(url) as ws:
    data = ws.receive_json()

  assert data["customMetadata"] == {"keep": 1}


def test_run_live_websocket_missing_app_name_raises_error(
    test_app, monkeypatch
):
  """Test that /run_live websocket connection fails when app_name and ADK_DEFAULT_APP_NAME are both missing."""
  from fastapi.websockets import WebSocketDisconnect

  monkeypatch.delenv("ADK_DEFAULT_APP_NAME", raising=False)

  url = "/run_live?user_id=user&session_id=session&modalities=AUDIO"

  with pytest.raises(WebSocketDisconnect) as exc_info:
    with test_app.websocket_connect(url) as ws:
      ws.receive_json()
  assert exc_info.value.code == 1008


def test_is_single_agent_directory(tmp_path):
  """Verify that is_single_agent_directory only identifies directories with agent.py or root_agent.yaml."""
  from google.adk.cli.utils.agent_loader import is_single_agent_directory

  # Directory with agent.py (should be identified as agent)
  agent_py_dir = tmp_path / "agent_py_dir"
  agent_py_dir.mkdir()
  (agent_py_dir / "agent.py").write_text("root_agent = 'dummy'")
  assert is_single_agent_directory(str(agent_py_dir)) is True

  # Directory with root_agent.yaml (should be identified as agent)
  yaml_dir = tmp_path / "yaml_dir"
  yaml_dir.mkdir()
  (yaml_dir / "root_agent.yaml").write_text("root_agent: dummy")
  assert is_single_agent_directory(str(yaml_dir)) is True

  # Normal directory or standard package with __init__.py only (should NOT be identified as agent)
  normal_pkg = tmp_path / "normal_pkg"
  normal_pkg.mkdir()
  (normal_pkg / "__init__.py").write_text(
      "from .app import App\nimport something"
  )
  assert is_single_agent_directory(str(normal_pkg)) is False


def test_agent_loader_single_agent_mode(tmp_path):
  """Verify that AgentLoader automatically detects and configures single agent mode."""
  agent_folder = tmp_path / "my_test_agent"
  agent_folder.mkdir()
  (agent_folder / "agent.py").write_text("root_agent = 'dummy'")

  loader = fast_api_module.AgentLoader(str(agent_folder))

  assert loader._is_single_agent is True
  assert loader._single_agent_name == "my_test_agent"
  assert loader.agents_dir == str(tmp_path)
  assert loader.list_agents() == ["my_test_agent"]


def test_single_agent_mode_detection(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Verify that pointing agents_dir to a single agent folder enables single agent mode."""
  agent_folder = tmp_path / "my_only_agent"
  agent_folder.mkdir()
  (agent_folder / "agent.py").write_text("root_agent = None")

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
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
  ):
    app = get_fast_api_app(
        agents_dir=str(agent_folder),
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=None,
        a2a=False,
        host="127.0.0.1",
        port=8000,
    )
    client = TestClient(app)

    response = client.get("/list-apps")
    assert response.status_code == 200
    assert response.json() == ["my_only_agent"]


def test_single_agent_mode_loads_services_module_from_agent_dir(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Verify a services module in the agent folder registers custom services."""
  agent_folder = tmp_path / "my_only_agent"
  agent_folder.mkdir()
  (agent_folder / "agent.py").write_text("root_agent = None")
  (agent_folder / "services.py").write_text(
      "from google.adk.cli.service_registry import get_service_registry\n"
      "\n"
      "\n"
      "def _custom_session_factory(uri, **kwargs):\n"
      "  return 'custom-session-service'\n"
      "\n"
      "\n"
      "get_service_registry().register_session_service(\n"
      "    'customscheme', _custom_session_factory\n"
      ")\n"
  )

  original_sys_path = list(sys.path)
  sys.modules.pop("services", None)

  try:
    # A fresh registry can only know the scheme if the agent's services.py ran.
    with (
        patch.object(
            service_registry_module, "_service_registry_instance", None
        ),
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
            "LocalEvalSetsManager",
            autospec=True,
            return_value=mock_eval_sets_manager,
        ),
        patch.object(
            fast_api_module,
            "LocalEvalSetResultsManager",
            autospec=True,
            return_value=mock_eval_set_results_manager,
        ),
    ):
      get_fast_api_app(
          agents_dir=str(agent_folder),
          web=True,
          session_service_uri="",
          artifact_service_uri="",
          memory_service_uri="",
          allow_origins=None,
          a2a=False,
          host="127.0.0.1",
          port=8000,
      )

      registry = service_registry_module.get_service_registry()
      assert (
          registry.create_session_service("customscheme://db")
          == "custom-session-service"
      )
  finally:
    sys.modules.pop("services", None)
    sys.path[:] = original_sys_path


def test_single_agent_mode_sets_default_app(
    tmp_path,
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Verify that in single agent mode, the agent is used as default app."""
  # Set environment variable to something else, but single mode should take precedence.
  monkeypatch.setenv("ADK_DEFAULT_APP_NAME", "some_other_app")

  agent_folder = tmp_path / "my_only_agent"
  agent_folder.mkdir()
  (agent_folder / "agent.py").write_text("root_agent = None")

  # Setup session data in the in-memory service
  async def setup_session():
    await mock_session_service.create_session(
        app_name="my_only_agent",
        user_id="test_user",
        session_id="test_session",
        state={},
    )

  asyncio.run(setup_session())

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
          "LocalEvalSetsManager",
          autospec=True,
          return_value=mock_eval_sets_manager,
      ),
      patch.object(
          fast_api_module,
          "LocalEvalSetResultsManager",
          autospec=True,
          return_value=mock_eval_set_results_manager,
      ),
  ):
    app = get_fast_api_app(
        agents_dir=str(agent_folder),
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=None,
        a2a=False,
        host="127.0.0.1",
        port=8000,
    )
    client = TestClient(app)

    # Accessing /users/{user_id}/sessions/{session_id} should work because of rewrite
    response = client.get("/users/test_user/sessions/test_session")
    assert response.status_code == 200
    assert response.json()["id"] == "test_session"


def test_agent_run_disconnect_aborts_run(
    test_app, create_test_session, monkeypatch
):
  """Test that /run endpoint aborts agent execution on client disconnect.

  Verifies that when the client connection is dropped during an active agent
  run:
  1. The background agent execution generator task is cancelled.
  2. The endpoint returns a clean 499 (Client Closed Request) status code.
  """
  import starlette.requests

  info = create_test_session
  trigger_disconnect: dict[str, bool] = {"value": False}
  was_cancelled: dict[str, bool] = {"value": False}

  async def run_async_mock(
      self,
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del (
        self,
        user_id,
        session_id,
        invocation_id,
        new_message,
        state_delta,
        run_config,
    )
    try:
      # Yield first pulse event
      yield _event_1()
      # Simulate connection drop mid-run
      trigger_disconnect["value"] = True
      # Run a long async operation to allow the monitor to trigger cancellation
      await asyncio.sleep(1.0)
      yield _event_2()
    except asyncio.CancelledError:
      was_cancelled["value"] = True
      raise

  monkeypatch.setattr(Runner, "run_async", run_async_mock)

  # Monkeypatch starlette.requests.Request.__init__ to inject simulated disconnect
  original_init = starlette.requests.Request.__init__

  def custom_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    original_receive = self._receive
    call_count = 0

    async def mock_receive():
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        return await original_receive()

      # Subsequent calls block until simulated connection drop is triggered
      while not trigger_disconnect["value"]:
        await asyncio.sleep(0.01)
      return {"type": "http.disconnect"}

    self._receive = mock_receive
    self.__dict__["receive"] = mock_receive

  monkeypatch.setattr(starlette.requests.Request, "__init__", custom_init)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": False,
  }

  # When standard /run POST request is initiated and mid-run connection drop occurs
  response = test_app.post("/run", json=payload)

  # Then the response status should be 499 and the running generator was cancelled
  assert response.status_code == 499
  assert was_cancelled["value"] is True


async def test_agent_run_disconnect_seals_dangling_call_and_runs_after_run(
    create_test_session,
    mock_session_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Tests /run disconnect seals dangling FunctionCall and runs after_run."""
  info = create_test_session
  captured_contexts = []
  tool_in_flight = asyncio.Event()
  after_run_flag = asyncio.Event()

  monkeypatch.setattr(Runner, "run_async", _ORIGINAL_RUNNER_RUN_ASYNC)
  loaded_app = _slow_tool_app(
      info,
      captured_contexts,
      tool_in_flight,
      after_run_flag,
      call_id="call_run_1",
  )
  monkeypatch.setattr(
      mock_agent_loader, "load_agent", lambda app_name: loaded_app
  )

  client = _create_test_client(
      mock_session_service,
      InMemoryArtifactService(),
      InMemoryMemoryService(),
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
  )
  app = client.app
  handler = None
  for route in app.routes:
    if route.path == "/run":
      handler = route.endpoint
      break
  assert handler is not None

  req = RunAgentRequest(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
      new_message={"role": "user", "parts": [{"text": "Run slow tool"}]},
      streaming=False,
  )

  async def receive():
    await tool_in_flight.wait()
    return {"type": "http.disconnect"}

  request = starlette.requests.Request(
      {
          "type": "http",
          "method": "POST",
          "path": "/run",
          "headers": [],
          "asgi": {"spec_version": "2.1"},
      },
      receive=receive,
  )

  response = await handler(req, request)
  assert response.status_code == 499
  assert len(captured_contexts) == 1
  assert captured_contexts[0].is_aborted is True
  assert after_run_flag.is_set()

  session = await mock_session_service.get_session(
      app_name=info["app_name"],
      user_id=info["user_id"],
      session_id=info["session_id"],
  )
  abort_events = [
      e for e in session.events if e.error_code == "INVOCATION_ABORTED"
  ]
  assert len(abort_events) == 1
  frs = abort_events[0].get_function_responses()
  assert len(frs) == 1
  assert frs[0].id == "call_run_1"
  assert frs[0].name == "slow_tool"


#################################################
# Gemini Enterprise Tests
#################################################


def test_gemini_app_not_found_raises(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    monkeypatch,
):
  """Test get_fast_api_app raises ValueError if gemini_enterprise_app_name not found."""
  monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
  mock_agent_loader.list_agents = MagicMock(return_value=["test_app"])
  with pytest.raises(ValueError, match="not found in dir"):
    _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        gemini_enterprise_app_name="nonexistent_app",
    )


def test_gemini_reasoning_engine_success(test_app_with_gemini_enterprise):
  """Test POST /api/reasoning_engine success case."""
  response = test_app_with_gemini_enterprise.post(
      "/api/reasoning_engine",
      json={"class_method": "get_session", "input": {"arg1": 1}},
  )
  assert response.status_code == 200
  assert response.json() == {
      "output": {"result": "success", "kwargs": {"arg1": 1}}
  }


def test_gemini_reasoning_engine_missing_class_method(
    test_app_with_gemini_enterprise,
):
  """Test POST /api/reasoning_engine with missing class_method."""
  response = test_app_with_gemini_enterprise.post(
      "/api/reasoning_engine",
      json={"input": {"arg1": 1}},
  )
  assert response.status_code == 400


def test_gemini_stream_reasoning_engine_success(
    test_app_with_gemini_enterprise,
):
  """Test POST /api/stream_reasoning_engine success case."""
  response = test_app_with_gemini_enterprise.post(
      "/api/stream_reasoning_engine",
      json={"class_method": "stream_query", "input": {"arg1": 1}},
  )
  assert response.status_code == 200
  lines = response.text.strip().split("\n")
  assert len(lines) == 2
  assert json.loads(lines[0]) == {"chunk": 1, "kwargs": {"arg1": 1}}
  assert json.loads(lines[1]) == {"chunk": 2, "kwargs": {"arg1": 1}}


def test_gemini_stream_reasoning_engine_missing_class_method(
    test_app_with_gemini_enterprise,
):
  """Test POST /api/stream_reasoning_engine with missing class_method."""
  response = test_app_with_gemini_enterprise.post(
      "/api/stream_reasoning_engine",
      json={"input": {"arg1": 1}},
  )
  assert response.status_code == 400


def test_gemini_stream_reasoning_engine_sync_generator(
    test_app_with_gemini_enterprise_sync_stream,
):
  """Regression test: a synchronous streaming class_method must not raise.

  A sync generator is adapted to an async iterator via run_in_threadpool. The
  adapter must not rely on catching StopIteration across the await boundary,
  since Python (PEP 479) converts an escaping StopIteration into
  RuntimeError("coroutine raised StopIteration") after the final chunk.
  """
  response = test_app_with_gemini_enterprise_sync_stream.post(
      "/api/stream_reasoning_engine",
      json={"class_method": "stream_query", "input": {"arg1": 1}},
  )
  assert response.status_code == 200
  lines = response.text.strip().split("\n")
  assert len(lines) == 2
  assert json.loads(lines[0]) == {"chunk": 1, "kwargs": {"arg1": 1}}
  assert json.loads(lines[1]) == {"chunk": 2, "kwargs": {"arg1": 1}}


def test_run_eval_request_live_fields_default():
  """RunEvalRequest defaults to non-live mode."""
  from google.adk.cli.dev_server import RunEvalRequest

  req = RunEvalRequest(eval_case_ids=["a"], eval_metrics=[])

  assert req.live_model_config is None
  assert req.user_simulator_config is None


def test_run_eval_request_accepts_live_and_audio_config():
  """RunEvalRequest accepts live flags and an audio user-simulator config."""
  from google.adk.cli.dev_server import RunEvalRequest

  req = RunEvalRequest.model_validate({
      "evalCaseIds": ["a"],
      "evalMetrics": [],
      "liveModelConfig": {"timeoutSeconds": 600},
      "userSimulatorConfig": {"type": "llm_audio", "audioModel": "cloud_tts"},
  })

  assert req.live_model_config.timeout_seconds == 600
  # The request keeps the raw mapping (OpenAPI-safe); it is validated into the
  # typed union inside `run_eval`.
  assert req.user_simulator_config == {
      "type": "llm_audio",
      "audioModel": "cloud_tts",
  }


def test_run_eval_request_config_validates_into_typed_union():
  """A request config mapping is validated into the typed union like `run_eval`.

  The request holds the config as a raw mapping; `run_eval` validates it via
  `TypeAdapter(UserSimulatorConfig)`. This exercises that same path.
  """
  from google.adk.cli.dev_server import RunEvalRequest
  from google.adk.evaluation.eval_config import _UserSimulatorConfig
  from google.adk.evaluation.simulation._llm_audio_user_simulator import LlmAudioUserSimulatorConfig
  from pydantic import TypeAdapter

  req = RunEvalRequest.model_validate({
      "evalCaseIds": ["a"],
      "evalMetrics": [],
      "userSimulatorConfig": {"type": "llm_audio", "audioModel": "cloud_tts"},
  })
  config = TypeAdapter(_UserSimulatorConfig).validate_python(
      req.user_simulator_config
  )

  assert isinstance(config, LlmAudioUserSimulatorConfig)
  assert config.type == "llm_audio"
  assert config.audio_model == "cloud_tts"


def test_run_eval_request_unknown_simulator_type_rejected_on_validation():
  """An unknown `type` passes request parsing but fails `run_eval` validation.

  The raw mapping is accepted by the request model, but the union validation
  `run_eval` performs rejects an unknown discriminator.
  """
  from google.adk.cli.dev_server import RunEvalRequest
  from google.adk.evaluation.eval_config import _UserSimulatorConfig
  from pydantic import TypeAdapter
  from pydantic import ValidationError

  req = RunEvalRequest.model_validate({
      "evalCaseIds": ["a"],
      "evalMetrics": [],
      "userSimulatorConfig": {"type": "not_a_real_simulator"},
  })

  with pytest.raises(ValidationError):
    TypeAdapter(_UserSimulatorConfig).validate_python(req.user_simulator_config)


#################################################
# Agent Identity Finalize Tests
#################################################


def test_finalize_agent_identity_credentials_success(test_app):
  """Test successful credential finalization and Base64 padding decoding."""
  import base64

  from google.cloud import iamconnectorcredentials_v1alpha

  raw_bytes = b"test-validation-state-bytes"
  # Unpadded url-safe base64 string
  b64_str = base64.urlsafe_b64encode(raw_bytes).decode("utf-8").rstrip("=")

  with (
      patch.object(
          iamconnectorcredentials_v1alpha,
          "IAMConnectorCredentialsServiceClient",
          autospec=True,
      ) as mock_client_cls,
      patch.object(
          iamconnectorcredentials_v1alpha,
          "FinalizeCredentialsRequest",
          autospec=True,
      ) as mock_req_cls,
  ):
    mock_client = mock_client_cls.return_value
    mock_client.finalize_credentials.return_value = None

    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": b64_str,
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_req_cls.assert_called_once_with(
        connector="projects/p/locations/l/connectors/c",
        user_id="user-123",
        user_id_validation_state=raw_bytes,
        consent_nonce="nonce-456",
    )
    mock_client.finalize_credentials.assert_called_once()


def test_finalize_agent_identity_credentials_invalid_base64(test_app):
  """Test error handling when user_id_validation_state is invalid Base64."""
  from google.cloud import iamconnectorcredentials_v1alpha

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ):
    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "!!!invalid_base64!!!",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 400
    assert (
        "Invalid base64 user_id_validation_state" in response.json()["detail"]
    )


def test_finalize_agent_identity_credentials_missing_dependency(test_app):
  """Test error handling when google-cloud-iamconnectorcredentials is not installed."""
  with patch.dict(
      "sys.modules", {"google.cloud.iamconnectorcredentials_v1alpha": None}
  ):
    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 500
    assert "Agent Identity support requires" in response.json()["detail"]


def test_finalize_agent_identity_credentials_invalid_argument_error(test_app):
  """Test backend InvalidArgument API error handling (400 response)."""
  from google.cloud import iamconnectorcredentials_v1alpha

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ) as mock_client_cls:
    mock_client = mock_client_cls.return_value
    mock_client.finalize_credentials.side_effect = InvalidArgument(
        "Invalid consent nonce"
    )

    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "invalid-nonce",
        },
    )
    assert response.status_code == 400
    assert "Invalid credentials request" in response.json()["detail"]


def test_finalize_agent_identity_credentials_api_call_error(test_app):
  """Test backend GoogleAPICallError error handling with status code propagation."""
  from google.cloud import iamconnectorcredentials_v1alpha

  err = GoogleAPICallError("Permission denied")
  err.code = 403

  with patch.object(
      iamconnectorcredentials_v1alpha,
      "IAMConnectorCredentialsServiceClient",
      autospec=True,
  ) as mock_client_cls:
    mock_client = mock_client_cls.return_value
    mock_client.finalize_credentials.side_effect = err

    response = test_app.post(
        "/agent-identity/finalize",
        json={
            "connector_name": "projects/p/locations/l/connectors/c",
            "user_id": "user-123",
            "user_id_validation_state": "dGVzdA",
            "consent_nonce": "nonce-456",
        },
    )
    assert response.status_code == 403
    assert "Failed to finalize credentials" in response.json()["detail"]


#################################################
# Span Exporter Tests
#################################################


def _readable_span(name, *, trace_id, span_id=1, attributes=None):
  """Builds a finished span suitable for feeding a SpanExporter."""
  from opentelemetry.sdk.trace import ReadableSpan
  from opentelemetry.trace import SpanContext

  return ReadableSpan(
      name=name,
      context=SpanContext(trace_id=trace_id, span_id=span_id, is_remote=False),
      attributes=attributes or {},
  )


def test_api_server_span_exporter_records_only_llm_and_tool_spans():
  """Only call_llm / send_data / execute_tool* spans are kept, by event id."""
  from google.adk.cli.api_server import ApiServerSpanExporter
  from opentelemetry.sdk.trace.export import SpanExportResult

  trace_dict = {}
  exporter = ApiServerSpanExporter(trace_dict)

  spans = [
      _readable_span(
          "call_llm",
          trace_id=11,
          span_id=1,
          attributes={"gcp.vertex.agent.event_id": "llm-event"},
      ),
      _readable_span(
          "send_data",
          trace_id=12,
          span_id=2,
          attributes={"gcp.vertex.agent.event_id": "data-event"},
      ),
      _readable_span(
          "execute_tool my_tool",
          trace_id=13,
          span_id=3,
          attributes={"gcp.vertex.agent.event_id": "tool-event"},
      ),
      _readable_span(
          "invocation",
          trace_id=14,
          span_id=4,
          attributes={"gcp.vertex.agent.event_id": "unrelated-event"},
      ),
  ]

  assert exporter.export(spans) == SpanExportResult.SUCCESS

  assert sorted(trace_dict) == ["data-event", "llm-event", "tool-event"]
  # The exporter augments the span attributes with its trace/span identifiers,
  # which is what the /debug/trace endpoint hands back to the UI.
  assert trace_dict["llm-event"]["trace_id"] == 11
  assert trace_dict["llm-event"]["span_id"] == 1
  assert trace_dict["tool-event"]["trace_id"] == 13


def test_api_server_span_exporter_skips_span_without_event_id():
  """A traced span carrying no event id cannot be keyed, so it is dropped."""
  from google.adk.cli.api_server import ApiServerSpanExporter

  trace_dict = {}
  exporter = ApiServerSpanExporter(trace_dict)

  exporter.export([
      _readable_span(
          "call_llm",
          trace_id=21,
          attributes={"gcp.vertex.agent.session_id": "session-a"},
      )
  ])

  assert trace_dict == {}


def test_in_memory_exporter_returns_only_spans_of_requested_session():
  """Spans are indexed per session id and looked up by trace id."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)

  span_a1 = _readable_span(
      "call_llm",
      trace_id=101,
      span_id=1,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  span_a2 = _readable_span(
      "execute_tool my_tool",
      trace_id=101,
      span_id=2,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  span_b = _readable_span(
      "call_llm",
      trace_id=202,
      span_id=3,
      attributes={"gcp.vertex.agent.session_id": "session-b"},
  )

  exporter.export([span_a1, span_a2, span_b])

  # Both session-a spans share a trace, so the trace id is recorded once.
  assert session_trace_dict == {"session-a": [101], "session-b": [202]}
  assert exporter.get_finished_spans("session-a") == [span_a1, span_a2]
  assert exporter.get_finished_spans("session-b") == [span_b]
  assert exporter.get_finished_spans("session-never-seen") == []


def test_in_memory_exporter_falls_back_to_conversation_id():
  """A span with no agent session id is indexed by the conversation id."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)

  conversation_span = _readable_span(
      "call_llm",
      trace_id=303,
      span_id=1,
      attributes={"gen_ai.conversation.id": "conversation-1"},
  )
  unattributed_span = _readable_span("call_llm", trace_id=404, span_id=2)

  exporter.export([conversation_span, unattributed_span])

  assert session_trace_dict == {"conversation-1": [303]}
  assert exporter.get_finished_spans("conversation-1") == [conversation_span]


def test_in_memory_exporter_clear_drops_spans_but_keeps_session_index():
  """clear() forgets the spans; the session -> trace id index is untouched."""
  from google.adk.cli.api_server import InMemoryExporter

  session_trace_dict = {}
  exporter = InMemoryExporter(session_trace_dict)
  span = _readable_span(
      "call_llm",
      trace_id=505,
      attributes={"gcp.vertex.agent.session_id": "session-a"},
  )
  exporter.export([span])
  assert exporter.get_finished_spans("session-a") == [span]

  exporter.clear()

  assert exporter.get_finished_spans("session-a") == []
  assert session_trace_dict == {"session-a": [505]}


def test_setup_telemetry_guards_add_span_processor_on_non_sdk_provider(
    monkeypatch,
):
  """A non-SDK TracerProvider lacking add_span_processor does not raise."""
  from unittest.mock import MagicMock

  from google.adk.cli.api_server import _setup_telemetry
  from opentelemetry import trace

  non_sdk_provider = MagicMock(spec=trace.TracerProvider)
  del non_sdk_provider.add_span_processor
  monkeypatch.setattr(trace, "get_tracer_provider", lambda: non_sdk_provider)

  exporter = MagicMock()
  _setup_telemetry(otel_to_cloud=False, internal_exporters=[exporter])


def test_setup_telemetry_adds_span_processors_when_supported(monkeypatch):
  """A TracerProvider with add_span_processor registers the internal exporters."""
  from unittest.mock import MagicMock

  from google.adk.cli.api_server import _setup_telemetry
  from opentelemetry import trace

  provider = MagicMock()
  monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

  exporter = MagicMock()
  _setup_telemetry(otel_to_cloud=False, internal_exporters=[exporter])
  provider.add_span_processor.assert_called_once_with(exporter)


#################################################
# Request-body plumbing tests
#################################################


def test_create_session_applies_body_session_id_state_and_events(
    test_app, test_session_info
):
  """CreateSessionRequest drives the id, the state and the seeded events."""
  base_url = (
      f"/apps/{test_session_info['app_name']}"
      f"/users/{test_session_info['user_id']}/sessions"
  )
  response = test_app.post(
      base_url,
      json={
          "session_id": "seeded_session",
          "state": {"greeting": "hello"},
          "events": [
              {
                  "author": "user",
                  "invocationId": "inv-1",
                  "content": {"role": "user", "parts": [{"text": "hi there"}]},
              },
          ],
      },
  )

  assert response.status_code == 200
  created = response.json()
  assert created["id"] == "seeded_session"
  assert created["state"] == {"greeting": "hello"}

  fetched = test_app.get(f"{base_url}/seeded_session")
  assert fetched.status_code == 200
  events = fetched.json()["events"]
  assert [event["content"]["parts"][0]["text"] for event in events] == [
      "hi there"
  ]


def test_patch_memory_unknown_session_returns_404(
    test_app, test_session_info, mock_memory_service
):
  """A request naming a missing session must not reach the memory service."""
  url = (
      f"/apps/{test_session_info['app_name']}"
      f"/users/{test_session_info['user_id']}/memory"
  )

  response = test_app.patch(url, json={"session_id": "no_such_session"})

  assert response.status_code == 404
  assert response.json()["detail"] == "Session not found"
  mock_memory_service.add_session_to_memory.assert_not_called()


#################################################
# ApiServer vs DevServer endpoint surface
#################################################


def test_dev_only_endpoints_absent_when_web_disabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """web=False serves ApiServer only: no eval / debug / graph routes."""
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
  )

  dev_only_paths = [
      "/config/telemetry",
      "/dev/apps/test_app/eval-sets",
      "/dev/apps/test_app/eval-results",
      "/dev/apps/test_app/metrics-info",
      "/dev/apps/test_app/tests",
      "/dev/apps/test_app/graph",
      "/dev/apps/test_app/debug/trace/some-event",
  ]
  for path in dev_only_paths:
    assert client.get(path).status_code == 404, path

  # The production endpoints are still there.
  assert client.get("/health").status_code == 200
  assert client.get("/list-apps").status_code == 200


def _installed_internal_exporters(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    *,
    web: bool,
) -> list[str]:
  """Names of the span exporters the server registers on the tracer provider."""
  with patch.object(
      api_server_module, "_setup_telemetry", autospec=True
  ) as mock_setup_telemetry:
    _create_test_client(
        mock_session_service,
        mock_artifact_service,
        mock_memory_service,
        mock_agent_loader,
        mock_eval_sets_manager,
        mock_eval_set_results_manager,
        web=web,
    )
  processors = mock_setup_telemetry.call_args.kwargs["internal_exporters"]
  return [type(processor.span_exporter).__name__ for processor in processors]


def test_span_buffers_not_filled_when_web_disabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Nothing reads the in-memory spans on web=False, so nothing writes them."""
  assert (
      _installed_internal_exporters(
          mock_session_service,
          mock_artifact_service,
          mock_memory_service,
          mock_agent_loader,
          mock_eval_sets_manager,
          mock_eval_set_results_manager,
          web=False,
      )
      == []
  )


def test_span_buffers_filled_when_web_enabled(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """The dev server's trace views need the spans, so both exporters run."""
  assert _installed_internal_exporters(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=True,
  ) == ["ApiServerSpanExporter", "InMemoryExporter"]


def test_app_info_rejects_special_agent_only_in_api_server_mode(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Internal `__` apps reach the dev server, but not the api server."""
  api_only_client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      web=False,
  )

  blocked = api_only_client.get("/apps/__internal_assistant/app-info")
  assert blocked.status_code == 403
  assert "internal special agents" in blocked.json()["detail"]

  # Same request on a loopback-bound dev server gets past the guard and is
  # answered on the merits of the loaded agent (which here is not an LlmAgent).
  dev_client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      bind_host="127.0.0.1",
  )
  allowed = dev_client.get(
      "/apps/__internal_assistant/app-info",
      headers={"host": "127.0.0.1:8000"},
  )
  assert allowed.status_code == 400
  assert allowed.json()["detail"] == "Root agent is not an LlmAgent"


def test_dev_endpoint_rejects_app_name_that_is_not_an_identifier(
    builder_test_client,
):
  """_get_agent_dir only accepts dot-separated Python identifiers."""
  ok = builder_test_client.get("/dev/apps/test_app/tests")
  assert ok.status_code == 200
  assert ok.json() == []

  nested_ok = builder_test_client.get("/dev/apps/pkg.test_app/tests")
  assert nested_ok.status_code == 200

  for bad_name in ("bad-name", "1app", "app%20name"):
    rejected = builder_test_client.get(f"/dev/apps/{bad_name}/tests")
    assert rejected.status_code == 400, bad_name
    assert "must be valid" in rejected.json()["detail"]


#################################################
# Eval endpoint plumbing
#################################################


def test_add_session_to_eval_set_builds_eval_case_from_session(
    test_app, test_session_info, mock_eval_sets_manager
):
  """AddSessionToEvalSetRequest turns a live session into an eval case."""
  app_name = test_session_info["app_name"]
  user_id = test_session_info["user_id"]
  mock_eval_sets_manager.create_eval_set(
      app_name=app_name, eval_set_id="my_eval_set"
  )

  sessions_url = f"/apps/{app_name}/users/{user_id}/sessions"
  created = test_app.post(
      sessions_url,
      json={
          "session_id": "eval_source_session",
          "events": [
              {
                  "author": "user",
                  "invocationId": "inv-1",
                  "content": {
                      "role": "user",
                      "parts": [{"text": "what is 2+2?"}],
                  },
              },
              {
                  "author": "dummy agent",
                  "invocationId": "inv-1",
                  "content": {"role": "model", "parts": [{"text": "4"}]},
              },
          ],
      },
  )
  assert created.status_code == 200

  response = test_app.post(
      f"/dev/apps/{app_name}/eval-sets/my_eval_set/add-session",
      json={
          "eval_id": "my_eval_case",
          "session_id": "eval_source_session",
          "user_id": user_id,
      },
  )
  assert response.status_code == 200

  eval_case = mock_eval_sets_manager.get_eval_case(
      app_name, "my_eval_set", "my_eval_case"
  )
  assert eval_case is not None
  assert eval_case.session_input.app_name == app_name
  assert eval_case.session_input.user_id == user_id
  assert [
      part.text
      for invocation in eval_case.conversation
      for part in invocation.user_content.parts
  ] == ["what is 2+2?"]


@pytest.mark.xfail(
    strict=True,
    reason="add-session maps ValueError, but the managers raise NotFoundError",
)
def test_add_session_to_eval_set_unknown_eval_set_is_a_client_error(
    test_app, create_test_session
):
  """Adding to an eval set that never existed is a client error, not a 500."""
  info = create_test_session

  response = test_app.post(
      f"/dev/apps/{info['app_name']}/eval-sets/missing_eval_set/add-session",
      json={
          "eval_id": "case-1",
          "session_id": info["session_id"],
          "user_id": info["user_id"],
      },
  )

  assert 400 <= response.status_code < 500


def test_get_eval_result_returns_saved_eval_set_result(
    test_app, mock_eval_set_results_manager
):
  """The eval-results endpoint renames EvalSetResult to EvalResult as-is."""
  mock_eval_set_results_manager.save_eval_set_result(
      "test_app", "my_eval_set", []
  )

  response = test_app.get(
      "/dev/apps/test_app/eval-results/test_app_my_eval_set_eval_result"
  )

  assert response.status_code == 200
  data = response.json()
  assert data["evalSetResultId"] == "test_app_my_eval_set_eval_result"
  assert data["evalSetId"] == "my_eval_set"


def test_create_eval_set_legacy_route_creates_eval_set(
    test_app, mock_eval_sets_manager
):
  """The deprecated create-eval-set route should create an empty eval set."""
  response = test_app.post("/dev/apps/test_app/eval_sets/legacy_eval_set")

  assert response.status_code == 200
  assert (
      mock_eval_sets_manager.get_eval_set("test_app", "legacy_eval_set")
      is not None
  )


def test_agent_run_sse_deferred_with_streaming_returns_422(
    test_app, create_test_session
):
  """Deferred plus SSE is rejected up front, not mid-stream.

  A deferred create returns an interaction id rather than a result, so it
  cannot stream. The run config is built before the response starts so the
  caller gets a status code instead of a 200 that breaks partway through.

  Args:
    test_app: The FastAPI test client.
    create_test_session: Fixture creating the session the request targets.
  """
  payload = {
      "app_name": create_test_session["app_name"],
      "user_id": create_test_session["user_id"],
      "session_id": create_test_session["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": True,
      "service_tier": "deferred",
  }

  response = test_app.post("/run_sse", json=payload)

  assert response.status_code == 422
  assert "cannot be used with StreamingMode.SSE" in response.json()["detail"]


def test_agent_run_sse_deferred_without_streaming_is_allowed(
    test_app, create_test_session, monkeypatch
):
  """Deferred is fine on /run_sse as long as the run is not streaming."""
  info = create_test_session

  async def run_async_stub(
      self,  # pylint: disable=unused-argument
      *,
      user_id: str,
      session_id: str,
      invocation_id: Optional[str] = None,
      new_message: Optional[types.Content] = None,
      state_delta: Optional[dict[str, Any]] = None,
      run_config: Optional[RunConfig] = None,
  ):
    del user_id, session_id, invocation_id, new_message, state_delta
    assert run_config.service_tier == "deferred"
    yield Event(
        author="dummy agent",
        invocation_id="invocation_id",
        content=types.Content(role="model", parts=[types.Part(text="hi")]),
    )

  monkeypatch.setattr(Runner, "run_async", run_async_stub)

  payload = {
      "app_name": info["app_name"],
      "user_id": info["user_id"],
      "session_id": info["session_id"],
      "new_message": {"role": "user", "parts": [{"text": "Hello agent"}]},
      "streaming": False,
      "service_tier": "deferred",
  }

  response = test_app.post("/run_sse", json=payload)

  assert response.status_code == 200


def test_runtime_config_endpoint_shadows_static_file(tmp_path):
  """The in-memory config must win over the file still shipped in the package.

  ApiServer registers this route before mounting StaticFiles at "/dev-ui/".
  Starlette matches in registration order, so moving the route after the mount
  would silently serve the stale on-disk file instead -- with a 200 and no
  error. Assert on the payload, not just the status code.
  """
  app = get_fast_api_app(
      agents_dir=str(tmp_path), web=True, url_prefix="/custom"
  )

  # The prefix is stripped by whatever mounts the app (reverse proxy, gateway,
  # or an outer Starlette Mount); ADK registers its routes unprefixed.
  outer = Starlette(routes=[Mount("/custom", app)])
  response = TestClient(outer).get(
      "/custom/dev-ui/assets/config/runtime-config.json"
  )

  assert response.status_code == 200
  body = response.json()
  # A stale file on disk would report "" here.
  assert body["backendUrl"] == "/custom"
  assert "telemetry" in body
  assert response.headers["cache-control"] == "no-store"


def test_runtime_config_endpoint_does_not_write_to_disk(tmp_path):
  """Serving the config must not touch the installed package directory."""
  import google.adk.cli as cli_package

  config_path = (
      Path(cli_package.__file__).parent
      / "browser"
      / "assets"
      / "config"
      / "runtime-config.json"
  )
  before = config_path.read_bytes() if config_path.exists() else None

  app = get_fast_api_app(
      agents_dir=str(tmp_path), web=True, url_prefix="/prefix"
  )
  TestClient(app).get("/dev-ui/assets/config/runtime-config.json")

  after = config_path.read_bytes() if config_path.exists() else None
  assert after == before


def test_runtime_config_rejects_half_specified_logo(tmp_path):
  """--logo-text without --logo-image-url is a config error, not a silent drop."""
  with pytest.raises(ValueError, match="Both --logo-text and --logo-image-url"):
    get_fast_api_app(agents_dir=str(tmp_path), web=True, logo_text="ACME")


@pytest.mark.parametrize(
    ("url_prefix", "expected_root_path"),
    [
        (None, ""),
        ("", ""),
        ("adk", "/adk"),
        ("adk/", "/adk"),
        ("/adk", "/adk"),
        ("/adk/", "/adk"),
        ("host:8000/adk", "/host:8000/adk"),
        ("https://host", ""),
        ("https://host/", ""),
        ("https://host/adk", "/adk"),
        ("https://host/adk/", "/adk"),
    ],
)
def test_url_prefix_propagated_to_fastapi_root_path(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    url_prefix: str | None,
    expected_root_path: str,
):
  """FastAPI root_path is extracted from url_prefix."""
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      url_prefix=url_prefix,
  )
  assert client.app.root_path == expected_root_path


def test_url_prefix_propagated_to_docs_openapi_url(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
):
  """Swagger UI /docs references the prefix-qualified openapi.json."""
  client = _create_test_client(
      mock_session_service,
      mock_artifact_service,
      mock_memory_service,
      mock_agent_loader,
      mock_eval_sets_manager,
      mock_eval_set_results_manager,
      url_prefix="/adk",
  )
  response = client.get("/docs")
  assert response.status_code == 200
  assert "/adk/openapi.json" in response.text


if __name__ == "__main__":
  pytest.main(["-xvs", __file__])
