# Copyright 2025 Google LLC
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
import sys
import tempfile
import time
from typing import Any
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.evaluation.in_memory_eval_sets_manager import InMemoryEvalSetsManager
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.sessions.state import State
from google.genai import types
from pydantic import BaseModel
import pytest

# Configure logging to help diagnose server startup issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("google_adk." + __name__)


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
async def dummy_run_live(self, session, live_request_queue):
  yield _event_1()
  await asyncio.sleep(0)

  yield _event_2()
  await asyncio.sleep(0)

  yield _event_3()


async def dummy_run_async(
    self,
    user_id,
    session_id,
    new_message,
    state_delta=None,
    run_config: Optional[RunConfig] = None,
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
      return root_agent

    def list_agents(self):
      return ["test_app"]

  return MockAgentLoader(".")


@pytest.fixture
def mock_session_service():
  """Create an in-memory session service instance for testing."""
  return InMemorySessionService()


@pytest.fixture
def mock_artifact_service():
  """Create a mock artifact service."""

  # Storage for artifacts
  artifacts = {}

  class MockArtifactService:

    async def load_artifact(
        self, app_name, user_id, session_id, filename, version=None
    ):
      """Load an artifact by filename."""
      key = f"{app_name}:{user_id}:{session_id}:{filename}"
      if key not in artifacts:
        return None

      if version is not None:
        # Get a specific version
        for v in artifacts[key]:
          if v["version"] == version:
            return v["artifact"]
        return None

      # Get the latest version
      return sorted(artifacts[key], key=lambda x: x["version"])[-1]["artifact"]

    async def list_artifact_keys(self, app_name, user_id, session_id):
      """List artifact names for a session."""
      prefix = f"{app_name}:{user_id}:{session_id}:"
      return [
          k.split(":")[-1] for k in artifacts.keys() if k.startswith(prefix)
      ]

    async def list_versions(self, app_name, user_id, session_id, filename):
      """List versions of an artifact."""
      key = f"{app_name}:{user_id}:{session_id}:{filename}"
      if key not in artifacts:
        return []
      return [a["version"] for a in artifacts[key]]

    async def delete_artifact(self, app_name, user_id, session_id, filename):
      """Delete an artifact."""
      key = f"{app_name}:{user_id}:{session_id}:{filename}"
      if key in artifacts:
        del artifacts[key]

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

  # Patch multiple services and signal handlers
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.InMemorySessionService",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryArtifactService",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryMemoryService",
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
  ):
    # Get the FastAPI app, but don't actually run it
    app = get_fast_api_app(
        agents_dir=".",
        web=True,
        session_service_uri="",
        artifact_service_uri="",
        memory_service_uri="",
        allow_origins=["*"],
        a2a=False,  # Disable A2A for most tests
        host="127.0.0.1",
        port=8000,
    )

    # Create a TestClient that doesn't start a real server
    client = TestClient(app)

    return client


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
@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)
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
        "author": "test",
        "capabilities": ["text"],
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
@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)
def test_app_with_a2a(
    mock_session_service,
    mock_artifact_service,
    mock_memory_service,
    mock_agent_loader,
    mock_eval_sets_manager,
    mock_eval_set_results_manager,
    temp_agents_dir_with_a2a,
):
  """Create a TestClient for the FastAPI app with A2A enabled."""

  # Mock A2A related classes
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.InMemorySessionService",
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryArtifactService",
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryMemoryService",
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
      patch("a2a.server.tasks.InMemoryTaskStore") as mock_task_store,
      patch(
          "google.adk.a2a.executor.a2a_agent_executor.A2aAgentExecutor"
      ) as mock_executor,
      patch(
          "a2a.server.request_handlers.DefaultRequestHandler"
      ) as mock_handler,
      patch("a2a.server.apps.A2AStarletteApplication") as mock_a2a_app,
  ):
    # Configure mocks
    mock_task_store.return_value = MagicMock()
    mock_executor.return_value = MagicMock()
    mock_handler.return_value = MagicMock()

    # Mock A2AStarletteApplication
    mock_app_instance = MagicMock()
    mock_app_instance.routes.return_value = (
        []
    )  # Return empty routes for testing
    mock_a2a_app.return_value = mock_app_instance

    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(temp_agents_dir_with_a2a)

    try:
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
    finally:
      os.chdir(original_cwd)


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


def test_create_eval_set(test_app, test_session_info):
  """Test creating an eval set."""
  url = f"/apps/{test_session_info['app_name']}/eval_sets/test_eval_set_id"
  response = test_app.post(url)

  # Verify the response
  assert response.status_code == 200


def test_list_eval_sets(test_app, create_test_eval_set):
  """Test get eval set."""
  info = create_test_eval_set
  url = f"/apps/{info['app_name']}/eval_sets"
  response = test_app.get(url)

  # Verify the response
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, list)
  assert len(data) == 1
  assert data[0] == "test_eval_set_id"


def test_get_eval_set_result_not_found(test_app):
  """Test getting an eval set result that doesn't exist."""
  url = "/apps/test_app_name/eval_results/test_eval_result_id_not_found"
  response = test_app.get(url)
  assert response.status_code == 404


def test_run_eval(test_app, create_test_eval_set):
  """Test running an eval."""

  # Helper function to verify eval case result.
  def verify_eval_case_result(actual_eval_case_result):
    expected_eval_case_result = {
        "evalSetId": "test_eval_set_id",
        "evalId": "test_eval_case_id",
        "finalEvalStatus": 1,
        "overallEvalMetricResults": [{
            "metricName": "tool_trajectory_avg_score",
            "threshold": 0.5,
            "score": 1.0,
            "evalStatus": 1,
            "details": {},
        }],
    }
    for k, v in expected_eval_case_result.items():
      assert actual_eval_case_result[k] == v

  info = create_test_eval_set
  url = f"/apps/{info['app_name']}/eval_sets/test_eval_set_id/run_eval"
  payload = {
      "eval_ids": ["test_eval_case_id"],
      "eval_metrics": [
          {"metric_name": "tool_trajectory_avg_score", "threshold": 0.5}
      ],
  }
  response = test_app.post(url, json=payload)

  # Verify the response
  assert response.status_code == 200

  data = response.json()
  assert len(data) == 1
  verify_eval_case_result(data[0])

  # Verify the eval set result is saved via get_eval_result endpoint.
  url = f"/apps/{info['app_name']}/eval_results/{info['app_name']}_test_eval_set_id_eval_result"
  response = test_app.get(url)
  assert response.status_code == 200
  data = response.json()
  assert isinstance(data, dict)
  assert data["evalSetId"] == "test_eval_set_id"
  assert (
      data["evalSetResultId"]
      == f"{info['app_name']}_test_eval_set_id_eval_result"
  )
  assert len(data["evalCaseResults"]) == 1
  verify_eval_case_result(data["evalCaseResults"][0])

  # Verify the eval set result is saved via list_eval_results endpoint.
  url = f"/apps/{info['app_name']}/eval_results"
  response = test_app.get(url)
  assert response.status_code == 200
  data = response.json()
  assert data == [f"{info['app_name']}_test_eval_set_id_eval_result"]


def test_list_metrics_info(test_app):
  """Test listing metrics info."""
  url = "/apps/test_app/metrics-info"
  response = test_app.get(url)

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


def test_debug_trace(test_app):
  """Test the debug trace endpoint."""
  # This test will likely return 404 since we haven't set up trace data,
  # but it tests that the endpoint exists and handles missing traces correctly.
  url = "/debug/trace/nonexistent-event"
  response = test_app.get(url)

  # Verify we get a 404 for a nonexistent trace
  assert response.status_code == 404
  logger.info("Debug trace test completed successfully")


def test_get_event_graph_returns_dot_src_for_app_agent():
  """Ensure graph endpoint unwraps App instances before building the graph."""
  from google.adk.cli.adk_web_server import AdkWebServer

  root_agent = DummyAgent(name="dummy_agent")
  app_agent = App(name="test_app", root_agent=root_agent)

  class Loader:

    def load_agent(self, app_name):
      return app_agent

    def list_agents(self):
      return [app_agent.name]

  session_service = AsyncMock()
  session = Session(
      id="session_id",
      app_name="test_app",
      user_id="user",
      state={},
      events=[Event(author="dummy_agent")],
  )
  event_id = session.events[0].id
  session_service.get_session.return_value = session

  adk_web_server = AdkWebServer(
      agent_loader=Loader(),
      session_service=session_service,
      memory_service=MagicMock(),
      artifact_service=MagicMock(),
      credential_service=MagicMock(),
      eval_sets_manager=MagicMock(),
      eval_set_results_manager=MagicMock(),
      agents_dir=".",
  )

  fast_api_app = adk_web_server.get_fast_api_app(
      setup_observer=lambda _observer, _server: None,
      tear_down_observer=lambda _observer, _server: None,
  )

  client = TestClient(fast_api_app)
  response = client.get(
      f"/apps/test_app/users/user/sessions/session_id/events/{event_id}/graph"
  )
  assert response.status_code == 200
  assert "dotSrc" in response.json()


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)
def test_a2a_agent_discovery(test_app_with_a2a):
  """Test that A2A agents are properly discovered and configured."""
  # This test mainly verifies that the A2A setup doesn't break the app
  response = test_app_with_a2a.get("/list-apps")
  assert response.status_code == 200
  logger.info("A2A agent discovery test passed")


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)
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


if __name__ == "__main__":
  pytest.main(["-xvs", __file__])
