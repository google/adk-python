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
import logging
import time
from typing import Any
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.evaluation.eval_case import EvalCase
from google.adk.evaluation.eval_case import Invocation
from google.adk.evaluation.eval_result import EvalSetResult
from google.adk.evaluation.eval_set import EvalSet
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.genai import types
from pydantic import BaseModel
import pytest

# Configure logging to help diagnose server startup issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    run_config: RunConfig = RunConfig(),
):
  yield _event_1()
  await asyncio.sleep(0)

  yield _event_2()
  await asyncio.sleep(0)

  yield _event_3()


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


# Mock for the run_evals function, tailored for test_run_eval
async def mock_run_evals_for_fast_api(*args, **kwargs):
  # This is what the test_run_eval expects for its assertions
  yield _MockEvalCaseResult(
      eval_set_id="test_eval_set_id",  # Matches expected in verify_eval_case_result
      eval_id="test_eval_case_id",  # Matches expected
      final_eval_status=1,  # Matches expected (assuming 1 is PASSED)
      user_id="test_user",  # Placeholder, adapt if needed
      session_id="test_session_for_eval_case",  # Placeholder
      overall_eval_metric_results=[{  # Matches expected
          "metricName": "tool_trajectory_avg_score",
          "threshold": 0.5,
          "score": 1.0,
          "evalStatus": 1,
      }],
      # Provide other fields if RunEvalResult or subsequent processing needs them
      eval_metric_results=[],
      eval_metric_result_per_invocation=[],
  )


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

  return MockAgentLoader(".")


@pytest.fixture
def mock_session_service():
  """Create a mock session service that uses an in-memory dictionary."""

  # In-memory database to store sessions during testing
  session_data = {
      "test_app": {
          "test_user": {
              "test_session": {
                  "id": "test_session",
                  "app_name": "test_app",
                  "user_id": "test_user",
                  "events": [],
                  "state": {},
                  "created_at": time.time(),
              }
          }
      }
  }

  # Mock session service class that operates on the in-memory database
  class MockSessionService:

    async def get_session(self, app_name, user_id, session_id):
      """Retrieve a session by ID."""
      if (
          app_name in session_data
          and user_id in session_data[app_name]
          and session_id in session_data[app_name][user_id]
      ):
        return session_data[app_name][user_id][session_id]
      return None

    async def create_session(
        self, app_name, user_id, state=None, session_id=None
    ):
      """Create a new session."""
      if session_id is None:
        session_id = f"session_{int(time.time())}"

      # Initialize app_name and user_id if they don't exist
      if app_name not in session_data:
        session_data[app_name] = {}
      if user_id not in session_data[app_name]:
        session_data[app_name][user_id] = {}

      # Create the session
      session = {
          "id": session_id,
          "app_name": app_name,
          "user_id": user_id,
          "events": [],
          "state": state or {},
      }

      session_data[app_name][user_id][session_id] = session
      return session

    async def list_sessions(self, app_name, user_id):
      """List all sessions for a user."""
      if app_name not in session_data or user_id not in session_data[app_name]:
        return {"sessions": []}

      return ListSessionsResponse(
          sessions=list(session_data[app_name][user_id].values())
      )

    async def delete_session(self, app_name, user_id, session_id):
      """Delete a session."""
      if (
          app_name in session_data
          and user_id in session_data[app_name]
          and session_id in session_data[app_name][user_id]
      ):
        del session_data[app_name][user_id][session_id]

  # Return an instance of our mock service
  return MockSessionService()


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
  return MagicMock()


@pytest.fixture
def mock_eval_sets_manager():
  """Create a mock eval sets manager."""

  # Storage for eval sets.
  eval_sets = {}

  class MockEvalSetsManager:
    """Mock eval sets manager."""

    def create_eval_set(self, app_name, eval_set_id):
      """Create an eval set."""
      if app_name not in eval_sets:
        eval_sets[app_name] = {}

      if eval_set_id in eval_sets[app_name]:
        raise ValueError(f"Eval set {eval_set_id} already exists.")

      eval_sets[app_name][eval_set_id] = EvalSet(
          eval_set_id=eval_set_id, eval_cases=[]
      )
      return eval_set_id

    def get_eval_set(self, app_name, eval_set_id):
      """Get an eval set."""
      if app_name not in eval_sets:
        raise ValueError(f"App {app_name} not found.")
      if eval_set_id not in eval_sets[app_name]:
        raise ValueError(f"Eval set {eval_set_id} not found in app {app_name}.")
      return eval_sets[app_name][eval_set_id]

    def list_eval_sets(self, app_name):
      """List eval sets."""
      if app_name not in eval_sets:
        raise ValueError(f"App {app_name} not found.")
      return list(eval_sets[app_name].keys())

    def add_eval_case(self, app_name, eval_set_id, eval_case):
      """Add an eval case to an eval set."""
      if app_name not in eval_sets:
        raise ValueError(f"App {app_name} not found.")
      if eval_set_id not in eval_sets[app_name]:
        raise ValueError(f"Eval set {eval_set_id} not found in app {app_name}.")
      eval_sets[app_name][eval_set_id].eval_cases.append(eval_case)

  return MockEvalSetsManager()


@pytest.fixture
def mock_eval_set_results_manager():
  """Create a mock eval set results manager."""

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
      patch(
          "google.adk.cli.cli_eval.run_evals",  # Patch where it's imported in fast_api.py
          new=mock_run_evals_for_fast_api,
      ),
  ):
    # Get the FastAPI app, but don't actually run it
    app = get_fast_api_app(
        agents_dir=".", web=True, session_db_url="", allow_origins=["*"]
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

  logger.info(f"Created test session: {session['id']}")
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
  assert data[2]["interrupted"] == True

  logger.info("Agent run test completed successfully")


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


def test_debug_trace(test_app):
  """Test the debug trace endpoint."""
  # This test will likely return 404 since we haven't set up trace data,
  # but it tests that the endpoint exists and handles missing traces correctly.
  url = "/debug/trace/nonexistent-event"
  response = test_app.get(url)

  # Verify we get a 404 for a nonexistent trace
  assert response.status_code == 404
  logger.info("Debug trace test completed successfully")


if __name__ == "__main__":
  pytest.main(["-xvs", __file__])
