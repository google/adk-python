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
import os
import sys
import time
import types as ptypes
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi.testclient import TestClient
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.run_config import RunConfig
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.utils import envs
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.genai import types
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


# Set up dummy module and add to sys.modules
dummy_module = ptypes.ModuleType("test_agent")
dummy_module.agent = ptypes.SimpleNamespace(
    root_agent=DummyAgent(name="dummy_agent")
)
sys.modules["test_app"] = dummy_module

# Try to load environment variables, with a fallback for testing
try:
  envs.load_dotenv_for_agent("test_app", ".")
except Exception as e:
  logger.warning(f"Could not load environment variables: {e}")
  # Create a basic .env file if needed
  if not os.path.exists(".env"):
    with open(".env", "w") as f:
      f.write("# Test environment variables\n")


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
def test_app(mock_session_service, mock_artifact_service, mock_memory_service):
  """Create a TestClient for the FastAPI app without starting a server."""

  # Patch multiple services and signal handlers
  with (
      patch("signal.signal", return_value=None),
      patch(
          "google.adk.cli.fast_api.InMemorySessionService",  # Changed this line
          return_value=mock_session_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryArtifactService",  # Make consistent
          return_value=mock_artifact_service,
      ),
      patch(
          "google.adk.cli.fast_api.InMemoryMemoryService",  # Make consistent
          return_value=mock_memory_service,
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
