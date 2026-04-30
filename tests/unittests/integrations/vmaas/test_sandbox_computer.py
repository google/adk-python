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

"""Unit tests for the AgentEngineSandboxComputer class."""

import time
import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations.vmaas.sandbox_computer import _STATE_KEY_ACCESS_TOKEN
from google.adk.integrations.vmaas.sandbox_computer import _STATE_KEY_AGENT_ENGINE_NAME
from google.adk.integrations.vmaas.sandbox_computer import _STATE_KEY_SANDBOX_NAME
from google.adk.integrations.vmaas.sandbox_computer import _STATE_KEY_TOKEN_EXPIRY
from google.adk.integrations.vmaas.sandbox_computer import AgentEngineSandboxComputer
from google.adk.tools.computer_use.base_computer import ComputerEnvironment
from google.adk.tools.computer_use.base_computer import ComputerState


class TestAgentEngineSandboxComputer(unittest.IsolatedAsyncioTestCase):
  """Tests for AgentEngineSandboxComputer."""

  def setUp(self):
    """Set up test fixtures."""
    self.project_id = "test-project"
    self.location = "us-central1"
    self.service_account = "sa@test-project.iam.gserviceaccount.com"

  def test_init(self):
    """Test computer initialization."""
    computer = AgentEngineSandboxComputer(
        project_id=self.project_id,
        location=self.location,
        service_account_email=self.service_account,
    )

    self.assertEqual(computer._project_id, self.project_id)
    self.assertEqual(computer._location, self.location)
    self.assertEqual(computer._service_account_email, self.service_account)
    self.assertEqual(computer._screen_size, (1280, 720))

  def test_init_with_byos(self):
    """Test initialization with bring-your-own-sandbox."""
    agent_engine_name = (
        "projects/test/locations/us-central1/reasoningEngines/123"
    )
    sandbox_name = f"{agent_engine_name}/sandboxEnvironments/456"

    computer = AgentEngineSandboxComputer(
        project_id=self.project_id,
        sandbox_name=sandbox_name,
    )

    # Agent engine name should be extracted from sandbox_name
    self.assertEqual(computer._agent_engine_name, agent_engine_name)
    self.assertEqual(computer._sandbox_name, sandbox_name)

  def test_init_with_vertexai_client(self):
    """Test initialization with provided vertexai client."""
    mock_client = MagicMock()
    computer = AgentEngineSandboxComputer(vertexai_client=mock_client)
    self.assertEqual(computer._client, mock_client)

  async def test_screen_size(self):
    """Test screen_size returns hardcoded size."""
    computer = AgentEngineSandboxComputer()
    result = await computer.screen_size()
    self.assertEqual(result, (1280, 720))

  async def test_environment(self):
    """Test environment returns ENVIRONMENT_BROWSER."""
    computer = AgentEngineSandboxComputer()
    result = await computer.environment()
    self.assertEqual(result, ComputerEnvironment.ENVIRONMENT_BROWSER)

  async def test_ensure_agent_engine_with_sandbox_name(self):
    """Test _ensure_agent_engine extracts agent engine from sandbox_name."""
    agent_engine_name = (
        "projects/test/locations/us-central1/reasoningEngines/123"
    )
    sandbox_name = f"{agent_engine_name}/sandboxEnvironments/456"
    computer = AgentEngineSandboxComputer(sandbox_name=sandbox_name)
    computer._session_state = {}

    result = await computer._ensure_agent_engine()

    self.assertEqual(result, agent_engine_name)
    # Should not have touched session state
    self.assertNotIn(_STATE_KEY_AGENT_ENGINE_NAME, computer._session_state)

  async def test_ensure_agent_engine_from_session_state(self):
    """Test _ensure_agent_engine uses session state value."""
    agent_engine_name = (
        "projects/test/locations/us-central1/reasoningEngines/123"
    )
    computer = AgentEngineSandboxComputer()
    computer._session_state = {_STATE_KEY_AGENT_ENGINE_NAME: agent_engine_name}

    result = await computer._ensure_agent_engine()

    self.assertEqual(result, agent_engine_name)

  @patch("google.adk.integrations.vmaas.sandbox_computer.asyncio.to_thread")
  @patch.object(AgentEngineSandboxComputer, "_get_client")
  async def test_ensure_agent_engine_creates_new(
      self, mock_get_client, mock_to_thread
  ):
    """Test _ensure_agent_engine creates new agent engine."""
    new_engine_name = "projects/test/locations/us-central1/reasoningEngines/new"

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_engine = MagicMock()
    mock_engine.api_resource.name = new_engine_name
    mock_to_thread.return_value = mock_engine

    computer = AgentEngineSandboxComputer(project_id=self.project_id)
    computer._session_state = {}

    result = await computer._ensure_agent_engine()

    self.assertEqual(result, new_engine_name)
    self.assertEqual(
        computer._session_state[_STATE_KEY_AGENT_ENGINE_NAME], new_engine_name
    )

  @patch("google.adk.integrations.vmaas.sandbox_computer.asyncio.to_thread")
  @patch.object(AgentEngineSandboxComputer, "_get_client")
  async def test_get_sandbox_with_constructor_value(
      self, mock_get_client, mock_to_thread
  ):
    """Test _get_sandbox uses constructor value (BYOS mode)."""
    sandbox_name = "projects/test/sandboxEnvironments/123"

    mock_sandbox = MagicMock()
    mock_sandbox.name = sandbox_name
    mock_to_thread.return_value = mock_sandbox

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer(sandbox_name=sandbox_name)
    computer._session_state = {}

    result_name, result_sandbox = await computer._get_sandbox()

    self.assertEqual(result_name, sandbox_name)
    self.assertEqual(result_sandbox, mock_sandbox)

  @patch("google.adk.integrations.vmaas.sandbox_computer.asyncio.to_thread")
  @patch.object(AgentEngineSandboxComputer, "_get_client")
  async def test_get_sandbox_from_session_state(
      self, mock_get_client, mock_to_thread
  ):
    """Test _get_sandbox uses session state value."""
    sandbox_name = "projects/test/sandboxEnvironments/123"

    mock_sandbox = MagicMock()
    mock_to_thread.return_value = mock_sandbox

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {_STATE_KEY_SANDBOX_NAME: sandbox_name}

    result_name, result_sandbox = await computer._get_sandbox()

    self.assertEqual(result_name, sandbox_name)
    self.assertEqual(result_sandbox, mock_sandbox)

  async def test_get_access_token_cached(self):
    """Test _get_access_token uses cached token."""
    sandbox_name = "projects/test/sandboxEnvironments/123"
    cached_token = "cached_token_123"
    # Set expiry far in the future
    token_expiry = time.time() + 3600

    computer = AgentEngineSandboxComputer()
    computer._session_state = {
        _STATE_KEY_ACCESS_TOKEN: cached_token,
        _STATE_KEY_TOKEN_EXPIRY: token_expiry,
    }

    result = await computer._get_access_token(sandbox_name)

    self.assertEqual(result, cached_token)

  @patch("google.adk.integrations.vmaas.sandbox_computer.asyncio.to_thread")
  @patch.object(AgentEngineSandboxComputer, "_get_client")
  async def test_get_access_token_generates_new_when_expired(
      self, mock_get_client, mock_to_thread
  ):
    """Test _get_access_token generates new token when expired."""
    sandbox_name = "projects/test/sandboxEnvironments/123"
    new_token = "new_token_456"
    # Set expiry in the past
    token_expiry = time.time() - 100

    mock_to_thread.return_value = new_token
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer(
        service_account_email=self.service_account
    )
    computer._session_state = {
        _STATE_KEY_ACCESS_TOKEN: "old_token",
        _STATE_KEY_TOKEN_EXPIRY: token_expiry,
    }

    result = await computer._get_access_token(sandbox_name)

    self.assertEqual(result, new_token)
    self.assertEqual(
        computer._session_state[_STATE_KEY_ACCESS_TOKEN], new_token
    )

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_click_at(self, mock_get_client):
    """Test click_at method."""
    mock_client = AsyncMock()
    mock_client.click_at = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.click_at(100, 200)

    mock_client.click_at.assert_called_once_with(100, 200)
    self.assertIsInstance(result, ComputerState)
    self.assertEqual(result.screenshot, b"png_data")
    self.assertEqual(result.url, "https://example.com")

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_hover_at(self, mock_get_client):
    """Test hover_at method."""
    mock_client = AsyncMock()
    mock_client.hover_at = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.hover_at(150, 250)

    mock_client.hover_at.assert_called_once_with(150, 250)
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_type_text_at(self, mock_get_client):
    """Test type_text_at method."""
    mock_client = AsyncMock()
    mock_client.type_text_at = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.type_text_at(
        100,
        200,
        "hello",
        press_enter=True,
        clear_before_typing=False,
    )

    mock_client.type_text_at.assert_called_once_with(
        x=100,
        y=200,
        text="hello",
        press_enter=True,
        clear_before_typing=False,
    )
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_scroll_document(self, mock_get_client):
    """Test scroll_document method."""
    mock_client = AsyncMock()
    mock_client.scroll_at = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.scroll_document("down")

    # Should scroll at center of screen
    mock_client.scroll_at.assert_called_once_with(640, 360, "down", 400)
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_scroll_at(self, mock_get_client):
    """Test scroll_at method."""
    mock_client = AsyncMock()
    mock_client.scroll_at = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.scroll_at(100, 200, "up", 500)

    mock_client.scroll_at.assert_called_once_with(100, 200, "up", 500)
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_navigate(self, mock_get_client):
    """Test navigate method."""
    mock_client = AsyncMock()
    mock_client.navigate = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://newsite.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.navigate("https://newsite.com")

    mock_client.navigate.assert_called_once_with("https://newsite.com")
    self.assertIsInstance(result, ComputerState)
    self.assertEqual(result.url, "https://newsite.com")

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_search(self, mock_get_client):
    """Test search method navigates to search engine."""
    mock_client = AsyncMock()
    mock_client.navigate = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(
        return_value="https://www.google.com"
    )
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer(
        search_engine_url="https://www.google.com"
    )
    computer._session_state = {}

    result = await computer.search()

    mock_client.navigate.assert_called_once_with("https://www.google.com")
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_go_back(self, mock_get_client):
    """Test go_back method."""
    mock_client = AsyncMock()
    mock_client.go_back = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://prev.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.go_back()

    mock_client.go_back.assert_called_once()
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_go_forward(self, mock_get_client):
    """Test go_forward method."""
    mock_client = AsyncMock()
    mock_client.go_forward = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://next.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.go_forward()

    mock_client.go_forward.assert_called_once()
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_key_combination(self, mock_get_client):
    """Test key_combination method."""
    mock_client = AsyncMock()
    mock_client.key_combination = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.key_combination(["control", "c"])

    mock_client.key_combination.assert_called_once_with(["control", "c"])
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_drag_and_drop(self, mock_get_client):
    """Test drag_and_drop method."""
    mock_client = AsyncMock()
    mock_client.drag_and_drop = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.drag_and_drop(10, 20, 100, 200)

    mock_client.drag_and_drop.assert_called_once_with(10, 20, 100, 200)
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_wait(self, mock_get_client):
    """Test wait method."""
    mock_client = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    # Use a short wait time for testing
    start_time = time.time()
    result = await computer.wait(1)
    elapsed = time.time() - start_time

    self.assertGreaterEqual(elapsed, 0.9)  # Allow some margin
    self.assertIsInstance(result, ComputerState)

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_current_state(self, mock_get_client):
    """Test current_state method."""
    mock_client = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="https://example.com")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.current_state()

    self.assertIsInstance(result, ComputerState)
    self.assertEqual(result.screenshot, b"png_data")
    self.assertEqual(result.url, "https://example.com")

  @patch.object(AgentEngineSandboxComputer, "_get_sandbox_client")
  async def test_open_web_browser(self, mock_get_client):
    """Test open_web_browser method returns current state."""
    mock_client = AsyncMock()
    mock_client.get_screenshot = AsyncMock(return_value=b"png_data")
    mock_client.get_current_url = AsyncMock(return_value="about:blank")
    mock_get_client.return_value = mock_client

    computer = AgentEngineSandboxComputer()
    computer._session_state = {}

    result = await computer.open_web_browser()

    # open_web_browser is a no-op for sandbox, just returns current state
    self.assertIsInstance(result, ComputerState)

  async def test_initialize_is_noop(self):
    """Test initialize does nothing (lazy provisioning)."""
    computer = AgentEngineSandboxComputer()
    # Should not raise
    await computer.initialize()

  async def test_close_is_noop(self):
    """Test close does nothing (TTL-based cleanup)."""
    computer = AgentEngineSandboxComputer()
    # Should not raise
    await computer.close()


if __name__ == "__main__":
  unittest.main()
