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

"""Unit tests for the SandboxClient class."""

import base64
import json
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations.vmaas.sandbox_client import SandboxClient


def _make_response(data: dict) -> MagicMock:
  """Create a mock HttpResponse with a JSON body."""
  response = MagicMock()
  response.body = json.dumps(data)
  return response


class TestSandboxClient(unittest.IsolatedAsyncioTestCase):
  """Tests for SandboxClient."""

  def setUp(self):
    """Set up test fixtures."""
    self.mock_vertexai_client = MagicMock()
    self.mock_sandbox = MagicMock()
    self.access_token = "test_token_12345"
    self.client = SandboxClient(
        vertexai_client=self.mock_vertexai_client,
        sandbox=self.mock_sandbox,
        access_token=self.access_token,
    )

  def test_init(self):
    """Test client initialization."""
    self.assertEqual(self.client._client, self.mock_vertexai_client)
    self.assertEqual(self.client._sandbox, self.mock_sandbox)
    self.assertEqual(self.client._access_token, self.access_token)

  def test_update_access_token(self):
    """Test updating access token."""
    new_token = "new_token_67890"
    self.client.update_access_token(new_token)
    self.assertEqual(self.client._access_token, new_token)

  @patch("asyncio.to_thread")
  async def test_make_cdp_request(self, mock_to_thread):
    """Test making a single CDP request."""
    mock_to_thread.return_value = _make_response({"result": "success"})

    result = await self.client.make_cdp_request(
        "Page.navigate", {"url": "https://example.com"}
    )

    self.assertEqual(result, {"result": "success"})
    mock_to_thread.assert_called_once()
    call_args = mock_to_thread.call_args
    # First positional arg is the send_command method
    self.assertEqual(
        call_args[0][0],
        self.mock_vertexai_client.agent_engines.sandboxes.send_command,
    )
    # Check keyword args
    self.assertEqual(call_args[1]["http_method"], "POST")
    self.assertEqual(call_args[1]["path"], "cdp")
    self.assertEqual(call_args[1]["access_token"], self.access_token)
    self.assertEqual(call_args[1]["sandbox_environment"], self.mock_sandbox)
    self.assertEqual(
        call_args[1]["request_dict"],
        {"command": "Page.navigate", "params": {"url": "https://example.com"}},
    )

  @patch("asyncio.to_thread")
  async def test_make_cdp_batch_request_with_batch_endpoint(
      self, mock_to_thread
  ):
    """Test making a batch CDP request using batch endpoint."""
    mock_to_thread.return_value = _make_response(
        {"results": [{"status": "success"}, {"status": "success"}]}
    )

    commands = [
        {
            "command": "Input.dispatchMouseEvent",
            "params": {"type": "mousePressed"},
        },
        {
            "command": "Input.dispatchMouseEvent",
            "params": {"type": "mouseReleased"},
        },
    ]
    result = await self.client.make_cdp_batch_request(commands)

    # Should have 2 results from batch
    self.assertEqual(len(result), 2)
    # Should have made 1 call to /cdps
    self.assertEqual(mock_to_thread.call_count, 1)
    call_args = mock_to_thread.call_args
    self.assertEqual(call_args[1]["path"], "cdps")

  @patch("asyncio.to_thread")
  async def test_make_cdp_batch_request_fallback_sequential(
      self, mock_to_thread
  ):
    """Test batch CDP falls back to sequential when batch endpoint fails."""

    # First call (to /cdps) fails with 404
    # Subsequent calls (to /cdp) succeed
    def side_effect(*args, **kwargs):
      if kwargs.get("path") == "cdps":
        raise Exception("404 Not Found")
      return _make_response({"result": "ok"})

    mock_to_thread.side_effect = side_effect

    commands = [
        {"command": "Command1", "params": {}},
        {"command": "Command2", "params": {}},
    ]
    result = await self.client.make_cdp_batch_request(commands)

    # Should have 2 results (sequential fallback)
    self.assertEqual(len(result), 2)
    self.assertEqual(result[0]["status"], "success")
    self.assertEqual(result[1]["status"], "success")
    # Should have made 3 calls (1 failed /cdps + 2 sequential /cdp)
    self.assertEqual(mock_to_thread.call_count, 3)

  @patch("asyncio.to_thread")
  async def test_get_screenshot(self, mock_to_thread):
    """Test capturing a screenshot."""
    # Create a simple PNG-like base64 data
    png_data = b"\x89PNG\r\n\x1a\n"
    base64_data = base64.b64encode(png_data).decode()

    mock_to_thread.return_value = _make_response({"data": base64_data})

    result = await self.client.get_screenshot()

    self.assertEqual(result, png_data)
    call_args = mock_to_thread.call_args
    self.assertEqual(
        call_args[1]["request_dict"]["command"], "Page.captureScreenshot"
    )

  @patch("asyncio.to_thread")
  async def test_get_current_url(self, mock_to_thread):
    """Test getting the current URL."""
    mock_to_thread.return_value = _make_response({
        "active_tab_id": "tab1",
        "all_tabs": [
            {"id": "tab1", "url": "https://example.com", "title": "Example"},
        ],
    })

    result = await self.client.get_current_url()

    self.assertEqual(result, "https://example.com")
    call_args = mock_to_thread.call_args
    self.assertEqual(call_args[1]["path"], "tabs")
    self.assertEqual(call_args[1]["http_method"], "GET")

  @patch("asyncio.to_thread")
  async def test_get_current_url_no_active_tab(self, mock_to_thread):
    """Test getting URL when no tab is active."""
    mock_to_thread.return_value = _make_response({
        "active_tab_id": None,
        "all_tabs": [],
    })

    result = await self.client.get_current_url()

    self.assertIsNone(result)

  @patch("asyncio.to_thread")
  async def test_navigate(self, mock_to_thread):
    """Test navigating to a URL."""
    mock_to_thread.return_value = _make_response({"frameId": "frame123"})

    result = await self.client.navigate("https://example.com")

    self.assertEqual(result, {"frameId": "frame123"})
    call_args = mock_to_thread.call_args
    self.assertEqual(
        call_args[1]["request_dict"],
        {"command": "Page.navigate", "params": {"url": "https://example.com"}},
    )

  @patch("asyncio.to_thread")
  async def test_click_at(self, mock_to_thread):
    """Test clicking at coordinates."""
    # Return success for batch endpoint
    mock_to_thread.return_value = _make_response({"results": [{}, {}]})

    await self.client.click_at(100, 200)

    # Should call batch endpoint
    call_args = mock_to_thread.call_args
    self.assertEqual(call_args[1]["path"], "cdps")
    commands = call_args[1]["request_dict"]["commands"]
    self.assertEqual(len(commands), 2)
    self.assertEqual(commands[0]["params"]["type"], "mousePressed")
    self.assertEqual(commands[1]["params"]["type"], "mouseReleased")

  @patch("asyncio.to_thread")
  async def test_hover_at(self, mock_to_thread):
    """Test hovering at coordinates."""
    mock_to_thread.return_value = _make_response({})

    await self.client.hover_at(150, 250)

    call_args = mock_to_thread.call_args
    self.assertEqual(
        call_args[1]["request_dict"]["params"],
        {"type": "mouseMoved", "x": 150, "y": 250},
    )

  @patch("asyncio.to_thread")
  async def test_scroll_at_down(self, mock_to_thread):
    """Test scrolling down."""
    mock_to_thread.return_value = _make_response({})

    await self.client.scroll_at(100, 200, "down", 300)

    call_args = mock_to_thread.call_args
    params = call_args[1]["request_dict"]["params"]
    self.assertEqual(params["type"], "mouseWheel")
    self.assertEqual(params["x"], 100)
    self.assertEqual(params["y"], 200)
    self.assertEqual(params["deltaX"], 0)
    self.assertEqual(params["deltaY"], 300)  # Positive for down

  @patch("asyncio.to_thread")
  async def test_scroll_at_up(self, mock_to_thread):
    """Test scrolling up."""
    mock_to_thread.return_value = _make_response({})

    await self.client.scroll_at(100, 200, "up", 300)

    call_args = mock_to_thread.call_args
    params = call_args[1]["request_dict"]["params"]
    self.assertEqual(params["deltaY"], -300)  # Negative for up

  @patch("asyncio.to_thread")
  async def test_go_back(self, mock_to_thread):
    """Test navigating back."""
    # First call returns navigation history, second navigates
    mock_to_thread.side_effect = [
        _make_response({
            "currentIndex": 1,
            "entries": [
                {"id": 1, "url": "https://first.com"},
                {"id": 2, "url": "https://second.com"},
            ],
        }),
        _make_response({}),  # Navigation response
    ]

    result = await self.client.go_back()

    self.assertTrue(result)
    self.assertEqual(mock_to_thread.call_count, 2)

  @patch("asyncio.to_thread")
  async def test_go_back_at_beginning(self, mock_to_thread):
    """Test navigating back when at beginning of history."""
    mock_to_thread.return_value = _make_response({
        "currentIndex": 0,
        "entries": [{"id": 1, "url": "https://first.com"}],
    })

    result = await self.client.go_back()

    self.assertFalse(result)
    # Should only call once (to get history)
    self.assertEqual(mock_to_thread.call_count, 1)

  @patch("asyncio.to_thread")
  async def test_type_text_with_clear_and_enter(self, mock_to_thread):
    """Test typing text with clear and enter options."""
    # Return success for batch endpoint
    mock_to_thread.return_value = _make_response({"results": [{}] * 7})

    await self.client.type_text(
        "hello", press_enter=True, clear_before_typing=True
    )

    # Should have: Ctrl+A down, Ctrl+A up, Delete down, Delete up,
    # insertText, Enter down, Enter up = 7 commands in batch
    call_args = mock_to_thread.call_args
    commands = call_args[1]["request_dict"]["commands"]
    self.assertEqual(len(commands), 7)

  @patch("asyncio.to_thread")
  async def test_key_combination(self, mock_to_thread):
    """Test pressing key combinations."""
    mock_to_thread.return_value = _make_response({"results": [{}] * 4})

    await self.client.key_combination(["control", "c"])

    # Should have: Control down, c down, c up, Control up = 4 commands
    call_args = mock_to_thread.call_args
    commands = call_args[1]["request_dict"]["commands"]
    self.assertEqual(len(commands), 4)

  @patch("asyncio.to_thread")
  async def test_drag_and_drop(self, mock_to_thread):
    """Test drag and drop operation."""
    mock_to_thread.return_value = _make_response({"results": [{}] * 4})

    await self.client.drag_and_drop(10, 20, 100, 200)

    # Should have: mouseMoved (start), mousePressed, mouseMoved (end),
    # mouseReleased = 4 commands
    call_args = mock_to_thread.call_args
    commands = call_args[1]["request_dict"]["commands"]
    self.assertEqual(len(commands), 4)

  @patch("asyncio.to_thread")
  async def test_health_check_healthy(self, mock_to_thread):
    """Test health check when sandbox is healthy."""
    mock_to_thread.return_value = _make_response({"status": "healthy"})

    result = await self.client.health_check()

    self.assertTrue(result)
    call_args = mock_to_thread.call_args
    self.assertEqual(call_args[1]["http_method"], "GET")
    self.assertEqual(call_args[1]["path"], "")

  @patch("asyncio.to_thread")
  async def test_health_check_unhealthy(self, mock_to_thread):
    """Test health check when sandbox is unhealthy."""
    mock_to_thread.return_value = _make_response({"status": "unhealthy"})

    result = await self.client.health_check()

    self.assertFalse(result)

  @patch("asyncio.to_thread")
  async def test_health_check_exception(self, mock_to_thread):
    """Test health check when request fails."""
    mock_to_thread.side_effect = Exception("Connection failed")

    result = await self.client.health_check()

    self.assertFalse(result)


if __name__ == "__main__":
  unittest.main()
