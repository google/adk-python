"""
Fixtures for UI tests with mocked WebSocket responses.

These tests use Playwright to test the frontend UI components
with a mock WebSocket server that simulates backend responses.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock

from playwright.sync_api import Page
from playwright.sync_api import Route
from playwright.sync_api import WebSocket
import pytest


# Sample event data for mocking
def create_mock_event(
    event_type: str,
    iteration: int = 1,
    event_id: int = 0,
    **metadata: Any,
) -> dict:
  """Create a mock event matching the format from web.py."""
  icons = {
      "rlm.run.start": "play_arrow",
      "rlm.run.end": "stop",
      "rlm.iteration.start": "loop",
      "rlm.iteration.end": "check_circle",
      "rlm.llm.start": "psychology",
      "rlm.llm.end": "psychology",
      "rlm.code.found": "code",
      "rlm.code.start": "terminal",
      "rlm.code.end": "terminal",
      "rlm.final.detected": "star",
      "rlm.final.answer": "check",
  }
  colors = {
      "rlm.run.start": "#7AA2F7",
      "rlm.run.end": "#9ECE6A",
      "rlm.iteration.start": "#7AA2F7",
      "rlm.iteration.end": "#565F89",
      "rlm.llm.start": "#BB9AF7",
      "rlm.llm.end": "#BB9AF7",
      "rlm.code.found": "#9ECE6A",
      "rlm.code.start": "#7DCFFF",
      "rlm.code.end": "#7DCFFF",
      "rlm.final.detected": "#E0AF68",
      "rlm.final.answer": "#E0AF68",
  }
  label = event_type.replace("rlm.", "").replace(".", " ").title()

  return {
      "id": event_id,
      "type": "event",
      "event_type": event_type,
      "iteration": iteration,
      "timestamp": 0.1 * (event_id + 1),
      "icon": icons.get(event_type, "circle"),
      "color": colors.get(event_type, "#A9B1D6"),
      "label": label,
      "metadata": metadata,
  }


def create_mock_session(
    session_id: str = "test-session-123",
    title: str = "Test Session",
    model: str = "gemini-3-pro-preview",
    conversation: list | None = None,
    events: list | None = None,
    files: list | None = None,
) -> dict:
  """Create a mock session response."""
  return {
      "type": "status_response",
      "session_id": session_id,
      "title": title,
      "model": model,
      "sub_model": model,
      "max_iterations": 30,
      "files": files or [],
      "conversation": conversation or [],
      "events": events or [],
  }


def create_mock_sessions_list(sessions: list[dict] | None = None) -> dict:
  """Create a mock sessions list response."""
  if sessions is None:
    sessions = [
        {
            "session_id": "session-1",
            "title": "First Session",
            "updated_at": "2024-01-15T10:00:00",
            "message_count": 2,
        },
        {
            "session_id": "session-2",
            "title": "Second Session",
            "updated_at": "2024-01-14T09:00:00",
            "message_count": 5,
        },
    ]
  return {
      "type": "sessions_list",
      "sessions": sessions,
  }


class MockWebSocketServer:
  """Mock WebSocket server for UI testing."""

  def __init__(self):
    self.messages_received: list[dict] = []
    self.messages_to_send: list[dict] = []
    self.auto_responses: dict[str, list[dict]] = {}
    self.connected = False
    self._ws: WebSocket | None = None

  def queue_message(self, message: dict):
    """Queue a message to be sent to the client."""
    self.messages_to_send.append(message)

  def queue_messages(self, messages: list[dict]):
    """Queue multiple messages."""
    self.messages_to_send.extend(messages)

  def set_auto_response(self, action: str, responses: list[dict]):
    """Set automatic responses for a given action."""
    self.auto_responses[action] = responses

  def get_received_messages(self) -> list[dict]:
    """Get all messages received from the client."""
    return self.messages_received.copy()

  def clear(self):
    """Clear all queued messages and received messages."""
    self.messages_received.clear()
    self.messages_to_send.clear()


class WebSocketInterceptor:
  """Intercept and mock WebSocket connections in Playwright."""

  # JavaScript code for WebSocket mock - defined once as class attribute
  MOCK_WS_SCRIPT = """
        // Store original WebSocket
        window._OriginalWebSocket = window.WebSocket;
        window._mockWsMessages = [];
        window._mockWsReceived = [];
        window._mockWsConnected = false;
        window._mockWsAutoResponses = {};
        window._mockWs = null;

        // Create mock WebSocket class
        class MockWebSocket {
            constructor(url) {
                this.url = url;
                this.readyState = 0; // CONNECTING
                this.onopen = null;
                this.onclose = null;
                this.onerror = null;
                this.onmessage = null;
                window._mockWs = this;

                // Auto-connect after a small delay
                setTimeout(() => {
                    this.readyState = 1; // OPEN
                    window._mockWsConnected = true;
                    if (this.onopen) {
                        this.onopen({ type: 'open' });
                    }
                    // Send any queued messages
                    this._processQueue();
                }, 50);
            }

            send(data) {
                const parsed = JSON.parse(data);
                window._mockWsReceived.push(parsed);

                // Check for auto-responses
                const action = parsed.action;
                if (window._mockWsAutoResponses[action]) {
                    const responses = window._mockWsAutoResponses[action];
                    responses.forEach((resp, i) => {
                        setTimeout(() => {
                            if (this.onmessage) {
                                this.onmessage({ data: JSON.stringify(resp) });
                            }
                        }, 10 * (i + 1));
                    });
                }
            }

            close() {
                this.readyState = 3; // CLOSED
                window._mockWsConnected = false;
                if (this.onclose) {
                    this.onclose({ type: 'close' });
                }
            }

            _processQueue() {
                while (window._mockWsMessages.length > 0) {
                    const msg = window._mockWsMessages.shift();
                    if (this.onmessage) {
                        this.onmessage({ data: JSON.stringify(msg) });
                    }
                }
            }

            _receiveMessage(data) {
                if (this.onmessage) {
                    this.onmessage({ data: JSON.stringify(data) });
                }
            }
        }

        MockWebSocket.CONNECTING = 0;
        MockWebSocket.OPEN = 1;
        MockWebSocket.CLOSING = 2;
        MockWebSocket.CLOSED = 3;

        window.WebSocket = MockWebSocket;
    """

  def __init__(self, page: Page):
    self.page = page
    self.mock_server = MockWebSocketServer()
    self._setup_complete = False
    self._pending_auto_responses: dict[str, list[dict]] = {}

  def setup(self):
    """Set up WebSocket interception via add_init_script (runs before page load)."""
    if self._setup_complete:
      return

    # Use add_init_script so the mock is injected BEFORE page JavaScript runs
    self.page.add_init_script(self.MOCK_WS_SCRIPT)
    self._setup_complete = True

  def set_auto_response(self, action: str, responses: list[dict]):
    """
    Set automatic responses for a given action.

    IMPORTANT: Call this BEFORE page.goto() for responses needed during
    initial connection (get_status, list_sessions).
    """
    self._pending_auto_responses[action] = responses
    # Add an init script to set this auto-response before page JS runs
    script = (
        f"window._mockWsAutoResponses[{json.dumps(action)}] ="
        f" {json.dumps(responses)};"
    )
    self.page.add_init_script(script)

  def set_auto_response_after_load(self, action: str, responses: list[dict]):
    """Set automatic responses after page has loaded."""
    self.page.evaluate(
        "(data) => { window._mockWsAutoResponses[data.action] ="
        " data.responses; }",
        {"action": action, "responses": responses},
    )

  def send_message(self, message: dict):
    """Send a message from the mock server to the client."""
    self.page.evaluate(
        """(msg) => {
                if (window._mockWs && window._mockWs.readyState === 1) {
                    window._mockWs._receiveMessage(msg);
                } else {
                    window._mockWsMessages.push(msg);
                }
            }""",
        message,
    )

  def send_messages(self, messages: list[dict], delay_ms: int = 10):
    """Send multiple messages with delay between them."""
    for i, msg in enumerate(messages):
      self.page.evaluate(
          f"""(msg) => {{
                    setTimeout(() => {{
                        if (window._mockWs && window._mockWs.readyState === 1) {{
                            window._mockWs._receiveMessage(msg);
                        }}
                    }}, {i * delay_ms});
                }}""",
          msg,
      )

  def get_received_messages(self) -> list[dict]:
    """Get all messages received by the mock server."""
    return self.page.evaluate("() => window._mockWsReceived || []")

  def is_connected(self) -> bool:
    """Check if WebSocket is connected."""
    return self.page.evaluate("() => window._mockWsConnected || false")

  def wait_for_connection(self, timeout: int = 5000):
    """Wait for WebSocket connection to be established."""
    self.page.wait_for_function(
        "() => window._mockWsConnected === true",
        timeout=timeout,
    )


@pytest.fixture
def mock_ws(page: Page) -> WebSocketInterceptor:
  """
  Fixture that provides a WebSocket interceptor for mocking.

  Usage:
      def test_example(mock_ws, page):
          mock_ws.setup()
          mock_ws.set_auto_response("get_status", [create_mock_session()])
          page.goto("http://localhost:8000")
          mock_ws.wait_for_connection()
  """
  return WebSocketInterceptor(page)


@pytest.fixture
def connected_page(page: Page, mock_ws: WebSocketInterceptor) -> Page:
  """
  Fixture that provides a page with mocked WebSocket already connected.

  Sets up default auto-responses for initial connection handshake.
  """
  mock_ws.setup()

  # Set up default auto-responses
  mock_ws.set_auto_response("get_status", [create_mock_session()])
  mock_ws.set_auto_response("list_sessions", [create_mock_sessions_list()])

  return page


@pytest.fixture
def mock_query_response() -> list[dict]:
  """
  Fixture that provides a sequence of events for a mock query response.
  """
  return [
      {"type": "query_start", "prompt": "What is 2+2?"},
      create_mock_event("rlm.run.start", iteration=0, event_id=0),
      create_mock_event("rlm.iteration.start", iteration=1, event_id=1),
      create_mock_event("rlm.llm.start", iteration=1, event_id=2),
      create_mock_event(
          "rlm.llm.end",
          iteration=1,
          event_id=3,
          response_preview="Let me calculate 2+2...",
      ),
      create_mock_event(
          "rlm.code.found",
          iteration=1,
          event_id=4,
          code="result = 2 + 2\nFINAL_VAR('result')",
      ),
      create_mock_event("rlm.code.start", iteration=1, event_id=5),
      create_mock_event(
          "rlm.code.end",
          iteration=1,
          event_id=6,
          output="4",
      ),
      create_mock_event("rlm.final.detected", iteration=1, event_id=7),
      create_mock_event("rlm.iteration.end", iteration=1, event_id=8),
      create_mock_event("rlm.run.end", iteration=1, event_id=9),
      {
          "type": "query_complete",
          "elapsed_seconds": 1.5,
          "total_events": 10,
          "final_answer": "4",
          "title": "What is 2+2?",
      },
  ]


# HTML content for a minimal test server (used when we don't want to run full app)
MINIMAL_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <div id="status">Loading...</div>
    <script>
        const ws = new WebSocket('ws://localhost:8000/ws/test');
        ws.onopen = () => document.getElementById('status').textContent = 'Connected';
        ws.onclose = () => document.getElementById('status').textContent = 'Disconnected';
    </script>
</body>
</html>
"""


@pytest.fixture(scope="session")
def live_server():
  """
  Start a live FastAPI server for UI tests.

  This starts the actual web server on a random available port.
  The server is shared across all tests in the session for efficiency.
  """
  import os
  import socket
  import subprocess
  import sys
  import tempfile

  # Find an available port
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]

  # Use a temp database for isolation
  db_file = tempfile.mktemp(suffix=".db")
  db_url = f"sqlite+aiosqlite:///{db_file}"

  # Start the server in a subprocess
  env = os.environ.copy()
  env["RLM_DB_URL"] = db_url

  # Start uvicorn via subprocess
  proc = subprocess.Popen(
      [
          sys.executable,
          "-m",
          "uvicorn",
          "adk_rlm.web:app",
          "--host",
          "127.0.0.1",
          "--port",
          str(port),
      ],
      env=env,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )

  # Wait for server to be ready
  import urllib.error
  import urllib.request

  max_retries = 30
  for i in range(max_retries):
    try:
      urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
      break
    except (urllib.error.URLError, ConnectionRefusedError):
      time.sleep(0.2)
  else:
    proc.terminate()
    raise RuntimeError(f"Server did not start within {max_retries * 0.2}s")

  url = f"http://127.0.0.1:{port}"

  yield url

  # Cleanup
  proc.terminate()
  proc.wait(timeout=5)

  # Remove temp database
  if os.path.exists(db_file):
    os.unlink(db_file)
