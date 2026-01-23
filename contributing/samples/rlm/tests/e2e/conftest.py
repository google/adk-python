"""
Fixtures for E2E web tests with real server and mocked LLM.

These tests use Playwright with a real FastAPI server but mock the LLM
calls to provide predictable, fast responses without API costs.
"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from typing import Generator
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from playwright.sync_api import Page
import pytest


@pytest.fixture(scope="session")
def e2e_server() -> Generator[str, None, None]:
  """
  Start a real FastAPI server for E2E tests.

  This starts the actual web server on a random available port with:
  - Isolated SQLite database (temp file)
  - Real WebSocket connections
  - Real session persistence

  The LLM calls are mocked at the application level via environment variable.
  """
  import urllib.error
  import urllib.request

  # Find an available port
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]

  # Use a temp database for isolation
  db_fd, db_path = tempfile.mkstemp(suffix=".db")
  os.close(db_fd)
  db_url = f"sqlite+aiosqlite:///{db_path}"

  # Create temp log directory
  log_dir = tempfile.mkdtemp(prefix="rlm_test_logs_")

  # Start the server in a subprocess
  env = os.environ.copy()
  env["RLM_DB_URL"] = db_url
  env["RLM_LOG_DIR"] = log_dir
  env["RLM_MODEL"] = "gemini-3-flash-preview"  # Use faster model for tests
  env["RLM_MAX_ITERATIONS"] = "5"  # Limit iterations for tests

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
  max_retries = 50
  for i in range(max_retries):
    try:
      urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
      break
    except (urllib.error.URLError, ConnectionRefusedError):
      time.sleep(0.2)
  else:
    stdout, stderr = proc.communicate(timeout=5)
    proc.terminate()
    raise RuntimeError(
        f"Server did not start within {max_retries * 0.2}s\n"
        f"stdout: {stdout.decode()}\n"
        f"stderr: {stderr.decode()}"
    )

  url = f"http://127.0.0.1:{port}"

  yield url

  # Cleanup
  proc.terminate()
  try:
    proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
    proc.kill()

  # Remove temp database
  if os.path.exists(db_path):
    os.unlink(db_path)

  # Remove temp log directory
  import shutil

  if os.path.exists(log_dir):
    shutil.rmtree(log_dir, ignore_errors=True)


@pytest.fixture
def e2e_page(page: Page, e2e_server: str) -> Page:
  """
  Provide a page connected to the E2E server.

  This fixture navigates to the server and waits for WebSocket connection.
  """
  page.goto(e2e_server)

  # Wait for WebSocket connection
  page.wait_for_function(
      "() => document.querySelector('#status-badge')?.textContent ==="
      " 'Connected'",
      timeout=10000,
  )

  return page


class MockLLMResponse:
  """Mock LLM response for testing."""

  def __init__(
      self,
      text: str,
      has_code: bool = False,
      code: str | None = None,
      is_final: bool = False,
  ):
    self.text = text
    self.has_code = has_code
    self.code = code
    self.is_final = is_final

  def to_response_text(self) -> str:
    """Generate response text that mimics LLM output."""
    if self.is_final:
      return f"FINAL({self.text})"

    if self.has_code and self.code:
      return f"Let me calculate that.\n\n```python\n{self.code}\n```"

    return self.text


# Pre-defined mock responses for common test scenarios
MOCK_RESPONSES = {
    "simple_math": [
        MockLLMResponse(
            text="4",
            has_code=True,
            code="result = 2 + 2\nFINAL_VAR('result')",
            is_final=False,
        ),
    ],
    "multi_step": [
        MockLLMResponse(
            text="First, let me break this down.",
            has_code=True,
            code="step1 = 10 * 2\nprint(f'Step 1: {step1}')",
        ),
        MockLLMResponse(
            text="Now let me finish.",
            has_code=True,
            code="final_result = step1 + 5\nFINAL_VAR('final_result')",
        ),
    ],
    "no_code": [
        MockLLMResponse(
            text="The capital of France is Paris.",
            is_final=True,
        ),
    ],
    "error_recovery": [
        MockLLMResponse(
            text="Let me try.",
            has_code=True,
            code="result = undefined_variable",  # Will error
        ),
        MockLLMResponse(
            text="Let me fix that.",
            has_code=True,
            code="result = 42\nFINAL_VAR('result')",
        ),
    ],
}


def create_mock_llm_client():
  """Create a mock Gemini client for testing."""
  from unittest.mock import AsyncMock
  from unittest.mock import MagicMock

  mock_client = MagicMock()
  mock_aio = MagicMock()
  mock_models = MagicMock()

  response_queue = []

  async def mock_generate_content(*args, **kwargs):
    """Mock generate_content that returns queued responses."""
    if response_queue:
      response = response_queue.pop(0)
    else:
      # Default response if queue is empty
      response = MockLLMResponse("FINAL(No response configured)", is_final=True)

    mock_response = MagicMock()
    mock_response.text = response.to_response_text()
    return mock_response

  mock_models.generate_content = AsyncMock(side_effect=mock_generate_content)
  mock_aio.models = mock_models
  mock_client.aio = mock_aio

  # Attach response queue for test manipulation
  mock_client._response_queue = response_queue

  return mock_client


@pytest.fixture
def mock_llm():
  """
  Fixture that provides a mock LLM client.

  Usage:
      def test_example(mock_llm, e2e_page):
          mock_llm.queue_responses(MOCK_RESPONSES["simple_math"])
          # ... run test
  """

  class MockLLMManager:

    def __init__(self):
      self.responses: list[MockLLMResponse] = []

    def queue_response(self, response: MockLLMResponse):
      """Queue a single response."""
      self.responses.append(response)

    def queue_responses(self, responses: list[MockLLMResponse]):
      """Queue multiple responses."""
      self.responses.extend(responses)

    def clear(self):
      """Clear all queued responses."""
      self.responses.clear()

  return MockLLMManager()


# Helper functions for E2E tests


def wait_for_query_complete(page: Page, timeout: int = 30000):
  """Wait for a query to complete (processing indicator hidden)."""
  # First wait for processing to start (not hidden)
  try:
    page.wait_for_function(
        "() =>"
        " !document.querySelector('#processing')?.classList.contains('hidden')",
        timeout=5000,
    )
  except Exception:
    # Processing might have already started and finished
    pass

  # Then wait for processing to finish (hidden again)
  page.wait_for_function(
      "() =>"
      " document.querySelector('#processing')?.classList.contains('hidden')",
      timeout=timeout,
  )


def wait_for_answer(page: Page, timeout: int = 30000):
  """Wait for an answer panel to appear."""
  page.wait_for_selector(".answer-panel", timeout=timeout)


def get_answer_text(page: Page) -> str:
  """Get the text from the last answer panel."""
  answer = page.locator(".answer-text").last
  return answer.text_content() or ""


def get_event_count(page: Page) -> int:
  """Get the current event count from the UI."""
  count_text = page.locator("#event-count").text_content() or "0 events"
  return int(count_text.split()[0])


def submit_query(page: Page, query: str):
  """Submit a query via the input field."""
  prompt_input = page.locator("#prompt-input")
  prompt_input.fill(query)
  prompt_input.press("Enter")


def get_message_count(page: Page) -> int:
  """Get the number of messages in the conversation."""
  return page.locator(".message").count()


def get_session_count(page: Page) -> int:
  """Get the number of sessions in the sidebar."""
  return page.locator(".session-item").count()
