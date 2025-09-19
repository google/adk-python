"""HTTP client for interacting with the ADK web server."""

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

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from typing import Any
from typing import AsyncGenerator
from typing import Dict
from typing import Literal
from typing import Optional

import httpx

from ...events.event import Event
from ...sessions.session import Session
from ..adk_web_server import RunAgentRequest

logger = logging.getLogger("google_adk." + __name__)


class AdkWebServerClient:
  """HTTP client for interacting with the ADK web server for conformance tests.

  Usage patterns:

    # Pattern 1: Manual lifecycle management
    client = AdkWebServerClient()
    session = await client.create_session(app_name="app", user_id="user")
    async for event in client.run_agent(request):
        # Process events...
    await client.close()  # Optional explicit cleanup

    # Pattern 2: Automatic cleanup with context manager (recommended)
    async with AdkWebServerClient() as client:
        session = await client.create_session(app_name="app", user_id="user")
        async for event in client.run_agent(request):
            # Process events...
        # Client automatically closed here
  """

  def __init__(
      self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0
  ):
    """Initialize the ADK web server client for conformance testing.

    Args:
      base_url: Base URL of the ADK web server (default: http://127.0.0.1:8000)
      timeout: Request timeout in seconds (default: 30.0)
    """
    self.base_url = base_url.rstrip("/")
    self.timeout = timeout
    self._client: Optional[httpx.AsyncClient] = None

  @asynccontextmanager
  async def _get_client(self) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Get or create an HTTP client with proper lifecycle management.

    Returns:
      AsyncGenerator yielding the HTTP client instance.
    """
    if self._client is None:
      self._client = httpx.AsyncClient(
          base_url=self.base_url,
          timeout=httpx.Timeout(self.timeout),
      )
    try:
      yield self._client
    finally:
      pass  # Keep client alive for reuse

  async def close(self) -> None:
    """Close the HTTP client and clean up resources."""
    if self._client:
      await self._client.aclose()
      self._client = None

  async def __aenter__(self) -> "AdkWebServerClient":
    """Async context manager entry.

    Returns:
      The client instance for use in the async context.
    """
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # pylint: disable=unused-argument
    """Async context manager exit that closes the HTTP client."""
    await self.close()

  async def get_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> Session:
    """Retrieve a specific session from the ADK web server.

    Args:
      app_name: Name of the application
      user_id: User identifier
      session_id: Session identifier

    Returns:
      The requested Session object

    Raises:
      httpx.HTTPStatusError: If the request fails or session not found
    """
    async with self._get_client() as client:
      response = await client.get(
          f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
      )
      response.raise_for_status()
      return Session.model_validate(response.json())

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[Dict[str, Any]] = None,
  ) -> Session:
    """Create a new session in the ADK web server.

    Args:
      app_name: Name of the application
      user_id: User identifier
      state: Optional initial state for the session

    Returns:
      The newly created Session object

    Raises:
      httpx.HTTPStatusError: If the request fails
    """
    async with self._get_client() as client:
      payload = {}
      if state is not None:
        payload["state"] = state

      response = await client.post(
          f"/apps/{app_name}/users/{user_id}/sessions",
          json=payload,
      )
      response.raise_for_status()
      return Session.model_validate(response.json())

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Delete a session from the ADK web server.

    Args:
      app_name: Name of the application
      user_id: User identifier
      session_id: Session identifier to delete

    Raises:
      httpx.HTTPStatusError: If the request fails or session not found
    """
    async with self._get_client() as client:
      response = await client.delete(
          f"/apps/{app_name}/users/{user_id}/sessions/{session_id}"
      )
      response.raise_for_status()

  async def run_agent(
      self,
      request: RunAgentRequest,
      mode: Optional[Literal["record", "replay"]] = None,
      test_case_dir: Optional[str] = None,
      user_message_index: Optional[int] = None,
  ) -> AsyncGenerator[Event, None]:
    """Run an agent with streaming Server-Sent Events response.

    Args:
      request: The RunAgentRequest containing agent execution parameters
      mode: Optional conformance mode ("record" or "replay") to trigger recording
      test_case_dir: Optional test case directory path for conformance recording
      user_message_index: Optional user message index for conformance recording

    Yields:
      Event objects streamed from the agent execution

    Raises:
      ValueError: If mode is provided but test_case_dir or user_message_index is None
      httpx.HTTPStatusError: If the request fails
      json.JSONDecodeError: If event data cannot be parsed
    """
    # Add recording parameters to state_delta for conformance tests
    if mode:
      if test_case_dir is None or user_message_index is None:
        raise ValueError(
            "test_case_dir and user_message_index must be provided when mode is"
            " specified"
        )

      # Modify request state_delta in place
      if request.state_delta is None:
        request.state_delta = {}

      if mode == "replay":
        request.state_delta["_adk_replay_config"] = {
            "dir": str(test_case_dir),
            "user_message_index": user_message_index,
        }
      else:  # record mode
        request.state_delta["_adk_recordings_config"] = {
            "dir": str(test_case_dir),
            "user_message_index": user_message_index,
        }

    async with self._get_client() as client:
      async with client.stream(
          "POST",
          "/run_sse",
          json=request.model_dump(by_alias=True, exclude_none=True),
      ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
          if line.startswith("data:") and (data := line[5:].strip()):
            event_data = json.loads(data)
            yield Event.model_validate(event_data)
          else:
            logger.debug("Non data line received: %s", line)
