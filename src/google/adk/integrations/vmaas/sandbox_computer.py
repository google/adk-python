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

"""Vertex AI Agent Engine Sandbox Computer implementation.

This module provides a BaseComputer implementation that uses Vertex AI
Agent Engine Computer Use Sandbox as the remote browser environment.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from ...features import experimental
from ...features import FeatureName
from ...tools.computer_use.base_computer import BaseComputer
from ...tools.computer_use.base_computer import ComputerEnvironment
from ...tools.computer_use.base_computer import ComputerState
from .sandbox_client import SandboxClient

if TYPE_CHECKING:
  import vertexai

  from ...tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

# Session state keys for sharing resources across sessions
_STATE_KEY_AGENT_ENGINE_NAME = "_vmaas_agent_engine_name"
_STATE_KEY_SANDBOX_NAME = "_vmaas_sandbox_name"
_STATE_KEY_ACCESS_TOKEN = "_vmaas_access_token"
_STATE_KEY_TOKEN_EXPIRY = "_vmaas_token_expiry"

# Default token timeout in seconds
_DEFAULT_TOKEN_TIMEOUT = 3600

# Buffer time before token expiry to trigger refresh (60 seconds)
_TOKEN_REFRESH_BUFFER = 60


@experimental(FeatureName.COMPUTER_USE)
class AgentEngineSandboxComputer(BaseComputer):
  """Computer implementation using Vertex AI Agent Engine Sandbox.

  This class provides a remote browser environment backed by Vertex AI
  Computer Use Sandbox. It supports:
  - Auto-provisioning of agent engines and sandboxes
  - Bring-your-own-sandbox (BYOS) mode
  - Session-aware resource sharing via session_state property
  - Automatic token refresh on expiry

  When used with ComputerUseToolset, the session_state property is
  automatically bound to tool_context.state before each tool call,
  enabling state sharing across invocations and agent server instances.

  Example usage:
    ```python
    from google.adk.integrations.vmaas import AgentEngineSandboxComputer
    from google.adk.tools.computer_use import ComputerUseToolset

    computer = AgentEngineSandboxComputer(
        project_id="my-project",
        service_account_email="sa@my-project.iam.gserviceaccount.com",
    )
    toolset = ComputerUseToolset(computer=computer)
    agent = Agent(tools=[toolset], ...)
    ```
  """

  def __init__(
      self,
      *,
      project_id: str | None = None,
      location: str = "us-central1",
      service_account_email: str | None = None,
      sandbox_name: str | None = None,
      sandbox_ttl_seconds: int = 3600,
      search_engine_url: str = "https://www.google.com",
      vertexai_client: "vertexai.Client | None" = None,
  ):
    """Initialize the sandbox computer.

    Args:
      project_id: GCP project ID. If None, uses Application Default
        Credentials project.
      location: Vertex AI location (default: us-central1).
      service_account_email: Service account email for token generation.
        Must have roles/iam.serviceAccountTokenCreator permission.
        If None, attempts to use ADC service account.
      sandbox_name: Existing sandbox resource name (BYOS mode). If provided,
        the agent engine name is extracted from it. If None, creates new
        agent engine and sandbox on demand.
        Format: projects/{project}/locations/{location}/reasoningEngines/{id}/sandboxEnvironments/{id}
      sandbox_ttl_seconds: TTL for auto-created sandboxes (default: 1 hour).
      search_engine_url: URL to navigate to for search() method.
      vertexai_client: Optional Vertex AI client instance. If None, creates
        one lazily using project_id and location.
    """
    self._project_id = project_id
    self._location = location
    self._service_account_email = service_account_email
    self._sandbox_name = sandbox_name
    self._sandbox_ttl_seconds = sandbox_ttl_seconds
    self._search_engine_url = search_engine_url
    self._screen_size = (1280, 720)

    # Extract agent engine name from sandbox_name if provided
    self._agent_engine_name = None
    if sandbox_name:
      # Format: projects/.../reasoningEngines/.../sandboxEnvironments/...
      parts = sandbox_name.split("/sandboxEnvironments/")
      if len(parts) == 2:
        self._agent_engine_name = parts[0]

    # Vertex client (lazy-initialized if not provided)
    self._client = vertexai_client

    # Session state for sharing sandbox/tokens across invocations
    self._session_state: dict[str, Any] | None = None

  async def prepare(self, tool_context: "ToolContext") -> None:
    """Bind session state for sandbox resource sharing."""
    self._session_state = tool_context.state

  def _get_client(self) -> "vertexai.Client":
    """Get or create the Vertex AI client."""
    if self._client is None:
      import vertexai

      self._client = vertexai.Client(
          project=self._project_id, location=self._location
      )
    return self._client

  async def _ensure_agent_engine(self) -> str:
    """Ensure an agent engine exists, creating one if needed.

    Returns:
      The agent engine resource name.
    """
    # Check if provided in constructor
    if self._agent_engine_name:
      return self._agent_engine_name

    # Check session state
    agent_engine_name = self._session_state.get(_STATE_KEY_AGENT_ENGINE_NAME)
    if agent_engine_name:
      return agent_engine_name

    # Create new agent engine
    logger.info("Creating new agent engine...")
    client = self._get_client()

    agent_engine = await asyncio.to_thread(client.agent_engines.create)
    agent_engine_name = agent_engine.api_resource.name

    # Store in session state for sharing
    self._session_state[_STATE_KEY_AGENT_ENGINE_NAME] = agent_engine_name
    logger.info("Created agent engine: %s", agent_engine_name)

    return agent_engine_name

  async def _get_sandbox(self) -> tuple[str, Any]:
    """Get the sandbox, creating one if needed.

    Returns:
      Tuple of (sandbox_name, sandbox_object).
    """
    client = self._get_client()

    # Check if provided in constructor (BYOS mode)
    if self._sandbox_name:
      # Get sandbox object from name
      sandbox = await asyncio.to_thread(
          client.agent_engines.sandboxes.get, name=self._sandbox_name
      )
      return self._sandbox_name, sandbox

    # Check session state for existing sandbox
    sandbox_name = self._session_state.get(_STATE_KEY_SANDBOX_NAME)
    if sandbox_name:
      sandbox = await asyncio.to_thread(
          client.agent_engines.sandboxes.get, name=sandbox_name
      )
      return sandbox_name, sandbox

    # Ensure agent engine exists first
    agent_engine_name = await self._ensure_agent_engine()

    # Create new sandbox
    logger.info(
        "Creating new sandbox under agent engine: %s", agent_engine_name
    )

    from vertexai import types

    operation = await asyncio.to_thread(
        client.agent_engines.sandboxes.create,
        spec={"computer_use_environment": {}},
        name=agent_engine_name,
        config=types.CreateAgentEngineSandboxConfig(
            display_name="adk_computer_use_sandbox"
        ),
    )

    sandbox_name = operation.response.name

    # Store in session state for sharing
    self._session_state[_STATE_KEY_SANDBOX_NAME] = sandbox_name
    logger.info("Created sandbox: %s", sandbox_name)

    return sandbox_name, operation.response

  async def _get_access_token(self, sandbox_name: str) -> str:
    """Get or refresh the access token for the sandbox.

    Args:
      sandbox_name: The sandbox resource name.

    Returns:
      The access token.
    """
    # Check session state
    token = self._session_state.get(_STATE_KEY_ACCESS_TOKEN)
    expiry = self._session_state.get(_STATE_KEY_TOKEN_EXPIRY, 0)
    if token and time.time() < expiry - _TOKEN_REFRESH_BUFFER:
      return token

    # Generate new token
    logger.debug("Generating new access token for sandbox: %s", sandbox_name)
    client = self._get_client()

    token = await asyncio.to_thread(
        client.agent_engines.sandboxes.generate_access_token,
        service_account_email=self._service_account_email,
        sandbox_id=sandbox_name,
        timeout=_DEFAULT_TOKEN_TIMEOUT,
    )

    # Store in session state
    self._session_state[_STATE_KEY_ACCESS_TOKEN] = token
    self._session_state[_STATE_KEY_TOKEN_EXPIRY] = (
        time.time() + _DEFAULT_TOKEN_TIMEOUT
    )

    return token

  async def _get_sandbox_client(self) -> SandboxClient:
    """Get a sandbox client, ensuring sandbox exists and token is valid.

    Returns:
      A configured SandboxClient.
    """
    sandbox_name, sandbox = await self._get_sandbox()

    try:
      token = await self._get_access_token(sandbox_name)
    except Exception as e:
      # Token generation failed - clear cached token and retry
      logger.warning("Token generation failed, clearing cache: %s", e)
      self._session_state[_STATE_KEY_ACCESS_TOKEN] = None
      self._session_state[_STATE_KEY_TOKEN_EXPIRY] = 0
      token = await self._get_access_token(sandbox_name)

    return SandboxClient(
        vertexai_client=self._get_client(),
        sandbox=sandbox,
        access_token=token,
    )

  async def _get_current_state(self) -> ComputerState:
    """Get the current state with screenshot and URL.

    Returns:
      The current ComputerState.
    """
    client = await self._get_sandbox_client()
    screenshot = await client.get_screenshot()
    url = await client.get_current_url()
    return ComputerState(screenshot=screenshot, url=url)

  # =========================================================================
  # BaseComputer interface implementation
  # =========================================================================

  async def screen_size(self) -> tuple[int, int]:
    """Returns the screen size of the environment."""
    return self._screen_size

  async def environment(self) -> ComputerEnvironment:
    """Returns the environment type."""
    return ComputerEnvironment.ENVIRONMENT_BROWSER

  async def open_web_browser(self) -> ComputerState:
    """Opens the web browser.

    For sandbox, the browser is always running. This is effectively a no-op
    that returns the current state.
    """
    return await self._get_current_state()

  async def click_at(self, x: int, y: int) -> ComputerState:
    """Clicks at a specific x, y coordinate."""
    client = await self._get_sandbox_client()
    await client.click_at(x, y)
    return await self._get_current_state()

  async def hover_at(self, x: int, y: int) -> ComputerState:
    """Hovers at a specific x, y coordinate."""
    client = await self._get_sandbox_client()
    await client.hover_at(x, y)
    return await self._get_current_state()

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = True,
      clear_before_typing: bool = True,
  ) -> ComputerState:
    """Types text at a specific x, y coordinate."""
    client = await self._get_sandbox_client()
    await client.type_text_at(
        x=x,
        y=y,
        text=text,
        press_enter=press_enter,
        clear_before_typing=clear_before_typing,
    )
    return await self._get_current_state()

  async def scroll_document(
      self,
      direction: Literal["up", "down", "left", "right"],
  ) -> ComputerState:
    """Scrolls the entire webpage."""
    client = await self._get_sandbox_client()
    # Scroll at center of screen
    center_x = self._screen_size[0] // 2
    center_y = self._screen_size[1] // 2
    # Use a reasonable default magnitude
    magnitude = 400
    await client.scroll_at(center_x, center_y, direction, magnitude)
    return await self._get_current_state()

  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> ComputerState:
    """Scrolls at a specific coordinate."""
    client = await self._get_sandbox_client()
    await client.scroll_at(x, y, direction, magnitude)
    return await self._get_current_state()

  async def wait(self, seconds: int) -> ComputerState:
    """Waits for n seconds."""
    await asyncio.sleep(seconds)
    return await self._get_current_state()

  async def go_back(self) -> ComputerState:
    """Navigates back in browser history."""
    client = await self._get_sandbox_client()
    await client.go_back()
    return await self._get_current_state()

  async def go_forward(self) -> ComputerState:
    """Navigates forward in browser history."""
    client = await self._get_sandbox_client()
    await client.go_forward()
    return await self._get_current_state()

  async def search(self) -> ComputerState:
    """Navigates to the search engine home page."""
    client = await self._get_sandbox_client()
    await client.navigate(self._search_engine_url)
    return await self._get_current_state()

  async def navigate(self, url: str) -> ComputerState:
    """Navigates to a URL."""
    client = await self._get_sandbox_client()
    await client.navigate(url)
    return await self._get_current_state()

  async def key_combination(self, keys: list[str]) -> ComputerState:
    """Presses a combination of keys."""
    client = await self._get_sandbox_client()
    await client.key_combination(keys)
    return await self._get_current_state()

  async def drag_and_drop(
      self,
      x: int,
      y: int,
      destination_x: int,
      destination_y: int,
  ) -> ComputerState:
    """Drag and drop from one coordinate to another."""
    client = await self._get_sandbox_client()
    await client.drag_and_drop(x, y, destination_x, destination_y)
    return await self._get_current_state()

  async def current_state(self) -> ComputerState:
    """Returns the current state."""
    return await self._get_current_state()

  async def initialize(self) -> None:
    """Initialize the computer.

    This is a no-op for sandbox as provisioning happens lazily on first use.
    """
    pass

  async def close(self) -> None:
    """Cleanup resources.

    Note: Sandboxes are cleaned up via TTL by the sandbox service.
    This method does not delete the sandbox to preserve state across
    agent restarts within the TTL window.
    """
    pass
