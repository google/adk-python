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

"""Low-level client for Vertex AI Computer Use Sandbox CDP commands.

This module provides functions to interact with the sandbox browser via
Chrome DevTools Protocol (CDP) commands sent through the Vertex AI SDK.
"""

from __future__ import annotations

import base64
import logging
from typing import Any
from typing import Literal
from typing import TYPE_CHECKING

from ...features import experimental
from ...features import FeatureName

if TYPE_CHECKING:
  import vertexai

logger = logging.getLogger("google_adk." + __name__)

# CDP command constants
_CDP_COMMAND_PAGE_CAPTURE_SCREENSHOT = "Page.captureScreenshot"
_CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT = "Input.dispatchMouseEvent"
_CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT = "Input.dispatchKeyEvent"
_CDP_COMMAND_INPUT_INSERT_TEXT = "Input.insertText"
_CDP_COMMAND_PAGE_GET_NAV_HISTORY = "Page.getNavigationHistory"
_CDP_COMMAND_PAGE_NAV_TO_HISTORY = "Page.navigateToHistoryEntry"
_CDP_COMMAND_PAGE_NAVIGATE = "Page.navigate"

# Key mapping from user-friendly names to CDP key values
_META_KEY_MAP = {
    "BACKSPACE": "BackSpace",
    "TAB": "Tab",
    "RETURN": "Enter",
    "ENTER": "Enter",
    "SHIFT": "Shift_L",
    "CONTROL": "Control_L",
    "ALT": "Alt_L",
    "ESCAPE": "Escape",
    "SPACE": "space",
    "PAGEUP": "Page_Up",
    "PAGE_UP": "Page_Up",
    "PAGEDOWN": "Page_Down",
    "PAGE_DOWN": "Page_Down",
    "END": "End",
    "HOME": "Home",
    "LEFT": "Left",
    "UP": "Up",
    "RIGHT": "Right",
    "DOWN": "Down",
    "INSERT": "Insert",
    "DELETE": "Delete",
    "SEMICOLON": "semicolon",
    "EQUALS": "equal",
    "MULTIPLY": "asterisk",
    "ADD": "plus",
    "SEPARATOR": "KP_Separator",
    "SUBTRACT": "minus",
    "DECIMAL": "period",
    "DIVIDE": "slash",
    "F1": "F1",
    "F2": "F2",
    "F3": "F3",
    "F4": "F4",
    "F5": "F5",
    "F6": "F6",
    "F7": "F7",
    "F8": "F8",
    "F9": "F9",
    "F10": "F10",
    "F11": "F11",
    "F12": "F12",
    "COMMAND": "Super_L",
}

# Modifier key to CDP modifier bitmask mapping
_MODIFIER_MAP = {
    "CONTROL": 2,
    "ALT": 1,
    "SHIFT": 8,
    "COMMAND": 4,
    "SUPER": 4,
}


@experimental(FeatureName.COMPUTER_USE)
class SandboxClient:
  """Client for interacting with Vertex AI Computer Use Sandbox via SDK."""

  def __init__(
      self,
      vertexai_client: "vertexai.Client",
      sandbox: Any,
      access_token: str,
  ):
    """Initialize the sandbox client.

    Args:
      vertexai_client: The Vertex AI client instance.
      sandbox: The sandbox object from vertexai SDK (SandboxEnvironment).
      access_token: The access token for authenticating with the sandbox.
    """
    self._client = vertexai_client
    self._sandbox = sandbox
    self._access_token = access_token

  def _parse_response(self, response: Any) -> dict[str, Any]:
    """Parse the response from send_command.

    Args:
      response: The HttpResponse from send_command.

    Returns:
      The parsed JSON response as a dict.
    """
    import json

    if hasattr(response, "body") and response.body:
      return json.loads(response.body)
    return {}

  def update_access_token(self, access_token: str) -> None:
    """Update the access token.

    Args:
      access_token: The new access token.
    """
    self._access_token = access_token

  async def make_cdp_request(
      self,
      command: str,
      params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Make a single CDP request to the sandbox.

    Args:
      command: The CDP command to execute (e.g., "Page.navigate").
      params: Optional parameters for the CDP command.

    Returns:
      The CDP command response.

    Raises:
      Exception: If the request fails.
    """
    import asyncio

    params = params if params is not None else {}
    request_dict = {"command": command, "params": params}

    response = await asyncio.to_thread(
        self._client.agent_engines.sandboxes.send_command,
        http_method="POST",
        path="cdp",
        access_token=self._access_token,
        sandbox_environment=self._sandbox,
        request_dict=request_dict,
    )
    return self._parse_response(response)

  async def make_cdp_batch_request(
      self,
      commands: list[dict[str, Any]],
      stop_on_error: bool = True,
  ) -> list[dict[str, Any]]:
    """Execute multiple CDP commands.

    First tries the batch endpoint (/cdps), falls back to sequential
    execution if batch is not available.

    Args:
      commands: List of CDP commands, each with "command" and "params" keys.
      stop_on_error: Whether to stop processing on first error.

    Returns:
      List of results for each command.
    """
    import asyncio

    # Try batch endpoint first
    try:
      request_dict = {"commands": commands, "stop_on_error": stop_on_error}
      response = await asyncio.to_thread(
          self._client.agent_engines.sandboxes.send_command,
          http_method="POST",
          path="cdps",
          access_token=self._access_token,
          sandbox_environment=self._sandbox,
          request_dict=request_dict,
      )
      parsed = self._parse_response(response)
      return parsed.get("results", [])
    except Exception as e:
      # Batch endpoint not available, fall back to sequential
      if "404" in str(e) or "not found" in str(e).lower():
        logger.debug("Batch CDP endpoint not available, using sequential")
      else:
        logger.warning("Batch CDP failed: %s, falling back to sequential", e)

    # Sequential fallback
    results = []
    for cmd in commands:
      try:
        result = await self.make_cdp_request(
            cmd["command"], cmd.get("params", {})
        )
        results.append({"status": "success", "result": result})
      except Exception as e:
        results.append({"status": "error", "error": str(e)})
        if stop_on_error:
          break
    return results

  async def get_screenshot(self, max_retries: int = 3) -> bytes:
    """Capture a screenshot of the current page.

    This method includes retry logic to handle transient errors that can occur
    during page navigation (e.g., "Execution context was destroyed").

    Args:
      max_retries: Maximum number of retry attempts (default: 3).

    Returns:
      The screenshot as PNG bytes.
    """
    import asyncio

    last_error = None
    for attempt in range(max_retries):
      try:
        response = await self.make_cdp_request(
            _CDP_COMMAND_PAGE_CAPTURE_SCREENSHOT
        )
        return base64.b64decode(response["data"])
      except Exception as e:
        last_error = e
        # Check if it's a transient navigation error
        error_str = str(e).lower()
        if "context was destroyed" in error_str or "navigation" in error_str:
          if attempt < max_retries - 1:
            logger.debug(
                "Retrying get_screenshot after navigation error (attempt %d)",
                attempt + 1,
            )
            await asyncio.sleep(0.5)  # Wait for page to stabilize
            continue
        raise

    # If we exhausted retries, raise the last error
    if last_error:
      raise last_error
    return b""

  async def get_current_url(self, max_retries: int = 3) -> str | None:
    """Get the URL of the currently active tab.

    This method includes retry logic to handle transient errors that can occur
    during page navigation (e.g., "Execution context was destroyed").

    Args:
      max_retries: Maximum number of retry attempts (default: 3).

    Returns:
      The current URL, or None if no active tab.
    """
    import asyncio

    last_error = None
    for attempt in range(max_retries):
      try:
        response = await asyncio.to_thread(
            self._client.agent_engines.sandboxes.send_command,
            http_method="GET",
            path="tabs",
            access_token=self._access_token,
            sandbox_environment=self._sandbox,
        )
        parsed = self._parse_response(response)

        active_tab_id = parsed.get("active_tab_id")
        if active_tab_id is None:
          return None

        for tab in parsed.get("all_tabs", []):
          if tab.get("id") == active_tab_id:
            return tab.get("url")

        return None
      except Exception as e:
        last_error = e
        # Check if it's a transient navigation error
        error_str = str(e).lower()
        if "context was destroyed" in error_str or "navigation" in error_str:
          if attempt < max_retries - 1:
            logger.debug(
                "Retrying get_current_url after navigation error (attempt %d)",
                attempt + 1,
            )
            await asyncio.sleep(0.5)  # Wait for page to stabilize
            continue
        raise

    # If we exhausted retries, raise the last error
    if last_error:
      raise last_error
    return None

  async def navigate(self, url: str) -> dict[str, Any]:
    """Navigate to a URL.

    Args:
      url: The URL to navigate to.

    Returns:
      The CDP response.
    """
    return await self.make_cdp_request(_CDP_COMMAND_PAGE_NAVIGATE, {"url": url})

  async def click_at(self, x: int, y: int) -> None:
    """Click at a specific coordinate.

    Args:
      x: The x-coordinate.
      y: The y-coordinate.
    """
    commands = [
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {
                "type": "mousePressed",
                "button": "left",
                "x": x,
                "y": y,
                "clickCount": 1,
            },
        },
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {
                "type": "mouseReleased",
                "button": "left",
                "x": x,
                "y": y,
                "clickCount": 1,
            },
        },
    ]
    await self.make_cdp_batch_request(commands)

  async def hover_at(self, x: int, y: int) -> None:
    """Hover at a specific coordinate.

    Args:
      x: The x-coordinate.
      y: The y-coordinate.
    """
    await self.make_cdp_request(
        _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
        {"type": "mouseMoved", "x": x, "y": y},
    )

  async def type_text(
      self,
      text: str,
      press_enter: bool = False,
      clear_before_typing: bool = False,
  ) -> None:
    """Type text at the currently focused element.

    Args:
      text: The text to type.
      press_enter: Whether to press Enter after typing.
      clear_before_typing: Whether to clear existing content first.
    """
    commands = []

    if clear_before_typing:
      # Ctrl+A to select all
      commands.extend([
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyDown",
                  "modifiers": 2,  # Ctrl
                  "windowsVirtualKeyCode": 65,  # A
                  "key": "A",
              },
          },
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyUp",
                  "windowsVirtualKeyCode": 65,
                  "key": "A",
              },
          },
          # Delete to clear
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyDown",
                  "windowsVirtualKeyCode": 46,  # Delete
                  "key": "Delete",
              },
          },
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyUp",
                  "windowsVirtualKeyCode": 46,
                  "key": "Delete",
              },
          },
      ])

    if text:
      commands.append({
          "command": _CDP_COMMAND_INPUT_INSERT_TEXT,
          "params": {"text": text},
      })

    if press_enter:
      commands.extend([
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyDown",
                  "windowsVirtualKeyCode": 13,
                  "key": "Enter",
              },
          },
          {
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {
                  "type": "keyUp",
                  "windowsVirtualKeyCode": 13,
                  "key": "Enter",
              },
          },
      ])

    if commands:
      await self.make_cdp_batch_request(commands)

  async def type_text_at(
      self,
      x: int,
      y: int,
      text: str,
      press_enter: bool = False,
      clear_before_typing: bool = False,
  ) -> None:
    """Click at a coordinate and type text.

    Args:
      x: The x-coordinate to click.
      y: The y-coordinate to click.
      text: The text to type.
      press_enter: Whether to press Enter after typing.
      clear_before_typing: Whether to clear existing content first.
    """
    await self.click_at(x, y)
    await self.type_text(text, press_enter, clear_before_typing)

  async def scroll_at(
      self,
      x: int,
      y: int,
      direction: Literal["up", "down", "left", "right"],
      magnitude: int,
  ) -> None:
    """Scroll at a specific coordinate.

    Args:
      x: The x-coordinate.
      y: The y-coordinate.
      direction: The scroll direction.
      magnitude: The scroll amount in pixels.
    """
    direction = direction.lower()
    sign = -1 if direction in ("left", "up") else 1
    delta_x = sign * magnitude if direction in ("left", "right") else 0
    delta_y = sign * magnitude if direction in ("up", "down") else 0

    await self.make_cdp_request(
        _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
        {
            "type": "mouseWheel",
            "x": x,
            "y": y,
            "deltaX": delta_x,
            "deltaY": delta_y,
        },
    )

  async def go_back(self) -> bool:
    """Navigate back in browser history.

    Returns:
      True if navigation was successful, False if at beginning of history.
    """
    response = await self.make_cdp_request(_CDP_COMMAND_PAGE_GET_NAV_HISTORY)
    current_index = response.get("currentIndex", 0)

    if current_index > 0:
      entry_id = response["entries"][current_index - 1]["id"]
      await self.make_cdp_request(
          _CDP_COMMAND_PAGE_NAV_TO_HISTORY, {"entryId": entry_id}
      )
      return True
    return False

  async def go_forward(self) -> bool:
    """Navigate forward in browser history.

    Returns:
      True if navigation was successful, False if at end of history.
    """
    response = await self.make_cdp_request(_CDP_COMMAND_PAGE_GET_NAV_HISTORY)
    current_index = response.get("currentIndex", 0)
    entries = response.get("entries", [])

    if current_index < len(entries) - 1:
      entry_id = entries[current_index + 1]["id"]
      await self.make_cdp_request(
          _CDP_COMMAND_PAGE_NAV_TO_HISTORY, {"entryId": entry_id}
      )
      return True
    return False

  async def key_combination(self, keys: list[str]) -> None:
    """Press a combination of keys.

    Args:
      keys: List of keys to press (e.g., ["control", "c"]).
    """
    commands = []
    modifiers_down = []

    for key in keys:
      upper_key = key.upper()
      is_modifier = upper_key in ("CONTROL", "ALT", "SHIFT", "COMMAND", "SUPER")

      if is_modifier:
        cdp_key = _META_KEY_MAP.get(upper_key, key)
        commands.append({
            "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
            "params": {"type": "keyDown", "key": cdp_key},
        })
        modifiers_down.append(cdp_key)
      elif upper_key in _META_KEY_MAP:
        # Special key like Enter, Backspace
        cdp_key = _META_KEY_MAP[upper_key]
        params_down = {"type": "keyDown", "key": cdp_key}
        params_up = {"type": "keyUp", "key": cdp_key}
        if cdp_key == "Enter":
          params_down["windowsVirtualKeyCode"] = 13
          params_up["windowsVirtualKeyCode"] = 13
        commands.append({
            "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
            "params": params_down,
        })
        commands.append({
            "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
            "params": params_up,
        })
      else:
        # Regular character
        if len(key) == 1:
          commands.append({
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {"type": "keyDown", "text": key},
          })
          commands.append({
              "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
              "params": {"type": "keyUp", "text": key},
          })
        else:
          # Word/sentence - use insertText
          commands.append({
              "command": _CDP_COMMAND_INPUT_INSERT_TEXT,
              "params": {"text": key},
          })

    # Release modifiers in reverse order
    for cdp_key in reversed(modifiers_down):
      commands.append({
          "command": _CDP_COMMAND_INPUT_DISPATCH_KEY_EVENT,
          "params": {"type": "keyUp", "key": cdp_key},
      })

    if commands:
      await self.make_cdp_batch_request(commands)

  async def drag_and_drop(self, x1: int, y1: int, x2: int, y2: int) -> None:
    """Drag from one coordinate to another.

    Args:
      x1: Starting x-coordinate.
      y1: Starting y-coordinate.
      x2: Ending x-coordinate.
      y2: Ending y-coordinate.
    """
    commands = [
        # Move to start position
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {"type": "mouseMoved", "x": x1, "y": y1},
        },
        # Press left mouse button
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {
                "type": "mousePressed",
                "button": "left",
                "x": x1,
                "y": y1,
                "clickCount": 1,
            },
        },
        # Move to end position (drag)
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {"type": "mouseMoved", "x": x2, "y": y2},
        },
        # Release left mouse button
        {
            "command": _CDP_COMMAND_INPUT_DISPATCH_MOUSE_EVENT,
            "params": {
                "type": "mouseReleased",
                "button": "left",
                "x": x2,
                "y": y2,
                "clickCount": 1,
            },
        },
    ]
    await self.make_cdp_batch_request(commands)

  async def health_check(self) -> bool:
    """Check if the sandbox is healthy.

    Returns:
      True if healthy, False otherwise.
    """
    import asyncio

    try:
      response = await asyncio.to_thread(
          self._client.agent_engines.sandboxes.send_command,
          http_method="GET",
          path="",
          access_token=self._access_token,
          sandbox_environment=self._sandbox,
      )
      parsed = self._parse_response(response)
      return parsed.get("status") == "healthy"
    except Exception as e:
      logger.warning("Sandbox health check failed: %s", e)
      return False
