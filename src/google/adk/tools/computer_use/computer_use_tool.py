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

import base64
import logging
from typing import Any
from typing import Callable

from google.genai import types
from typing_extensions import override

from ...models.llm_request import LlmRequest
from ...utils.feature_decorator import experimental
from ..function_tool import FunctionTool
from ..tool_context import ToolContext
from .base_computer import ComputerState

logger = logging.getLogger("google_adk." + __name__)


@experimental
class ComputerUseTool(FunctionTool):
  """A tool that wraps computer control functions for use with LLMs.

  This tool automatically normalizes coordinates from a virtual coordinate space
  (by default 1000x1000) to the actual screen size. This allows LLMs to work
  with a consistent coordinate system regardless of the actual screen dimensions,
  making their output more predictable and easier to handle.
  """

  def __init__(
      self,
      *,
      func: Callable[..., Any],
      screen_size: tuple[int, int],
      virtual_screen_size: tuple[int, int] = (1000, 1000),
  ):
    """Initialize the ComputerUseTool.

    Args:
      func: The computer control function to wrap.
      screen_size: The actual screen size as (width, height) in pixels.
        This represents the real dimensions of the target screen/display.
      virtual_screen_size: The virtual coordinate space dimensions as (width, height)
        that the LLM uses to specify coordinates. Coordinates from the LLM are
        automatically normalized from this virtual space to the actual screen_size.
        Default is (1000, 1000), meaning the LLM thinks it's working with a
        1000x1000 pixel screen regardless of the actual screen dimensions.

    Raises:
      ValueError: If screen_size or virtual_screen_size is not a valid tuple
        of positive integers.
    """
    super().__init__(func=func)
    self._screen_size = screen_size
    self._coordinate_space = virtual_screen_size

    # Validate screen size
    if not isinstance(screen_size, tuple) or len(screen_size) != 2:
      raise ValueError("screen_size must be a tuple of (width, height)")
    if screen_size[0] <= 0 or screen_size[1] <= 0:
      raise ValueError("screen_size dimensions must be positive")

    # Validate virtual screen size
    if (
        not isinstance(virtual_screen_size, tuple)
        or len(virtual_screen_size) != 2
    ):
      raise ValueError("virtual_screen_size must be a tuple of (width, height)")
    if virtual_screen_size[0] <= 0 or virtual_screen_size[1] <= 0:
      raise ValueError("virtual_screen_size dimensions must be positive")

  def _normalize_x(self, x: int) -> int:
    """Normalize x coordinate from virtual screen space to actual screen width."""
    if not isinstance(x, (int, float)):
      raise ValueError(f"x coordinate must be numeric, got {type(x)}")

    normalized = int(x / self._coordinate_space[0] * self._screen_size[0])
    # Clamp to screen bounds
    return max(0, min(normalized, self._screen_size[0] - 1))

  def _normalize_y(self, y: int) -> int:
    """Normalize y coordinate from virtual screen space to actual screen height."""
    if not isinstance(y, (int, float)):
      raise ValueError(f"y coordinate must be numeric, got {type(y)}")

    normalized = int(y / self._coordinate_space[1] * self._screen_size[1])
    # Clamp to screen bounds
    return max(0, min(normalized, self._screen_size[1] - 1))

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Run the computer control function with normalized coordinates."""

    try:
      # Normalize coordinates if present
      if "x" in args:
        original_x = args["x"]
        args["x"] = self._normalize_x(args["x"])
        logger.debug("Normalized x: %s -> %s", original_x, args["x"])

      if "y" in args:
        original_y = args["y"]
        args["y"] = self._normalize_y(args["y"])
        logger.debug("Normalized y: %s -> %s", original_y, args["y"])

      # Handle destination coordinates for drag and drop
      if "destination_x" in args:
        original_dest_x = args["destination_x"]
        args["destination_x"] = self._normalize_x(args["destination_x"])
        logger.debug(
            "Normalized destination_x: %s -> %s",
            original_dest_x,
            args["destination_x"],
        )

      if "destination_y" in args:
        original_dest_y = args["destination_y"]
        args["destination_y"] = self._normalize_y(args["destination_y"])
        logger.debug(
            "Normalized destination_y: %s -> %s",
            original_dest_y,
            args["destination_y"],
        )

      # Execute the actual computer control function
      result = await super().run_async(args=args, tool_context=tool_context)

      # Process the result if it's an EnvironmentState
      if isinstance(result, ComputerState):
        return {
            "image": {
                "mimetype": "image/png",
                "data": base64.b64encode(result.screenshot).decode("utf-8"),
            },
            "url": result.url,
        }

      return result

    except Exception as e:
      logger.error("Error in ComputerUseTool.run_async: %s", e)
      raise

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """ComputerUseToolset will add this tool to the LLM request and add computer
    use configuration to the LLM request."""
    pass
