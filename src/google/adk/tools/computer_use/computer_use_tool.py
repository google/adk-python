# Copyright 2026 Google LLC
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

from typing_extensions import override

from ...features import experimental
from ...features import FeatureName
from ...models.llm_request import LlmRequest
from ..function_tool import FunctionTool
from ..tool_context import ToolContext
from .base_computer import ComputerState

logger = logging.getLogger("google_adk." + __name__)


@experimental(FeatureName.COMPUTER_USE)
class ComputerUseTool(FunctionTool):
  """A tool that wraps computer control functions for use with LLMs.

  This tool automatically normalizes coordinates from a virtual coordinate space
  (by default 1000x1000) to the actual screen size. This allows LLMs to work
  with a consistent coordinate system regardless of the actual screen
  dimensions, making their output more predictable and easier to handle.
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
      screen_size: The actual screen size as (width, height) in pixels. This
        represents the real dimensions of the target screen/display.
      virtual_screen_size: The virtual coordinate space dimensions as (width,
        height) that the LLM uses to specify coordinates. Coordinates from the
        LLM are automatically normalized from this virtual space to the actual
        screen_size. Default is (1000, 1000), meaning the LLM thinks it's
        working with a 1000x1000 pixel screen regardless of the actual screen
        dimensions.

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
    """Normalizes x coordinate from virtual space to actual screen width."""
    if not isinstance(x, (int, float)):
      raise ValueError(f"x coordinate must be numeric, got {type(x)}")

    normalized = int(x / self._coordinate_space[0] * self._screen_size[0])
    # Clamp to screen bounds
    return max(0, min(normalized, self._screen_size[0] - 1))

  def _normalize_y(self, y: int) -> int:
    """Normalizes y coordinate from virtual space to actual screen height."""
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

    # Check for safety confirmation if required by the model
    if not tool_context.tool_confirmation:
      safety_decision = args.get("safety_decision")
      if safety_decision:
        decision = safety_decision.get("decision")
        explanation = safety_decision.get("explanation")

        if decision == "require_confirmation":
          hint = (
              explanation
              or "This computer use action requires safety confirmation."
          )
          tool_context.request_confirmation(hint=hint)
          tool_context.actions.skip_summarization = True
          return {
              "error": (
                  "This tool call requires confirmation, please approve or"
                  " reject."
              )
          }
    elif not tool_context.tool_confirmation.confirmed:
      return {"error": "This tool call is rejected."}

    try:
      # Normalize coordinates if present. Legacy actions use x/y and
      # destination_x/destination_y; Gemini 3.x drag_and_drop uses
      # start_x/start_y and end_x/end_y.
      for x_field_name in ("x", "destination_x", "start_x", "end_x"):
        if x_field_name in args:
          original_x = args[x_field_name]
          args[x_field_name] = self._normalize_x(original_x)
          logger.debug(
              "Normalized %s: %s -> %s",
              x_field_name,
              original_x,
              args[x_field_name],
          )

      for y_field_name in ("y", "destination_y", "start_y", "end_y"):
        if y_field_name in args:
          original_y = args[y_field_name]
          args[y_field_name] = self._normalize_y(original_y)
          logger.debug(
              "Normalized %s: %s -> %s",
              y_field_name,
              original_y,
              args[y_field_name],
          )

      accepts_start_coords = (
          self._spec.has_signature
          and "start_x" in self._spec.signature.parameters
      )
      if not accepts_start_coords:
        if "start_x" in args and "x" not in args:
          args["x"] = args.pop("start_x")
        if "start_y" in args and "y" not in args:
          args["y"] = args.pop("start_y")
        if "end_x" in args and "destination_x" not in args:
          args["destination_x"] = args.pop("end_x")
        if "end_y" in args and "destination_y" not in args:
          args["destination_y"] = args.pop("end_y")

      # Execute the actual computer control function
      result = await super().run_async(args=args, tool_context=tool_context)

      # Process the result if it's an EnvironmentState
      response = result
      if isinstance(result, ComputerState):
        if result.screenshot is None:
          raise ValueError(
              "A computer action returned a state with no screenshot"
          )
        response = {
            "image": {
                "mimetype": "image/png",
                "data": base64.b64encode(result.screenshot).decode("utf-8"),
            },
            "url": result.url,
        }

      if (
          tool_context.tool_confirmation
          and tool_context.tool_confirmation.confirmed
      ):
        if not isinstance(response, dict):
          response = {"result": response}
        response["safety_acknowledgement"] = "true"

      return response

    except Exception as e:
      logger.error("Error in ComputerUseTool.run_async: %s", e)
      raise

  @override
  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Configures no tools directly; handled by ComputerUseToolset.

    Keep this as a no-op so FunctionTool declarations for both legacy and modern
    computer use aliases are registered in tools_dict/canonical_tools without
    being serialized into llm_request.config.tools.
    """
    pass
