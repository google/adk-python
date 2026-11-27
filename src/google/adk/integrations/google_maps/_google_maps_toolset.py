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

"""ADK toolset for Google Maps Platform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import override

from . import _tools
from ...agents.readonly_context import ReadonlyContext
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.function_tool import FunctionTool
from ...tools.tool_context import ToolContext
from ._config import default_config
from ._config import GoogleMapsConfig


class GoogleMapsToolset(BaseToolset):
  """Toolset for Street View, Address Validation, and Geocoding.

  The tool names are:
    - get_street_view_image
    - validate_address
    - reverse_geocode
  """

  def __init__(
      self,
      config: GoogleMapsConfig | None = None,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
  ) -> None:
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self.config = config or default_config

  def _get_tool_functions(self) -> list[Callable[..., Any]]:
    """Returns async tool callables bound to ``self.config``."""
    cfg = self.config

    async def get_street_view_image(
        latitude: float,
        longitude: float,
        tool_context: ToolContext,
    ) -> str:
      """Gets an outdoor Street View image of a location.

      Checks for outdoor coverage first, points the camera at the location,
      and saves the image as a session artifact.

      Args:
          latitude: Latitude in decimal degrees.
          longitude: Longitude in decimal degrees.
          tool_context: ADK ToolContext injected by the runner.
      """
      return await _tools.get_street_view_image(
          latitude=latitude,
          longitude=longitude,
          tool_context=tool_context,
          config=cfg,
      )

    async def validate_address(
        raw_address: str,
        tool_context: ToolContext,
    ) -> str:
      """Validates and standardizes a street address.

      Uses the Google Address Validation API. Reports whether the address is
      single-family or multi-unit, and whether a unit number such as an
      apartment or suite is missing.

      Args:
          raw_address: The street address to validate.
          tool_context: ADK ToolContext injected by the runner.
      """
      return await _tools.validate_address(
          raw_address=raw_address,
          tool_context=tool_context,
          config=cfg,
      )

    async def reverse_geocode(
        latitude: float,
        longitude: float,
        tool_context: ToolContext,
    ) -> str:
      """Finds the street address at a latitude and longitude.

      Args:
          latitude: Latitude in decimal degrees.
          longitude: Longitude in decimal degrees.
          tool_context: ADK ToolContext injected by the runner.
      """
      return await _tools.reverse_geocode(
          latitude=latitude,
          longitude=longitude,
          tool_context=tool_context,
          config=cfg,
      )

    return [
        get_street_view_image,
        validate_address,
        reverse_geocode,
    ]

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns ADK ``FunctionTool`` instances selected by ``tool_filter``."""
    all_tools: list[BaseTool] = [
        FunctionTool(func=func) for func in self._get_tool_functions()
    ]
    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self) -> None:
    """Cleans up resources held by the toolset."""
    return None
