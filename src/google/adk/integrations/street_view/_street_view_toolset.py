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

"""Configurable ADK BaseToolset for Google Street View and Maps Platform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.function_tool import FunctionTool
from ...tools.tool_context import ToolContext
from ._config import default_config
from ._config import GoogleMapsConfig
from ._tools import _execute_property_classification
from ._tools import _execute_reverse_geocode
from ._tools import _execute_street_view_fallback


class StreetViewToolset(BaseToolset):
  """ADK Toolset bundling Street View, Address Validation, and Geocoding tools.

  The tool names are:
    - street_view_fallback_tool
    - property_classification_tool
    - reverse_geocode_tool
  """

  def __init__(
      self,
      config: GoogleMapsConfig | None = None,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      tool_name_prefix: str | None = None,
      include_street_view: bool = True,
      include_property_classification: bool = True,
      include_reverse_geocode: bool = True,
  ) -> None:
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
    self.config = config or default_config
    self.include_street_view = include_street_view
    self.include_property_classification = include_property_classification
    self.include_reverse_geocode = include_reverse_geocode

  def get_tool_functions(self) -> list[Callable[..., Any]]:
    """Returns async tool callables bound to ``self.config``."""
    cfg = self.config
    funcs: list[Callable[..., Any]] = []

    if self.include_street_view:

      async def street_view_fallback_tool(
          latitude: float,
          longitude: float,
          tool_context: ToolContext,
      ) -> str:
        """Queries Google Street View for an outdoor visual baseline.

        Use this tool when no historical ground-truth photos exist for an
        address (cold-start). Performs a pre-flight metadata check before
        downloading imagery and saves the resulting panorama into ADK session
        artifacts and state.

        Args:
            latitude: Target property latitude in decimal degrees.
            longitude: Target property longitude in decimal degrees.
            tool_context: ADK ToolContext injected by the runner.
        """
        return await _execute_street_view_fallback(
            latitude=latitude,
            longitude=longitude,
            tool_context=tool_context,
            config=cfg,
        )

      funcs.append(street_view_fallback_tool)

    if self.include_property_classification:

      async def property_classification_tool(
          raw_address: str,
          tool_context: ToolContext,
      ) -> str:
        """Standardizes an address and classifies SINGLE_FAMILY vs MULTI_UNIT.

        Detects omitted apartment/suite numbers via the Google Address
        Validation API and stores the property classification in ADK session
        state to set spatial decision boundaries.

        Args:
            raw_address: Raw customer or recipient street address.
            tool_context: ADK ToolContext injected by the runner.
        """
        return await _execute_property_classification(
            raw_address=raw_address,
            tool_context=tool_context,
            config=cfg,
        )

      funcs.append(property_classification_tool)

    if self.include_reverse_geocode:

      async def reverse_geocode_tool(
          latitude: float,
          longitude: float,
          tool_context: ToolContext,
      ) -> str:
        """Resolves GPS telemetry coordinates into a physical street address.

        Saves the resolved street context into ADK session state for comparison
        against the intended recipient address.

        Args:
            latitude: GPS latitude in decimal degrees.
            longitude: GPS longitude in decimal degrees.
            tool_context: ADK ToolContext injected by the runner.
        """
        return await _execute_reverse_geocode(
            latitude=latitude,
            longitude=longitude,
            tool_context=tool_context,
            config=cfg,
        )

      funcs.append(reverse_geocode_tool)

    return funcs

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Returns ADK ``FunctionTool`` instances selected by ``tool_filter``."""
    all_tools: list[BaseTool] = [
        FunctionTool(func=func) for func in self.get_tool_functions()
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
