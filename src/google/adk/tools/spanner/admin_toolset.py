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

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.spanner import admin_tool
from typing_extensions import override

from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from .settings import SpannerToolSettings
from .spanner_credentials import SpannerCredentialsConfig

DEFAULT_SPANNER_TOOL_NAME_PREFIX = "spanner"


@experimental(FeatureName.SPANNER_ADMIN_TOOLSET)
class SpannerAdminToolset(BaseToolset):
  """A toolset containing tools for interacting with Spanner admin tasks.

  The tool names are:
    - spanner_list_instances
    - spanner_get_instance
    - spanner_create_database
    - spanner_list_databases
    - spanner_create_instance
    - spanner_list_instance_configs
    - spanner_get_instance_config
  """

  def __init__(
      self,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      credentials_config: SpannerCredentialsConfig | None = None,
      spanner_tool_settings: SpannerToolSettings | None = None,
  ):
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_SPANNER_TOOL_NAME_PREFIX,
    )
    self._credentials_config = credentials_config
    self._tool_settings = (
        spanner_tool_settings
        if spanner_tool_settings
        else SpannerToolSettings()
    )

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: ReadonlyContext
  ) -> bool:
    if self.tool_filter is None:
      return True

    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)

    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter

    return False

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Get tools from the toolset."""
    all_tools = [
        GoogleTool(
            func=func,
            credentials_config=self._credentials_config,
            tool_settings=self._tool_settings,
        )
        for func in [
            # Admin tools
            admin_tool.create_database,
            admin_tool.list_instances,
            admin_tool.get_instance,
            admin_tool.list_databases,
            admin_tool.create_instance,
            admin_tool.list_instance_configs,
            admin_tool.get_instance_config,
        ]
    ]

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self):
    pass
