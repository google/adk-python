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

from typing import List
from typing import Optional
from typing import Union

from google.adk.agents.readonly_context import ReadonlyContext
from typing_extensions import override

from . import metadata_tool
from . import query_tool
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from ...utils.feature_decorator import experimental
from .settings import Capabilities
from .settings import SpannerToolSettings
from .spanner_credentials import SpannerCredentialsConfig

DEFAULT_SPANNER_TOOL_NAME_PREFIX = "spanner"


@experimental
class SpannerToolset(BaseToolset):
  """Spanner Toolset contains tools for interacting with Spanner data, database and table information.

  The tool names are:
    - spanner_list_table_names
    - spanner_list_table_indexes
    - spanner_list_table_index_columns
    - spanner_list_named_schemas
    - spanner_get_table_schema
    - spanner_execute_sql
  """

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      credentials_config: Optional[SpannerCredentialsConfig] = None,
      spanner_tool_settings: Optional[SpannerToolSettings] = None,
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
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[BaseTool]:
    """Get tools from the toolset."""
    all_tools = [
        GoogleTool(
            func=func,
            credentials_config=self._credentials_config,
            tool_settings=self._tool_settings,
        )
        for func in [
            # Metadata tools
            metadata_tool.list_table_names,
            metadata_tool.list_table_indexes,
            metadata_tool.list_table_index_columns,
            metadata_tool.list_named_schemas,
            metadata_tool.get_table_schema,
        ]
    ]

    # Query tools
    if (
        self._tool_settings
        and Capabilities.DATA_READ in self._tool_settings.capabilities
    ):
      all_tools.append(
          GoogleTool(
              func=query_tool.execute_sql,
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
          )
      )

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self):
    pass
