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

from typing import Any
from typing import Callable
from typing import List

from google.adk.agents.readonly_context import ReadonlyContext
from typing_extensions import override

from . import data_agent_tool
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from .config import DataAgentToolConfig
from .credentials import DataAgentCredentialsConfig


class DataAgentToolset(BaseToolset):
  """Data Agent Toolset contains tools for interacting with data agents."""

  def __init__(
      self,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      credentials_config: DataAgentCredentialsConfig | None = None,
      data_agent_tool_config: DataAgentToolConfig | None = None,
  ):
    super().__init__(tool_filter=tool_filter)
    self._credentials_config = credentials_config
    self._tool_settings = (
        data_agent_tool_config
        if data_agent_tool_config
        else DataAgentToolConfig()
    )

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: ReadonlyContext | None
  ) -> bool:
    # Unlike the base implementation, an empty tool_filter selects no tools.
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
  ) -> List[BaseTool]:
    funcs: list[Callable[..., Any]] = [
        data_agent_tool.list_accessible_data_agents,
        data_agent_tool.get_data_agent_info,
        data_agent_tool.ask_data_agent,
    ]
    if self._tool_settings.enable_data_agent_modification:
      funcs.append(data_agent_tool.create_data_agent)
      funcs.append(data_agent_tool.delete_data_agent)
      funcs.append(data_agent_tool.update_data_agent)

    all_tools = [
        GoogleTool(
            func=func,
            credentials_config=self._credentials_config,
            tool_settings=self._tool_settings,
        )
        for func in funcs
    ]

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self) -> None:
    pass
