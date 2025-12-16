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

from google.adk.agents.readonly_context import ReadonlyContext
from typing_extensions import override

from . import client
from . import message_tool
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from .config import PubSubToolConfig
from .pubsub_credentials import PubSubCredentialsConfig


@experimental(FeatureName.PUBSUB_TOOLSET)
class PubSubToolset(BaseToolset):
  """Pub/Sub Toolset contains tools for interacting with Pub/Sub topics and subscriptions."""

  def __init__(
      self,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      credentials_config: PubSubCredentialsConfig | None = None,
      pubsub_tool_config: PubSubToolConfig | None = None,
  ):
    """Initializes the PubSubToolset.

    Args:
        tool_filter: A predicate or list of tool names to filter the tools in
          the toolset. If None, all tools are included.
        credentials_config: The credentials configuration to use for
          authenticating with Google Cloud.
        pubsub_tool_config: The configuration for the Pub/Sub tools.
    """
    super().__init__(tool_filter=tool_filter)
    self._credentials_config = credentials_config
    self._tool_settings = (
        pubsub_tool_config if pubsub_tool_config else PubSubToolConfig()
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
            message_tool.publish_message,
            message_tool.pull_messages,
            message_tool.acknowledge_messages,
        ]
    ]

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self):
    """Clean up resources used by the toolset."""
    client.cleanup_clients()
