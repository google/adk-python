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

"""Toolset definition for Eventarc."""

from __future__ import annotations

from typing import Any

from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.google_tool import GoogleTool
from ._config import EventarcCredentialsConfig
from ._config import EventarcToolConfig
from ._message_tool import publish_message


@experimental(FeatureName.EVENTARC_TOOLSET)
class EventarcToolset(BaseToolset):
  """Toolset for interacting with Google Cloud Eventarc."""

  def __init__(
      self,
      *,
      tool_config: EventarcToolConfig | None = None,
      credentials_config: EventarcCredentialsConfig | None = None,
      tool_name_prefix: str | None = None,
      **kwargs: Any,
  ):
    """Initializes the Eventarc toolset.

    Args:
        tool_config: Configuration for the Eventarc tool.
        credentials_config: Configuration for Google Cloud credentials.
        tool_name_prefix: Prefix to apply to the tool name.
        **kwargs: Additional arguments passed to the base class.
    """
    super().__init__(tool_name_prefix=tool_name_prefix, **kwargs)
    self.tool_config = tool_config or EventarcToolConfig()
    self.credentials_config = credentials_config or EventarcCredentialsConfig()

    self._publish_message_tool = GoogleTool(
        func=publish_message,
        credentials_config=self.credentials_config,
        tool_settings=self.tool_config,
    )
    self._tools = [self._publish_message_tool]

  async def get_tools(self, readonly_context: Any = None) -> list[BaseTool]:
    """Returns the list of enabled tools in this toolset.

    Args:
        readonly_context: Context to determine if tool is selected.

    Returns:
        A list of BaseTool objects.
    """
    return [
        tool
        for tool in self._tools
        if self._is_tool_selected(tool, readonly_context)
    ]
