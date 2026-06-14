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

from typing_extensions import override

from . import storage_tool
from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from .gcs_credentials import GCSCredentialsConfig
from .settings import Capabilities
from .settings import GCSToolSettings

DEFAULT_GCS_TOOL_NAME_PREFIX = "gcs"


@experimental(FeatureName.GCS_TOOLSET)
class GCSToolset(BaseToolset):
  """GCS Toolset contains tools for interacting with GCS storage.

  The tool names are:
    - get_bucket
    - create_object
    - get_object_data
    - get_object_metadata
    - list_objects
    - delete_objects
  """

  def __init__(
      self,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      credentials_config: GCSCredentialsConfig | None = None,
      gcs_tool_settings: GCSToolSettings | None = None,
  ):
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_GCS_TOOL_NAME_PREFIX,
    )
    self._credentials_config = credentials_config
    self._tool_settings = (
        gcs_tool_settings if gcs_tool_settings else GCSToolSettings()
    )

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Get tools from the toolset."""
    all_tools = []

    if self._tool_settings and (
        Capabilities.READ_ONLY in self._tool_settings.capabilities
        or Capabilities.READ_WRITE in self._tool_settings.capabilities
    ):
      all_tools.extend([
          GoogleTool(
              func=func,
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
          )
          for func in [
              storage_tool.get_bucket,
              storage_tool.get_object_data,
              storage_tool.get_object_metadata,
              storage_tool.list_objects,
          ]
      ])

    if (
        self._tool_settings
        and Capabilities.READ_WRITE in self._tool_settings.capabilities
    ):
      all_tools.extend([
          GoogleTool(
              func=func,
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
          )
          for func in [
              storage_tool.create_object,
              storage_tool.delete_objects,
          ]
      ])

    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]
