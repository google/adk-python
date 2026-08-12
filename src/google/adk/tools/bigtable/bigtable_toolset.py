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

import inspect
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

from google.adk.agents.readonly_context import ReadonlyContext
from google.auth.credentials import Credentials
from pydantic import BaseModel
from typing_extensions import override

from . import metadata_tool
from . import query_tool
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from ..tool_context import ToolContext
from .bigtable_credentials import BigtableCredentialsConfig
from .settings import BigtableToolSettings


class BigtableParameterizedViewTool(GoogleTool):
  """Wrapper FunctionTool for Bigtable execute_sql query tool that passes view parameters.

  This tool wraps the Bigtable query tool to automatically resolve and inject
  parameters from the ToolContext (e.g. user_id) into the query's
  view_parameters. The parameter names to resolve are configured via
  view_parameter_names.

  Example:
      If a parameterized view `purchase_history_pv` was created with the query:
      `SELECT * FROM purchases WHERE user_id = VIEW_PARAMETERS('user_id')`

      By configuring `view_parameter_names=["user_id"]`, the wrapper will
      resolve the `user_id` value from the `tool_context.user_id` at runtime and
      pass it as `view_parameters={"user_id": user_id}`.
      This securely restricts query execution to the logged-in user's data
      without exposing the `user_id` parameter to the LLM.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      credentials_config: Optional[BigtableCredentialsConfig] = None,
      tool_settings: Optional[BigtableToolSettings] = None,
      view_parameter_names: Optional[List[str]] = None,
  ):
    """Initializes the BigtableParameterizedViewTool.

    Args:
        func: The Bigtable query function to wrap.
        credentials_config: The credentials configuration.
        tool_settings: The tool settings.
        view_parameter_names: A list of parameter names to resolve from
          tool_context and pass into view_parameters. This is configured on the
          toolset (BigtableToolset) and forwarded here.
    """
    super().__init__(
        func=func,
        credentials_config=credentials_config,
        tool_settings=tool_settings,
    )
    self.name = "execute_sql_parameterized"
    self.description = (
        "Execute a GoogleSQL query from a Bigtable table using parameterized"
        " views to securely check permissions."
    )
    self.view_parameter_names = view_parameter_names
    # Exclude from being parsed and exposed to the LLM when generating tool schemas
    self._ignore_params.append("_view_parameters")

  @override
  async def _run_async_with_credential(
      self,
      credentials: Credentials,
      tool_settings: BaseModel,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if "_view_parameters" in signature.parameters and self.view_parameter_names:
      view_params = {}
      for param_name in self.view_parameter_names:
        # 1. Check if it's a strongly-typed top-level property (like 'user_id')
        if (val := getattr(tool_context, param_name, None)) is not None:
          view_params[param_name] = val
        # 2. Fallback to checking application-level session state
        elif tool_context.state and param_name in tool_context.state:
          view_params[param_name] = tool_context.state[param_name]

      args_to_call["_view_parameters"] = view_params
    return await super()._run_async_with_credential(
        credentials, tool_settings, args_to_call, tool_context
    )


DEFAULT_BIGTABLE_TOOL_NAME_PREFIX = "bigtable"


@experimental(FeatureName.BIGTABLE_TOOLSET)
class BigtableToolset(BaseToolset):
  """Bigtable Toolset contains tools for interacting with Bigtable data and metadata.

  The tool names are:
    - bigtable_list_instances
    - bigtable_get_instance_info
    - bigtable_list_tables
    - bigtable_get_table_info
    - bigtable_list_clusters
    - bigtable_get_cluster_info
    - bigtable_execute_sql
  """

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      credentials_config: Optional[BigtableCredentialsConfig] = None,
      bigtable_tool_settings: Optional[BigtableToolSettings] = None,
      view_parameter_names: Optional[List[str]] = None,
  ):
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_BIGTABLE_TOOL_NAME_PREFIX,
    )
    self._credentials_config = credentials_config
    self._tool_settings = (
        bigtable_tool_settings
        if bigtable_tool_settings
        else BigtableToolSettings()
    )
    self.view_parameter_names = view_parameter_names

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
            metadata_tool.list_instances,
            metadata_tool.get_instance_info,
            metadata_tool.list_tables,
            metadata_tool.get_table_info,
            metadata_tool.list_clusters,
            metadata_tool.get_cluster_info,
            query_tool.execute_sql,
        ]
    ]
    if self.view_parameter_names:
      all_tools.append(
          BigtableParameterizedViewTool(
              func=query_tool.execute_sql,
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
              view_parameter_names=self.view_parameter_names,
          )
      )
    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]
