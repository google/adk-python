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

import inspect
from typing import Any
from typing import Callable
from typing import Optional

from google.auth.credentials import Credentials
from typing_extensions import override

from ...utils.feature_decorator import experimental
from ..function_tool import FunctionTool
from ..tool_context import ToolContext
from .bigquery_credentials import BigQueryCredentialsConfig
from .bigquery_credentials import BigQueryCredentialsManager
from .config import BigQueryToolConfig


@experimental
class BigQueryTool(FunctionTool):
  """GoogleApiTool class for tools that call Google APIs.

  This class is for developers to handcraft customized Google API tools rather
  than auto generate Google API tools based on API specs.

  This class handles all the OAuth complexity, credential management,
  and common Google API patterns so subclasses can focus on their
  specific functionality.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      credentials_config: Optional[BigQueryCredentialsConfig] = None,
      bigquery_tool_config: Optional[BigQueryToolConfig] = None,
  ):
    """Initialize the Google API tool.

    Args:
        func: callable that impelments the tool's logic, can accept one
          'credential" parameter
        credentials_config: credentials config used to call Google API. If None,
          then we don't hanlde the auth logic
    """
    super().__init__(func=func)
    self._ignore_params.append("credentials")
    self._ignore_params.append("config")
    self._credentials_manager = (
        BigQueryCredentialsManager(credentials_config)
        if credentials_config
        else None
    )
    self._tool_config = (
        bigquery_tool_config if bigquery_tool_config else BigQueryToolConfig()
    )

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    """Main entry point for tool execution with credential handling.

    This method handles all the OAuth complexity and then delegates
    to the subclass's run_async_with_credential method.
    """
    try:
      # Get valid credentials
      credentials = (
          await self._credentials_manager.get_valid_credentials(tool_context)
          if self._credentials_manager
          else None
      )

      if credentials is None and self._credentials_manager:
        # OAuth flow in progress
        return (
            "User authorization is required to access Google services for"
            f" {self.name}. Please complete the authorization flow."
        )

      # Execute the tool's specific logic with valid credentials

      return await self._run_async_with_credential(
          credentials, self._tool_config, args, tool_context
      )

    except Exception as ex:
      return {
          "status": "ERROR",
          "error_details": str(ex),
      }

  async def _run_async_with_credential(
      self,
      credentials: Credentials,
      tool_config: BigQueryToolConfig,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    """Execute the tool's specific logic with valid credentials.

    Args:
        credentials: Valid Google OAuth credentials
        args: Arguments passed to the tool
        tool_context: Tool execution context

    Returns:
        The result of the tool execution
    """
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if "credentials" in signature.parameters:
      args_to_call["credentials"] = credentials
    if "config" in signature.parameters:
      args_to_call["config"] = tool_config
    return await super().run_async(args=args_to_call, tool_context=tool_context)
