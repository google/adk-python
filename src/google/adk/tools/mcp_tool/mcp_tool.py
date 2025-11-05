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

import base64
import inspect
import logging
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union
import warnings

from fastapi.openapi.models import APIKeyIn
from google.genai.types import FunctionDeclaration
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from .._gemini_schema_util import _to_gemini_schema
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp.types import Tool as McpBaseTool
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        "MCP Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e


from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ...auth.auth_tool import AuthConfig
from ..base_authenticated_tool import BaseAuthenticatedTool
#  import
from ..tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


class McpTool(BaseAuthenticatedTool):
  """Turns an MCP Tool into an ADK Tool.

  Internally, the tool initializes from a MCP Tool, and uses the MCP Session to
  call the tool.

  Note: For API key authentication, only header-based API keys are supported.
  Query and cookie-based API keys will result in authentication errors.
  """

  def __init__(
      self,
      *,
      mcp_tool: McpBaseTool,
      mcp_session_manager: MCPSessionManager,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      require_confirmation: Union[bool, Callable[..., bool]] = False,
      header_provider: Optional[
          Callable[[ReadonlyContext], Dict[str, str]]
      ] = None,
  ):
    """Initializes an McpTool.

    This tool wraps an MCP Tool interface and uses a session manager to
    communicate with the MCP server.

    Args:
        mcp_tool: The MCP tool to wrap.
        mcp_session_manager: The MCP session manager to use for communication.
        auth_scheme: The authentication scheme to use.
        auth_credential: The authentication credential to use.
        require_confirmation: Whether this tool requires confirmation. A boolean
          or a callable that takes the function's arguments and returns a
          boolean. If the callable returns True, the tool will require
          confirmation from the user.

    Raises:
        ValueError: If mcp_tool or mcp_session_manager is None.
    """
    super().__init__(
        name=mcp_tool.name,
        description=mcp_tool.description if mcp_tool.description else "",
        auth_config=AuthConfig(
            auth_scheme=auth_scheme, raw_auth_credential=auth_credential
        )
        if auth_scheme
        else None,
    )
    self._mcp_tool = mcp_tool
    self._mcp_session_manager = mcp_session_manager
    self._require_confirmation = require_confirmation
    self._header_provider = header_provider

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    input_schema = self._mcp_tool.inputSchema
    parameters = _to_gemini_schema(input_schema)
    function_decl = FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters=parameters,
    )
    return function_decl

  @property
  def raw_mcp_tool(self) -> McpBaseTool:
    """Returns the raw MCP tool."""
    return self._mcp_tool

  async def _invoke_callable(
      self, target: Callable[..., Any], args_to_call: dict[str, Any]
  ) -> Any:
    """Invokes a callable, handling both sync and async cases."""

    # Functions are callable objects, but not all callable objects are functions
    # checking coroutine function is not enough. We also need to check whether
    # Callable's __call__ function is a coroutine funciton
    is_async = inspect.iscoroutinefunction(target) or (
        hasattr(target, "__call__")
        and inspect.iscoroutinefunction(target.__call__)
    )
    if is_async:
      return await target(**args_to_call)
    else:
      return target(**args_to_call)

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    if isinstance(self._require_confirmation, Callable):
      require_confirmation = await self._invoke_callable(
          self._require_confirmation, args
      )
    else:
      require_confirmation = bool(self._require_confirmation)

    if require_confirmation:
      if not tool_context.tool_confirmation:
        args_to_show = args.copy()
        if "tool_context" in args_to_show:
          args_to_show.pop("tool_context")

        tool_context.request_confirmation(
            hint=(
                f"Please approve or reject the tool call {self.name}() by"
                " responding with a FunctionResponse with an expected"
                " ToolConfirmation payload."
            ),
        )
        return {
            "error": (
                "This tool call requires confirmation, please approve or"
                " reject."
            )
        }
      elif not tool_context.tool_confirmation.confirmed:
        return {"error": "This tool call is rejected."}
    return await super().run_async(args=args, tool_context=tool_context)

  @retry_on_closed_resource
  @override
  async def _run_async_impl(
      self, *, args, tool_context: ToolContext, credential: AuthCredential
  ) -> Dict[str, Any]:
    """Runs the tool asynchronously.

    Args:
        args: The arguments as a dict to pass to the tool.
        tool_context: The tool context of the current invocation.

    Returns:
        Any: The response from the tool.
    """
    # Extract headers from credential for session pooling
    auth_headers = await self._get_headers(tool_context, credential)
    dynamic_headers = None
    if self._header_provider:
      dynamic_headers = self._header_provider(
          ReadonlyContext(tool_context._invocation_context)
      )

    headers: Dict[str, str] = {}
    if auth_headers:
      headers.update(auth_headers)
    if dynamic_headers:
      headers.update(dynamic_headers)
    final_headers = headers if headers else None

    # Get the session from the session manager
    session = await self._mcp_session_manager.create_session(
        headers=final_headers
    )

    response = await session.call_tool(self._mcp_tool.name, arguments=args)
    return response.model_dump(exclude_none=True, mode="json")

  async def _get_headers(
      self, tool_context: ToolContext, credential: AuthCredential
  ) -> Optional[dict[str, str]]:
    """Extracts authentication headers from credentials.

    Args:
        tool_context: The tool context of the current invocation.
        credential: The authentication credential to process.

    Returns:
        Dictionary of headers to add to the request, or None if no auth.

    Raises:
        ValueError: If API key authentication is configured for non-header location.
    """
    headers: Optional[dict[str, str]] = None
    if credential:
      if credential.oauth2:
        headers = {"Authorization": f"Bearer {credential.oauth2.access_token}"}
      elif credential.http:
        # Handle HTTP authentication schemes
        if (
            credential.http.scheme.lower() == "bearer"
            and credential.http.credentials.token
        ):
          headers = {
              "Authorization": f"Bearer {credential.http.credentials.token}"
          }
        elif credential.http.scheme.lower() == "basic":
          # Handle basic auth
          if (
              credential.http.credentials.username
              and credential.http.credentials.password
          ):

            credentials = f"{credential.http.credentials.username}:{credential.http.credentials.password}"
            encoded_credentials = base64.b64encode(
                credentials.encode()
            ).decode()
            headers = {"Authorization": f"Basic {encoded_credentials}"}
        elif credential.http.credentials.token:
          # Handle other HTTP schemes with token
          headers = {
              "Authorization": (
                  f"{credential.http.scheme} {credential.http.credentials.token}"
              )
          }
      elif credential.api_key:
        if (
            not self._credentials_manager
            or not self._credentials_manager._auth_config
        ):
          error_msg = (
              "Cannot find corresponding auth scheme for API key credential"
              f" {credential}"
          )
          logger.error(error_msg)
          raise ValueError(error_msg)
        elif (
            self._credentials_manager._auth_config.auth_scheme.in_
            != APIKeyIn.header
        ):
          error_msg = (
              "McpTool only supports header-based API key authentication."
              " Configured location:"
              f" {self._credentials_manager._auth_config.auth_scheme.in_}"
          )
          logger.error(error_msg)
          raise ValueError(error_msg)
        else:
          headers = {
              self._credentials_manager._auth_config.auth_scheme.name: (
                  credential.api_key
              )
          }
      elif credential.service_account:
        # Service accounts should be exchanged for access tokens before reaching this point
        logger.warning(
            "Service account credentials should be exchanged before MCP"
            " session creation"
        )

    return headers


class MCPTool(McpTool):
  """Deprecated name, use `McpTool` instead."""

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "MCPTool class is deprecated, use `McpTool` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)
