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
import logging
from typing import Optional

from fastapi.openapi.models import APIKeyIn
from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .._gemini_schema_util import _to_gemini_schema
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_closed_resource

# Attempt to import MCP Tool from the MCP library, and hints user to upgrade
# their Python version to 3.10 if it fails.
try:
  from mcp.types import Tool as McpBaseTool
except ImportError as e:
  import sys

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


class MCPTool(BaseAuthenticatedTool):
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
  ):
    """Initializes an MCPTool.

    This tool wraps an MCP Tool interface and uses a session manager to
    communicate with the MCP server.

    Args:
        mcp_tool: The MCP tool to wrap.
        mcp_session_manager: The MCP session manager to use for communication.
        auth_scheme: The authentication scheme to use.
        auth_credential: The authentication credential to use.

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

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    schema_dict = self._mcp_tool.inputSchema
    parameters = _to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @retry_on_closed_resource
  @override
  async def _run_async_impl(
      self, *, args, tool_context: ToolContext, credential: AuthCredential
  ):
    """Runs the tool asynchronously.

    Args:
        args: The arguments as a dict to pass to the tool.
        tool_context: The tool context of the current invocation.

    Returns:
        Any: The response from the tool.
    """
    # Extract headers from credential for session pooling
    headers = await self._get_headers(tool_context, credential)

    # Get the session from the session manager
    session = await self._mcp_session_manager.create_session(headers=headers)

    response = await session.call_tool(self.name, arguments=args)
    return response

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
              "MCPTool only supports header-based API key authentication."
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
