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

import asyncio
import base64
import inspect
import logging
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import Union
import warnings

from fastapi.openapi.models import APIKeyIn
from google.genai.types import FunctionDeclaration
from mcp.shared.session import ProgressFnT
from mcp.types import Tool as McpBaseTool
from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from ...auth.auth_tool import AuthConfig
from ...features import FeatureName
from ...features import is_feature_enabled
from .._gemini_schema_util import _to_gemini_schema
from ..base_authenticated_tool import BaseAuthenticatedTool
from ..tool_context import ToolContext
from .mcp_session_manager import MCPSessionManager
from .mcp_session_manager import retry_on_errors

logger = logging.getLogger("google_adk." + __name__)


@runtime_checkable
class ProgressCallbackFactory(Protocol):
  """Factory protocol for creating per-tool progress callbacks.

  This protocol allows users to create different progress callbacks for
  different tools based on tool name and runtime context. The factory receives
  the tool name, a CallbackContext for accessing and modifying session state,
  and additional keyword arguments for forward compatibility.

  Example usage::

    def my_callback_factory(
        tool_name: str,
        *,
        callback_context: CallbackContext | None = None,
        **kwargs
    ) -> ProgressFnT | None:
      session_id = callback_context.session.id if callback_context else "N/A"

      async def callback(progress, total, message):
        print(f"[{tool_name}] Session {session_id}: {progress}/{total}")
        # Can modify state in the callback
        if callback_context:
          callback_context.state['last_progress'] = progress

      return callback

    toolset = McpToolset(
        connection_params=...,
        progress_callback=my_callback_factory,
    )

  Note:
    The **kwargs parameter is required for forward compatibility. Future
    versions may pass additional parameters. Implementations should accept
    **kwargs even if they don't use them.
  """

  def __call__(
      self,
      tool_name: str,
      *,
      callback_context: Optional[CallbackContext] = None,
      **kwargs: Any,
  ) -> Optional[ProgressFnT]:
    """Create a progress callback for a specific tool.

    Args:
      tool_name: The name of the MCP tool.
      callback_context: The callback context providing access to session,
        state, artifacts, and other runtime information. Allows modifying
        state via ctx.state['key'] = value. May be None if not available.
      **kwargs: Additional keyword arguments for future extensibility.
        Implementations should accept **kwargs for forward compatibility.

    Returns:
      A progress callback function, or None if no callback is needed
      for this tool.
    """
    ...


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
      progress_callback: Optional[
          Union[ProgressFnT, ProgressCallbackFactory]
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
        header_provider: Optional function to provide dynamic headers.
        progress_callback: Optional callback to receive progress notifications
          from MCP server during long-running tool execution. Can be either:

          - A ``ProgressFnT`` callback that receives (progress, total, message).
            This callback will be used for all invocations.

          - A ``ProgressCallbackFactory`` that creates per-invocation callbacks.
            The factory receives (tool_name, callback_context, **kwargs) and
            returns a ProgressFnT or None. This allows callbacks to access
            and modify runtime context like session state.

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
    self._progress_callback = progress_callback

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Gets the function declaration for the tool.

    Returns:
        FunctionDeclaration: The Gemini function declaration for the tool.
    """
    input_schema = self._mcp_tool.inputSchema
    output_schema = self._mcp_tool.outputSchema
    if is_feature_enabled(FeatureName.JSON_SCHEMA_FOR_FUNC_DECL):
      function_decl = FunctionDeclaration(
          name=self.name,
          description=self.description,
          parameters_json_schema=input_schema,
          response_json_schema=output_schema,
      )
    else:
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
    # Callable's __call__ function is a coroutine function
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
      args_to_call = args.copy()
      try:
        signature = inspect.signature(self._require_confirmation)
        valid_params = set(signature.parameters.keys())
        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

        if "tool_context" in valid_params or has_kwargs:
          args_to_call["tool_context"] = tool_context

        # Filter args_to_call only if there's no **kwargs
        if not has_kwargs:
          # Add tool_context to valid_params if it was added to args_to_call
          if "tool_context" in args_to_call:
            valid_params.add("tool_context")
          args_to_call = {
              k: v for k, v in args_to_call.items() if k in valid_params
          }
      except ValueError:
        args_to_call = args

      require_confirmation = await self._invoke_callable(
          self._require_confirmation, args_to_call
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

  @retry_on_errors
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

    # Resolve progress callback (may be a factory that needs runtime context)
    resolved_callback = self._resolve_progress_callback(tool_context)

    response = await session.call_tool(
        self._mcp_tool.name,
        arguments=args,
        progress_callback=resolved_callback,
    )
    return response.model_dump(exclude_none=True, mode="json")

  def _resolve_progress_callback(
      self, tool_context: ToolContext
  ) -> Optional[ProgressFnT]:
    """Resolve the progress callback for the current invocation.

    If progress_callback is a ProgressCallbackFactory, call it to create
    a callback with runtime context. Otherwise, return the callback directly.

    Args:
      tool_context: The tool context for the current invocation.

    Returns:
      The resolved progress callback, or None if not configured.
    """
    if (
        not hasattr(self, "_progress_callback")
        or self._progress_callback is None
    ):
      return None

    # Determine if callback is a factory by checking if it's a coroutine
    # function. ProgressFnT is an async function, while ProgressCallbackFactory
    # is a sync function that returns an async function.
    if asyncio.iscoroutinefunction(self._progress_callback):
      return self._progress_callback

    # If it's a regular callable (not async), treat it as a factory
    if callable(self._progress_callback) and not inspect.iscoroutinefunction(
        self._progress_callback
    ):
      return self._progress_callback(self.name, callback_context=tool_context)

    return self._progress_callback

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
        ValueError: If API key authentication is configured for non-header
        location.
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
                  f"{credential.http.scheme}"
                  f" {credential.http.credentials.token}"
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
