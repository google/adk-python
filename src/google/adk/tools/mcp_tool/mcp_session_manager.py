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

import asyncio
from contextlib import AsyncExitStack
from datetime import timedelta
import functools
import hashlib
import json
import logging
import sys
from typing import Any
from typing import Dict
from typing import Optional
from typing import TextIO
from typing import Union

import anyio
from pydantic import BaseModel

try:
  from mcp import ClientSession
  from mcp import StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
  from mcp.client.streamable_http import streamablehttp_client
except ImportError as e:

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

logger = logging.getLogger('google_adk.' + __name__)


class StdioConnectionParams(BaseModel):
  """Parameters for the MCP Stdio connection.

  Attributes:
      server_params: Parameters for the MCP Stdio server.
      timeout: Timeout in seconds for establishing the connection to the MCP
        stdio server.
  """

  server_params: StdioServerParameters
  timeout: float = 5.0


class SseConnectionParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/sse.py

  Attributes:
      url: URL for the MCP SSE server.
      headers: Headers for the MCP SSE connection.
      timeout: Timeout in seconds for establishing the connection to the MCP SSE
        server.
      sse_read_timeout: Timeout in seconds for reading data from the MCP SSE
        server.
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5.0
  sse_read_timeout: float = 60 * 5.0


class StreamableHTTPConnectionParams(BaseModel):
  """Parameters for the MCP Streamable HTTP connection.

  See MCP Streamable HTTP Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py

  Attributes:
      url: URL for the MCP Streamable HTTP server.
      headers: Headers for the MCP Streamable HTTP connection.
      timeout: Timeout in seconds for establishing the connection to the MCP
        Streamable HTTP server.
      sse_read_timeout: Timeout in seconds for reading data from the MCP
        Streamable HTTP server.
      terminate_on_close: Whether to terminate the MCP Streamable HTTP server
        when the connection is closed.
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5.0
  sse_read_timeout: float = 60 * 5.0
  terminate_on_close: bool = True


def retry_on_closed_resource(func):
  """Decorator to automatically retry action when MCP session is closed.

  When MCP session was closed, the decorator will automatically retry the
  action once. The create_session method will handle creating a new session
  if the old one was disconnected.

  Args:
      func: The function to decorate.

  Returns:
      The decorated function.
  """

  @functools.wraps(func)  # Preserves original function metadata
  async def wrapper(self, *args, **kwargs):
    try:
      return await func(self, *args, **kwargs)
    except (anyio.ClosedResourceError, anyio.BrokenResourceError):
      # If the session connection is closed or unusable, we will retry the
      # function to reconnect to the server. create_session will handle
      # detecting and replacing disconnected sessions.
      logger.info('Retrying %s due to closed resource', func.__name__)
      return await func(self, *args, **kwargs)

  return wrapper


class MCPSessionManager:
  """Manages MCP client sessions.

  This class provides methods for creating and initializing MCP client sessions,
  handling different connection parameters (Stdio and SSE) and supporting
  session pooling based on authentication headers.
  """

  def __init__(
      self,
      connection_params: Union[
          StdioServerParameters,
          StdioConnectionParams,
          SseConnectionParams,
          StreamableHTTPConnectionParams,
      ],
      errlog: TextIO = sys.stderr,
  ):
    """Initializes the MCP session manager.

    Args:
        connection_params: Parameters for the MCP connection (Stdio, SSE or
          Streamable HTTP). Stdio by default also has a 5s read timeout as other
          parameters but it's not configurable for now.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
    """
    if isinstance(connection_params, StdioServerParameters):
      # So far timeout is not configurable. Given MCP is still evolving, we
      # would expect stdio_client to evolve to accept timeout parameter like
      # other client.
      logger.warning(
          'StdioServerParameters is not recommended. Please use'
          ' StdioConnectionParams.'
      )
      self._connection_params = StdioConnectionParams(
          server_params=connection_params,
          timeout=5,
      )
    else:
      self._connection_params = connection_params
    self._errlog = errlog

    # Session pool: maps session keys to (session, exit_stack) tuples
    self._sessions: Dict[str, tuple[ClientSession, AsyncExitStack]] = {}

    # Lock to prevent race conditions in session creation
    self._session_lock = asyncio.Lock()

  def _generate_session_key(
      self, merged_headers: Optional[Dict[str, str]] = None
  ) -> str:
    """Generates a session key based on connection params and merged headers.

    For StdioConnectionParams, returns a constant key since headers are not
    supported. For SSE and StreamableHTTP connections, generates a key based
    on the provided merged headers.

    Args:
        merged_headers: Already merged headers (base + additional).

    Returns:
        A unique session key string.
    """
    if isinstance(self._connection_params, StdioConnectionParams):
      # For stdio connections, headers are not supported, so use constant key
      return 'stdio_session'

    # For SSE and StreamableHTTP connections, use merged headers
    if merged_headers:
      headers_json = json.dumps(merged_headers, sort_keys=True)
      headers_hash = hashlib.md5(headers_json.encode()).hexdigest()
      return f'session_{headers_hash}'
    else:
      return 'session_no_headers'

  def _merge_headers(
      self, additional_headers: Optional[Dict[str, str]] = None
  ) -> Optional[Dict[str, str]]:
    """Merges base connection headers with additional headers.

    Args:
        additional_headers: Optional headers to merge with connection headers.

    Returns:
        Merged headers dictionary, or None if no headers are provided.
    """
    if isinstance(self._connection_params, StdioConnectionParams) or isinstance(
        self._connection_params, StdioServerParameters
    ):
      # Stdio connections don't support headers
      return None

    base_headers = {}
    if (
        hasattr(self._connection_params, 'headers')
        and self._connection_params.headers
    ):
      base_headers = self._connection_params.headers.copy()

    if additional_headers:
      base_headers.update(additional_headers)

    return base_headers

  def _is_session_disconnected(self, session: ClientSession) -> bool:
    """Checks if a session is disconnected or closed.

    Args:
        session: The ClientSession to check.

    Returns:
        True if the session is disconnected, False otherwise.
    """
    return session._read_stream._closed or session._write_stream._closed

  def _create_client(self, merged_headers: Optional[Dict[str, str]] = None):
    """Creates an MCP client based on the connection parameters.

    Args:
        merged_headers: Optional headers to include in the connection.
                       Only applicable for SSE and StreamableHTTP connections.

    Returns:
        The appropriate MCP client instance.

    Raises:
        ValueError: If the connection parameters are not supported.
    """
    if isinstance(self._connection_params, StdioConnectionParams):
      client = stdio_client(
          server=self._connection_params.server_params,
          errlog=self._errlog,
      )
    elif isinstance(self._connection_params, SseConnectionParams):
      client = sse_client(
          url=self._connection_params.url,
          headers=merged_headers,
          timeout=self._connection_params.timeout,
          sse_read_timeout=self._connection_params.sse_read_timeout,
      )
    elif isinstance(self._connection_params, StreamableHTTPConnectionParams):
      client = streamablehttp_client(
          url=self._connection_params.url,
          headers=merged_headers,
          timeout=timedelta(seconds=self._connection_params.timeout),
          sse_read_timeout=timedelta(
              seconds=self._connection_params.sse_read_timeout
          ),
          terminate_on_close=self._connection_params.terminate_on_close,
      )
    else:
      raise ValueError(
          'Unable to initialize connection. Connection should be'
          ' StdioServerParameters or SseServerParams, but got'
          f' {self._connection_params}'
      )
    return client

  async def create_session(
      self, headers: Optional[Dict[str, str]] = None
  ) -> ClientSession:
    """Creates and initializes an MCP client session.

    This method will check if an existing session for the given headers
    is still connected. If it's disconnected, it will be cleaned up and
    a new session will be created.

    Args:
        headers: Optional headers to include in the session. These will be
                merged with any existing connection headers. Only applicable
                for SSE and StreamableHTTP connections.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    # Merge headers once at the beginning
    merged_headers = self._merge_headers(headers)

    # Generate session key using merged headers
    session_key = self._generate_session_key(merged_headers)

    # Use async lock to prevent race conditions
    async with self._session_lock:
      # Check if we have an existing session
      if session_key in self._sessions:
        session, exit_stack = self._sessions[session_key]

        # Check if the existing session is still connected
        if not self._is_session_disconnected(session):
          # Session is still good, return it
          return session
        else:
          # Session is disconnected, clean it up
          logger.info('Cleaning up disconnected session: %s', session_key)
          try:
            await exit_stack.aclose()
          except Exception as e:
            logger.warning('Error during disconnected session cleanup: %s', e)
          finally:
            del self._sessions[session_key]

      # Create a new session (either first time or replacing disconnected one)
      exit_stack = AsyncExitStack()
      timeout_in_seconds = (
          self._connection_params.timeout
          if hasattr(self._connection_params, 'timeout')
          else None
      )

      try:
        client = self._create_client(merged_headers)

        transports = await asyncio.wait_for(
            exit_stack.enter_async_context(client),
            timeout=timeout_in_seconds,
        )
        # The streamable http client returns a GetSessionCallback in addition to the
        # read/write MemoryObjectStreams needed to build the ClientSession, we limit
        # then to the two first values to be compatible with all clients.
        if isinstance(self._connection_params, StdioConnectionParams):
          session = await exit_stack.enter_async_context(
              ClientSession(
                  *transports[:2],
                  read_timeout_seconds=timedelta(seconds=timeout_in_seconds),
              )
          )
        else:
          session = await exit_stack.enter_async_context(
              ClientSession(*transports[:2])
          )
        await asyncio.wait_for(session.initialize(), timeout=timeout_in_seconds)

        # Store session and exit stack in the pool
        self._sessions[session_key] = (session, exit_stack)
        logger.debug('Created new session: %s', session_key)
        return session

      except Exception as e:
        # If session creation fails, clean up the exit stack
        if exit_stack:
          try:
            await exit_stack.aclose()
          except Exception as exit_stack_error:
            logger.warning(
                'Error during session creation cleanup: %s', exit_stack_error
            )
        raise ConnectionError(f'Failed to create MCP session: {e}') from e

  async def close(self):
    """Closes all sessions and cleans up resources."""
    async with self._session_lock:
      for session_key in list(self._sessions.keys()):
        _, exit_stack = self._sessions[session_key]
        try:
          await exit_stack.aclose()
        except Exception as e:
          # Log the error but don't re-raise to avoid blocking shutdown
          print(
              'Warning: Error during MCP session cleanup for'
              f' {session_key}: {e}',
              file=self._errlog,
          )
        finally:
          del self._sessions[session_key]


SseServerParams = SseConnectionParams

StreamableHTTPServerParams = StreamableHTTPConnectionParams
