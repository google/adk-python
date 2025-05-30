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

from contextlib import AsyncExitStack
from datetime import timedelta
import functools
import logging
import sys
from typing import Any
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
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

logger = logging.getLogger('google_adk.' + __name__)


class SseServerParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/sse.py
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5
  sse_read_timeout: float = 60 * 5


class StreamableHTTPServerParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/streamable_http.py
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5
  sse_read_timeout: float = 60 * 5
  terminate_on_close: bool = True


def retry_on_closed_resource(async_reinit_func_name: str):
  """Decorator to automatically reinitialize session and retry action.

  When MCP session was closed, the decorator will automatically recreate the
  session and retry the action with the same parameters.

  Note:
  1. async_reinit_func_name is the name of the class member function that
  reinitializes the MCP session.
  2. Both the decorated function and the async_reinit_func_name must be async
  functions.

  Usage:
  class MCPTool:
      ...
      async def create_session(self):
          self.session = ...

      @retry_on_closed_resource('create_session')
      async def use_session(self):
          await self.session.call_tool()

  Args:
      async_reinit_func_name: The name of the async function to recreate session.

  Returns:
      The decorated function.
  """

  def decorator(func):
    @functools.wraps(func)  # Preserves original function metadata
    async def wrapper(self, *args, **kwargs):
      try:
        return await func(self, *args, **kwargs)
      except anyio.ClosedResourceError as close_err:
        try:
          if hasattr(self, async_reinit_func_name) and callable(
              getattr(self, async_reinit_func_name)
          ):
            async_init_fn = getattr(self, async_reinit_func_name)
            await async_init_fn()
          else:
            raise ValueError(
                f'Function {async_reinit_func_name} does not exist in decorated'
                ' class. Please check the function name in'
                ' retry_on_closed_resource decorator.'
            ) from close_err
        except Exception as reinit_err:
          raise RuntimeError(
              f'Error reinitializing: {reinit_err}'
          ) from reinit_err
        return await func(self, *args, **kwargs)

    return wrapper

  return decorator


class MCPSessionManager:
  """Manages MCP client sessions.

  This class provides methods for creating and initializing MCP client sessions,
  handling different connection parameters (Stdio and SSE).
  """

  def __init__(
      self,
      connection_params: Union[
          StdioServerParameters, SseServerParams, StreamableHTTPServerParams
      ],
      errlog: TextIO = sys.stderr,
  ):
    """Initializes the MCP session manager.

    Args:
        connection_params: Parameters for the MCP connection (Stdio, SSE or Streamable HTTP).
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
    """
    self._connection_params = connection_params
    self._errlog = errlog
    # Each session manager maintains its own exit stack for proper cleanup
    self._exit_stack: Optional[AsyncExitStack] = None
    self._session: Optional[ClientSession] = None

  async def create_session(self) -> ClientSession:
    """Creates and initializes an MCP client session.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    if self._session is not None:
      return self._session

    # Create a new exit stack for this session
    self._exit_stack = AsyncExitStack()

    try:
      if isinstance(self._connection_params, StdioServerParameters):
        client = stdio_client(
            server=self._connection_params, errlog=self._errlog
        )
      elif isinstance(self._connection_params, SseServerParams):
        client = sse_client(
            url=self._connection_params.url,
            headers=self._connection_params.headers,
            timeout=self._connection_params.timeout,
            sse_read_timeout=self._connection_params.sse_read_timeout,
        )
      elif isinstance(self._connection_params, StreamableHTTPServerParams):
        client = streamablehttp_client(
            url=self._connection_params.url,
            headers=self._connection_params.headers,
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

      transports = await self._exit_stack.enter_async_context(client)
      # The streamable http client returns a GetSessionCallback in addition to the read/write MemoryObjectStreams
      # needed to build the ClientSession, we limit then to the two first values to be compatible with all clients.
      session = await self._exit_stack.enter_async_context(
          ClientSession(*transports[:2])
      )
      await session.initialize()

      self._session = session
      return session

    except Exception:
      # If session creation fails, clean up the exit stack
      if self._exit_stack:
        await self._exit_stack.aclose()
        self._exit_stack = None
      raise

  async def close(self):
    """Closes the session and cleans up resources."""
    if self._exit_stack:
      try:
        await self._exit_stack.aclose()
      except Exception as e:
        # Log the error but don't re-raise to avoid blocking shutdown
        print(
            f'Warning: Error during MCP session cleanup: {e}', file=self._errlog
        )
      finally:
        self._exit_stack = None
        self._session = None
