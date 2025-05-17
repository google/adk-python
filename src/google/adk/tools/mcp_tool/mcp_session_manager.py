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

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import functools
import logging
import sys
from typing import Any, Optional, TextIO
import anyio
from pydantic import BaseModel

try:
  from mcp import ClientSession, StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
except ImportError as e:
  import sys

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    raise e

logger = logging.getLogger(__name__)


class SseServerParams(BaseModel):
  """Parameters for the MCP SSE connection.

  See MCP SSE Client documentation for more details.
  https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/sse.py
  """

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5
  sse_read_timeout: float = 60 * 5


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
    @functools.wraps(
        func
    )  # Preserves original function metadata (name, docstring)
    async def wrapper(self, *args, **kwargs):
      try:
        return await func(self, *args, **kwargs)
      except anyio.ClosedResourceError:
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
            )
        except Exception as reinit_err:
          raise RuntimeError(
              f'Error reinitializing: {reinit_err}'
          ) from reinit_err
        return await func(self, *args, **kwargs)

    return wrapper

  return decorator


@asynccontextmanager
async def tracked_stdio_client(server, errlog, process=None):
  """A wrapper around stdio_client that ensures proper process tracking and cleanup."""
  our_process = process

  # If no process was provided, create one
  if our_process is None:
    our_process = await asyncio.create_subprocess_exec(
        server.command,
        *server.args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=errlog,
    )

  # Use the original stdio_client, but ensure process cleanup
  try:
    async with stdio_client(server=server, errlog=errlog) as client:
      yield client, our_process
  finally:
    # Ensure the process is properly terminated if it still exists
    if our_process and our_process.returncode is None:
      try:
        logger.info(
            f'Terminating process {our_process.pid} from tracked_stdio_client'
        )
        our_process.terminate()
        try:
          await asyncio.wait_for(our_process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
          # Force kill if it doesn't terminate quickly
          if our_process.returncode is None:
            logger.warning(f'Forcing kill of process {our_process.pid}')
            our_process.kill()
      except ProcessLookupError:
        # Process already gone, that's fine
        logger.info(f'Process {our_process.pid} already terminated')


class MCPSessionManager:
  """Manages MCP client sessions.

  This class provides methods for creating and initializing MCP client sessions,
  handling different connection parameters (Stdio and SSE).
  """

  def __init__(
      self,
      connection_params: StdioServerParameters | SseServerParams,
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> ClientSession:
    """Initializes the MCP session manager.

    Example usage:
    ```
    mcp_session_manager = MCPSessionManager(
        connection_params=connection_params,
        exit_stack=exit_stack,
    )
    session = await mcp_session_manager.create_session()
    ```

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE).
        exit_stack: AsyncExitStack to manage the session lifecycle.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
    """

    self._connection_params = connection_params
    self._exit_stack = exit_stack
    self._errlog = errlog
    self._process = None  # Track the subprocess
    self._active_processes = set()  # Track all processes created
    self._active_file_handles = set()  # Track file handles

  async def create_session(
      self,
  ) -> tuple[ClientSession, Optional[asyncio.subprocess.Process]]:
    """Creates a new MCP session and tracks the associated process."""
    session, process = await self._initialize_session(
        connection_params=self._connection_params,
        exit_stack=self._exit_stack,
        errlog=self._errlog,
    )
    self._process = process  # Store reference to process

    # Track the process
    if process:
      self._active_processes.add(process)

    return session, process

  @classmethod
  async def _initialize_session(
      cls,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> tuple[ClientSession, Optional[asyncio.subprocess.Process]]:
    """Initializes an MCP client session.

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE).
        exit_stack: AsyncExitStack to manage the session lifecycle.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    process = None

    if isinstance(connection_params, StdioServerParameters):
      # For stdio connections, we need to track the subprocess
      client, process = await cls._create_stdio_client(
          server=connection_params,
          errlog=errlog,
          exit_stack=exit_stack,
      )
    elif isinstance(connection_params, SseServerParams):
      # For SSE connections, create the client without a subprocess
      client = sse_client(
          url=connection_params.url,
          headers=connection_params.headers,
          timeout=connection_params.timeout,
          sse_read_timeout=connection_params.sse_read_timeout,
      )
    else:
      raise ValueError(
          'Unable to initialize connection. Connection should be'
          ' StdioServerParameters or SseServerParams, but got'
          f' {connection_params}'
      )

    # Create the session with the client
    transports = await exit_stack.enter_async_context(client)
    session = await exit_stack.enter_async_context(ClientSession(*transports))
    await session.initialize()

    return session, process

  @staticmethod
  async def _create_stdio_client(
      server: StdioServerParameters,
      errlog: TextIO,
      exit_stack: AsyncExitStack,
  ) -> tuple[Any, asyncio.subprocess.Process]:
    """Create stdio client and return both the client and process.

    This implementation adapts to how the MCP stdio_client is created.
    The actual implementation may need to be adjusted based on the MCP library
    structure.
    """
    # Create the subprocess directly so we can track it
    process = await asyncio.create_subprocess_exec(
        server.command,
        *server.args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=errlog,
    )

    # Create the stdio client using the MCP library
    try:
      # Method 1: Try using the existing process if stdio_client supports it
      client = stdio_client(server=server, errlog=errlog, process=process)
    except TypeError:
      # Method 2: If the above doesn't work, let stdio_client create its own process
      # and we'll need to terminate both processes later
      logger.warning(
          'Using stdio_client with its own process - may lead to duplicate'
          ' processes'
      )
      client = stdio_client(server=server, errlog=errlog)

    return client, process

  async def _emergency_cleanup(self):
    """Perform emergency cleanup of resources when normal cleanup fails."""
    logger.info('Performing emergency cleanup of MCPSessionManager resources')

    # Clean up any tracked processes
    for proc in list(self._active_processes):
      try:
        if proc and proc.returncode is None:
          logger.info(f'Emergency termination of process {proc.pid}')
          proc.terminate()
          try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
          except asyncio.TimeoutError:
            logger.warning(f"Process {proc.pid} didn't terminate, forcing kill")
            proc.kill()
        self._active_processes.remove(proc)
      except Exception as e:
        logger.error(f'Error during process cleanup: {e}')

    # Clean up any tracked file handles
    for handle in list(self._active_file_handles):
      try:
        if not handle.closed:
          logger.info('Closing file handle')
          handle.close()
        self._active_file_handles.remove(handle)
      except Exception as e:
        logger.error(f'Error closing file handle: {e}')
