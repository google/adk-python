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
from contextlib import asynccontextmanager

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

# Attempt to import MCP and cbor2 at the top level
try:
  from mcp import ClientSession
  from mcp import StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
  from mcp.shared import message as mcp_messages
  # Removed MCPJsonEncoder import as it caused ModuleNotFoundError
  import cbor2
  MCP_DEPENDENCIES_AVAILABLE = True

except ImportError as e:
  MCP_DEPENDENCIES_AVAILABLE = False
  # Simplified atexit_ensure_closeddependency check as mcp.common.json was problematic
  if 'mcp.shared.message' in str(e) or 'cbor2' in str(e):
    pass

  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    if not ('mcp.shared.message' in str(e) or 'cbor2' in str(e)):
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
      connection_params: Union[
          StdioServerParameters, SseServerParams, StreamableHTTPServerParams
      ],
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ):
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
        connection_params: Parameters for the MCP connection (Stdio, SSE or
          Streamable HTTP). Stdio by default also has a 5s read timeout as other
          parameters but it's not configurable for now.
          
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
      connection_params: Union[
        StdioServerParameters, SseServerParams, StreamableHTTPServerParams
      ],
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> tuple[ClientSession, Optional[asyncio.subprocess.Process]]:
    """Initializes an MCP client session.

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE pr StreamableHTTP).
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
     elif isinstance(connection_params, StreamableHTTPServerParams):
       client = streamablehttp_client(
           url=connection_params.url,
           headers=connection_params.headers,
           timeout=timedelta(seconds=connection_params.timeout),
           sse_read_timeout=timedelta(
               seconds=connection_params.sse_read_timeout
           ),
           terminate_on_close=connection_params.terminate_on_close,
       )
    else:
      raise ValueError(
          'Unable to initialize connection. Connection should be'
          ' StdioServerParameters or SseServerParams, or StreamableHTTPServerParams but got'
          f' {connection_params}'
      )

    if self._session is not None:
      return self._session

    # Create a new exit stack for this session
    self._exit_stack = AsyncExitStack()

    try:
      if isinstance(self._connection_params, StdioServerParameters):
        # So far timeout is not configurable. Given MCP is still evolving, we
        # would expect stdio_client to evolve to accept timeout parameter like
        # other client.
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

    # Create the session with the client
    transports = await exit_stack.enter_async_context(client)
    session = await exit_stack.enter_async_context(ClientSession(*transports[:2]))
    await session.initialize()

    return session, process

  @classmethod
  async def _create_stdio_client(
      cls,
      server: StdioServerParameters,
      errlog: TextIO,
      exit_stack: AsyncExitStack, # exit_stack is not used by stdio_client directly but kept for signature consistency if needed later
  ) -> tuple[Any, Optional[asyncio.subprocess.Process]]:
    """Create stdio client and return the client and potentially a process."""
    # Simplified: Always use mcp.client.stdio.stdio_client for stdio connections
    # This avoids the complexities and brittleness of the manual StreamWriterWrapper path.
    # stdio_client itself handles subprocess creation and management if not given one.
    
    # stdio_client is an async context manager that yields the client and manages the process.
    # To fit the expected return type (client, process), we need to manage the process separately if stdio_client doesn't return it.
    # However, mcp.client.stdio.stdio_client *does* manage its own process internally if not given one.
    # For simplicity and robustness, we let stdio_client manage its process.
    # The `tracked_stdio_client` context manager can wrap this if finer-grained process tracking by ADK is still desired.
    # For now, return None for the process, as stdio_client handles it internally.
    
    # The stdio_client itself is the async context manager yielding the (reader, writer) transport tuple.
    # We return the stdio_client instance itself, which ClientSession will then enter.
    client = stdio_client(server=server, errlog=errlog)
    process = None # stdio_client manages its own process
    
    # This entire block is replaced by the simpler stdio_client usage above.
    # if sys.platform == "win32":
    #     ...
    # else:
    #     ...

    if client is None: # Should not happen if stdio_client() call is successful
        raise RuntimeError("MCP Client (stdio_client) was not initialized in _create_stdio_client")

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
