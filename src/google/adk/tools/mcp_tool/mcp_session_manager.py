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
import functools
import logging
import sys
from typing import Any
from typing import Optional
from typing import TextIO

import anyio
from pydantic import BaseModel

# Attempt to import MCP and cbor2 at the top level
try:
  from mcp import ClientSession
  from mcp import StdioServerParameters
  from mcp.client.sse import sse_client
  from mcp.client.stdio import stdio_client
  from mcp.shared import message as mcp_messages # Moved here
  import cbor2  # Moved here
  MCP_AND_CBOR2_AVAILABLE = True
except ImportError as e:
  MCP_AND_CBOR2_AVAILABLE = False # Flag to indicate if critical imports failed
  # We will still raise the original ImportError if MCP core components are missing,
  # but log if mcp_messages or cbor2 specifically failed for the StreamWriterWrapper.
  # The original ImportError for ClientSession etc. is more critical.
  if 'mcp.shared.message' in str(e) or 'cbor2' in str(e):
    # Log this specific failure for debugging StreamWriterWrapper context
    # This logger might not be configured yet if this is the first thing failing.
    # Consider a simple print for absolute surety if logger isn't set up.
    # print(f"WARNING: MCPSessionManager: Failed to import mcp.shared.message or cbor2: {e}")
    # Fall-through to the more general ImportError check below.
    pass

  # Original check for critical MCP components
  if sys.version_info < (3, 10):
    raise ImportError(
        'MCP Tool requires Python 3.10 or above. Please upgrade your Python'
        ' version.'
    ) from e
  else:
    # If it's not one of the optional ones and not a Python version issue, re-raise.
    if not ('mcp.shared.message' in str(e) or 'cbor2' in str(e)):
        raise e
    # If only mcp_messages or cbor2 failed, MCP_AND_CBOR2_AVAILABLE will be False,
    # and the StreamWriterWrapper will know not to attempt encoding.

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

  @classmethod
  async def _create_stdio_client(
      cls,
      server: StdioServerParameters,
      errlog: TextIO,
      exit_stack: AsyncExitStack,
  ) -> tuple[Any, Optional[asyncio.subprocess.Process]]:
    """Create stdio client and return the client and potentially a process."""
    process = None
    client = None
    current_loop = None
    current_policy = None

    try:
        current_loop = asyncio.get_running_loop()
        current_policy = asyncio.get_event_loop_policy()
        logger.info(
            "ADK_WEB_DEBUG: mcp_session_manager._create_stdio_client - "
            f"Before subprocess creation: Loop type: {type(current_loop)}, Policy type: {type(current_policy)}"
        )
    except Exception as e_log:
        logger.error(f"ADK_WEB_DEBUG: Error getting loop/policy info: {e_log}")

    if sys.platform == "win32":
        logger.info(
            "ADK_WEB_DEBUG: mcp_session_manager._create_stdio_client - Windows platform detected. "
            f"Attempting direct asyncio.create_subprocess_exec for command: {server.command} {' '.join(server.args)}"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                server.command,
                *server.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=errlog,
            )
            logger.info(f"ADK_WEB_DEBUG: Successfully created subprocess with PID {process.pid} on Windows.")

            if not (process and process.stdin and process.stdout):
                raise Exception("Subprocess stdin/stdout not available after creation on Windows direct path")

            # Check if necessary modules for StreamWriterWrapper were loaded at the top level
            if not MCP_AND_CBOR2_AVAILABLE:
                logger.error("StreamWriterWrapper cannot function: mcp.shared.message or cbor2 failed to import at module level.")
                # Fallback or raise an error, as the wrapper won't work correctly.
                # For now, let's attempt fallback to stdio_client if these are missing,
                # as the custom path will fail.
                raise ImportError("Missing dependencies for StreamWriterWrapper (mcp.shared.message or cbor2)")

            # Wrapper for asyncio.StreamWriter to provide a .send() method
            class StreamWriterWrapper:
                def __init__(self, writer: asyncio.StreamWriter):
                    self._writer = writer
                
                async def send(self, data: Any): # data is likely SessionMessage from MCP
                    logger.debug(f"StreamWriterWrapper: send called with data of type {type(data)}")
                    encoded_data: bytes
                    if isinstance(data, bytes):
                        encoded_data = data
                    elif isinstance(data, mcp_messages.SessionMessage): # mcp_messages should be available from top-level import
                        try:
                            if not hasattr(data, 'model_dump') or not callable(data.model_dump):
                                logger.error(f"StreamWriterWrapper: SessionMessage of type {type(data)} does not have model_dump.")
                                raise TypeError(f"SessionMessage type {type(data)} cannot be auto-encoded as it lacks model_dump.")
                            json_model = data.model_dump(mode='json')
                            encoded_data = cbor2.dumps(json_model) # cbor2 should be available from top-level import
                            logger.debug(f"StreamWriterWrapper: Encoded SessionMessage to {len(encoded_data)} CBOR bytes.")
                        except Exception as e_encode:
                            logger.error(f"StreamWriterWrapper: Failed to encode SessionMessage ({type(data)}): {e_encode}", exc_info=True)
                            raise TypeError(f"StreamWriterWrapper: Could not encode {type(data)} to bytes.") from e_encode
                    else:
                        logger.error(f"StreamWriterWrapper: Received unencodable/unexpected type {type(data)}. Expecting bytes or mcp.shared.message.SessionMessage.")
                        raise TypeError(f"StreamWriterWrapper: Data must be bytes or an encodable mcp.shared.message.SessionMessage, got {type(data)}.")

                    self._writer.write(encoded_data) # This must receive bytes.
                    await self._writer.drain()
                    logger.debug(f"StreamWriterWrapper: send completed for data of type {type(data)} (encoded to {len(encoded_data)} bytes of type {type(encoded_data)})")
                
                def can_write_eof(self) -> bool:
                    return self._writer.can_write_eof()

                def write_eof(self):
                    return self._writer.write_eof()

                def close(self):
                    return self._writer.close()

                async def wait_closed(self):
                    return await self._writer.wait_closed()
                
                def get_extra_info(self, name, default=None):
                    return self._writer.get_extra_info(name, default)
                
                # Add other necessary StreamWriter methods if MCP requires them
                # For now, __getattr__ can be a fallback but explicit is better if known.
                # def __getattr__(self, name):
                #     # Delegate other attributes to the original writer
                #     return getattr(self._writer, name)

            @asynccontextmanager
            async def manual_stdio_transport(proc_stdout, proc_stdin_writer):
                logger.info("ADK_WEB_DEBUG: manual_stdio_transport entering with proc_stdout and proc_stdin_writer...")
                wrapped_writer = StreamWriterWrapper(proc_stdin_writer)
                try:
                    yield (proc_stdout, wrapped_writer) # (reader, wrapped_writer)
                finally:
                    logger.info("ADK_WEB_DEBUG: manual_stdio_transport exiting...")
            
            client = manual_stdio_transport(process.stdout, process.stdin)
            logger.info(f"ADK_WEB_DEBUG: Created manual_stdio_transport with StreamWriterWrapper for PID {process.pid} on Windows.")

        except NotImplementedError as e_ni:
            logger.error(
                f"ADK_WEB_DEBUG: mcp_session_manager._create_stdio_client - Windows - "
                f"asyncio.create_subprocess_exec failed with NotImplementedError: {e_ni}. "
                f"Loop type: {type(current_loop)}, Policy type: {type(current_policy)}",
                exc_info=True
            )
            raise # Re-raise to indicate critical failure
        except Exception as e_subproc:
            logger.error(
                f"ADK_WEB_DEBUG: mcp_session_manager._create_stdio_client - Windows - "
                f"Error during direct asyncio.create_subprocess_exec: {e_subproc}. Falling back.",
                exc_info=True
            )
            client = stdio_client(server=server, errlog=errlog)
            process = None 
            logger.info("ADK_WEB_DEBUG: Successfully used mcp.client.stdio.stdio_client as fallback on Windows.")
    else:
        logger.info(
            "ADK_WEB_DEBUG: mcp_session_manager._create_stdio_client - Non-Windows platform. "
            f"Using mcp.client.stdio.stdio_client for command: {server.command} {' '.join(server.args)}"
        )
        client = stdio_client(server=server, errlog=errlog)
        process = None 
        logger.info("ADK_WEB_DEBUG: Successfully used mcp.client.stdio.stdio_client on non-Windows platform.")

    if client is None:
        raise RuntimeError("ADK_WEB_DEBUG: MCP Client (stdio_client or manual wrapper) was not initialized in _create_stdio_client")

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
