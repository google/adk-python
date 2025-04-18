from contextlib import AsyncExitStack
import functools
import sys
from typing import Any, TextIO
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
    self.connection_params = connection_params
    self.exit_stack = exit_stack
    self.errlog = errlog

  async def create_session(self) -> ClientSession:
    return await MCPSessionManager.initialize_session(
        connection_params=self.connection_params,
        exit_stack=self.exit_stack,
        errlog=self.errlog,
    )

  @classmethod
  async def initialize_session(
      cls,
      *,
      connection_params: StdioServerParameters | SseServerParams,
      exit_stack: AsyncExitStack,
      errlog: TextIO = sys.stderr,
  ) -> ClientSession:
    """Initializes an MCP client session.

    Args:
        connection_params: Parameters for the MCP connection (Stdio or SSE).
        exit_stack: AsyncExitStack to manage the session lifecycle.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.

    Returns:
        ClientSession: The initialized MCP client session.
    """
    if isinstance(connection_params, StdioServerParameters):
      client = stdio_client(server=connection_params, errlog=errlog)
    elif isinstance(connection_params, SseServerParams):
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

    transports = await exit_stack.enter_async_context(client)
    session = await exit_stack.enter_async_context(ClientSession(*transports))
    await session.initialize()
    return session
