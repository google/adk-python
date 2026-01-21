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
import logging
from typing import AsyncContextManager
from typing import Optional

from mcp import ClientSession

logger = logging.getLogger('google_adk.' + __name__)


class SessionContext:
  """Represents the context of a single MCP session within a dedicated task.

  AnyIO's TaskGroup/CancelScope requires that the start and end of a scope
  occur within the same task. Since MCP clients use AnyIO internally, we need
  to ensure that the client's entire lifecycle (creation, usage, and cleanup)
  happens within a single dedicated task.

  This class spawns a background task that:
  1. Enters the MCP client's async context and initializes the session
  2. Signals readiness via an asyncio.Event
  3. Waits for a close signal
  4. Cleans up the client within the same task

  This ensures CancelScope constraints are satisfied regardless of which
  task calls start() or close().

  Can be used in two ways:
  1. Direct method calls: start() and close()
  2. As an async context manager: async with lifecycle as session: ...
  """

  def __init__(
      self,
      client: AsyncContextManager,
      timeout: Optional[float],
      sse_read_timeout: Optional[float],
      is_stdio: bool = False,
  ):
    """
    Args:
        client: An MCP client context manager (e.g., from streamablehttp_client,
            sse_client, or stdio_client).
        timeout: Timeout in seconds for connection and initialization.
        sse_read_timeout: Timeout in seconds for reading data from the MCP SSE
            server.
        is_stdio: Whether this is a stdio connection (affects read timeout).
    """
    self._client = client
    self._timeout = timeout
    self._sse_read_timeout = sse_read_timeout
    self._is_stdio = is_stdio
    self._session: Optional[ClientSession] = None
    self._ready_event = asyncio.Event()
    self._close_event = asyncio.Event()
    self._task: Optional[asyncio.Task] = None
    self._task_lock = asyncio.Lock()

  @property
  def session(self) -> Optional[ClientSession]:
    """Get the managed ClientSession, if available."""
    return self._session

  async def start(self) -> ClientSession:
    """Start the runner and wait for the session to be ready.

    Returns:
        The initialized ClientSession.

    Raises:
        ConnectionError: If session creation fails.
    """
    async with self._task_lock:
      if self._session:
        logger.debug(
            'Session has already been created, returning existing session'
        )
        return self._session

      if self._close_event.is_set():
        raise ConnectionError(
            'Failed to create MCP session: session already closed'
        )

      if not self._task:
        self._task = asyncio.create_task(self._run())

    await self._ready_event.wait()

    if self._task.cancelled():
      raise ConnectionError('Failed to create MCP session: task cancelled')

    if self._task.done() and self._task.exception():
      raise ConnectionError(
          f'Failed to create MCP session: {self._task.exception()}'
      ) from self._task.exception()

    return self._session

  async def close(self):
    """Signal the context task to close and wait for cleanup."""
    # Set the close event to signal the task to close.
    # Even if start has not been called, we need to set the close event
    # to signal the task to close right away.
    async with self._task_lock:
      self._close_event.set()

    # If start has not been called, only set the close event and return
    if not self._task:
      return

    if not self._ready_event.is_set():
      self._task.cancel()

    try:
      await asyncio.wait_for(self._task, timeout=self._timeout)
    except asyncio.TimeoutError:
      logger.warning('Failed to close MCP session: task timed out')
      self._task.cancel()
    except asyncio.CancelledError:
      pass
    except Exception as e:
      logger.warning(f'Failed to close MCP session: {e}')

  async def __aenter__(self) -> ClientSession:
    return await self.start()

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()

  async def _run(self):
    """Run the complete session context within a single task."""
    try:
      async with AsyncExitStack() as exit_stack:
        transports = await asyncio.wait_for(
            exit_stack.enter_async_context(self._client),
            timeout=self._timeout,
        )
        # The streamable http client returns a GetSessionCallback in addition
        # to the read/write MemoryObjectStreams needed to build the
        # ClientSession. We limit to the first two values to be compatible
        # with all clients.
        if self._is_stdio:
          session = await exit_stack.enter_async_context(
              ClientSession(
                  *transports[:2],
                  read_timeout_seconds=timedelta(seconds=self._timeout)
                  if self._timeout is not None
                  else None,
              )
          )
        else:
          # For SSE and Streamable HTTP clients, use the sse_read_timeout
          # instead of the connection timeout as the read_timeout for the session.
          session = await exit_stack.enter_async_context(
              ClientSession(
                  *transports[:2],
                  read_timeout_seconds=timedelta(seconds=self._sse_read_timeout)
                  if self._sse_read_timeout is not None
                  else None,
              )
          )
        await asyncio.wait_for(session.initialize(), timeout=self._timeout)
        logger.debug('Session has been successfully initialized')

        self._session = session
        self._ready_event.set()

        # Wait for close signal - the session remains valid while we wait
        await self._close_event.wait()
    except BaseException as e:
      logger.warning(f'Error on session runner task: {e}')
      raise
    finally:
      self._ready_event.set()
      self._close_event.set()
