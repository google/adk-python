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
from typing import Any
from typing import AsyncContextManager
from typing import Coroutine
from typing import Optional
from typing import TypeVar

from mcp import ClientSession
from mcp import SamplingCapability
from mcp.client.session import SamplingFnT

from ...features import FeatureName
from ...features import is_feature_enabled

logger = logging.getLogger('google_adk.' + __name__)

_T = TypeVar('_T')


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
      *,
      sampling_callback: Optional[SamplingFnT] = None,
      sampling_capabilities: Optional[SamplingCapability] = None,
  ):
    """
    Args:
        client: An MCP client context manager (e.g., from streamablehttp_client,
            sse_client, or stdio_client).
        timeout: Timeout in seconds for connection and initialization.
        sse_read_timeout: Timeout in seconds for reading data from the MCP SSE
            server.
        is_stdio: Whether this is a stdio connection (affects read timeout).
        sampling_callback: Optional callback to handle sampling requests from the
            MCP server.
        sampling_capabilities: Optional capabilities for sampling.
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
    self._sampling_callback = sampling_callback
    self._sampling_capabilities = sampling_capabilities

  @property
  def session(self) -> Optional[ClientSession]:
    """Get the managed ClientSession, if available."""
    return self._session

  @property
  def _is_task_alive(self) -> bool:
    """Whether the background session task is currently running.

    Returns True only when the task has been started and has not yet completed.
    Returns False if the task has not been started or has finished.
    """
    return self._task is not None and not self._task.done()

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

    # Pre-fix code returned `self._session` here directly (typed as
    # ClientSession even though it could in theory be None). Adding an
    # explicit None check is safer but introduces a new exception path,
    # so we gate it behind the feature flag to keep flag-OFF byte-for-byte
    # compatible with pre-fix behavior.
    if (
        is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING)  # pylint: disable=protected-access
        and self._session is None
    ):
      raise ConnectionError('Failed to create MCP session: unknown error')

    return self._session  # type: ignore[return-value]

  async def _run_guarded(self, coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine while monitoring the background session task.

    Races the given coroutine against the background task. If the task
    dies first (e.g. transport crash from a non-2xx HTTP response), the
    coroutine is cancelled and the original error is raised immediately
    instead of hanging until a read timeout expires.

    Args:
        coro: The coroutine to run (e.g. session.call_tool(...)).

    Returns:
        The result of the coroutine.

    Raises:
        ConnectionError: If the background task has already died or dies
            during execution, wrapping the original exception.
    """
    if self._task is None:
      coro.close()
      raise ConnectionError('MCP session task has not been started')

    if self._task.done():
      exc = self._task.exception() if not self._task.cancelled() else None
      # Close the coroutine to avoid "was never awaited" warnings.
      coro.close()
      raise ConnectionError(
          f'MCP session task has already terminated: {exc}'
      ) from exc

    coro_task = asyncio.ensure_future(coro)

    done, _ = await asyncio.wait(
        [coro_task, self._task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if coro_task in done:
      # If the coroutine itself raised, the exception propagates as-is
      # (not wrapped in ConnectionError). This is intentional so callers
      # can distinguish tool-level errors (McpError) from transport-level
      # crashes (ConnectionError).
      return coro_task.result()

    # The background task finished first, indicating a transport crash.
    # Cancel the in-flight tool call and surface the original error.
    coro_task.cancel()
    try:
      await coro_task
    except BaseException:
      pass

    exc = self._task.exception() if not self._task.cancelled() else None
    raise ConnectionError(f'MCP session connection lost: {exc}') from exc

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
        if is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING):  # pylint: disable=protected-access
          # Post-fix: do NOT wrap in asyncio.wait_for. The MCP client uses
          # AnyIO TaskGroup/CancelScope internally, which must be entered
          # and exited in the same task. asyncio.wait_for runs its target
          # in a nested task and can cancel from a different task on
          # timeout, producing "Attempted to exit cancel scope in a
          # different task" errors. The connection-establishment timeout
          # is still enforced by MCPSessionManager.create_session via its
          # outer asyncio.wait_for around
          # exit_stack.enter_async_context(SessionContext(...)).
          transports = await exit_stack.enter_async_context(self._client)
        else:
          # Pre-fix behavior: wrap with asyncio.wait_for so the inner
          # context entry has its own timeout. Callers that depend on
          # this inner timeout firing rely on this path; without it,
          # mocks that delay `__aenter__` cause tests to time out at the
          # test framework limit instead of the configured per-step timeout.
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
                  sampling_callback=self._sampling_callback,
                  sampling_capabilities=self._sampling_capabilities,
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
                  sampling_callback=self._sampling_callback,
                  sampling_capabilities=self._sampling_capabilities,
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
