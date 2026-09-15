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
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
from datetime import timedelta
import logging
from types import TracebackType
from typing import Any
from typing import Coroutine
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import TypeVar

import anyio

from ...dependencies._mcp import ClientSession
from ...dependencies._mcp import ElicitationFnT
from ...dependencies._mcp import IS_MCP_SDK_V2
from ...dependencies._mcp import McpError
from ...dependencies._mcp import NotificationBinding
from ...dependencies._mcp import ResultClaim
from ...dependencies._mcp import SamplingCapability
from ...dependencies._mcp import SamplingFnT
from ...features import FeatureName
from ...features import is_feature_enabled

logger = logging.getLogger('google_adk.' + __name__)

_T = TypeVar('_T')

# The SDK's own ceiling for one `server/discover` probe. Used here as the cap
# on the whole negotiation budget, so a slow probe cannot starve the
# `initialize()` fallback that follows it.
_DISCOVER_TIMEOUT_SECONDS = 10.0


def _read_timeout(seconds: Optional[float]) -> Optional[float | timedelta]:
  """Converts a timeout in seconds to the type ``ClientSession`` expects.

  ADK carries every timeout as float seconds. MCP SDK 1.x wants a
  ``timedelta`` here, while 2.x wants the float. Neither accepts the other, and
  the wrong one does not fail at the call: it fails later, in arithmetic the
  SDK does on the value. Converting in one place keeps that difference to a
  single function.

  Args:
    seconds: The timeout in seconds, or None for no timeout.

  Returns:
    The timeout in the form the installed SDK expects, or None.
  """
  if seconds is None:
    return None
  if IS_MCP_SDK_V2:
    return seconds
  return timedelta(seconds=seconds)


def _format_exception(exc: BaseException | None) -> str:
  """Formats an exception into a readable string representation.

  This handles `ExceptionGroup` (by flattening inner exceptions) and optionally
  extracts HTTP response bodies for network-related errors, truncating them
  to 1000 characters to prevent log/context overflow.

  Args:
    exc: The exception to format.

  Returns:
    A formatted string representing the exception and its pertinent details.
  """
  if exc is None:
    return 'None'
  if hasattr(exc, 'exceptions') and getattr(exc, 'exceptions'):
    return ' | '.join(_format_exception(e) for e in exc.exceptions)
  if hasattr(exc, 'response') and exc.response is not None:
    try:
      response_text = exc.response.text
      if len(response_text) > 1000:
        response_text = response_text[:1000] + '... [truncated]'
      return f'{exc} (Response: {response_text})'
    except Exception:
      pass
  return str(exc)


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
      client: AbstractAsyncContextManager[Any],
      timeout: float | None,
      sse_read_timeout: float | None,
      is_stdio: bool = False,
      *,
      sampling_callback: SamplingFnT | None = None,
      sampling_capabilities: SamplingCapability | None = None,
      elicitation_callback: ElicitationFnT | None = None,
      extensions: dict[str, dict[str, Any]] | None = None,
      result_claims: Mapping[str, Sequence[ResultClaim]] | None = None,
      notification_bindings: Sequence[NotificationBinding] | None = None,
  ):
    """Initializes SessionContext.

    Args:
      client: An MCP client context manager (e.g., from streamablehttp_client,
        sse_client, or stdio_client).
      timeout: Timeout in seconds for connection and initialization. This is the
        budget for the whole bring-up -- entering the client's context and
        running ``initialize()`` -- not a separate allowance for each step.
      sse_read_timeout: Timeout in seconds for reading data from the MCP SSE
        server.
      is_stdio: Whether this is a stdio connection (affects read timeout).
      sampling_callback: Optional callback to handle sampling requests from the
        MCP server.
      sampling_capabilities: Optional capabilities for sampling.
      elicitation_callback: Optional callback to handle elicitation requests
        from the MCP server (``elicitation/create``).
      extensions: MCP extensions this client advertises, keyed by extension
        identifier. Supplying any extension argument also makes the session
        negotiate with ``server/discover`` first, because extensions are only
        live on a modern connection.
      result_claims: Non-core ``tools/call`` result shapes to accept, keyed by
        the identifier of the extension that defines them.
      notification_bindings: Handlers for extension notifications.
    """
    self._client = client
    self._timeout = timeout
    self._sse_read_timeout = sse_read_timeout
    self._is_stdio = is_stdio
    self._session: ClientSession | None = None
    self._ready_event = asyncio.Event()
    self._close_event = asyncio.Event()
    self._task: asyncio.Task[None] | None = None
    self._task_lock = asyncio.Lock()
    self._sampling_callback = sampling_callback
    self._sampling_capabilities = sampling_capabilities
    self._elicitation_callback = elicitation_callback
    self._extensions = extensions
    self._result_claims = result_claims
    self._notification_bindings = notification_bindings

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

    The wait is bounded by ``timeout``, which covers connecting and
    initializing together. A connect that eats most of the budget therefore
    leaves ``initialize()`` less of it.

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

        def _retrieve_exception(t: asyncio.Task[None]) -> None:
          if not t.cancelled():
            t.exception()

        self._task.add_done_callback(_retrieve_exception)

    if (
        is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING)  # pylint: disable=protected-access
        and self._timeout is not None
    ):
      # `_ready_event` is a plain asyncio.Event, so bounding this wait only
      # cancels a bare future waiter and never crosses an AnyIO cancel
      # scope. The scopes live inside `self._task` and are unwound there,
      # in the task that entered them -- the same thing `close()` does for
      # an abandoned start.
      try:
        await asyncio.wait_for(self._ready_event.wait(), timeout=self._timeout)
      except asyncio.TimeoutError as e:
        self._task.cancel()
        raise ConnectionError(
            'Failed to create MCP session: timed out after'
            f' {self._timeout}s waiting for the session to become ready'
        ) from e
    else:
      await self._ready_event.wait()

    if self._task.cancelled():
      raise ConnectionError('Failed to create MCP session: task cancelled')

    if self._task.done() and self._task.exception():
      raise ConnectionError(
          'Failed to create MCP session:'
          f' {_format_exception(self._task.exception())}'
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

  async def _run_guarded(
      self,
      coro: Coroutine[Any, Any, _T],
      *,
      propagate_cancel: bool = False,
  ) -> _T:
    """Run a coroutine while monitoring the background session task.

    Races the given coroutine against the background task. If the task
    dies first (e.g. transport crash from a non-2xx HTTP response), the
    coroutine is cancelled and the original error is raised immediately
    instead of hanging until a read timeout expires.

    Args:
        coro: The coroutine to run (e.g. session.call_tool(...)).
        propagate_cancel: Whether to cancel ``coro`` when this call is itself
            cancelled. ``asyncio.wait`` does not cancel what it waits on, so
            by default a cancelled caller leaves ``coro`` running detached.
            That is tolerable for a single request, which the transport will
            eventually fail, but not for a coroutine that has cleanup of its
            own to do on the wire: it would never be told to run it. Off by
            default so the existing call path keeps its current semantics.

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
          f'MCP session task has already terminated: {_format_exception(exc)}'
      ) from exc

    coro_task = asyncio.ensure_future(coro)

    try:
      done, _ = await asyncio.wait(
          [coro_task, self._task],
          return_when=asyncio.FIRST_COMPLETED,
      )
    except asyncio.CancelledError:
      if propagate_cancel and not coro_task.done():
        coro_task.cancel()
        try:
          await coro_task
        except BaseException:
          pass
      raise

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
    raise ConnectionError(
        f'MCP session connection lost: {_format_exception(exc)}'
    ) from exc

  async def close(self) -> None:
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

  async def __aexit__(
      self,
      exc_type: type[BaseException] | None,
      exc_val: BaseException | None,
      exc_tb: TracebackType | None,
  ) -> None:
    await self.close()

  @property
  def _extension_kwargs(self) -> dict[str, Any]:
    """The extension arguments to hand `ClientSession`, when there are any.

    Spread rather than passed as three `None`s: MCP SDK 1.x declares none of
    these parameters, and naming one there is a `TypeError`. Nothing
    configures an extension on 1.x -- the opt-in is refused before a session
    manager exists -- so this is empty and the construction below is the one
    1.x has always made.
    """
    if not self._wants_extensions:
      return {}
    return {
        'extensions': self._extensions,
        'result_claims': self._result_claims,
        'notification_bindings': self._notification_bindings,
    }

  @property
  def _wants_extensions(self) -> bool:
    """Whether any MCP extension was configured on this session."""
    return bool(
        self._extensions or self._result_claims or self._notification_bindings
    )

  async def _negotiate(self, session: ClientSession) -> None:
    """Brings `session` up, preferring `server/discover` when it can matter.

    `initialize()` always performs the pre-2026 handshake, and an extension
    capability has nowhere to ride on that wire -- the SDK drops
    claim-bearing identifiers from the advertisement at legacy protocol
    versions, so a claim registered on this session would never fire. Only
    `server/discover` reaches a version where extensions are live.

    A server that predates `server/discover` answers it with an error, so the
    probe is bounded and falls back rather than failing the session. The
    bound matters: the probe and the fallback share one bring-up budget, and
    an unbounded probe against a server that simply never answers would spend
    all of it and leave nothing for `initialize()`.
    """
    if not self._wants_extensions:
      await session.initialize()
      return

    budget = min(
        self._timeout or _DISCOVER_TIMEOUT_SECONDS, _DISCOVER_TIMEOUT_SECONDS
    )
    try:
      with anyio.fail_after(budget / 2):
        session.adopt(await session.discover())
      return
    except (McpError, RuntimeError, TimeoutError) as e:
      # RuntimeError is what `adopt` raises when the server and this client
      # share no modern protocol version.
      logger.warning(
          'MCP extensions were requested but server/discover is unavailable'
          ' (%s); falling back to initialize(). Extensions, including tasks,'
          ' stay inactive on this session.',
          e,
      )
    await session.initialize()

  async def _run(self) -> None:
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
          # is enforced by `start()`, which bounds its wait on
          # `_ready_event` -- an asyncio.Event, so bounding it never
          # cancels across a cancel scope. (create_session's outer
          # asyncio.wait_for only exists on the flag-off path.)
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
                  read_timeout_seconds=_read_timeout(self._timeout),
                  sampling_callback=self._sampling_callback,
                  sampling_capabilities=self._sampling_capabilities,
                  elicitation_callback=self._elicitation_callback,
                  **self._extension_kwargs,
              )
          )
        else:
          # For SSE and Streamable HTTP clients, use the sse_read_timeout
          # instead of the connection timeout as the read_timeout for the session.
          session = await exit_stack.enter_async_context(
              ClientSession(
                  *transports[:2],
                  read_timeout_seconds=_read_timeout(self._sse_read_timeout),
                  sampling_callback=self._sampling_callback,
                  sampling_capabilities=self._sampling_capabilities,
                  elicitation_callback=self._elicitation_callback,
                  **self._extension_kwargs,
              )
          )
        # pylint: disable-next=protected-access
        if is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING):
          # Use anyio.fail_after to keep session.initialize within the AnyIO
          # cancel scope instead of asyncio.wait_for which runs in a nested
          # task.
          with anyio.fail_after(self._timeout):
            await self._negotiate(session)
        else:
          await asyncio.wait_for(
              self._negotiate(session), timeout=self._timeout
          )
        logger.debug('Session has been successfully initialized')

        self._session = session
        self._ready_event.set()

        # Wait for close signal - the session remains valid while we wait
        await self._close_event.wait()
    except Exception as e:
      logger.warning('Error on session runner task: %s', e)
      raise
    finally:
      self._ready_event.set()
      self._close_event.set()
