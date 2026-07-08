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
from collections import deque
from contextlib import AbstractAsyncContextManager
from contextlib import AsyncExitStack
import contextvars
import functools
import hashlib
import json
import logging
import os
import sys
import threading
from typing import Any
from typing import AsyncIterator
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import TextIO
import urllib.parse

import google.auth
import google.auth.credentials
from google.auth.transport.requests import Request
import httpx

try:
  from google.auth.aio.credentials import Credentials as AsyncCredentials
  from google.auth.aio.transport.sessions import AsyncAuthorizedSession

  _AIO_SUPPORTED = True
except ImportError:

  class AsyncCredentials:  # pylint: disable=g-bad-classes
    pass

  class AsyncAuthorizedSession:  # pylint: disable=g-bad-classes
    pass

  _AIO_SUPPORTED = False

from mcp import ClientSession
from mcp import SamplingCapability
from mcp import StdioServerParameters
from mcp.client.session import SamplingFnT
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client as _create_mcp_http_client
from mcp.client.streamable_http import McpHttpClientFactory
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel
from pydantic import ConfigDict

try:
  from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

  _HAS_HTTPX_INSTRUMENTOR = True
except (ImportError, AttributeError):
  _HAS_HTTPX_INSTRUMENTOR = False

from ...features import FeatureName
from ...features import is_feature_enabled
from .session_context import SessionContext

logger = logging.getLogger('google_adk.' + __name__)

_MAX_LOG_BODY_LENGTH = 1000


def create_mcp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
  """Creates MCP HTTP client and instruments it when OTel is available."""
  client = _create_mcp_http_client(
      headers=headers,
      timeout=timeout,
      auth=auth,
  )
  if _HAS_HTTPX_INSTRUMENTOR:
    HTTPXClientInstrumentor.instrument_client(client)
  return client


_http_debug_var: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar('_http_debug_var', default=None)
)


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
  sensitive_keys = {'authorization', 'cookie', 'set-cookie', 'x-goog-api-key'}
  return {
      k: '<redacted>' if k.lower() in sensitive_keys else v
      for k, v in headers.items()
  }


class _StreamableHttpClientWrapper:
  """Wrapper to manage the lifecycle of a pre-created HTTP client with streamable_http_client."""

  def __init__(
      self,
      url: str,
      http_client: httpx.AsyncClient,
      terminate_on_close: bool = True,
  ):
    self.url = url
    self.http_client = http_client
    self.terminate_on_close = terminate_on_close
    self.ctx_mgr = streamable_http_client(
        url=url,
        http_client=http_client,
        terminate_on_close=terminate_on_close,
    )

  async def __aenter__(self) -> Any:
    # If http_client is a Mock, it might not have __aenter__ but mock async methods can be used
    if hasattr(self.http_client, '__aenter__'):
      await self.http_client.__aenter__()
    try:
      return await self.ctx_mgr.__aenter__()
    except Exception:
      if hasattr(self.http_client, '__aexit__'):
        await self.http_client.__aexit__(None, None, None)
      raise

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    try:
      await self.ctx_mgr.__aexit__(exc_type, exc_val, exc_tb)
    finally:
      if hasattr(self.http_client, '__aexit__'):
        await self.http_client.__aexit__(exc_type, exc_val, exc_tb)


def _has_cancelled_error_context(exc: BaseException) -> bool:
  """Returns True if `exc` is/was caused by `asyncio.CancelledError`.

  Cancellation can be translated into other exceptions during teardown (e.g.
  connection errors) while still retaining the original cancellation in an
  exception's context chain.
  """

  seen: set[int] = set()
  queue = deque([exc])
  while queue:
    current = queue.popleft()
    if id(current) in seen:
      continue
    seen.add(id(current))
    if isinstance(current, asyncio.CancelledError):
      return True
    if current.__cause__ is not None:
      queue.append(current.__cause__)
    if current.__context__ is not None:
      queue.append(current.__context__)
  return False


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
      httpx_client_factory: Factory function to create a custom HTTPX client. If
        not provided, a default factory will be used.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5.0
  sse_read_timeout: float = 60 * 5.0
  httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client


@runtime_checkable
class CheckableMcpHttpClientFactory(McpHttpClientFactory, Protocol):
  pass


class _DebugHttpxClientFactory:
  """A factory wrapper that hooks into the httpx.AsyncClient responses to capture debug info."""

  def __init__(
      self,
      base_factory: CheckableMcpHttpClientFactory,
      session_manager: MCPSessionManager | None = None,
  ):
    self._base_factory = base_factory
    self._session_manager = session_manager

  def __call__(
      self,
      headers: dict[str, str] | None = None,
      timeout: httpx.Timeout | None = None,
      auth: httpx.Auth | None = None,
  ) -> httpx.AsyncClient:
    client = self._base_factory(headers=headers, timeout=timeout, auth=auth)
    if hasattr(client, 'event_hooks') and isinstance(client.event_hooks, dict):
      client.event_hooks.setdefault('response', []).append(self._response_hook)
    return client

  def _extract_session_id(self, response: httpx.Response) -> str | None:
    query_params = urllib.parse.parse_qs(
        urllib.parse.urlparse(str(response.url)).query
    )
    return (
        query_params.get('sessionId', [None])[0]
        or query_params.get('session_id', [None])[0]
    )

  async def _response_hook(self, response: httpx.Response):
    debug_list = None
    if self._session_manager is not None:
      session_id = self._extract_session_id(response)
      if session_id:
        debug_list = self._session_manager._get_active_debug_list_by_session_id(
            session_id
        )

    if debug_list is None:
      debug_list = _http_debug_var.get(None)

    if debug_list is None:
      return

    content_type = response.headers.get('content-type', '')
    is_sse = 'text/event-stream' in content_type

    request_body = None
    if response.request.content:
      try:
        request_body = response.request.content.decode(
            'utf-8', errors='replace'
        )
        if len(request_body) > _MAX_LOG_BODY_LENGTH:
          request_body = request_body[:_MAX_LOG_BODY_LENGTH] + '... [truncated]'
      except Exception:  # pylint: disable=broad-exception-caught
        request_body = '<binary>'

    if not is_sse:
      try:
        await response.aread()
        response_body = response.text
        if len(response_body) > _MAX_LOG_BODY_LENGTH:
          response_body = (
              response_body[:_MAX_LOG_BODY_LENGTH] + '... [truncated]'
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        response_body = f'<failed to read body: {e}>'
    else:
      response_body = '<SSE stream>'

    debug_info = {
        'url': str(response.url),
        'status_code': response.status_code,
        'method': response.request.method,
        'request_headers': _redact_headers(dict(response.request.headers)),
        'request_body': request_body,
        'response_headers': _redact_headers(dict(response.headers)),
        'response_body': response_body,
    }
    debug_list.append(debug_info)


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
      httpx_client_factory: Factory function to create a custom HTTPX client. If
        not provided, a default factory will be used.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  url: str
  headers: dict[str, Any] | None = None
  timeout: float = 5.0
  sse_read_timeout: float = 60 * 5.0
  terminate_on_close: bool = True
  httpx_client_factory: CheckableMcpHttpClientFactory = create_mcp_http_client


def retry_on_errors(func):
  """Decorator to automatically retry action when MCP session errors occur.

  When MCP session errors occur, the decorator will automatically retry the
  action once. The create_session method will handle creating a new session
  if the old one was disconnected.

  Cancellation is not retried and must be allowed to propagate. In async
  runtimes, cancellation may surface as `asyncio.CancelledError` or as another
  exception while the task is cancelling.

  Args:
      func: The function to decorate.

  Returns:
      The decorated function.
  """

  @functools.wraps(func)  # Preserves original function metadata
  async def wrapper(self, *args, **kwargs):
    try:
      return await func(self, *args, **kwargs)
    except Exception as e:
      task = asyncio.current_task()
      if task is not None:
        cancelling = getattr(task, 'cancelling', None)
        if cancelling is not None and cancelling() > 0:
          raise
      if _has_cancelled_error_context(e):
        raise
      # If an error is thrown, we will retry the function to reconnect to the
      # server. create_session will handle detecting and replacing disconnected
      # sessions.
      logger.info('Retrying %s due to error: %s', func.__name__, e)
      return await func(self, *args, **kwargs)

  return wrapper


class _RefreshableAsyncCredentials(AsyncCredentials):
  """Adapter to refresh sync credentials asynchronously."""

  def __init__(
      self,
      creds: google.auth.credentials.Credentials,
      target_host: str | None = None,
  ):
    super().__init__()
    self._creds = creds
    self._target_host = target_host
    self._lock = asyncio.Lock()

  async def before_request(
      self,
      _request: Any,
      _method: str,
      url: str,
      headers: dict[str, str],
  ) -> None:
    if self._target_host:
      parsed_url = urllib.parse.urlparse(url)
      if parsed_url.netloc != self._target_host:
        logger.debug(
            'Skipping token injection for redirect to %s', parsed_url.netloc
        )
        return

    if any(k.lower() == 'authorization' for k in headers):
      logger.debug('Authorization header already present, not overwriting')
      return

    async with self._lock:
      await asyncio.to_thread(self._refresh_sync)
    if self._creds.token:
      headers['Authorization'] = f'Bearer {self._creds.token}'

  def _refresh_sync(self) -> None:
    if self._creds.expired or not self._creds.token:
      self._creds.refresh(Request())


class _GoogleAuthAsyncByteStream(httpx.AsyncByteStream):
  """Adapter to bridge google-auth Response.content with httpx.AsyncByteStream."""

  def __init__(self, auth_response: Any):
    self._auth_response = auth_response

  async def __aiter__(self) -> AsyncIterator[bytes]:
    async for chunk in self._auth_response.content():
      yield chunk

  async def aclose(self) -> None:
    await self._auth_response.close()


class _GoogleAuthAsyncTransport(httpx.AsyncBaseTransport):
  """Adapter to bridge google-auth AsyncAuthorizedSession with httpx.AsyncBaseTransport."""

  def __init__(self, auth_session: Any):
    self._auth_session = auth_session

  async def handle_async_request(
      self, request: httpx.Request
  ) -> httpx.Response:
    content = await request.aread()
    headers_dict = dict(request.headers)

    timeout_val = 30.0
    if request.extensions and 'timeout' in request.extensions:
      timeout_dict = request.extensions['timeout']
      if 'read' in timeout_dict and timeout_dict['read'] is not None:
        timeout_val = timeout_dict['read']

    if request.headers.get('accept') == 'text/event-stream':
      # google-auth-aio translates timeout to aiohttp ClientTimeout(total=timeout).
      # For SSE streams, we disable the total timeout (setting it to 0.0) to
      # prevent aiohttp from forcibly closing the stream after sse_read_timeout.
      timeout_val = 0.0

    auth_response: Any = await self._auth_session.request(
        method=request.method,
        url=str(request.url),
        data=content if content else None,
        headers=headers_dict,
        timeout=timeout_val,
    )

    # google-auth-aio uses aiohttp internally, which automatically handles
    # decompression and decodes chunked transfer encoding, but leaves the
    # headers intact. We must strip these headers so httpx doesn't attempt
    # to decompress or parse chunked framing again on the raw stream.
    response_headers = {
        k: v
        for k, v in auth_response.headers.items()
        if k.lower()
        not in ('content-encoding', 'content-length', 'transfer-encoding')
    }

    return httpx.Response(
        status_code=auth_response.status_code,
        headers=response_headers,
        stream=_GoogleAuthAsyncByteStream(auth_response),
    )

  async def aclose(self) -> None:
    await self._auth_session.close()


class _SharedAsyncTransport(httpx.AsyncBaseTransport):
  """Wrapper transport that prevents the wrapped transport from being closed."""

  def __init__(self, transport: httpx.AsyncBaseTransport):
    self._transport = transport

  async def handle_async_request(
      self, request: httpx.Request
  ) -> httpx.Response:
    return await self._transport.handle_async_request(request)

  async def aclose(self) -> None:
    pass


def _create_mtls_client_factory(
    mtls_transport: httpx.AsyncBaseTransport,
) -> CheckableMcpHttpClientFactory:
  """Returns a factory that creates httpx.AsyncClient using the mtls_transport."""

  def factory(
      headers: dict[str, Any] | None = None,
      timeout: httpx.Timeout | None = None,
      auth: httpx.Auth | None = None,
  ) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        auth=auth,
        timeout=timeout,
        transport=_SharedAsyncTransport(mtls_transport),
        follow_redirects=True,
    )

  return factory


class MCPSessionManager:
  """Manages MCP client sessions.

  This class provides methods for creating and initializing MCP client sessions,
  handling different connection parameters (Stdio and SSE) and supporting
  session pooling based on authentication headers.
  """

  def __init__(
      self,
      connection_params: (
          StdioServerParameters
          | StdioConnectionParams
          | SseConnectionParams
          | StreamableHTTPConnectionParams
      ),
      errlog: TextIO = sys.stderr,
      *,
      sampling_callback: SamplingFnT | None = None,
      sampling_capabilities: SamplingCapability | None = None,
  ):
    """Initializes the MCP session manager.

    Args:
        connection_params: Parameters for the MCP connection (Stdio, SSE or
          Streamable HTTP). Stdio by default also has a 5s read timeout as other
          parameters but it's not configurable for now.
        errlog: (Optional) TextIO stream for error logging. Use only for
          initializing a local stdio MCP session.
        sampling_callback: Optional callback to handle sampling requests from the
          MCP server.
        sampling_capabilities: Optional capabilities for sampling.
    """
    self._sampling_callback = sampling_callback
    self._sampling_capabilities = sampling_capabilities

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

    # Session pool: maps session keys to (session, exit_stack, loop) tuples.
    # Kept as a tuple for backward-compatibility with downstream tests
    # that construct or unpack entries directly.
    self._sessions: dict[
        str, tuple[ClientSession, AsyncExitStack, asyncio.AbstractEventLoop]
    ] = {}

    # Sibling pool: maps session keys to their SessionContext. Stored
    # separately from `_sessions` so the tuple shape above stays stable.
    # Used by McpTool to access `_run_guarded` for transport-crash detection.
    self._session_contexts: dict[str, SessionContext] = {}

    # Map of event loops to their respective locks to prevent race conditions
    # across different event loops in session creation.
    self._session_lock_map: dict[asyncio.AbstractEventLoop, asyncio.Lock] = {}
    self._lock_map_lock = threading.Lock()
    self._session_id_to_key: dict[str, str] = {}
    self._active_debug_lists: dict[str, list[dict[str, Any]]] = {}

    # Cache for mTLS transports per event loop to avoid re-creation.
    self._mtls_transports: dict[
        asyncio.AbstractEventLoop, _GoogleAuthAsyncTransport
    ] = {}

  def _make_on_session_created(self, session_key: str) -> Callable[[str], None]:
    def on_session_created(session_id: str):
      logger.debug('Session created: %s -> %s', session_id, session_key)
      self._session_id_to_key[session_id] = session_key

    return on_session_created

  def _set_active_debug_list(
      self, session_key: str, debug_list: list[dict[str, Any]]
  ):
    self._active_debug_lists[session_key] = debug_list

  def _get_active_debug_list_by_session_id(
      self, session_id: str
  ) -> list[dict[str, Any]] | None:
    session_key = self._session_id_to_key.get(session_id)
    if session_key:
      return self._active_debug_lists.get(session_key)
    return None

  @property
  def _session_lock(self) -> asyncio.Lock:
    """Returns an asyncio.Lock bound to the current event loop."""
    current_loop = asyncio.get_running_loop()
    with self._lock_map_lock:
      if current_loop not in self._session_lock_map:
        self._session_lock_map[current_loop] = asyncio.Lock()
      return self._session_lock_map[current_loop]

  async def _get_mtls_transport(self) -> _GoogleAuthAsyncTransport | None:
    """Attempts to create a _GoogleAuthAsyncTransport for mTLS, caching it per loop."""
    if isinstance(self._connection_params, StdioConnectionParams):
      return None

    if not _AIO_SUPPORTED:
      logger.debug('google.auth.aio not available, mTLS not configured')
      return None

    use_client_cert = (
        os.environ.get('GOOGLE_API_USE_CLIENT_CERTIFICATE', 'true').lower()
        == 'true'
    )
    if not use_client_cert:
      return None

    current_loop = asyncio.get_running_loop()
    if current_loop in self._mtls_transports:
      return self._mtls_transports[current_loop]

    try:
      scopes = ['https://www.googleapis.com/auth/cloud-platform']
      sync_credentials, _ = await asyncio.to_thread(
          google.auth.default, scopes=scopes
      )

      target_url = self._connection_params.url
      target_host = urllib.parse.urlparse(target_url).netloc

      credentials = _RefreshableAsyncCredentials(
          sync_credentials, target_host=target_host
      )
      auth_session = AsyncAuthorizedSession(credentials)
      await auth_session.configure_mtls_channel()

      if auth_session.is_mtls:
        logger.info('Successfully configured mTLS using AsyncAuthorizedSession')
        transport = _GoogleAuthAsyncTransport(auth_session)
        self._mtls_transports[current_loop] = transport
        return transport
      else:
        logger.warning(
            'mTLS was requested but AsyncAuthorizedSession channel is not mTLS'
        )
    except Exception as e:  # pylint: disable=broad-except
      logger.warning(
          'Failed to configure mTLS using AsyncAuthorizedSession: %s', e
      )
    return None

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

  def _get_session_context(
      self, headers: Optional[Dict[str, str]] = None
  ) -> Optional[SessionContext]:
    """Returns the SessionContext for the session matching the given headers.

    Note: This method reads from the session-context pool without acquiring
    ``_session_lock``. This is safe because it is called immediately after
    ``create_session()`` (which populates the entry under the lock) within
    the same task, and dict reads are atomic in CPython.

    Args:
        headers: Optional headers used to identify the session.

    Returns:
        The SessionContext if a matching session exists, None otherwise.
    """
    merged_headers = self._merge_headers(headers)
    session_key = self._generate_session_key(merged_headers)
    return self._session_contexts.get(session_key)

  async def _cleanup_session(
      self,
      session_key: str,
      exit_stack: AsyncExitStack,
      stored_loop: asyncio.AbstractEventLoop,
  ):
    """Cleans up a session, handling different event loops safely.

    Args:
        session_key: The session key to clean up.
        exit_stack: The AsyncExitStack managing the session resources.
        stored_loop: The event loop on which the session was created.
    """
    current_loop = asyncio.get_running_loop()
    try:
      if stored_loop is current_loop:
        await exit_stack.aclose()
      elif stored_loop.is_closed():
        logger.warning(
            f'Error cleaning up session {session_key}: original event loop'
            ' is closed, resources may be leaked.'
        )
      else:
        # The old loop is still running in another thread;
        # schedule cleanup on it.
        logger.info(
            f'Scheduling cleanup of session {session_key} on its original'
            ' event loop.'
        )
        future = asyncio.run_coroutine_threadsafe(
            exit_stack.aclose(), stored_loop
        )

        # Attach a callback so errors don't go unnoticed
        def cleanup_done(f: asyncio.Future):
          try:
            if f.exception():
              logger.warning(
                  f'Error cleaning up session {session_key} on original'
                  f' loop: {f.exception()}'
              )
          except Exception as e:
            logger.warning(
                f'Failed to check cleanup status for {session_key}: {e}'
            )

        future.add_done_callback(cleanup_done)
    except Exception as e:
      logger.warning(
          f'Error during session cleanup for {session_key}: {e}',
          exc_info=True,
      )
    finally:
      if session_key in self._sessions:
        del self._sessions[session_key]
      # Also drop the SessionContext reference so we don't leak the
      # SessionContext after its underlying session is gone.
      if session_key in self._session_contexts:
        del self._session_contexts[session_key]
      # Also clean up session ID mapping
      for sid, skey in list(self._session_id_to_key.items()):
        if skey == session_key:
          del self._session_id_to_key[sid]
      if session_key in self._active_debug_lists:
        del self._active_debug_lists[session_key]

  def _create_client(
      self,
      merged_headers: dict[str, str] | None = None,
      mtls_transport: httpx.AsyncBaseTransport | None = None,
      *,
      session_key: str | None = None,
  ) -> AbstractAsyncContextManager[Any]:
    """Creates an MCP client based on the connection parameters.

    Args:
        session_key: Optional session key for this client.
        merged_headers: Optional headers to include in the connection. Only
          applicable for SSE and StreamableHTTP connections.
        mtls_transport: Optional mTLS transport for the HTTP client.

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
      factory = self._connection_params.httpx_client_factory
      if mtls_transport:
        factory = _create_mtls_client_factory(mtls_transport)
      debug_factory = _DebugHttpxClientFactory(
          factory,
          session_manager=self,
      )
      on_session_created = None
      if session_key is not None:
        on_session_created = self._make_on_session_created(session_key)
      client = sse_client(
          url=self._connection_params.url,
          headers=merged_headers,
          timeout=self._connection_params.timeout,
          sse_read_timeout=self._connection_params.sse_read_timeout,
          httpx_client_factory=debug_factory,
          on_session_created=on_session_created,
      )
    elif isinstance(self._connection_params, StreamableHTTPConnectionParams):
      factory = self._connection_params.httpx_client_factory
      if mtls_transport:
        factory = _create_mtls_client_factory(mtls_transport)
      debug_factory = _DebugHttpxClientFactory(
          factory,
          session_manager=self,
      )
      http_client = debug_factory(
          headers=merged_headers,
          timeout=httpx.Timeout(
              self._connection_params.timeout,
              read=self._connection_params.sse_read_timeout,
          ),
      )
      client = _StreamableHttpClientWrapper(
          url=self._connection_params.url,
          http_client=http_client,
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
      self, headers: dict[str, str] | None = None
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
      # Register the active debug list for this session key if available in context
      debug_list = _http_debug_var.get(None)
      if debug_list is not None:
        self._set_active_debug_list(session_key, debug_list)

      # Check if we have an existing session
      if session_key in self._sessions:
        session, exit_stack, stored_loop = self._sessions[session_key]

        # Check if the existing session is still connected and bound to
        # the current loop. When the feature flag is on, we ALSO check the
        # SessionContext's background task: a crashed transport can leave
        # the session's read/write streams open even though the underlying
        # task has already died (e.g. after a 4xx/5xx HTTP response).
        # Without that extra check, callers would reuse a dead session and
        # hang on the next call. The check is gated because it triggers
        # session re-creation in some test mocks where `_task` looks
        # "not alive" but the streams are otherwise reusable.
        current_loop = asyncio.get_running_loop()
        if is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING):  # pylint: disable=protected-access
          ctx = self._session_contexts.get(session_key)
          ctx_alive = ctx is None or ctx._is_task_alive  # pylint: disable=protected-access
        else:
          ctx_alive = True  # Pre-fix: do not consult task aliveness
        if (
            stored_loop is current_loop
            and not self._is_session_disconnected(session)
            and ctx_alive
        ):
          # Session is still good, return it
          return session
        else:
          # Session is disconnected, dead, or from a different loop; clean up.
          logger.info(
              'Cleaning up session (disconnected or different loop): %s',
              session_key,
          )
          await self._cleanup_session(session_key, exit_stack, stored_loop)

      # Create a new session (either first time or replacing disconnected one)
      exit_stack = AsyncExitStack()
      timeout_in_seconds = (
          self._connection_params.timeout
          if hasattr(self._connection_params, 'timeout')
          else None
      )
      sse_read_timeout_in_seconds = (
          self._connection_params.sse_read_timeout
          if hasattr(self._connection_params, 'sse_read_timeout')
          else None
      )

      try:
        mtls_transport = await self._get_mtls_transport()
        client = self._create_client(
            merged_headers,
            mtls_transport=mtls_transport,
            session_key=session_key,
        )
        is_stdio = isinstance(self._connection_params, StdioConnectionParams)

        session_context = SessionContext(
            client=client,
            timeout=timeout_in_seconds,
            sse_read_timeout=sse_read_timeout_in_seconds,
            is_stdio=is_stdio,
            sampling_callback=self._sampling_callback,
            sampling_capabilities=self._sampling_capabilities,
        )

        if is_feature_enabled(FeatureName._MCP_GRACEFUL_ERROR_HANDLING):  # pylint: disable=protected-access
          session = await exit_stack.enter_async_context(session_context)
        else:
          session = await asyncio.wait_for(
              exit_stack.enter_async_context(session_context),
              timeout=timeout_in_seconds,
          )

        # Store session, exit stack, and loop in the pool. The pool storage
        # remains a tuple for backward-compatibility with downstream tests
        # that construct or unpack entries directly.
        self._sessions[session_key] = (
            session,
            exit_stack,
            asyncio.get_running_loop(),
        )
        # Track the SessionContext in a sibling dict so McpTool can call
        # `_run_guarded` on it. Stored separately to avoid changing the
        # shape of `_sessions` (which is a public-ish internal surface).
        self._session_contexts[session_key] = session_context
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

  def __getstate__(self):
    """Custom pickling to exclude non-picklable runtime objects."""
    state = self.__dict__.copy()
    # Remove unpicklable entries or those that shouldn't persist across pickle
    state['_sessions'] = {}
    state['_session_contexts'] = {}
    state['_session_lock_map'] = {}
    state['_mtls_transports'] = {}
    state['_session_id_to_key'] = {}
    state['_active_debug_lists'] = {}

    # Locks and file-like objects cannot be pickled
    state.pop('_lock_map_lock', None)
    state.pop('_errlog', None)

    return state

  def __setstate__(self, state):
    """Custom unpickling to restore state."""
    self.__dict__.update(state)
    # Re-initialize members that were not pickled
    self._sessions = {}
    self._session_contexts = {}
    self._session_lock_map = {}
    self._mtls_transports = {}
    self._session_id_to_key = {}
    self._active_debug_lists = {}
    self._lock_map_lock = threading.Lock()
    # If _errlog was removed during pickling, default to sys.stderr
    if not hasattr(self, '_errlog') or self._errlog is None:
      self._errlog = sys.stderr

  async def close(self):
    """Closes all sessions and cleans up resources."""
    async with self._session_lock:
      for session_key in list(self._sessions.keys()):
        _, exit_stack, stored_loop = self._sessions[session_key]
        await self._cleanup_session(session_key, exit_stack, stored_loop)

      for transport in self._mtls_transports.values():
        await transport.aclose()
      self._mtls_transports.clear()


SseServerParams = SseConnectionParams

StreamableHTTPServerParams = StreamableHTTPConnectionParams
