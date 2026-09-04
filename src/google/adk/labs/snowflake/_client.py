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

"""HTTP client for the Snowflake Cortex Agents REST API.

Thin ``httpx``-based access to the three calls the agent needs: creating a
thread, running the agent as an SSE stream, and cancelling a run. Credentials
come from a caller-supplied header provider on every request, so this module
never holds a token. Failures surface as typed errors that carry Snowflake's
error code and request id, never the request payload.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
import re
from typing import Any
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

from ._sse_parser import iter_sse_events
from ._sse_parser import SseEvent

if TYPE_CHECKING:
  from ...agents.readonly_context import ReadonlyContext

logger = logging.getLogger('google_adk.' + __name__)

HeaderProvider = Callable[
    ['ReadonlyContext'], 'dict[str, str] | Awaitable[dict[str, str]]'
]
"""Supplies the HTTP headers, typically ``Authorization``, for one request."""

_MAX_SNOWFLAKE_ID = 10**38 - 1
_DECIMAL_RE = re.compile(r'[0-9]+')
_DEFAULT_ORIGIN_APPLICATION = 'google_adk'
_SSE_MEDIA_TYPE = 'text/event-stream'
_JSON_MEDIA_TYPE = 'application/json'
_MAX_ERROR_MESSAGE_CHARS = 200
_AUTH_HINT = (
    ' Check that header_provider returns a valid Authorization header and,'
    ' for tokens other than OAuth, the matching'
    ' X-Snowflake-Authorization-Token-Type.'
)


class CortexClientError(Exception):
  """Base class for failures talking to the Cortex Agents REST API."""


class CortexApiError(CortexClientError):
  """Snowflake answered a request with an error, or with an unusable body."""

  def __init__(
      self,
      message: str,
      *,
      status_code: int,
      snowflake_code: str | None = None,
      request_id: str | None = None,
  ):
    super().__init__(message)
    self.status_code = status_code
    """The HTTP status Snowflake answered with."""
    self.snowflake_code = snowflake_code
    """Snowflake's own error code from the response body, if any."""
    self.request_id = request_id
    """Snowflake's request id, for quoting to Snowflake support."""


class CortexTransportError(CortexClientError):
  """A request never completed: connection failure, timeout, or a cut stream."""

  def __init__(self, message: str, *, timed_out: bool):
    super().__init__(message)
    self.timed_out = timed_out
    """Whether the failure was a timeout rather than a broken connection."""


def _parse_snowflake_id(value: str | int, field: str, *, minimum: int) -> int:
  """Converts a stored id into the integer Snowflake expects, strictly.

  The value is never echoed into the error: Snowflake ids stay out of logs.
  """
  if isinstance(value, bool):
    number = None
  elif isinstance(value, int):
    number = value
  elif isinstance(value, str) and _DECIMAL_RE.fullmatch(value):
    number = int(value)
  else:
    number = None
  if number is None or not minimum <= number <= _MAX_SNOWFLAKE_ID:
    raise ValueError(
        f'{field} must be a decimal integer between {minimum} and 10^38-1;'
        ' the stored Snowflake cursor is not usable.'
    )
  return number


def _media_type(response: httpx.Response) -> str:
  content_type: str = response.headers.get('content-type', '')
  return content_type.partition(';')[0].strip().lower()


class SnowflakeCortexClient:
  """Calls the Cortex Agents REST API for one configured Cortex Agent object.

  Each request asks ``header_provider`` for its headers, so credentials can be
  minted per invocation and are never stored here. The client can share an
  ``httpx.AsyncClient`` with the application or own one of its own, which
  ``aclose`` releases.
  """

  def __init__(
      self,
      *,
      account_url: str,
      database: str,
      schema_name: str,
      cortex_agent_name: str,
      header_provider: HeaderProvider,
      timeout: float,
      http_client: httpx.AsyncClient | None = None,
      origin_application: str = _DEFAULT_ORIGIN_APPLICATION,
  ):
    """Initializes the client.

    Args:
      account_url: Base URL of the Snowflake account.
      database: Database holding the Cortex Agent object.
      schema_name: Schema holding the Cortex Agent object.
      cortex_agent_name: Name of the Cortex Agent object.
      header_provider: Supplies the headers of each request, typically
        ``Authorization`` and ``X-Snowflake-Authorization-Token-Type``. May be
        sync or async.
      timeout: Seconds to wait on Snowflake for a connection and, during a
        run, between two SSE chunks.
      http_client: An ``httpx.AsyncClient`` to send through. When omitted, the
        client creates its own and closes it in ``aclose``.
      origin_application: Label Snowflake stores on threads this client
        creates. Snowflake accepts at most 16 UTF-8 bytes.
    """
    base = account_url.rstrip('/')
    self._threads_url = f'{base}/api/v2/cortex/threads'
    self._run_url = (
        f'{base}/api/v2/databases/{quote(database, safe="")}'
        f'/schemas/{quote(schema_name, safe="")}'
        f'/agents/{quote(cortex_agent_name, safe="")}:run'
    )
    self._cancel_url = f'{base}/api/v2/cortex/agent/runs/{{run_id}}/cancel'
    self._header_provider = header_provider
    # The read timeout is what bounds a stream that goes quiet between
    # chunks; connecting should never take as long as a run may.
    self._timeout = httpx.Timeout(timeout, connect=min(timeout, 30.0))
    self._http_client = http_client
    self._owns_http_client = http_client is None
    self._origin_application = origin_application

  async def create_thread(self, ctx: ReadonlyContext) -> str:
    """Creates a Snowflake thread for a new conversation.

    Args:
      ctx: The invocation's read-only context, passed to ``header_provider``.

    Returns:
      The new thread id as a decimal string.

    Raises:
      CortexApiError: Snowflake rejected the request or returned no thread id.
      CortexTransportError: Snowflake could not be reached in time.
    """
    response = await self._send(
        ctx,
        self._threads_url,
        json_body={'origin_application': self._origin_application},
        accept=_JSON_MEDIA_TYPE,
        operation='create thread',
    )
    try:
      payload = response.json()
    except ValueError:
      payload = None
    thread_id = payload.get('thread_id') if isinstance(payload, dict) else None
    if (
        isinstance(thread_id, bool)
        or not isinstance(thread_id, int)
        or not 1 <= thread_id <= _MAX_SNOWFLAKE_ID
    ):
      raise CortexApiError(
          'Snowflake created a thread but the response carried no usable'
          ' thread_id.',
          status_code=response.status_code,
          request_id=_request_id(response, payload),
      )
    return str(thread_id)

  @contextlib.asynccontextmanager
  async def run(
      self,
      ctx: ReadonlyContext,
      *,
      thread_id: str | int,
      parent_message_id: str | int,
      text: str,
  ) -> AsyncIterator[AsyncIterator[SseEvent]]:
    """Runs the Cortex Agent on one user message and streams its events.

    Use as ``async with client.run(...) as events``. Leaving the block closes
    the HTTP response, which is what stops Snowflake's stream when the
    consumer gives up early.

    Args:
      ctx: The invocation's read-only context, passed to ``header_provider``.
      thread_id: The Snowflake thread to append to.
      parent_message_id: The assistant message to continue from; ``0`` for
        the first turn of a thread.
      text: The user's message.

    Yields:
      The run's events in stream order, ending with the ``done`` event when
      Snowflake finished normally.

    Raises:
      ValueError: ``thread_id`` or ``parent_message_id`` is not a Snowflake id.
      CortexApiError: Snowflake rejected the run or did not answer with an
        event stream.
      CortexTransportError: Snowflake could not be reached, timed out, or the
        stream was cut before the run finished.
    """
    body = {
        'thread_id': _parse_snowflake_id(thread_id, 'thread_id', minimum=1),
        'parent_message_id': _parse_snowflake_id(
            parent_message_id, 'parent_message_id', minimum=0
        ),
        'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': text}]}
        ],
        'stream': True,
    }
    response = await self._send(
        ctx,
        self._run_url,
        json_body=body,
        accept=_SSE_MEDIA_TYPE,
        operation='run',
        stream=True,
    )
    try:
      media_type = _media_type(response)
      if media_type != _SSE_MEDIA_TYPE:
        await response.aread()
        raise CortexApiError(
            'Snowflake answered the run with'
            f' {media_type or "no"} content instead of {_SSE_MEDIA_TYPE}.',
            status_code=response.status_code,
            request_id=_request_id(response, None),
        )
      yield self._events(response)
    finally:
      await response.aclose()

  async def cancel_run(self, ctx: ReadonlyContext, run_id: str) -> bool:
    """Asks Snowflake to cancel a run, best effort.

    Args:
      ctx: The invocation's read-only context, passed to ``header_provider``.
      run_id: The run to cancel.

    Returns:
      Whether Snowflake acknowledged the cancel. A failure is logged by type
      only and never raised: by the time this is called the consumer has
      already gone.
    """
    try:
      response = await self._send(
          ctx,
          self._cancel_url.format(run_id=quote(run_id, safe='')),
          json_body=None,
          accept=_JSON_MEDIA_TYPE,
          operation='cancel',
      )
    except CortexClientError as e:
      status = getattr(e, 'status_code', None)
      logger.debug(
          'Best-effort cancel of a Snowflake Cortex run was refused: %s%s',
          type(e).__name__,
          f' (HTTP {status})' if status is not None else '',
      )
      return False
    await response.aclose()
    return True

  async def aclose(self) -> None:
    """Closes the HTTP client if this instance created it."""
    if self._owns_http_client and self._http_client is not None:
      await self._http_client.aclose()
      self._http_client = None

  def _client(self) -> httpx.AsyncClient:
    if self._http_client is None:
      self._http_client = httpx.AsyncClient(timeout=self._timeout)
    return self._http_client

  async def _headers(
      self, ctx: ReadonlyContext, *, accept: str
  ) -> dict[str, str]:
    provided = self._header_provider(ctx)
    if inspect.isawaitable(provided):
      provided = await provided
    headers = dict(provided or {})
    # Content negotiation is the client's to decide: a provider that sets
    # `Accept: application/json` would otherwise turn the run into a single
    # JSON body. `identity` keeps proxies from buffering the event stream.
    headers.update({
        'Content-Type': _JSON_MEDIA_TYPE,
        'Accept': accept,
        'Accept-Encoding': 'identity',
    })
    return headers

  async def _send(
      self,
      ctx: ReadonlyContext,
      url: str,
      *,
      json_body: dict[str, Any] | None,
      accept: str,
      operation: str,
      stream: bool = False,
  ) -> httpx.Response:
    client = self._client()
    request = client.build_request(
        'POST',
        url,
        json=json_body,
        headers=await self._headers(ctx, accept=accept),
        timeout=self._timeout,
    )
    try:
      response = await client.send(request, stream=stream)
    except httpx.TimeoutException as e:
      raise CortexTransportError(
          f'Snowflake did not answer the {operation} request in time.',
          timed_out=True,
      ) from e
    except httpx.RequestError as e:
      raise CortexTransportError(
          f'Snowflake could not be reached for the {operation} request.',
          timed_out=False,
      ) from e
    if response.is_success:
      return response
    try:
      await response.aread()
      raise _api_error(response, operation)
    finally:
      await response.aclose()

  async def _events(self, response: httpx.Response) -> AsyncIterator[SseEvent]:
    try:
      async for event in iter_sse_events(response.aiter_bytes()):
        yield event
    except httpx.TimeoutException as e:
      raise CortexTransportError(
          'Snowflake stopped sending events before the run finished.',
          timed_out=True,
      ) from e
    except httpx.RequestError as e:
      raise CortexTransportError(
          'The connection to Snowflake dropped before the run finished.',
          timed_out=False,
      ) from e


def _request_id(response: httpx.Response, payload: Any) -> str | None:
  header: str | None = response.headers.get('x-snowflake-request-id')
  if header:
    return header
  if isinstance(payload, dict) and payload.get('request_id') is not None:
    return str(payload['request_id'])
  return None


def _api_error(response: httpx.Response, operation: str) -> CortexApiError:
  try:
    payload = response.json()
  except ValueError:
    payload = None
  code = payload.get('code') if isinstance(payload, dict) else None
  message = payload.get('message') if isinstance(payload, dict) else None
  detail = (
      str(message)[:_MAX_ERROR_MESSAGE_CHARS]
      if message
      else 'no error message in the response'
  )
  text = (
      f'Snowflake rejected the {operation} request with HTTP'
      f' {response.status_code}'
  )
  if code is not None:
    text += f' (code {code})'
  text += f': {detail}.'
  if response.status_code in (401, 403):
    text += _AUTH_HINT
  return CortexApiError(
      text,
      status_code=response.status_code,
      snowflake_code=str(code) if code is not None else None,
      request_id=_request_id(response, payload),
  )
