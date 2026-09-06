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

"""Tests for the Snowflake Cortex REST client.

Drives ``SnowflakeCortexClient`` against an ``httpx.MockTransport`` standing in
for Snowflake, and verifies the requests it sends, the events it streams, and
the errors it raises when Snowflake or the network misbehaves.
"""

from __future__ import annotations

import json
from typing import Any
from typing import AsyncIterator
from typing import Callable
from unittest.mock import MagicMock

from google.adk.labs.snowflake._client import CortexApiError
from google.adk.labs.snowflake._client import CortexTransportError
from google.adk.labs.snowflake._client import SnowflakeCortexClient
from google.adk.labs.snowflake._sse_parser import SseEvent
import httpx
import pytest

_TOKEN = 'pat-secret-token-value'
_ACCOUNT_URL = 'https://acct.snowflakecomputing.com'
_RUN_PATH = (
    '/api/v2/databases/SALES_DB/schemas/ANALYTICS/agents/SALES_AGENT:run'
)
_STREAM = (
    b'event: metadata\n'
    b'data: {"metadata":{"role":"user","message_id":455}}\n\n'
    b'event: response.text.delta\n'
    b'data: {"content_index":0,"sequence_number":1,"text":"Hi"}\n\n'
    b'event: done\ndata: [DONE]\n\n'
)
# Where the first complete event ends: a cut here loses nothing.
_FIRST_EVENT_END = _STREAM.index(b'\n\n') + 2

_Handler = Callable[[httpx.Request], httpx.Response]


def _bearer_headers(ctx: Any) -> dict[str, str]:
  del ctx
  return {
      'Authorization': f'Bearer {_TOKEN}',
      'X-Snowflake-Authorization-Token-Type': 'PROGRAMMATIC_ACCESS_TOKEN',
  }


class _Chunks(httpx.AsyncByteStream):
  """A scripted response body: some chunks, then optionally a failure."""

  def __init__(
      self, chunks: list[bytes], *, then_raise: Exception | None = None
  ):
    self._chunks = chunks
    self._then_raise = then_raise
    self.closed = False

  async def __aiter__(self) -> AsyncIterator[bytes]:
    for chunk in self._chunks:
      yield chunk
    if self._then_raise is not None:
      raise self._then_raise

  async def aclose(self) -> None:
    self.closed = True


def _sse_response(body: httpx.AsyncByteStream) -> httpx.Response:
  return httpx.Response(
      200, stream=body, headers={'content-type': 'text/event-stream'}
  )


def _make_client(
    handler: _Handler,
    *,
    header_provider: Callable[..., Any] = _bearer_headers,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[SnowflakeCortexClient, list[httpx.Request]]:
  """A client wired to `handler`, plus the list of requests it received."""
  requests: list[httpx.Request] = []

  def _recording(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return handler(request)

  client = SnowflakeCortexClient(
      account_url=_ACCOUNT_URL,
      database='SALES_DB',
      schema_name='ANALYTICS',
      cortex_agent_name='SALES_AGENT',
      header_provider=header_provider,
      timeout=5.0,
      http_client=http_client
      or httpx.AsyncClient(transport=httpx.MockTransport(_recording)),
  )
  return client, requests


def _ctx() -> MagicMock:
  return MagicMock(name='ReadonlyContext')


# --- create_thread ------------------------------------------------------------


async def test_create_thread_posts_origin_and_returns_the_id_as_text():
  """The Threads API call carries the auth headers and yields a string id."""
  client, requests = _make_client(
      lambda request: httpx.Response(200, json={'thread_id': 1234567890})
  )

  thread_id = await client.create_thread(_ctx())

  (request,) = requests
  assert thread_id == '1234567890'
  assert request.url == f'{_ACCOUNT_URL}/api/v2/cortex/threads'
  assert json.loads(request.content) == {'origin_application': 'google_adk'}
  assert request.headers['authorization'] == f'Bearer {_TOKEN}'
  assert request.headers['accept'] == 'application/json'


async def test_async_header_provider_is_awaited():
  """A coroutine provider works the same as a plain callable."""

  async def _async_headers(ctx: Any) -> dict[str, str]:
    del ctx
    return {'Authorization': 'Bearer async-token'}

  client, requests = _make_client(
      lambda request: httpx.Response(200, json={'thread_id': 7}),
      header_provider=_async_headers,
  )

  await client.create_thread(_ctx())

  assert requests[0].headers['authorization'] == 'Bearer async-token'


async def test_header_provider_receives_the_context():
  """The provider is handed the invocation's context to mint headers from."""
  seen: list[Any] = []

  def _provider(ctx: Any) -> dict[str, str]:
    seen.append(ctx)
    return {}

  client, _ = _make_client(
      lambda request: httpx.Response(200, json={'thread_id': 7}),
      header_provider=_provider,
  )
  ctx = _ctx()

  await client.create_thread(ctx)

  assert seen == [ctx]


@pytest.mark.parametrize(
    'payload', [{'thread_id': 'abc'}, {'thread_id': 0}, {'thread_id': True}, {}]
)
async def test_create_thread_rejects_an_unusable_id(payload: dict[str, Any]):
  """A 2xx without a positive integer thread_id is an API error."""
  client, _ = _make_client(lambda request: httpx.Response(200, json=payload))

  with pytest.raises(CortexApiError, match='no usable thread_id'):
    await client.create_thread(_ctx())


async def test_create_thread_surfaces_snowflake_error_details_without_token():
  """An error carries status, code and request id but never the credential."""
  client, _ = _make_client(
      lambda request: httpx.Response(
          401,
          json={
              'code': '390144',
              'message': 'JWT token is invalid',
              'request_id': 'req-1',
          },
      )
  )

  with pytest.raises(CortexApiError) as info:
    await client.create_thread(_ctx())

  error = info.value
  assert (error.status_code, error.snowflake_code, error.request_id) == (
      401,
      '390144',
      'req-1',
  )
  assert 'JWT token is invalid' in str(error)
  assert 'X-Snowflake-Authorization-Token-Type' in str(error)
  assert _TOKEN not in str(error)


# --- run ----------------------------------------------------------------------


async def test_run_posts_the_documented_body_and_streams_events():
  """The run request carries ids, the message and stream=true; events flow."""
  body = _Chunks([_STREAM[:37], _STREAM[37:]])
  client, requests = _make_client(lambda request: _sse_response(body))

  async with client.run(
      _ctx(), thread_id='1234567890', parent_message_id='0', text='hello'
  ) as events:
    received = [event async for event in events]

  (request,) = requests
  assert request.url == f'{_ACCOUNT_URL}{_RUN_PATH}'
  assert json.loads(request.content) == {
      'thread_id': 1234567890,
      'parent_message_id': 0,
      'messages': [
          {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}
      ],
      'stream': True,
  }
  assert request.headers['accept'] == 'text/event-stream'
  assert request.headers['accept-encoding'] == 'identity'
  assert [e.event for e in received] == [
      'metadata',
      'response.text.delta',
      'done',
  ]
  assert received[-1].is_done
  assert body.closed


async def test_run_encodes_identifiers_and_tolerates_a_trailing_slash():
  """Object names are URL-encoded; a trailing slash on the account is fine."""
  requests: list[httpx.Request] = []

  def _handler(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    return _sse_response(_Chunks([b'event: done\ndata: [DONE]\n\n']))

  client = SnowflakeCortexClient(
      account_url=f'{_ACCOUNT_URL}/',
      database='MY DB',
      schema_name='S/1',
      cortex_agent_name='AGENT',
      header_provider=_bearer_headers,
      timeout=5.0,
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
  )

  async with client.run(
      _ctx(), thread_id=1, parent_message_id=0, text='x'
  ) as events:
    async for _ in events:
      pass

  assert str(requests[0].url) == (
      f'{_ACCOUNT_URL}/api/v2/databases/MY%20DB/schemas/S%2F1/agents/AGENT:run'
  )


@pytest.mark.parametrize(
    'ids',
    [
        {'thread_id': '0', 'parent_message_id': '0'},
        {'thread_id': '12a', 'parent_message_id': '0'},
        {'thread_id': '1', 'parent_message_id': '-1'},
        {'thread_id': '1', 'parent_message_id': ' 1'},
        {'thread_id': str(10**38), 'parent_message_id': '0'},
        {'thread_id': True, 'parent_message_id': '0'},
    ],
)
async def test_run_rejects_invalid_ids_before_sending(ids: dict[str, Any]):
  """A cursor that is not a strict Snowflake id never reaches the network."""
  client, requests = _make_client(lambda request: _sse_response(_Chunks([])))

  with pytest.raises(ValueError, match='decimal integer') as info:
    async with client.run(_ctx(), text='x', **ids):
      pass

  assert requests == []
  assert '12a' not in str(info.value)


@pytest.mark.parametrize('status', [401, 403, 429, 500, 503])
async def test_run_raises_on_error_status_before_yielding(status: int):
  """A non-2xx answer fails the run with its status before any event."""
  client, _ = _make_client(
      lambda request: httpx.Response(status, json={'message': 'nope'})
  )

  with pytest.raises(CortexApiError) as info:
    async with client.run(
        _ctx(), thread_id='1', parent_message_id='0', text='x'
    ):
      pytest.fail('the stream must not open on an error status')

  assert info.value.status_code == status
  assert 'nope' in str(info.value)


async def test_run_rejects_a_non_event_stream_answer():
  """A 200 that is not text/event-stream cannot be a run."""
  client, _ = _make_client(
      lambda request: httpx.Response(200, json={'content': []})
  )

  with pytest.raises(CortexApiError, match='instead of text/event-stream'):
    async with client.run(
        _ctx(), thread_id='1', parent_message_id='0', text='x'
    ):
      pytest.fail('the stream must not open without an event stream')


async def test_connect_timeout_is_a_transport_error():
  """Not reaching Snowflake in time is reported as a timeout."""

  def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout('slow', request=request)

  client, _ = _make_client(_timeout)

  with pytest.raises(CortexTransportError, match='in time') as info:
    async with client.run(
        _ctx(), thread_id='1', parent_message_id='0', text='x'
    ):
      pass

  assert info.value.timed_out is True


async def test_connection_failure_is_a_transport_error():
  """A refused connection is reported as unreachable, not as a timeout."""

  def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('refused', request=request)

  client, _ = _make_client(_refuse)

  with pytest.raises(
      CortexTransportError, match='could not be reached'
  ) as info:
    await client.create_thread(_ctx())

  assert info.value.timed_out is False


async def test_stream_cut_mid_run_is_a_transport_error():
  """A connection dropping between chunks fails the run, not just ends it."""
  body = _Chunks([_STREAM[:37]], then_raise=httpx.ReadError('dropped'))
  client, _ = _make_client(lambda request: _sse_response(body))

  with pytest.raises(CortexTransportError, match='dropped') as info:
    async with client.run(
        _ctx(), thread_id='1', parent_message_id='0', text='x'
    ) as events:
      async for _ in events:
        pass

  assert info.value.timed_out is False
  assert body.closed


async def test_read_timeout_mid_run_is_a_timeout():
  """Snowflake going quiet between chunks is reported as a timeout."""
  body = _Chunks([_STREAM[:37]], then_raise=httpx.ReadTimeout('quiet'))
  client, _ = _make_client(lambda request: _sse_response(body))

  with pytest.raises(CortexTransportError) as info:
    async with client.run(
        _ctx(), thread_id='1', parent_message_id='0', text='x'
    ) as events:
      async for _ in events:
        pass

  assert info.value.timed_out is True


async def test_stream_ending_without_done_is_left_to_the_caller():
  """A clean close before `[DONE]` yields what arrived and stops."""
  body = _Chunks([_STREAM[:_FIRST_EVENT_END]])
  client, _ = _make_client(lambda request: _sse_response(body))

  async with client.run(
      _ctx(), thread_id='1', parent_message_id='0', text='x'
  ) as events:
    received = [event async for event in events]

  assert [e.event for e in received] == ['metadata']
  assert not any(e.is_done for e in received)


async def test_leaving_the_run_early_closes_the_upstream_response():
  """A consumer that stops reading releases the Snowflake connection."""
  body = _Chunks([_STREAM])
  client, _ = _make_client(lambda request: _sse_response(body))

  async with client.run(
      _ctx(), thread_id='1', parent_message_id='0', text='x'
  ) as events:
    async for event in events:
      first = event
      break

  assert isinstance(first, SseEvent)
  assert body.closed


# --- cancel_run and lifecycle -------------------------------------------------


async def test_cancel_run_posts_to_the_cancel_endpoint():
  """A cancel is a POST to the documented run cancel path."""
  client, requests = _make_client(lambda request: httpx.Response(200))

  cancelled = await client.cancel_run(_ctx(), '123-455')

  (request,) = requests
  assert cancelled is True
  assert request.method == 'POST'
  assert (
      request.url == f'{_ACCOUNT_URL}/api/v2/cortex/agent/runs/123-455/cancel'
  )
  assert request.headers['authorization'] == f'Bearer {_TOKEN}'


async def test_cancel_of_a_finished_run_is_reported_not_raised():
  """Snowflake answering 409 for a run that already ended is not an error."""
  client, requests = _make_client(
      lambda request: httpx.Response(
          409, json={'code': '390201', 'message': 'run already completed'}
      )
  )

  cancelled = await client.cancel_run(_ctx(), '123-455')

  assert cancelled is False
  assert len(requests) == 1


async def test_cancel_run_failure_is_reported_not_raised():
  """Cancel is best effort: an error status or a dead network is `False`."""
  client, _ = _make_client(
      lambda request: httpx.Response(500, json={'message': 'busy'})
  )

  def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError('refused', request=request)

  unreachable, _ = _make_client(_refuse)

  assert await client.cancel_run(_ctx(), '1-1') is False
  assert await unreachable.cancel_run(_ctx(), '1-1') is False


async def test_aclose_leaves_a_shared_http_client_open():
  """The application's client is the application's to close."""
  shared = httpx.AsyncClient(
      transport=httpx.MockTransport(
          lambda request: httpx.Response(200, json={'thread_id': 1})
      )
  )
  client, _ = _make_client(
      lambda request: httpx.Response(200), http_client=shared
  )
  await client.create_thread(_ctx())

  await client.aclose()

  assert shared.is_closed is False
  await shared.aclose()


async def test_aclose_closes_an_owned_http_client(
    monkeypatch: pytest.MonkeyPatch,
):
  """Without a shared client, the one created on demand is closed."""
  created: list[httpx.AsyncClient] = []
  real_async_client = httpx.AsyncClient

  def _tracking(**kwargs: Any) -> httpx.AsyncClient:
    instance = real_async_client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'thread_id': 1})
        ),
        **kwargs,
    )
    created.append(instance)
    return instance

  monkeypatch.setattr(httpx, 'AsyncClient', _tracking)
  client = SnowflakeCortexClient(
      account_url=_ACCOUNT_URL,
      database='D',
      schema_name='S',
      cortex_agent_name='A',
      header_provider=_bearer_headers,
      timeout=5.0,
  )
  await client.create_thread(_ctx())

  await client.aclose()

  (owned,) = created
  assert owned.is_closed is True
