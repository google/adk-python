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

"""Tests for the optional Bearer-token auth middleware."""

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

from google.adk.cli.api_server import _BearerAuthMiddleware
import pytest


class _CollectingApp:
  """Minimal ASGI app stub that records whether it was invoked."""

  def __init__(self) -> None:
    self.called = False

  async def __call__(self, scope, receive, send) -> None:
    self.called = True
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [(b"content-type", b"text/plain")],
    })
    await send({
        "type": "http.response.body",
        "body": b"downstream-ok",
    })


class _ResponseCapture:
  """Captures the events the middleware emits via ``send``."""

  def __init__(self) -> None:
    self.events: List[dict] = []

  async def __call__(self, message: dict) -> None:
    self.events.append(message)

  @property
  def status(self) -> Optional[int]:
    for event in self.events:
      if event.get("type") == "http.response.start":
        return event.get("status")
    return None

  @property
  def headers(self) -> List[Tuple[bytes, bytes]]:
    for event in self.events:
      if event.get("type") == "http.response.start":
        return list(event.get("headers", []))
    return []

  @property
  def body(self) -> bytes:
    chunks = []
    for event in self.events:
      if event.get("type") == "http.response.body":
        chunks.append(event.get("body", b""))
    return b"".join(chunks)


async def _noop_receive():
  return {"type": "http.request", "body": b"", "more_body": False}


def _scope(
    path: str = "/run",
    method: str = "POST",
    auth: Optional[bytes] = None,
) -> dict[str, Any]:
  headers: List[Tuple[bytes, bytes]] = []
  if auth is not None:
    headers.append((b"authorization", auth))
  return {
      "type": "http",
      "method": method,
      "path": path,
      "raw_path": path.encode("latin-1"),
      "headers": headers,
  }


@pytest.mark.asyncio
async def test_token_unset_passes_request_through_unchanged():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token=None)

  await middleware(_scope(path="/run"), _noop_receive, send)

  assert downstream.called is True
  assert send.status == 200
  assert send.body == b"downstream-ok"


@pytest.mark.asyncio
async def test_empty_string_token_is_treated_as_unset():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="")

  await middleware(_scope(path="/run"), _noop_receive, send)

  assert downstream.called is True
  assert send.status == 200


@pytest.mark.asyncio
async def test_token_set_request_without_auth_header_is_rejected():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  await middleware(_scope(path="/run"), _noop_receive, send)

  assert downstream.called is False
  assert send.status == 401
  assert b'"authentication required"' in send.body
  assert any(h[0] == b"www-authenticate" for h in send.headers)


@pytest.mark.asyncio
async def test_token_set_wrong_bearer_is_rejected():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  await middleware(
      _scope(path="/run", auth=b"Bearer wrong-token"), _noop_receive, send
  )

  assert downstream.called is False
  assert send.status == 401


@pytest.mark.asyncio
async def test_token_set_correct_bearer_is_accepted():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  await middleware(
      _scope(path="/run", auth=b"Bearer secret-token"), _noop_receive, send
  )

  assert downstream.called is True
  assert send.status == 200
  assert send.body == b"downstream-ok"


@pytest.mark.parametrize("public_path", ["/health", "/version"])
@pytest.mark.asyncio
async def test_token_set_public_paths_are_always_open(public_path: str):
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  await middleware(_scope(path=public_path, method="GET"), _noop_receive, send)

  assert downstream.called is True
  assert send.status == 200


@pytest.mark.asyncio
async def test_non_http_scopes_are_passed_through_unchanged():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  scope = {
      "type": "websocket",
      "path": "/run",
      "headers": [],
  }
  await middleware(scope, _noop_receive, send)

  assert downstream.called is True


@pytest.mark.asyncio
async def test_session_route_requires_auth_when_token_set():
  downstream = _CollectingApp()
  send = _ResponseCapture()
  middleware = _BearerAuthMiddleware(downstream, token="secret-token")

  scope = _scope(
      path="/apps/example/users/alice/sessions/sess-1",
      method="POST",
  )
  await middleware(scope, _noop_receive, send)

  assert downstream.called is False
  assert send.status == 401
