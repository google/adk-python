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

"""Tests for DNS-rebinding protection in _OriginCheckMiddleware."""

import asyncio
from typing import Any
from typing import Optional

from google.adk.cli.api_server import _get_allowed_request_hosts
from google.adk.cli.api_server import _is_dns_rebinding_request
from google.adk.cli.api_server import _is_loopback_address
from google.adk.cli.api_server import _is_request_origin_allowed
from google.adk.cli.api_server import _OriginCheckMiddleware
import pytest


class TestIsLoopbackAddress:
  """Unit tests for _is_loopback_address."""

  @pytest.mark.parametrize(
      "host",
      [
          "127.0.0.1",
          "localhost",
          "::1",
          "[::1]",
          "127.0.0.1:8000",
          "localhost:8000",
          "[::1]:8000",
          "127.1.2.3",  # any 127.x.x.x is loopback
      ],
  )
  def test_loopback_hosts(self, host: str):
    assert _is_loopback_address(host), f"{host!r} should be loopback"

  @pytest.mark.parametrize(
      "host",
      [
          "evil.com",
          "127.evil.com",
          "0.0.0.0",
          "192.168.1.1",
          "10.0.0.1",
          "128.0.0.1",
          "",
      ],
  )
  def test_non_loopback_hosts(self, host: str):
    assert not _is_loopback_address(host), f"{host!r} should NOT be loopback"


class TestDnsRebindingProtection:
  """Tests that DNS-rebinding attacks are blocked when server is on loopback."""

  def _make_scope(
      self, server_host: str = "127.0.0.1", host_header: str = "127.0.0.1:8000"
  ) -> dict:
    """Build a minimal ASGI scope for testing."""
    return {
        "type": "http",
        "method": "POST",
        "server": (server_host, 8000),
        "headers": [
            (b"host", host_header.encode()),
        ],
        "scheme": "http",
    }

  # --- DNS rebinding scenarios (should be BLOCKED) ---

  def test_dns_rebinding_evil_origin_loopback_server_no_configured_origins(
      self,
  ):
    """Attacker page (evil.com) DNS-rebinds to 127.0.0.1 and sends a POST.

    Browser sends Origin: http://evil.com, Host: evil.com.
    Server is bound to 127.0.0.1.
    No explicit allow-origins configured.
    Expected: BLOCKED.
    """
    scope = self._make_scope(
        server_host="127.0.0.1", host_header="evil.com:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert (
        not result
    ), "DNS-rebinding from evil.com should be blocked on loopback server"

  def test_dns_rebinding_127_evil_origin(self):
    """Origin header host starts with '127.' but is a hostname (127.evil.com)."""
    scope = self._make_scope(
        server_host="127.0.0.1", host_header="127.evil.com:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://127.evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  def test_dns_rebinding_localhost_server(self):
    """Same attack, server bound as 'localhost'."""
    scope = self._make_scope(server_host="localhost", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  def test_dns_rebinding_ipv6_loopback_server(self):
    """Same attack, server bound to ::1."""
    scope = self._make_scope(server_host="::1", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  # --- Legitimate same-origin requests (should be ALLOWED) ---

  def test_same_origin_localhost_allowed(self):
    """Legitimate browser request from localhost UI to localhost server."""
    scope = self._make_scope(
        server_host="127.0.0.1", host_header="127.0.0.1:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://127.0.0.1:8000",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert result, "Same-origin localhost request should be allowed"

  def test_same_origin_localhost_named(self):
    """Browser opens http://localhost:8000 -> requests to localhost:8000."""
    scope = self._make_scope(
        server_host="127.0.0.1", host_header="localhost:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://localhost:8000",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert result

  # --- Explicit allow-origins configured (allow-list bypasses DNS guard) ---

  def test_explicit_allowlist_overrides_dns_rebinding_guard(self):
    """If the developer explicitly allows evil.com, it should be permitted."""
    scope = self._make_scope(server_host="127.0.0.1", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=["http://evil.com"],
        allowed_origin_regex=None,
        has_configured_allowed_origins=True,
    )
    assert result, "Explicitly allowed origin should still pass"

  # --- Non-loopback server (protection does not apply) ---

  def test_non_loopback_server_no_dns_guard(self):
    """Server bound to 0.0.0.0 — DNS guard must not interfere with same-origin check."""
    scope = self._make_scope(
        server_host="0.0.0.0", host_header="example.com:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://example.com:8000",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert result, "Same-origin on public server should be allowed"


def _make_http_scope(
    method: str = "GET",
    server_host: str = "127.0.0.1",
    host_header: Optional[str] = "127.0.0.1:8000",
    origin: Optional[str] = None,
    extra_headers: Optional[list[tuple[bytes, bytes]]] = None,
) -> dict[str, Any]:
  """Builds a minimal ASGI HTTP scope."""
  # server_host is the local end of the connection, which is what ASGI servers
  # report, not the address the server was told to bind.
  headers: list[tuple[bytes, bytes]] = []
  if host_header is not None:
    headers.append((b"host", host_header.encode()))
  if origin is not None:
    headers.append((b"origin", origin.encode()))
  headers.extend(extra_headers or [])
  return {
      "type": "http",
      "method": method,
      "server": (server_host, 8000),
      "headers": headers,
      "scheme": "http",
  }


class TestGetAllowedRequestHosts:
  """Unit tests for deriving accepted Host values from --allow_origins."""

  def test_no_configuration_accepts_nothing_extra(self):
    assert _get_allowed_request_hosts([]) == frozenset()

  def test_literal_origins_contribute_their_hosts(self):
    """Hosts are compared case-insensitively, so they are folded here."""
    assert _get_allowed_request_hosts(
        ["https://Proxy.Example.COM", "http://localhost:3000"]
    ) == frozenset({"proxy.example.com", "localhost"})

  def test_entry_without_a_host_contributes_nothing(self):
    """A scheme-less or unparsable entry has no hostname to vouch for."""
    assert (
        _get_allowed_request_hosts(["localhost:3000", "", "http://[::1"])
        == frozenset()
    )

  def test_only_wildcard_disables_the_guard(self):
    """A wildcard already says "accept anything" out loud."""
    assert _get_allowed_request_hosts(["*"]) is None


class TestIsDnsRebindingRequest:
  """Unit tests for the Host-header based DNS-rebinding guard."""

  def test_rebound_host_on_loopback_bind_is_rejected(self):
    """The attacker's domain in Host, while we are bound to loopback."""
    scope = _make_http_scope(host_header="evil.com:8000")
    assert _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())

  @pytest.mark.parametrize(
      "host_header", ["localhost:8000", "127.0.0.1:8000", "[::1]:8000"]
  )
  def test_loopback_host_is_accepted(self, host_header):
    scope = _make_http_scope(host_header=host_header)
    assert not _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())

  def test_forwarded_headers_cannot_vouch_for_the_host(self):
    """Regression: a rebound page can set these, so only Host can be trusted."""
    for spoofed in [
        (b"x-forwarded-host", b"127.0.0.1:8000"),
        (b"forwarded", b"proto=http;host=127.0.0.1:8000"),
        (b"x-forwarded-host", b"127.0.0.1, evil.com"),
    ]:
      scope = _make_http_scope(
          host_header="evil.com:8000", extra_headers=[spoofed]
      )
      assert _is_dns_rebinding_request(
          scope, "127.0.0.1", frozenset()
      ), f"{spoofed!r} must not overrule the Host header"

  def test_host_from_allow_origins_is_accepted(self):
    """A same-machine reverse proxy is named via --allow_origins."""
    scope = _make_http_scope(host_header="proxy.example.com")
    assert not _is_dns_rebinding_request(
        scope, "127.0.0.1", frozenset({"proxy.example.com"})
    )

  def test_allow_origins_does_not_vouch_for_other_hosts(self):
    """Regression: configuring an origin must not disable the guard wholesale."""
    scope = _make_http_scope(host_header="evil.com:8000")
    assert _is_dns_rebinding_request(
        scope, "127.0.0.1", frozenset({"proxy.example.com"})
    )

  def test_blanket_allow_origins_disables_the_guard(self):
    scope = _make_http_scope(host_header="evil.com:8000")
    assert not _is_dns_rebinding_request(scope, "127.0.0.1", None)

  @pytest.mark.parametrize("bind_host", ["0.0.0.0", "::", "192.168.1.5"])
  def test_non_loopback_bind_is_not_guarded(self, bind_host):
    """`adk deploy` binds a public interface to serve other hosts on purpose."""
    scope = _make_http_scope(host_header="my-service.run.app")
    assert not _is_dns_rebinding_request(scope, bind_host, frozenset())

  def test_wildcard_bind_reached_over_loopback_is_not_guarded(self):
    """Regression: scope["server"] is the accepted socket, not the bind.

    A wildcard bind reports 127.0.0.1 for a loopback connection, which is what
    a same-host proxy makes; keying off it would 403 every one.
    """
    scope = _make_http_scope(
        server_host="127.0.0.1", host_header="my-service.run.app"
    )
    assert not _is_dns_rebinding_request(scope, "0.0.0.0", frozenset())

  def test_unknown_bind_is_not_guarded(self):
    """Regression: guessing an embedded app's bind would 403 its own traffic."""
    scope = _make_http_scope(
        server_host="127.0.0.1", host_header="evil.com:8000"
    )
    assert not _is_dns_rebinding_request(scope, None, frozenset())

  @pytest.mark.parametrize(
      "host_header",
      [
          "127.0.0.1, evil.com",
          "evil.com, 127.0.0.1",
          "[::1].evil.com",
          "[::1]evil.com",
          "[::1",
          "127.0.0.1:8000.evil.com",
          "localhost:8000x",
          "[127.0.0.1]@evil.com",
      ],
  )
  def test_smuggled_host_is_rejected(self, host_header):
    """A single loopback-looking token must not vouch for the whole header."""
    scope = _make_http_scope(host_header=host_header)
    assert _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())

  def test_duplicate_host_headers_are_rejected(self):
    """The loopback one comes first, so only the singleton rule can reject."""
    scope = _make_http_scope(
        host_header="127.0.0.1:8000",
        extra_headers=[(b"host", b"evil.com:8000")],
    )
    assert _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())

  @pytest.mark.parametrize(
      "host_header", ["LOCALHOST:8000", "localhost.:8000", "LocalHost."]
  )
  def test_loopback_host_spellings_are_accepted(self, host_header):
    """Host names are case-insensitive and may carry the root dot."""
    scope = _make_http_scope(host_header=host_header)
    assert not _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())

  def test_missing_host_header_is_accepted(self):
    """Non-browser clients may omit Host; they are not a rebinding vector."""
    scope = _make_http_scope(host_header=None)
    assert not _is_dns_rebinding_request(scope, "127.0.0.1", frozenset())


class TestOriginCheckMiddleware:
  """End-to-end checks that reads are guarded, not just state-changing calls."""

  def _call(
      self,
      scope: dict[str, Any],
      bind_host: Optional[str] = "127.0.0.1",
      allow_origins: Optional[list[str]] = None,
  ) -> tuple[Optional[int], bool]:
    """Returns (status code, whether the wrapped app was reached)."""
    reached = False
    statuses: list[int] = []

    async def inner_app(scope, receive, send):
      del receive
      nonlocal reached
      reached = True
      if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(message):
      if message["type"] == "http.response.start":
        statuses.append(message["status"])

    async def receive():
      return {"type": "http.request", "body": b"", "more_body": False}

    middleware = _OriginCheckMiddleware(
        inner_app,
        has_configured_allowed_origins=bool(allow_origins),
        allowed_origins=allow_origins or [],
        allowed_origin_regex=None,
        bind_host=bind_host,
    )
    asyncio.run(middleware(scope, receive, send))
    return (statuses[0] if statuses else None), reached

  def test_proxy_host_named_in_allow_origins_is_served(self):
    """A loopback bind behind a same-machine proxy names the proxy origin."""
    status, reached = self._call(
        _make_http_scope(
            host_header="proxy.example.com",
            origin="https://proxy.example.com",
        ),
        allow_origins=["https://proxy.example.com"],
    )
    assert status == 200
    assert reached

  @pytest.mark.parametrize("origin", [None, "http://evil.com:8000"])
  @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST"])
  def test_rebound_host_is_blocked_for_every_method(self, method, origin):
    """Regression: reads, and requests without Origin, both skipped the check."""
    status, reached = self._call(
        _make_http_scope(
            method=method, host_header="evil.com:8000", origin=origin
        )
    )
    assert status == 403
    assert not reached

  @pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "POST"])
  def test_same_origin_dev_ui_still_allowed(self, method: str):
    status, reached = self._call(
        _make_http_scope(
            method=method,
            host_header="localhost:8000",
            origin="http://localhost:8000",
        )
    )
    assert status == 200
    assert reached

  def test_local_request_without_origin_allowed(self):
    """curl, the ADK CLI and same-origin browser reads send no Origin."""
    status, reached = self._call(_make_http_scope(host_header="127.0.0.1:8000"))
    assert status == 200
    assert reached

  def test_cross_origin_get_with_foreign_origin_is_blocked(self):
    """A read carrying a foreign Origin is no longer waved through."""
    status, reached = self._call(
        _make_http_scope(host_header="127.0.0.1:8000", origin="http://evil.com")
    )
    assert status == 403
    assert not reached

  def test_configured_origin_allowed_for_reads(self):
    status, reached = self._call(
        _make_http_scope(
            host_header="127.0.0.1:8000", origin="http://localhost:3000"
        ),
        allow_origins=["http://localhost:3000"],
    )
    assert status == 200
    assert reached

  def test_public_bind_same_origin_still_allowed(self):
    """`adk deploy` containers bind 0.0.0.0 and serve a real hostname."""
    status, reached = self._call(
        _make_http_scope(
            host_header="my-service.run.app",
            origin="http://my-service.run.app",
        ),
        bind_host="0.0.0.0",
    )
    assert status == 200
    assert reached

  def test_non_http_scope_is_passed_through(self):
    """Lifespan messages are not requests and carry nothing to validate."""
    scope = _make_http_scope(host_header="evil.com:8000")
    scope["type"] = "lifespan"
    _, reached = self._call(scope)
    assert reached
