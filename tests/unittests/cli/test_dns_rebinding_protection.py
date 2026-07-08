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

from google.adk.cli.api_server import _is_loopback_address
from google.adk.cli.api_server import _is_request_origin_allowed
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
