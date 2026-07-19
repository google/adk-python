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

import ipaddress
import socket

from google.adk.tools._ssrf_protection import is_blocked_address
from google.adk.tools._ssrf_protection import is_blocked_hostname
from google.adk.tools._ssrf_protection import resolve_host_addresses
from google.adk.tools._ssrf_protection import rewrite_url_host
from google.adk.tools._ssrf_protection import send_pinned_async
from google.adk.tools._ssrf_protection import validate_url
import httpx
import pytest


class TestIsBlockedHostname:

  def test_localhost_blocked(self):
    assert is_blocked_hostname("localhost")

  def test_localhost_trailing_dot(self):
    assert is_blocked_hostname("localhost.")

  def test_subdomain_localhost_blocked(self):
    assert is_blocked_hostname("foo.localhost")

  def test_case_insensitive(self):
    assert is_blocked_hostname("LOCALHOST")

  def test_normal_hostname_allowed(self):
    assert not is_blocked_hostname("example.com")

  def test_hostname_containing_localhost_allowed(self):
    assert not is_blocked_hostname("notlocalhost.com")


class TestIsBlockedAddress:

  def test_loopback_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("127.0.0.1"))

  def test_link_local_blocked(self):
    # 169.254.169.254 is the AWS / GCP / Azure metadata endpoint.
    assert is_blocked_address(ipaddress.ip_address("169.254.169.254"))

  def test_private_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("10.0.0.1"))
    assert is_blocked_address(ipaddress.ip_address("192.168.1.1"))
    assert is_blocked_address(ipaddress.ip_address("172.16.0.1"))

  def test_ipv6_loopback_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("::1"))

  def test_ipv6_link_local_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("fe80::1"))

  def test_ipv6_unique_local_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("fc00::1"))

  def test_global_allowed(self):
    assert not is_blocked_address(ipaddress.ip_address("8.8.8.8"))

  def test_ipv6_global_allowed(self):
    assert not is_blocked_address(ipaddress.ip_address("2001:4860:4860::8888"))


class TestResolveHostAddresses:

  def test_ip_literal_short_circuits(self):
    addrs = resolve_host_addresses("8.8.8.8")
    assert addrs == (ipaddress.ip_address("8.8.8.8"),)

  def test_ipv6_literal_short_circuits(self):
    addrs = resolve_host_addresses("2001:4860:4860::8888")
    assert addrs == (ipaddress.ip_address("2001:4860:4860::8888"),)

  def test_resolve_returns_all_records(self, monkeypatch):
    def fake(host, port, *args, **kwargs):
      return [
          (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
          (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", 0)),
      ]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    addrs = resolve_host_addresses("multi.example.com")
    assert addrs == (
        ipaddress.ip_address("8.8.8.8"),
        ipaddress.ip_address("8.8.4.4"),
    )

  def test_resolve_failure_raises_value_error(self, monkeypatch):
    def fake(host, port, *args, **kwargs):
      raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    with pytest.raises(ValueError, match="Unable to resolve host"):
      resolve_host_addresses("no-such-host.example")


@pytest.fixture
def patch_dns(monkeypatch):
  """Map a few example hostnames to canned addresses for validate_url tests."""

  responses = {
      "api.example.com": [("8.8.8.8", socket.AF_INET)],
      "rebinder.example.com": [
          ("8.8.4.4", socket.AF_INET),
          ("127.0.0.1", socket.AF_INET),
      ],
      "internal.example.com": [("10.0.0.5", socket.AF_INET)],
  }
  original = socket.getaddrinfo

  def fake(host, port, *args, **kwargs):
    if host in responses:
      return [
          (family, socket.SOCK_STREAM, 6, "", (ip, 0))
          for ip, family in responses[host]
      ]
    return original(host, port, *args, **kwargs)

  monkeypatch.setattr(socket, "getaddrinfo", fake)


class TestValidateUrl:

  def test_localhost_blocked(self):
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("http://localhost:8080/path")

  def test_loopback_ip_blocked(self):
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("http://127.0.0.1/path")

  def test_link_local_blocked(self):
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("http://169.254.169.254/latest/meta-data/")

  def test_private_ip_blocked(self):
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("http://10.0.0.1/internal")

  def test_ftp_scheme_blocked(self):
    with pytest.raises(ValueError, match="Unsupported url scheme"):
      validate_url("ftp://example.com/file")

  def test_file_scheme_blocked(self):
    with pytest.raises(ValueError, match="Unsupported url scheme"):
      validate_url("file:///etc/passwd")

  def test_no_hostname_blocked(self):
    with pytest.raises(ValueError, match="missing a hostname"):
      validate_url("http:///path")

  def test_public_url_allowed(self, patch_dns):
    target = validate_url("https://api.example.com/v1/resource")
    assert target.hostname == "api.example.com"
    assert target.scheme == "https"
    assert target.addresses == (ipaddress.ip_address("8.8.8.8"),)

  def test_rebinder_blocked_when_any_record_is_private(self, patch_dns):
    # rebinder.example.com resolves to one public IP and one loopback IP.
    # An attacker controlling DNS could flip records between this check and
    # the actual connect. Rejecting the hostname when any record is private
    # closes that window.
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("https://rebinder.example.com/x")

  def test_internal_hostname_blocked(self, patch_dns):
    with pytest.raises(ValueError, match="Blocked host"):
      validate_url("https://internal.example.com/admin")

  def test_validated_target_has_addresses(self, patch_dns):
    target = validate_url("https://api.example.com/v1/resource")
    assert len(target.addresses) == 1
    assert not any(is_blocked_address(a) for a in target.addresses)


class TestRewriteUrlHost:

  def test_basic_replacement(self):
    from urllib.parse import urlparse

    parsed = urlparse("https://api.example.com/v1/resource")
    assert rewrite_url_host(parsed, "8.8.8.8") == "https://8.8.8.8/v1/resource"

  def test_preserves_explicit_port(self):
    from urllib.parse import urlparse

    parsed = urlparse("https://api.example.com:8443/v1/resource")
    assert (
        rewrite_url_host(parsed, "8.8.8.8")
        == "https://8.8.8.8:8443/v1/resource"
    )

  def test_ipv6_brackets(self):
    from urllib.parse import urlparse

    parsed = urlparse("https://api.example.com/x")
    assert rewrite_url_host(parsed, "2001:db8::1") == "https://[2001:db8::1]/x"


class TestSendPinnedAsync:

  @pytest.mark.asyncio
  async def test_pins_url_and_sets_host_and_sni(self, patch_dns):
    captured: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
      captured.append(request)
      return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(mock_handler)
    target = validate_url("https://api.example.com/v1/resource")

    async with httpx.AsyncClient(transport=transport) as client:
      response = await send_pinned_async(
          client,
          target,
          method="GET",
      )

    assert response.status_code == 200
    assert len(captured) == 1
    sent = captured[0]
    # The URL should hit the validated IP literally so DNS at send time
    # can't redirect to a private IP.
    assert sent.url.host == "8.8.8.8"
    # The Host header keeps the original hostname so the remote server
    # routes the request to the right vhost.
    assert sent.headers["Host"] == "api.example.com"
    # The SNI extension keeps the original hostname for TLS cert validation.
    assert sent.extensions.get("sni_hostname") == "api.example.com"

  @pytest.mark.asyncio
  async def test_passes_method_and_body_through(self, patch_dns):
    captured: list[httpx.Request] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
      captured.append(request)
      return httpx.Response(201, json={"created": True})

    transport = httpx.MockTransport(mock_handler)
    target = validate_url("https://api.example.com/v1/users")

    async with httpx.AsyncClient(transport=transport) as client:
      response = await send_pinned_async(
          client,
          target,
          method="POST",
          json={"name": "alice"},
          headers={"X-Custom": "v"},
      )

    assert response.status_code == 201
    sent = captured[0]
    assert sent.method == "POST"
    assert sent.headers["X-Custom"] == "v"
    assert sent.headers["Host"] == "api.example.com"

  @pytest.mark.asyncio
  async def test_tries_next_address_on_connect_failure(self, monkeypatch):
    # Simulate a hostname that resolves to two public IPs. The first call
    # fails; the second succeeds. Both must already be in the validated
    # address list. This verifies the fallback walks the list rather than
    # giving up after the first error.

    def fake_getaddrinfo(host, port, *args, **kwargs):
      if host == "two.example.com":
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", 0)),
        ]
      raise socket.gaierror(8, "no")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    seen_hosts: list[str] = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
      seen_hosts.append(request.url.host)
      if request.url.host == "8.8.8.8":
        raise httpx.ConnectError("simulated connect failure")
      return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(mock_handler)
    target = validate_url("https://two.example.com/path")

    async with httpx.AsyncClient(transport=transport) as client:
      response = await send_pinned_async(client, target, method="GET")

    assert response.status_code == 200
    assert seen_hosts == ["8.8.8.8", "8.8.4.4"]
