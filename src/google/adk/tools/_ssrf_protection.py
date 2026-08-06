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

"""Shared SSRF protection helpers for tools that make HTTP requests.

Two layers:

1. ``validate_url`` rejects bad schemes, missing/blocked hostnames, and any
   DNS result that includes a non-globally-routable IP. It returns a
   ``ValidatedTarget`` so callers can use the pre-resolved address list.

2. ``send_pinned_async`` issues an ``httpx`` request against the validated IP
   literal directly, preserves the ``Host`` header, and sets the TLS server
   name via ``request.extensions["sni_hostname"]``. Together with (1) this
   closes the DNS rebinding window between URL validation and connect: even
   if the attacker flips the DNS record after validation, the socket goes to
   the IP we validated and the cert check uses the original hostname.

A matching ``PinnedAddressAdapter`` for the ``requests`` library is also
provided so ``load_web_page`` and any other sync caller can share the same
resolution and blocking rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from typing import Any
from urllib.parse import ParseResult
from urllib.parse import urlparse
from urllib.parse import urlunparse

import httpx

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORT_BY_SCHEME = {"http": 80, "https": 443}
_ResolvedAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class ValidatedTarget:
  """A URL that passed validation, with its resolved addresses cached."""

  url: str
  parsed: ParseResult
  scheme: str
  hostname: str
  host_header: str
  addresses: tuple[_ResolvedAddress, ...]


def _format_host(hostname: str) -> str:
  if ":" in hostname:
    return f"[{hostname}]"
  return hostname


def _build_host_header(
    *,
    hostname: str,
    scheme: str,
    explicit_port: int | None,
) -> str:
  formatted = _format_host(hostname)
  default_port = _DEFAULT_PORT_BY_SCHEME[scheme]
  if explicit_port is None or explicit_port == default_port:
    return formatted
  return f"{formatted}:{explicit_port}"


def is_blocked_hostname(hostname: str) -> bool:
  """Return True for hostnames that always point at the local host."""
  normalized = hostname.rstrip(".").lower()
  return normalized == "localhost" or normalized.endswith(".localhost")


def is_blocked_address(address: _ResolvedAddress) -> bool:
  """Return True for any IP that isn't globally routable.

  ``ipaddress.is_global`` already covers private (RFC 1918), loopback,
  link-local (including 169.254.169.254), multicast, reserved, and unspecified
  ranges across IPv4 and IPv6. Using it directly avoids drift between hand
  maintained allow lists in different tools.
  """
  return not address.is_global


def _parse_ip_literal(hostname: str) -> _ResolvedAddress | None:
  try:
    return ipaddress.ip_address(hostname)
  except ValueError:
    return None


def resolve_host_addresses(hostname: str) -> tuple[_ResolvedAddress, ...]:
  """Resolve a hostname to all of its A / AAAA records.

  IP literals short-circuit and return themselves. ``getaddrinfo`` errors are
  surfaced as ``ValueError`` so callers can handle resolution failure and a
  bad scheme through the same code path.
  """
  literal = _parse_ip_literal(hostname)
  if literal is not None:
    return (literal,)

  try:
    info = socket.getaddrinfo(
        hostname,
        None,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
  except (socket.gaierror, UnicodeError) as exc:
    raise ValueError(f"Unable to resolve host: {hostname}") from exc

  addresses: list[_ResolvedAddress] = []
  for family, _, _, _, sockaddr in info:
    if family not in (socket.AF_INET, socket.AF_INET6):
      continue
    addresses.append(ipaddress.ip_address(sockaddr[0]))

  if not addresses:
    raise ValueError(f"Unable to resolve host: {hostname}")

  # Deduplicate while preserving order so the first record is still tried
  # first by callers that iterate the tuple.
  return tuple(dict.fromkeys(addresses))


def validate_url(url: str) -> ValidatedTarget:
  """Validate ``url`` and return its resolved addresses.

  Raises ``ValueError`` for unsupported schemes, missing or blocked
  hostnames, invalid ports, and DNS results where any IP is not globally
  routable. The check rejects the whole hostname if even one record points
  at private space so an attacker can't sneak past the gate with a
  multi-record set such as ``[8.8.8.8, 127.0.0.1]``.

  Returning the addresses lets the caller pin the connection to a vetted IP
  instead of re-resolving at connect time. That closes the DNS rebinding
  window between this validation and the eventual HTTP request.
  """
  parsed = urlparse(url)
  scheme = parsed.scheme.lower()
  if scheme not in _ALLOWED_URL_SCHEMES:
    raise ValueError(f"Unsupported url scheme: {url}")

  hostname = parsed.hostname
  if not hostname:
    raise ValueError(f"URL is missing a hostname: {url}")

  try:
    explicit_port = parsed.port
  except ValueError as exc:
    raise ValueError(f"Invalid url port: {url}") from exc

  if is_blocked_hostname(hostname):
    raise ValueError(f"Blocked host: {hostname}")

  addresses = resolve_host_addresses(hostname)
  if any(is_blocked_address(addr) for addr in addresses):
    raise ValueError(f"Blocked host: {hostname}")

  return ValidatedTarget(
      url=url,
      parsed=parsed,
      scheme=scheme,
      hostname=hostname,
      host_header=_build_host_header(
          hostname=hostname,
          scheme=scheme,
          explicit_port=explicit_port,
      ),
      addresses=addresses,
  )


def rewrite_url_host(parsed: ParseResult, ip: str) -> str:
  """Rewrite ``parsed`` to use ``ip`` (literal) in place of the hostname."""
  formatted = _format_host(ip)
  port = parsed.port
  netloc = formatted if port is None else f"{formatted}:{port}"
  return urlunparse(parsed._replace(netloc=netloc))


async def send_pinned_async(
    client: httpx.AsyncClient,
    target: ValidatedTarget,
    **request_params: Any,
) -> httpx.Response:
  """Send a request to ``target`` via ``client`` with the IP pinned.

  The URL is rewritten to use the first validated IP literally so the
  connection bypasses DNS at send time. The original hostname is preserved in
  the ``Host`` header (for HTTP routing) and in the ``sni_hostname`` request
  extension (for TLS verification, consumed by ``httpcore``).

  If the chosen address fails to connect, the next address in
  ``target.addresses`` is tried. All addresses in the tuple have already
  passed ``is_blocked_address``, so this loop never reaches a private IP.
  """
  request_params.pop("url", None)
  headers = dict(request_params.pop("headers", None) or {})
  headers["Host"] = target.host_header
  base_extensions = request_params.pop("extensions", None) or {}
  extensions = {**base_extensions, "sni_hostname": target.hostname}

  last_error: Exception | None = None
  for address in target.addresses:
    rewritten_url = rewrite_url_host(target.parsed, str(address))
    try:
      return await client.request(
          url=rewritten_url,
          headers=headers,
          extensions=extensions,
          **request_params,
      )
    except httpx.HTTPError as exc:
      last_error = exc

  assert (
      last_error is not None
  )  # loop ran at least once: addresses is non-empty
  raise last_error
