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

"""Shared SSRF protection helpers for tools that make HTTP requests."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_ResolvedAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def is_blocked_hostname(hostname: str) -> bool:
  normalized = hostname.rstrip(".").lower()
  return normalized == "localhost" or normalized.endswith(".localhost")


def is_blocked_address(address: _ResolvedAddress) -> bool:
  return not address.is_global


def resolve_host_addresses(hostname: str) -> tuple[_ResolvedAddress, ...]:
  try:
    addr = ipaddress.ip_address(hostname)
    return (addr,)
  except ValueError:
    pass

  try:
    address_info = socket.getaddrinfo(
        hostname,
        None,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
  except (socket.gaierror, UnicodeError) as exc:
    raise ValueError(f"Unable to resolve host: {hostname}") from exc

  resolved: list[_ResolvedAddress] = []
  for family, _, _, _, sockaddr in address_info:
    if family not in (socket.AF_INET, socket.AF_INET6):
      continue
    resolved.append(ipaddress.ip_address(sockaddr[0]))

  if not resolved:
    raise ValueError(f"Unable to resolve host: {hostname}")

  return tuple(dict.fromkeys(resolved))


def validate_url(url: str) -> None:
  """Validate a URL against SSRF attacks.

  Raises ValueError if the URL targets a blocked host or scheme.
  """
  parsed = urlparse(url)
  scheme = parsed.scheme.lower()
  if scheme not in _ALLOWED_URL_SCHEMES:
    raise ValueError(f"Unsupported url scheme: {url}")

  hostname = parsed.hostname
  if not hostname:
    raise ValueError(f"URL is missing a hostname: {url}")

  if is_blocked_hostname(hostname):
    raise ValueError(f"Blocked host: {hostname}")

  resolved = resolve_host_addresses(hostname)
  if any(is_blocked_address(addr) for addr in resolved):
    raise ValueError(f"Blocked host: {hostname}")
