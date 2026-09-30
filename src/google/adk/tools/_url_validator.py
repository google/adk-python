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

"""Url checks shared by the tools that open a url the model supplied."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import socket
from urllib.parse import ParseResult
from urllib.parse import urlparse

_ALLOWED_URL_SCHEMES = frozenset({'http', 'https'})
_DEFAULT_PORT_BY_SCHEME = {'http': 80, 'https': 443}
# Hostnames that always designate the local machine or a metadata endpoint.
_BLOCKED_HOSTNAMES = frozenset({
    'localhost',
    'metadata',
    'metadata.goog',
})
# Hostname suffixes reserved for loopback, link-local and internal networks.
_BLOCKED_HOSTNAME_SUFFIXES = (
    '.localhost',
    '.local',
    '.internal',
    '.metadata.goog',
)
_ResolvedAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class _RequestTarget:
  parsed_url: ParseResult
  scheme: str
  hostname: str
  host_header: str


def _format_host(hostname: str) -> str:
  if ':' in hostname:
    return f'[{hostname}]'
  return hostname


def _default_port_for_scheme(scheme: str) -> int:
  return _DEFAULT_PORT_BY_SCHEME[scheme]


def _build_host_header(
    *, hostname: str, scheme: str, explicit_port: int | None
) -> str:
  formatted_hostname = _format_host(hostname)
  if explicit_port is None or explicit_port == _default_port_for_scheme(scheme):
    return formatted_hostname
  return f'{formatted_hostname}:{explicit_port}'


def _parse_request_target(url: str) -> _RequestTarget:
  parsed_url = urlparse(url)
  scheme = parsed_url.scheme.lower()
  if scheme not in _ALLOWED_URL_SCHEMES:
    raise ValueError(f'Unsupported url scheme: {url}')

  hostname = parsed_url.hostname
  if not hostname:
    raise ValueError(f'URL is missing a hostname: {url}')

  try:
    explicit_port = parsed_url.port
  except ValueError as exc:
    raise ValueError(f'Invalid url port: {url}') from exc

  return _RequestTarget(
      parsed_url=parsed_url,
      scheme=scheme,
      hostname=hostname,
      host_header=_build_host_header(
          hostname=hostname,
          scheme=scheme,
          explicit_port=explicit_port,
      ),
  )


def _parse_ip_literal(hostname: str) -> _ResolvedAddress | None:
  try:
    return ipaddress.ip_address(hostname)
  except ValueError:
    return None


def _is_blocked_hostname(hostname: str) -> bool:
  """Reports whether a name designates loopback or internal infrastructure.

  This check is purely lexical, so unlike the address checks it also applies
  when an outbound proxy performs the DNS resolution on our behalf.

  Args:
    hostname: The hostname parsed out of the requested url.

  Returns:
    True if the request must be refused without contacting the host.
  """
  normalized_hostname = hostname.rstrip('.').lower()
  if normalized_hostname in _BLOCKED_HOSTNAMES:
    return True
  return normalized_hostname.endswith(_BLOCKED_HOSTNAME_SUFFIXES)


_NAT64_WELL_KNOWN_PREFIX = ipaddress.ip_network('64:ff9b::/96')


def _embedded_ipv4(address: _ResolvedAddress) -> ipaddress.IPv4Address | None:
  """Returns the IPv4 address embedded in an IPv6 address, if any.

  ``is_global`` on the outer IPv6 address does not reflect the reachability of
  the embedded IPv4 target for IPv4-mapped (``::ffff:a.b.c.d``), IPv4-compatible
  (``::a.b.c.d``), 6to4 (``2002::/16``) and NAT64 (``64:ff9b::/96``) addresses.
  For example ``64:ff9b::169.254.169.254`` is reported as global but, on a
  network with NAT64, routes to the internal ``169.254.169.254`` metadata
  endpoint. Returning the embedded IPv4 lets the caller vet it directly.
  """
  if not isinstance(address, ipaddress.IPv6Address):
    return None
  if address.ipv4_mapped is not None:
    return address.ipv4_mapped
  if address.sixtofour is not None:
    return address.sixtofour
  if address in _NAT64_WELL_KNOWN_PREFIX:
    return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
  # IPv4-compatible ``::a.b.c.d`` (deprecated): top 96 bits zero, low 32 bits a
  # non-trivial IPv4 (excluding ``::`` and ``::1``).
  packed = int(address)
  if packed >> 32 == 0 and (packed & 0xFFFFFFFF) not in (0, 1):
    return ipaddress.IPv4Address(packed & 0xFFFFFFFF)
  return None


def _is_blocked_address(address: _ResolvedAddress) -> bool:
  if not address.is_global:
    return True
  # Reject IPv6 addresses that embed a non-global IPv4 target (NAT64,
  # IPv4-compatible, etc.), which `is_global` alone does not catch.
  embedded = _embedded_ipv4(address)
  return embedded is not None and not embedded.is_global


def _resolve_host_addresses(hostname: str) -> tuple[_ResolvedAddress, ...]:
  resolved_address = _parse_ip_literal(hostname)

  if resolved_address is not None:
    return (resolved_address,)

  try:
    address_info = socket.getaddrinfo(
        hostname,
        None,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
  except (socket.gaierror, UnicodeError) as exc:
    raise ValueError(f'Unable to resolve host: {hostname}') from exc

  resolved_addresses: list[_ResolvedAddress] = []
  for family, _, _, _, sockaddr in address_info:
    if family not in (socket.AF_INET, socket.AF_INET6):
      continue
    resolved_addresses.append(ipaddress.ip_address(sockaddr[0]))

  if not resolved_addresses:
    raise ValueError(f'Unable to resolve host: {hostname}')

  return tuple(resolved_addresses)


def _resolve_direct_addresses(hostname: str) -> tuple[_ResolvedAddress, ...]:
  resolved_addresses = tuple(dict.fromkeys(_resolve_host_addresses(hostname)))
  if any(_is_blocked_address(address) for address in resolved_addresses):
    raise ValueError(f'Blocked host: {hostname}')
  return resolved_addresses


def _reject_blocked_proxied_hostname(hostname: str) -> None:
  """Best-effort address check for a hostname that the proxy will resolve.

  The proxy performs the authoritative DNS resolution and opens the connection,
  so the local lookup here is advisory rather than a pin. It still refuses the
  common case where a public resolver maps the requested name onto a metadata,
  loopback or otherwise private address.

  A local resolution failure is not treated as an error: split-horizon DNS and
  egress-only networks legitimately leave the proxy as the only resolver, and
  failing closed there would break every such deployment. Those environments
  are covered by `_is_blocked_hostname` instead. A proxy that resolves a
  public-looking name to an internal address remains outside what a client can
  detect, and has to be constrained by the proxy's own egress policy.

  Args:
    hostname: The hostname that will be handed to the proxy.

  Raises:
    ValueError: If the local resolver maps the hostname to a non-global
      address.
  """
  try:
    resolved_addresses = _resolve_host_addresses(hostname)
  except ValueError:
    return
  if any(_is_blocked_address(address) for address in resolved_addresses):
    raise ValueError(f'Blocked host: {hostname}')
