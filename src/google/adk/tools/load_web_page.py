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

"""Tool for web browse."""

from dataclasses import dataclass
import ipaddress
import socket
from typing import Any
from urllib.parse import ParseResult
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.utils import get_environ_proxies
from requests.utils import select_proxy

_ALLOWED_URL_SCHEMES = frozenset({'http', 'https'})
_DEFAULT_PORT_BY_SCHEME = {'http': 80, 'https': 443}
_ResolvedAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


@dataclass(frozen=True)
class _RequestTarget:
  parsed_url: ParseResult
  scheme: str
  hostname: str
  host_header: str


class _PinnedAddressAdapter(HTTPAdapter):
  """Routes a request to a vetted IP while preserving the original host."""

  def __init__(
      self,
      *,
      rewritten_url: str,
      host_header: str,
      hostname: str,
  ) -> None:
    super().__init__()
    self._rewritten_url = rewritten_url
    self._host_header = host_header
    self._hostname = hostname

  def build_connection_pool_key_attributes(
      self,
      request: requests.PreparedRequest,
      verify: bool | str,
      cert: tuple[str, str] | str | None = None,
  ) -> tuple[dict[str, Any], dict[str, Any]]:
    host_params, pool_kwargs = super().build_connection_pool_key_attributes(
        request, verify, cert
    )
    if host_params['scheme'] == 'https':
      pool_kwargs['assert_hostname'] = self._hostname
      pool_kwargs['server_hostname'] = self._hostname
    return host_params, pool_kwargs

  def send(
      self,
      request: requests.PreparedRequest,
      stream: bool = False,
      timeout: Any = None,
      verify: bool | str = True,
      cert: tuple[str, str] | str | None = None,
      proxies: dict[str, str | None] | None = None,
  ) -> requests.Response:
    prepared_request = request.copy()
    prepared_request.headers['Host'] = self._host_header
    prepared_request.url = self._rewritten_url
    return super().send(
        prepared_request,
        stream=stream,
        timeout=timeout,
        verify=verify,
        cert=cert,
        proxies=proxies,
    )


def _failed_to_fetch_message(url: str) -> str:
  return f'Failed to fetch url: {url}'


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
  normalized_hostname = hostname.rstrip('.').lower()
  return normalized_hostname == 'localhost' or normalized_hostname.endswith(
      '.localhost'
  )


def _is_blocked_address(address: _ResolvedAddress) -> bool:
  return not address.is_global


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


def _get_proxy_url(url: str) -> str | None:
  proxies = get_environ_proxies(url)
  return select_proxy(url, proxies)


def _resolve_direct_addresses(hostname: str) -> tuple[_ResolvedAddress, ...]:
  resolved_addresses = tuple(dict.fromkeys(_resolve_host_addresses(hostname)))
  if any(_is_blocked_address(address) for address in resolved_addresses):
    raise ValueError(f'Blocked host: {hostname}')
  return resolved_addresses


def _rewrite_url_host(parsed_url: ParseResult, hostname: str) -> str:
  explicit_port = parsed_url.port
  formatted_hostname = _format_host(hostname)
  if explicit_port is None:
    rewritten_netloc = formatted_hostname
  else:
    rewritten_netloc = f'{formatted_hostname}:{explicit_port}'
  return parsed_url._replace(netloc=rewritten_netloc).geturl()


def _fetch_direct_response(
    *,
    url: str,
    target: _RequestTarget,
    resolved_addresses: tuple[_ResolvedAddress, ...],
) -> requests.Response:
  last_error: requests.RequestException | None = None
  for address in resolved_addresses:
    session = requests.Session()
    adapter = _PinnedAddressAdapter(
        rewritten_url=_rewrite_url_host(target.parsed_url, str(address)),
        host_header=target.host_header,
        hostname=target.hostname,
    )
    session.mount(f'{target.scheme}://', adapter)
    try:
      return session.get(
          url,
          allow_redirects=False,
          proxies={'http': None, 'https': None},
      )
    except requests.RequestException as exc:
      last_error = exc
    finally:
      session.close()

  if last_error is not None:
    raise last_error
  raise requests.RequestException(f'Unable to fetch url: {url}')


def _fetch_response(url: str) -> requests.Response:
  target = _parse_request_target(url)

  if _is_blocked_hostname(target.hostname):
    raise ValueError(f'Blocked host: {target.hostname}')

  parsed_ip_literal = _parse_ip_literal(target.hostname)
  if _get_proxy_url(url):
    # Proxies resolve the target hostname remotely, so only literal IPs and
    # localhost-style names can be rejected locally without breaking proxy use.
    if parsed_ip_literal is not None and _is_blocked_address(parsed_ip_literal):
      raise ValueError(f'Blocked host: {target.hostname}')
    return requests.get(url, allow_redirects=False)

  if parsed_ip_literal is not None:
    if _is_blocked_address(parsed_ip_literal):
      raise ValueError(f'Blocked host: {target.hostname}')
    return _fetch_direct_response(
        url=url,
        target=target,
        resolved_addresses=(parsed_ip_literal,),
    )

  resolved_addresses = _resolve_direct_addresses(target.hostname)
  return _fetch_direct_response(
      url=url,
      target=target,
      resolved_addresses=resolved_addresses,
  )


def load_web_page(url: str) -> str:
  """Fetches the content in the url and returns the text in it.

  Args:
      url (str): The url to browse.

  Returns:
      str: The text content of the url.
  """
  from bs4 import BeautifulSoup

  try:
    response = _fetch_response(url)
  except ValueError:
    return _failed_to_fetch_message(url)

  # Set allow_redirects=False to prevent SSRF attacks via redirection.
  if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'lxml')
    text = soup.get_text(separator='\n', strip=True)
  else:
    text = _failed_to_fetch_message(url)

  # Split the text into lines, filtering out very short lines
  # (e.g., single words or short subtitles)
  return '\n'.join(line for line in text.splitlines() if len(line.split()) > 3)
