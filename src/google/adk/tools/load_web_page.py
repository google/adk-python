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

import itertools
import time
from typing import Any
import unicodedata
from urllib.parse import ParseResult

import requests
from requests.adapters import HTTPAdapter

from ._url_validator import _format_host
from ._url_validator import _get_proxy_url
from ._url_validator import _is_blocked_address
from ._url_validator import _is_blocked_hostname
from ._url_validator import _parse_ip_literal
from ._url_validator import _parse_request_target
from ._url_validator import _reject_blocked_proxied_hostname
from ._url_validator import _RequestTarget
from ._url_validator import _resolve_direct_addresses
from ._url_validator import _ResolvedAddress

# Default timeout in seconds for HTTP requests. This bounds the connect phase
# and the gap between two received chunks, but not the total transfer time.
_DEFAULT_TIMEOUT_SECONDS = 30
# Total wall-clock budget in seconds for reading a response body. This bounds
# drip-feed responses that keep resetting the per-chunk read timeout.
_MAX_BODY_READ_SECONDS = 60
# Maximum number of response body bytes buffered in memory.
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
# Chunk size used while streaming a response body.
_RESPONSE_CHUNK_BYTES = 64 * 1024


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
      proxies: dict[str, str] | None = None,
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


def _declared_content_length(response: requests.Response) -> int:
  """Returns the declared body size, or 0 when the header is unusable.

  A missing, malformed or negative Content-Length yields 0, which leaves the
  cap to be enforced while streaming rather than up front.

  Args:
    response: The response whose headers to read.

  Returns:
    The declared body size in bytes, clamped to be non-negative.
  """
  raw_content_length = response.headers.get('Content-Length')
  if raw_content_length is None:
    return 0
  try:
    return max(0, int(raw_content_length))
  except ValueError:
    return 0


def _read_capped_content(response: requests.Response) -> bytes:
  """Buffers a streamed response body under a size and a time limit."""
  if _declared_content_length(response) > _MAX_RESPONSE_BYTES:
    raise ValueError(f'Response body is too large: {response.url}')

  deadline = time.monotonic() + _MAX_BODY_READ_SECONDS
  chunks: list[bytes] = []
  buffered_bytes = 0
  for chunk in response.iter_content(chunk_size=_RESPONSE_CHUNK_BYTES):
    buffered_bytes += len(chunk)
    if buffered_bytes > _MAX_RESPONSE_BYTES:
      raise ValueError(f'Response body is too large: {response.url}')
    if time.monotonic() > deadline:
      raise ValueError(f'Timed out reading response body: {response.url}')
    chunks.append(chunk)
  return b''.join(chunks)


def _read_successful_response(response: requests.Response) -> bytes:
  if response.status_code != 200:
    raise ValueError(f'Unexpected response status: {response.status_code}')
  return _read_capped_content(response)


def _rewrite_url_host(parsed_url: ParseResult, hostname: str) -> str:
  explicit_port = parsed_url.port
  formatted_hostname = _format_host(hostname)
  if explicit_port is None:
    rewritten_netloc = formatted_hostname
  else:
    rewritten_netloc = f'{formatted_hostname}:{explicit_port}'
  return parsed_url._replace(netloc=rewritten_netloc).geturl()


def _fetch_direct_content(
    *,
    url: str,
    target: _RequestTarget,
    resolved_addresses: tuple[_ResolvedAddress, ...],
) -> bytes:
  """Fetches a url over a connection pinned to an already vetted address."""
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
      # The body is read inside the session scope so that the size cap is
      # applied while streaming instead of after the whole body is buffered.
      with session.get(
          url,
          allow_redirects=False,
          # requests types reject None, which requests documents as no proxy.
          proxies={'http': None, 'https': None},  # type: ignore[dict-item]
          timeout=_DEFAULT_TIMEOUT_SECONDS,
          stream=True,
      ) as response:
        return _read_successful_response(response)
    except requests.RequestException as exc:
      last_error = exc
    finally:
      session.close()

  if last_error is not None:
    raise last_error
  raise requests.RequestException(f'Unable to fetch url: {url}')


def _fetch_proxied_content(url: str) -> bytes:
  with requests.get(
      url,
      allow_redirects=False,
      timeout=_DEFAULT_TIMEOUT_SECONDS,
      stream=True,
  ) as response:
    return _read_successful_response(response)


def _fetch_content(url: str) -> bytes:
  """Fetches a vetted url and returns its capped response body."""
  target = _parse_request_target(url)

  if _is_blocked_hostname(target.hostname):
    raise ValueError(f'Blocked host: {target.hostname}')

  parsed_ip_literal = _parse_ip_literal(target.hostname)
  if parsed_ip_literal is not None and _is_blocked_address(parsed_ip_literal):
    raise ValueError(f'Blocked host: {target.hostname}')

  if _get_proxy_url(url):
    # A proxy resolves the hostname remotely, so the connection cannot be
    # pinned to a vetted address. Screen the name against the local resolver
    # anyway; see `_reject_blocked_proxied_hostname` for why an unresolvable
    # name is still forwarded.
    if parsed_ip_literal is None:
      _reject_blocked_proxied_hostname(target.hostname)
    return _fetch_proxied_content(url)

  if parsed_ip_literal is not None:
    return _fetch_direct_content(
        url=url,
        target=target,
        resolved_addresses=(parsed_ip_literal,),
    )

  resolved_addresses = _resolve_direct_addresses(target.hostname)
  return _fetch_direct_content(
      url=url,
      target=target,
      resolved_addresses=resolved_addresses,
  )


# Scripts written without spaces between words. Each of their letters counts as
# a word when deciding whether a line is long enough to keep.
_UNSPACED_SCRIPT_RANGES = (
    (0x0E00, 0x0EFF),  # Thai and Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana and Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
    (0x20000, 0x3FFFF),  # Supplementary and Tertiary Ideographic Planes
)


def _is_unspaced_letter(char: str) -> bool:
  """Returns whether `char` is a letter of a script written without spaces."""
  code = ord(char)
  return any(
      low <= code <= high for low, high in _UNSPACED_SCRIPT_RANGES
  ) and unicodedata.category(char).startswith('L')


def _count_words(line: str) -> int:
  """Counts the words in a line of page text.

  Words are separated by whitespace. In a script written without spaces, such
  as Chinese, Japanese or Thai, each letter counts as a word, and the runs of
  other characters around those letters count only if they hold a letter or a
  digit, so that a full stop such as "。" is not counted as a word.
  """
  if line.isascii():
    return len(line.split())
  count = 0
  for token in line.split():
    if not any(_is_unspaced_letter(char) for char in token):
      count += 1
      continue
    for is_letter, group in itertools.groupby(token, _is_unspaced_letter):
      chars = list(group)
      if is_letter:
        count += len(chars)
      elif any(c.isalnum() for c in chars):
        count += 1
  return count


def load_web_page(url: str) -> str:
  """Fetches the content in the url and returns the text in it.

  Args:
      url (str): The url to browse.

  Returns:
      str: The text content of the url.
  """
  try:
    from bs4 import BeautifulSoup
    import lxml  # noqa: F401 -- verify lxml is available for the parser
  except ImportError as e:
    raise ImportError(
        'load_web_page requires the "beautifulsoup4" and "lxml" packages. '
        'Install them with: pip install google-adk[extensions]'
    ) from e

  # Requests are issued with allow_redirects=False to prevent SSRF attacks via
  # redirection, and the response body is capped to bound memory usage.
  try:
    content = _fetch_content(url)
  except (ValueError, requests.RequestException):
    return _failed_to_fetch_message(url)

  soup = BeautifulSoup(content, 'lxml')
  text = soup.get_text(separator='\n', strip=True)

  # Split the text into lines, filtering out very short lines
  # (e.g., single words or short subtitles)
  return '\n'.join(line for line in text.splitlines() if _count_words(line) > 3)
