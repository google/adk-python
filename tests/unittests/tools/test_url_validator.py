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
from unittest import mock

from google.adk.tools._url_validator import _embedded_ipv4
from google.adk.tools._url_validator import _is_blocked_address
from google.adk.tools._url_validator import _is_blocked_hostname
from google.adk.tools._url_validator import _parse_request_target
from google.adk.tools._url_validator import _resolve_direct_addresses
from google.adk.tools._url_validator import _resolve_host_addresses
import google.adk.tools._url_validator as url_validator
import pytest

_PUBLIC_IPV4 = '8.8.8.8'
_PUBLIC_IPV6 = '2001:4860:4860::8888'

_LOOPBACK_IPV4 = '127.0.0.1'
_LOOPBACK_IPV6 = '::1'
_PRIVATE_IPV4 = '10.0.0.1'
_METADATA_IPV4 = '169.254.169.254'

# IPv6 addresses embedding an IPv4. `ipaddress.is_global` does not always
# account for the embedded address, so the validator must check it.
_PUBLIC_VIA_NAT64 = f'64:ff9b::{_PUBLIC_IPV4}'
_METADATA_VIA_NAT64 = f'64:ff9b::{_METADATA_IPV4}'
_LOOPBACK_VIA_IPV4_MAPPED = f'::ffff:{_LOOPBACK_IPV4}'
_METADATA_VIA_IPV4_COMPATIBLE = f'::{_METADATA_IPV4}'
_METADATA_VIA_6TO4 = '2002:a9fe:a9fe::'  # 169.254.169.254 in hex.


def _fake_dns(monkeypatch: pytest.MonkeyPatch, *ipv4s: str) -> None:
  """Makes every DNS lookup return the given IPv4 addresses."""
  records = [
      (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, 0))
      for ip in ipv4s
  ]
  monkeypatch.setattr(
      url_validator.socket, 'getaddrinfo', mock.Mock(return_value=records)
  )


def _broken_dns(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
  """Makes every DNS lookup raise `error`."""
  monkeypatch.setattr(
      url_validator.socket, 'getaddrinfo', mock.Mock(side_effect=error)
  )


# --- _parse_request_target ---------------------------------------------------


@pytest.mark.parametrize(
    ('url', 'expected_hostname', 'expected_host_header'),
    [
        ('https://example.com:443/path', 'example.com', 'example.com'),
        ('http://example.com:8080/path', 'example.com', 'example.com:8080'),
        (
            f'http://[{_PUBLIC_IPV6}]:8080/',
            _PUBLIC_IPV6,
            f'[{_PUBLIC_IPV6}]:8080',
        ),
    ],
)
def test_parse_request_target_accepts_http_urls(
    url: str, expected_hostname: str, expected_host_header: str
):
  target = _parse_request_target(url)

  assert target.hostname == expected_hostname
  assert target.host_header == expected_host_header


@pytest.mark.parametrize(
    ('url', 'expected_error'),
    [
        ('file:///etc/passwd', 'Unsupported url scheme'),
        ('http:///missing-host', 'missing a hostname'),
        ('http://example.com:99999/', 'Invalid url port'),
    ],
)
def test_parse_request_target_rejects_invalid_urls(
    url: str, expected_error: str
):
  with pytest.raises(ValueError, match=expected_error):
    _parse_request_target(url)


# --- _is_blocked_hostname ----------------------------------------------------


@pytest.mark.parametrize(
    'hostname',
    [
        'localhost',
        'LOCALHOST.',
        'a.localhost',
        'metadata',
        'metadata.goog',
        'sub.metadata.goog',
        'instance.internal',
        'service.local',
    ],
)
def test_is_blocked_hostname_blocks_internal_names(hostname: str):
  assert _is_blocked_hostname(hostname)


@pytest.mark.parametrize('hostname', ['example.com', 'localhost.example.com'])
def test_is_blocked_hostname_allows_other_names(hostname: str):
  assert not _is_blocked_hostname(hostname)


# --- _embedded_ipv4 ----------------------------------------------------------


@pytest.mark.parametrize(
    ('ip', 'expected'),
    [
        (_LOOPBACK_VIA_IPV4_MAPPED, _LOOPBACK_IPV4),
        (_METADATA_VIA_6TO4, _METADATA_IPV4),
        (_METADATA_VIA_NAT64, _METADATA_IPV4),
        (_METADATA_VIA_IPV4_COMPATIBLE, _METADATA_IPV4),
    ],
)
def test_embedded_ipv4_extracts_the_wrapped_address(ip: str, expected: str):
  assert _embedded_ipv4(ipaddress.ip_address(ip)) == ipaddress.ip_address(
      expected
  )


@pytest.mark.parametrize(
    'ip', [_PUBLIC_IPV4, _PUBLIC_IPV6, '::', _LOOPBACK_IPV6]
)
def test_embedded_ipv4_returns_none_without_an_embedded_address(ip: str):
  assert _embedded_ipv4(ipaddress.ip_address(ip)) is None


# --- _is_blocked_address -----------------------------------------------------


@pytest.mark.parametrize('ip', [_PUBLIC_IPV4, _PUBLIC_IPV6, _PUBLIC_VIA_NAT64])
def test_is_blocked_address_allows_public_addresses(ip: str):
  assert not _is_blocked_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize(
    'ip', [_LOOPBACK_IPV4, _LOOPBACK_IPV6, _PRIVATE_IPV4, _METADATA_IPV4]
)
def test_is_blocked_address_blocks_non_public_addresses(ip: str):
  assert _is_blocked_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize(
    'ip',
    [
        _METADATA_VIA_NAT64,
        _LOOPBACK_VIA_IPV4_MAPPED,
        _METADATA_VIA_IPV4_COMPATIBLE,
        _METADATA_VIA_6TO4,
    ],
)
def test_is_blocked_address_blocks_ipv6_wrapping_non_public_ipv4(ip: str):
  assert _is_blocked_address(ipaddress.ip_address(ip))


# --- _resolve_host_addresses -------------------------------------------------


@pytest.mark.parametrize('ip', [_PUBLIC_IPV4, _PUBLIC_IPV6])
def test_resolve_host_addresses_returns_ip_literal_without_dns(
    monkeypatch, ip: str
):
  _broken_dns(monkeypatch, AssertionError('unexpected DNS lookup'))

  assert _resolve_host_addresses(ip) == (ipaddress.ip_address(ip),)


def test_resolve_host_addresses_reports_dns_failure(monkeypatch):
  _broken_dns(monkeypatch, socket.gaierror('Name or service not known'))

  with pytest.raises(ValueError, match='Unable to resolve host'):
    _resolve_host_addresses('example.com')


# --- _resolve_direct_addresses -----------------------------------------------


def test_resolve_direct_addresses_returns_unique_public_addresses(
    monkeypatch,
):
  _fake_dns(monkeypatch, _PUBLIC_IPV4, _PUBLIC_IPV4)

  assert _resolve_direct_addresses('example.com') == (
      ipaddress.ip_address(_PUBLIC_IPV4),
  )


def test_resolve_direct_addresses_blocks_host_with_any_non_public_address(
    monkeypatch,
):
  _fake_dns(monkeypatch, _PUBLIC_IPV4, _METADATA_IPV4)

  with pytest.raises(ValueError, match='Blocked host'):
    _resolve_direct_addresses('example.com')


def test_resolve_direct_addresses_blocks_non_public_ip_literal():
  with pytest.raises(ValueError, match='Blocked host'):
    _resolve_direct_addresses(_LOOPBACK_IPV4)
