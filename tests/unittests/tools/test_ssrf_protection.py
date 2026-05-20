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

import pytest

from google.adk.tools._ssrf_protection import is_blocked_address
from google.adk.tools._ssrf_protection import is_blocked_hostname
from google.adk.tools._ssrf_protection import validate_url

import ipaddress


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
    assert is_blocked_address(ipaddress.ip_address("169.254.169.254"))

  def test_private_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("10.0.0.1"))
    assert is_blocked_address(ipaddress.ip_address("192.168.1.1"))
    assert is_blocked_address(ipaddress.ip_address("172.16.0.1"))

  def test_ipv6_loopback_blocked(self):
    assert is_blocked_address(ipaddress.ip_address("::1"))

  def test_global_allowed(self):
    assert not is_blocked_address(ipaddress.ip_address("8.8.8.8"))


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

  @pytest.fixture(autouse=True)
  def _patch_dns(self, monkeypatch):
    import socket as _socket

    original = _socket.getaddrinfo

    def fake_getaddrinfo(host, port, *args, **kwargs):
      if host == "api.example.com":
        return [(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))]
      return original(host, port, *args, **kwargs)

    monkeypatch.setattr(_socket, "getaddrinfo", fake_getaddrinfo)

  def test_public_url_allowed(self):
    validate_url("https://api.example.com/v1/resource")
