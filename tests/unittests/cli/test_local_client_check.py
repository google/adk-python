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

"""Tests for _is_local_client, the peer-address gate on the Agent Builder."""

from typing import Any
from typing import Optional

from google.adk.cli.api_server import _is_local_client
import pytest


def _make_scope(
    client: Optional[tuple[str, int]] = ("127.0.0.1", 51234),
    headers: Optional[list[tuple[bytes, bytes]]] = None,
) -> dict[str, Any]:
  """Build a minimal ASGI scope for testing."""
  scope: dict[str, Any] = {
      "type": "http",
      "method": "POST",
      "headers": headers or [],
  }
  if client is not None:
    scope["client"] = client
  return scope


class TestIsLocalClient:
  """The peer address is the only signal a remote caller cannot forge."""

  @pytest.mark.parametrize(
      "client_host",
      ["127.0.0.1", "127.1.2.3", "::1", "localhost"],
  )
  def test_loopback_peers_are_local(self, client_host: str):
    assert _is_local_client(_make_scope(client=(client_host, 51234)))

  @pytest.mark.parametrize(
      "client_host",
      ["203.0.113.7", "192.168.1.5", "10.0.0.1", "0.0.0.0", "evil.com"],
  )
  def test_remote_peers_are_not_local(self, client_host: str):
    assert not _is_local_client(_make_scope(client=(client_host, 51234)))

  def test_missing_client_is_not_local(self):
    """ASGI servers may omit `client`; fail closed rather than open."""
    assert not _is_local_client(_make_scope(client=None))

  @pytest.mark.parametrize(
      "header_name",
      [b"forwarded", b"x-forwarded-for", b"x-forwarded-host"],
  )
  def test_forwarded_loopback_peer_is_not_local(self, header_name: bytes):
    """Through a proxy or tunnel the peer is the forwarder, not the caller."""
    scope = _make_scope(headers=[(header_name, b"203.0.113.7")])
    assert not _is_local_client(scope)

  def test_forged_origin_does_not_make_a_remote_peer_local(self):
    """Any non-browser client can set Origin, so it must not be trusted here."""
    scope = _make_scope(
        client=("203.0.113.7", 51234),
        headers=[(b"origin", b"http://127.0.0.1:8000")],
    )
    assert not _is_local_client(scope)
