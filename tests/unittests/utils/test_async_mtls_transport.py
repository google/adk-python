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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.utils import _async_mtls_transport as amt
import pytest

_AIO_SUPPORTED = amt._AIO_SUPPORTED


@pytest.mark.asyncio
async def test_create_transport_returns_none_when_aio_unsupported():
  with patch.object(amt, "_AIO_SUPPORTED", False):
    transport = await amt.create_google_auth_mtls_transport(
        "https://foo.googleapis.com/bar"
    )
  assert transport is None


@pytest.mark.asyncio
@pytest.mark.skipif(not _AIO_SUPPORTED, reason="google.auth.aio not supported")
async def test_create_transport_success_when_mtls():
  mock_session = AsyncMock()
  mock_session.is_mtls = True
  mock_session.configure_mtls_channel = AsyncMock()

  with (
      patch("google.auth.default", return_value=(Mock(), None)),
      patch.object(amt, "AsyncAuthorizedSession", return_value=mock_session),
      patch.object(amt, "_GoogleAuthAsyncTransport") as mock_transport_class,
  ):
    mock_transport_class.return_value = "the-transport"
    transport = await amt.create_google_auth_mtls_transport(
        "https://foo.googleapis.com/bar"
    )

  assert transport == "the-transport"
  mock_session.configure_mtls_channel.assert_awaited_once()
  mock_transport_class.assert_called_once_with(mock_session)


@pytest.mark.asyncio
@pytest.mark.skipif(not _AIO_SUPPORTED, reason="google.auth.aio not supported")
async def test_create_transport_returns_none_when_channel_not_mtls():
  mock_session = AsyncMock()
  mock_session.is_mtls = False
  mock_session.configure_mtls_channel = AsyncMock()

  with (
      patch("google.auth.default", return_value=(Mock(), None)),
      patch.object(amt, "AsyncAuthorizedSession", return_value=mock_session),
  ):
    transport = await amt.create_google_auth_mtls_transport(
        "https://foo.googleapis.com/bar"
    )

  assert transport is None


@pytest.mark.asyncio
async def test_create_transport_returns_none_on_exception():
  with patch("google.auth.default", side_effect=Exception("auth error")):
    transport = await amt.create_google_auth_mtls_transport(
        "https://foo.googleapis.com/bar"
    )
  assert transport is None


@pytest.mark.asyncio
async def test_refreshable_credentials_injects_token_for_target_host():
  creds = Mock()
  creds.token = "the-token"
  creds.expired = False
  refreshable = amt._RefreshableAsyncCredentials(
      creds, target_host="foo.googleapis.com"
  )

  headers: dict[str, str] = {}
  await refreshable.before_request(
      None, "GET", "https://foo.googleapis.com/bar", headers
  )
  assert headers["Authorization"] == "Bearer the-token"


@pytest.mark.asyncio
async def test_refreshable_credentials_skips_other_hosts():
  creds = Mock()
  creds.token = "the-token"
  creds.expired = False
  refreshable = amt._RefreshableAsyncCredentials(
      creds, target_host="foo.googleapis.com"
  )

  headers: dict[str, str] = {}
  await refreshable.before_request(
      None, "GET", "https://evil.example.com/bar", headers
  )
  assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_refreshable_credentials_does_not_overwrite_existing_header():
  creds = Mock()
  creds.token = "the-token"
  creds.expired = False
  refreshable = amt._RefreshableAsyncCredentials(creds)

  headers = {"Authorization": "Bearer preset"}
  await refreshable.before_request(
      None, "GET", "https://foo.googleapis.com/bar", headers
  )
  assert headers["Authorization"] == "Bearer preset"
