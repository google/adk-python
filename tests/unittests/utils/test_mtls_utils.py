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

"""Unit tests for _mtls_utils."""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.utils import _mtls_utils
from google.auth import exceptions as ga_exceptions
from google.auth.transport import mtls
import pytest

_DEFAULT_TEMPLATE = "service.{location}.rep.googleapis.com"
_MTLS_TEMPLATE = "service.{location}.rep.mtls.googleapis.com"
_LOCATION = "us-central1"


class TestMtlsUtils:
  """Tests for _mtls_utils functions."""

  @patch.object(mtls, "should_use_client_cert", autospec=True)
  def test_use_client_cert_effective_with_mtls_cert_true(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.return_value = True
    assert _mtls_utils.use_client_cert_effective() is True
    mock_should_use_client_cert.assert_called_once()

  @patch.object(mtls, "should_use_client_cert", autospec=True)
  def test_use_client_cert_effective_with_mtls_cert_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.return_value = False
    assert _mtls_utils.use_client_cert_effective() is False
    mock_should_use_client_cert.assert_called_once()

  @patch.object(mtls, "should_use_client_cert", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"})
  def test_use_client_cert_effective_fallback_true(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is True

  @patch.object(mtls, "should_use_client_cert", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "false"})
  def test_use_client_cert_effective_fallback_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is False

  @patch.object(mtls, "should_use_client_cert", autospec=True)
  @patch.dict("os.environ", {}, clear=True)
  def test_use_client_cert_effective_fallback_default_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is False

  @patch.object(_mtls_utils, "use_client_cert_effective", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "always"})
  def test_get_api_endpoint_always(self, mock_use_client_cert):
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _MTLS_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_not_called()

  @patch.object(_mtls_utils, "use_client_cert_effective", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "never"})
  def test_get_api_endpoint_never(self, mock_use_client_cert):
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _DEFAULT_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_not_called()

  @patch.object(_mtls_utils, "use_client_cert_effective", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  def test_get_api_endpoint_auto_with_cert(self, mock_use_client_cert):
    mock_use_client_cert.return_value = True
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _MTLS_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_called_once()

  @patch.object(_mtls_utils, "use_client_cert_effective", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  def test_get_api_endpoint_auto_without_cert(self, mock_use_client_cert):
    mock_use_client_cert.return_value = False
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _DEFAULT_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_called_once()

  @patch.object(_mtls_utils, "use_client_cert_effective", autospec=True)
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "invalid_value"})
  def test_get_api_endpoint_invalid_fallback_to_auto(
      self, mock_use_client_cert
  ):
    mock_use_client_cert.return_value = True
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _MTLS_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_called_once()

  @pytest.mark.parametrize(
      "url, expected",
      [
          ("https://oauth2.googleapis.com/token", True),
          ("https://openidconnect.googleapis.com/v1/userinfo", True),
          ("https://oauth2.mtls.googleapis.com/token", False),
          ("https://example.com/token", False),
          ("https://accounts.google.com/o/oauth2/v2/auth", False),
          (None, False),
          ("", False),
      ],
  )
  def test_is_non_mtls_googleapis_endpoint(self, url, expected):
    assert _mtls_utils.is_non_mtls_googleapis_endpoint(url) is expected

  @patch.dict("os.environ", {}, clear=True)
  @pytest.mark.parametrize(
      "url, expected",
      [
          (
              "https://oauth2.googleapis.com/token",
              "https://oauth2.mtls.googleapis.com/token",
          ),
          (
              "https://openidconnect.googleapis.com/v1/userinfo",
              "https://openidconnect.mtls.googleapis.com/v1/userinfo",
          ),
          # Non-Google providers are never rewritten.
          ("https://example.com/token", "https://example.com/token"),
          # Already-mTLS hosts are left alone.
          (
              "https://oauth2.mtls.googleapis.com/token",
              "https://oauth2.mtls.googleapis.com/token",
          ),
      ],
  )
  def test_effective_googleapis_endpoint_rewrites(self, url, expected):
    assert _mtls_utils.effective_googleapis_endpoint(url) == expected

  @patch.dict(
      "os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "never"}, clear=True
  )
  def test_effective_googleapis_endpoint_never_opts_out(self):
    url = "https://oauth2.googleapis.com/token"
    assert _mtls_utils.effective_googleapis_endpoint(url) == url

  @patch.dict("os.environ", {}, clear=True)
  def test_effective_googleapis_endpoint_preserves_query(self):
    url = "https://iam.googleapis.com/v1/token?foo=bar"
    assert (
        _mtls_utils.effective_googleapis_endpoint(url)
        == "https://iam.mtls.googleapis.com/v1/token?foo=bar"
    )

  @patch("google.auth.transport.mtls.has_default_client_cert_source")
  @patch("google.auth.transport._mtls_helper.get_client_cert_and_key")
  @patch("google.auth.transport.requests._MutualTlsAdapter")
  def test_configure_session_for_mtls_mounts_adapter(
      self, mock_adapter, mock_get_cert, mock_has_cert_source
  ):
    mock_has_cert_source.return_value = False
    mock_get_cert.return_value = (True, b"cert", b"key")
    session = MagicMock()

    result = _mtls_utils.configure_session_for_mtls(session)

    assert result is True
    mock_adapter.assert_called_once_with(b"cert", b"key")
    session.mount.assert_called_once_with("https://", mock_adapter.return_value)

  @patch("google.auth.transport.mtls.has_default_client_cert_source")
  @patch("google.auth.transport._mtls_helper.get_client_cert_and_key")
  def test_configure_session_for_mtls_no_cert(
      self, mock_get_cert, mock_has_cert_source
  ):
    mock_has_cert_source.return_value = False
    mock_get_cert.return_value = (False, None, None)
    session = MagicMock()

    result = _mtls_utils.configure_session_for_mtls(session)

    assert result is False
    session.mount.assert_not_called()

  @patch("google.auth.transport.mtls.has_default_client_cert_source")
  @patch("google.auth.transport._mtls_helper.get_client_cert_and_key")
  def test_configure_session_for_mtls_cert_error_falls_back(
      self, mock_get_cert, mock_has_cert_source
  ):
    mock_has_cert_source.return_value = False
    mock_get_cert.side_effect = ga_exceptions.ClientCertError("boom")
    session = MagicMock()

    result = _mtls_utils.configure_session_for_mtls(session)

    assert result is False
    session.mount.assert_not_called()


class TestMtlsClientCerts:
  """Tests for MtlsClientCerts."""

  @patch.object(mtls, "has_default_client_cert_source", autospec=True)
  def test_get_certs_no_default_source(self, mock_has_cert):
    mock_has_cert.return_value = False
    certs = _mtls_utils.MtlsClientCerts()
    cert_path, key_path, passphrase = certs.get_certs()
    assert cert_path is None
    assert key_path is None
    assert passphrase is None
    mock_has_cert.assert_called_once()

  @patch.object(mtls, "has_default_client_cert_source", autospec=True)
  @patch.object(mtls, "default_client_encrypted_cert_source", autospec=True)
  def test_get_certs_with_default_source(
      self, mock_encrypted_source, mock_has_cert
  ):
    mock_has_cert.return_value = True

    mock_cert_source = MagicMock()
    mock_cert_source.return_value = (None, None, b"test_passphrase")
    mock_encrypted_source.return_value = mock_cert_source

    certs = _mtls_utils.MtlsClientCerts()
    cert_path, key_path, passphrase = certs.get_certs()

    assert cert_path is not None
    assert key_path is not None
    assert passphrase == b"test_passphrase"

    assert os.path.exists(certs._tempdir.name)
    assert cert_path.startswith(certs._tempdir.name)
    assert key_path.startswith(certs._tempdir.name)

    # Getting certs again should return cached values without calling mtls again
    cert_path2, key_path2, passphrase2 = certs.get_certs()
    assert cert_path2 == cert_path
    assert key_path2 == key_path
    assert passphrase2 == passphrase
    mock_has_cert.assert_called_once()
    mock_encrypted_source.assert_called_once()

  @patch.object(mtls, "has_default_client_cert_source", autospec=True)
  @patch.object(mtls, "default_client_encrypted_cert_source", autospec=True)
  def test_get_certs_extraction_failure(
      self, mock_encrypted_source, mock_has_cert
  ):
    mock_has_cert.return_value = True
    mock_encrypted_source.side_effect = Exception("extraction failed")

    certs = _mtls_utils.MtlsClientCerts()
    with pytest.raises(
        RuntimeError, match="Failed to extract default client certificates"
    ):
      certs.get_certs()

    assert certs._tempdir is None

  @patch.object(mtls, "has_default_client_cert_source", autospec=True)
  @patch.object(mtls, "default_client_encrypted_cert_source", autospec=True)
  def test_close_cleans_up_tempdir(self, mock_encrypted_source, mock_has_cert):
    mock_has_cert.return_value = True
    mock_cert_source = MagicMock()
    mock_cert_source.return_value = (None, None, b"test_passphrase")
    mock_encrypted_source.return_value = mock_cert_source

    certs = _mtls_utils.MtlsClientCerts()
    certs.get_certs()

    tempdir_name = certs._tempdir.name
    assert os.path.exists(tempdir_name)

    certs.close()

    assert not os.path.exists(tempdir_name)
    assert certs._tempdir is None
    assert certs.cert_path is None
    assert certs.key_path is None
    assert certs.passphrase is None
    assert not certs._initialized
