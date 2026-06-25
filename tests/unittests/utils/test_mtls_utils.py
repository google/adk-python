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
import pytest

_DEFAULT_TEMPLATE = "service.{location}.rep.googleapis.com"
_MTLS_TEMPLATE = "service.{location}.rep.mtls.googleapis.com"
_LOCATION = "us-central1"


class TestMtlsUtils:
  """Tests for _mtls_utils functions."""

  @patch("google.auth.transport.mtls.should_use_client_cert")
  def test_use_client_cert_effective_with_mtls_cert_true(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.return_value = True
    assert _mtls_utils.use_client_cert_effective() is True
    mock_should_use_client_cert.assert_called_once()

  @patch("google.auth.transport.mtls.should_use_client_cert")
  def test_use_client_cert_effective_with_mtls_cert_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.return_value = False
    assert _mtls_utils.use_client_cert_effective() is False
    mock_should_use_client_cert.assert_called_once()

  @patch("google.auth.transport.mtls.should_use_client_cert")
  @patch.dict("os.environ", {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"})
  def test_use_client_cert_effective_fallback_true(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is True

  @patch("google.auth.transport.mtls.should_use_client_cert")
  @patch.dict("os.environ", {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "false"})
  def test_use_client_cert_effective_fallback_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is False

  @patch("google.auth.transport.mtls.should_use_client_cert")
  @patch.dict("os.environ", {}, clear=True)
  def test_use_client_cert_effective_fallback_default_false(
      self, mock_should_use_client_cert
  ):
    mock_should_use_client_cert.side_effect = AttributeError
    assert _mtls_utils.use_client_cert_effective() is False

  @patch("google.adk.utils._mtls_utils.use_client_cert_effective")
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "always"})
  def test_get_api_endpoint_always(self, mock_use_client_cert):
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _MTLS_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_not_called()

  @patch("google.adk.utils._mtls_utils.use_client_cert_effective")
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "never"})
  def test_get_api_endpoint_never(self, mock_use_client_cert):
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _DEFAULT_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_not_called()

  @patch("google.adk.utils._mtls_utils.use_client_cert_effective")
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  def test_get_api_endpoint_auto_with_cert(self, mock_use_client_cert):
    mock_use_client_cert.return_value = True
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _MTLS_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_called_once()

  @patch("google.adk.utils._mtls_utils.use_client_cert_effective")
  @patch.dict("os.environ", {"GOOGLE_API_USE_MTLS_ENDPOINT": "auto"})
  def test_get_api_endpoint_auto_without_cert(self, mock_use_client_cert):
    mock_use_client_cert.return_value = False
    endpoint = _mtls_utils.get_api_endpoint(
        _LOCATION, _DEFAULT_TEMPLATE, _MTLS_TEMPLATE
    )
    assert endpoint == _DEFAULT_TEMPLATE.format(location=_LOCATION)
    mock_use_client_cert.assert_called_once()

  @patch("google.adk.utils._mtls_utils.use_client_cert_effective")
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
