# Copyright 2025 Google LLC
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

"""Tests for the BaseAuthCredentialExchanger class."""

from typing import Optional
from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger
import pytest


class MockAuthCredentialExchanger(BaseAuthCredentialExchanger):

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    return AuthCredential(token="some-token")


class TestBaseAuthCredentialExchanger:
  """Tests for the BaseAuthCredentialExchanger class."""

  @pytest.fixture
  def base_exchanger(self):
    return BaseAuthCredentialExchanger()

  @pytest.fixture
  def auth_scheme(self):
    scheme = MagicMock(spec=AuthScheme)
    scheme.type = "apiKey"
    scheme.name = "x-api-key"
    return scheme

  def test_exchange_credential_not_implemented(
      self, base_exchanger, auth_scheme
  ):
    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.API_KEY, token="some-token"
    )
    with pytest.raises(NotImplementedError) as exc_info:
      base_exchanger.exchange_credential(auth_scheme, auth_credential)
    assert "Subclasses must implement exchange_credential." in str(
        exc_info.value
    )

  def test_auth_credential_missing_error(self):
    error_message = "Test missing credential"
    error = AuthCredentialMissingError(error_message)
    # assert error.message == error_message
    assert str(error) == error_message
