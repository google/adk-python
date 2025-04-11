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

"""Unit tests for AutoAuthCredentialExchanger."""

from typing import Dict
from typing import Optional
from typing import Type
from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.tools.openapi_tool.auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.oauth2_exchanger import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger
import pytest


class MockCredentialExchanger(BaseAuthCredentialExchanger):
  """Mock credential exchanger for testing."""

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Mock exchange credential method."""
    return auth_credential


@pytest.fixture
def auto_exchanger():
  """Fixture for creating an AutoAuthCredentialExchanger instance."""
  return AutoAuthCredentialExchanger()


@pytest.fixture
def auth_scheme():
  """Fixture for creating a mock AuthScheme instance."""
  scheme = MagicMock(spec=AuthScheme)
  return scheme


def test_init_with_custom_exchangers():
  """Test initialization with custom exchangers."""
  custom_exchangers: Dict[str, Type[BaseAuthCredentialExchanger]] = {
      AuthCredentialTypes.API_KEY: MockCredentialExchanger
  }

  auto_exchanger = AutoAuthCredentialExchanger(
      custom_exchangers=custom_exchangers
  )

  assert (
      auto_exchanger.exchangers[AuthCredentialTypes.API_KEY]
      == MockCredentialExchanger
  )
  assert (
      auto_exchanger.exchangers[AuthCredentialTypes.OPEN_ID_CONNECT]
      == OAuth2CredentialExchanger
  )


def test_exchange_credential_no_auth_credential(auto_exchanger, auth_scheme):
  """Test exchange_credential with no auth_credential."""

  assert auto_exchanger.exchange_credential(auth_scheme, None) is None


def test_exchange_credential_no_exchange(auto_exchanger, auth_scheme):
  """Test exchange_credential with NoExchangeCredentialExchanger."""
  auth_credential = AuthCredential(auth_type=AuthCredentialTypes.API_KEY)

  result = auto_exchanger.exchange_credential(auth_scheme, auth_credential)

  assert result == auth_credential


def test_exchange_credential_open_id_connect(auto_exchanger, auth_scheme):
  """Test exchange_credential with OpenID Connect scheme."""
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT
  )
  mock_exchanger = MagicMock(spec=OAuth2CredentialExchanger)
  mock_exchanger.exchange_credential.return_value = "exchanged_credential"
  auto_exchanger.exchangers[AuthCredentialTypes.OPEN_ID_CONNECT] = (
      lambda: mock_exchanger
  )

  result = auto_exchanger.exchange_credential(auth_scheme, auth_credential)

  assert result == "exchanged_credential"
  mock_exchanger.exchange_credential.assert_called_once_with(
      auth_scheme, auth_credential
  )


def test_exchange_credential_service_account(auto_exchanger, auth_scheme):
  """Test exchange_credential with Service Account scheme."""
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT
  )
  mock_exchanger = MagicMock(spec=ServiceAccountCredentialExchanger)
  mock_exchanger.exchange_credential.return_value = "exchanged_credential_sa"
  auto_exchanger.exchangers[AuthCredentialTypes.SERVICE_ACCOUNT] = (
      lambda: mock_exchanger
  )

  result = auto_exchanger.exchange_credential(auth_scheme, auth_credential)

  assert result == "exchanged_credential_sa"
  mock_exchanger.exchange_credential.assert_called_once_with(
      auth_scheme, auth_credential
  )


def test_exchange_credential_custom_exchanger(auto_exchanger, auth_scheme):
  """Test that exchange_credential calls the correct (custom) exchanger."""
  # Use a custom exchanger via the initialization
  mock_exchanger = MagicMock(spec=MockCredentialExchanger)
  mock_exchanger.exchange_credential.return_value = "custom_credential"
  auto_exchanger.exchangers[AuthCredentialTypes.API_KEY] = (
      lambda: mock_exchanger
  )
  auth_credential = AuthCredential(auth_type=AuthCredentialTypes.API_KEY)

  result = auto_exchanger.exchange_credential(auth_scheme, auth_credential)

  assert result == "custom_credential"
  mock_exchanger.exchange_credential.assert_called_once_with(
      auth_scheme, auth_credential
  )
