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

"""Unit tests for the CredentialExchangerRegistry."""

from typing import Optional
from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.exchanger.base_credential_exchanger import BaseCredentialExchanger
from google.adk.auth.exchanger.credential_exchanger_registry import CredentialExchangerRegistry
import pytest


class MockCredentialExchanger(BaseCredentialExchanger):
  """Mock credential exchanger for testing."""

  def __init__(self, exchange_result: Optional[AuthCredential] = None):
    self.exchange_result = exchange_result or AuthCredential(
        auth_type=AuthCredentialTypes.HTTP
    )

  def exchange(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Mock exchange method."""
    return self.exchange_result


class TestCredentialExchangerRegistry:
  """Test cases for CredentialExchangerRegistry."""

  def test_initialization(self):
    """Test that the registry initializes with an empty exchangers dictionary."""
    registry = CredentialExchangerRegistry()

    # Access the private attribute for testing
    assert hasattr(registry, '_exchangers')
    assert isinstance(registry._exchangers, dict)
    assert len(registry._exchangers) == 0

  def test_register_single_exchanger(self):
    """Test registering a single exchanger."""
    registry = CredentialExchangerRegistry()
    mock_exchanger = MockCredentialExchanger()

    registry.register(AuthCredentialTypes.API_KEY, mock_exchanger)

    # Verify the exchanger was registered
    retrieved_exchanger = registry.get_exchanger(AuthCredentialTypes.API_KEY)
    assert retrieved_exchanger is mock_exchanger

  def test_register_multiple_exchangers(self):
    """Test registering multiple exchangers for different credential types."""
    registry = CredentialExchangerRegistry()

    api_key_exchanger = MockCredentialExchanger()
    oauth2_exchanger = MockCredentialExchanger()
    service_account_exchanger = MockCredentialExchanger()

    registry.register(AuthCredentialTypes.API_KEY, api_key_exchanger)
    registry.register(AuthCredentialTypes.OAUTH2, oauth2_exchanger)
    registry.register(
        AuthCredentialTypes.SERVICE_ACCOUNT, service_account_exchanger
    )

    # Verify all exchangers were registered correctly
    assert (
        registry.get_exchanger(AuthCredentialTypes.API_KEY) is api_key_exchanger
    )
    assert (
        registry.get_exchanger(AuthCredentialTypes.OAUTH2) is oauth2_exchanger
    )
    assert (
        registry.get_exchanger(AuthCredentialTypes.SERVICE_ACCOUNT)
        is service_account_exchanger
    )

  def test_register_overwrites_existing_exchanger(self):
    """Test that registering an exchanger for an existing type overwrites the previous one."""
    registry = CredentialExchangerRegistry()

    first_exchanger = MockCredentialExchanger()
    second_exchanger = MockCredentialExchanger()

    # Register first exchanger
    registry.register(AuthCredentialTypes.API_KEY, first_exchanger)
    assert (
        registry.get_exchanger(AuthCredentialTypes.API_KEY) is first_exchanger
    )

    # Register second exchanger for the same type
    registry.register(AuthCredentialTypes.API_KEY, second_exchanger)
    assert (
        registry.get_exchanger(AuthCredentialTypes.API_KEY) is second_exchanger
    )
    assert (
        registry.get_exchanger(AuthCredentialTypes.API_KEY)
        is not first_exchanger
    )

  def test_get_exchanger_returns_correct_instance(self):
    """Test that get_exchanger returns the correct exchanger instance."""
    registry = CredentialExchangerRegistry()
    mock_exchanger = MockCredentialExchanger()

    registry.register(AuthCredentialTypes.HTTP, mock_exchanger)

    retrieved_exchanger = registry.get_exchanger(AuthCredentialTypes.HTTP)
    assert retrieved_exchanger is mock_exchanger
    assert isinstance(retrieved_exchanger, BaseCredentialExchanger)

  def test_get_exchanger_nonexistent_type_returns_none(self):
    """Test that get_exchanger returns None for non-existent credential types."""
    registry = CredentialExchangerRegistry()

    # Try to get an exchanger that was never registered
    result = registry.get_exchanger(AuthCredentialTypes.OAUTH2)
    assert result is None

  def test_get_exchanger_after_registration_and_removal(self):
    """Test behavior when an exchanger is registered and then the registry is cleared indirectly."""
    registry = CredentialExchangerRegistry()
    mock_exchanger = MockCredentialExchanger()

    # Register exchanger
    registry.register(AuthCredentialTypes.API_KEY, mock_exchanger)
    assert registry.get_exchanger(AuthCredentialTypes.API_KEY) is mock_exchanger

    # Clear the internal dictionary (simulating some edge case)
    registry._exchangers.clear()
    assert registry.get_exchanger(AuthCredentialTypes.API_KEY) is None

  def test_register_with_all_credential_types(self):
    """Test registering exchangers for all available credential types."""
    registry = CredentialExchangerRegistry()

    exchangers = {}
    credential_types = [
        AuthCredentialTypes.API_KEY,
        AuthCredentialTypes.HTTP,
        AuthCredentialTypes.OAUTH2,
        AuthCredentialTypes.OPEN_ID_CONNECT,
        AuthCredentialTypes.SERVICE_ACCOUNT,
    ]

    # Register an exchanger for each credential type
    for cred_type in credential_types:
      exchanger = MockCredentialExchanger()
      exchangers[cred_type] = exchanger
      registry.register(cred_type, exchanger)

    # Verify all exchangers can be retrieved
    for cred_type in credential_types:
      retrieved_exchanger = registry.get_exchanger(cred_type)
      assert retrieved_exchanger is exchangers[cred_type]

  def test_register_with_mock_exchanger_using_magicmock(self):
    """Test registering with a MagicMock exchanger."""
    registry = CredentialExchangerRegistry()
    mock_exchanger = MagicMock(spec=BaseCredentialExchanger)

    registry.register(AuthCredentialTypes.API_KEY, mock_exchanger)

    retrieved_exchanger = registry.get_exchanger(AuthCredentialTypes.API_KEY)
    assert retrieved_exchanger is mock_exchanger

  def test_registry_isolation(self):
    """Test that different registry instances are isolated from each other."""
    registry1 = CredentialExchangerRegistry()
    registry2 = CredentialExchangerRegistry()

    exchanger1 = MockCredentialExchanger()
    exchanger2 = MockCredentialExchanger()

    # Register different exchangers in different registry instances
    registry1.register(AuthCredentialTypes.API_KEY, exchanger1)
    registry2.register(AuthCredentialTypes.API_KEY, exchanger2)

    # Verify isolation
    assert registry1.get_exchanger(AuthCredentialTypes.API_KEY) is exchanger1
    assert registry2.get_exchanger(AuthCredentialTypes.API_KEY) is exchanger2
    assert (
        registry1.get_exchanger(AuthCredentialTypes.API_KEY) is not exchanger2
    )
    assert (
        registry2.get_exchanger(AuthCredentialTypes.API_KEY) is not exchanger1
    )

  def test_exchanger_functionality_through_registry(self):
    """Test that exchangers registered in the registry function correctly."""
    registry = CredentialExchangerRegistry()

    # Create a mock exchanger with specific return value
    expected_result = AuthCredential(auth_type=AuthCredentialTypes.HTTP)
    mock_exchanger = MockCredentialExchanger(exchange_result=expected_result)

    registry.register(AuthCredentialTypes.API_KEY, mock_exchanger)

    # Get the exchanger and test its functionality
    retrieved_exchanger = registry.get_exchanger(AuthCredentialTypes.API_KEY)
    input_credential = AuthCredential(auth_type=AuthCredentialTypes.API_KEY)

    result = retrieved_exchanger.exchange(input_credential)
    assert result is expected_result

  def test_register_none_exchanger(self):
    """Test that registering None as an exchanger works (edge case)."""
    registry = CredentialExchangerRegistry()

    # This should work but return None when retrieved
    registry.register(AuthCredentialTypes.API_KEY, None)

    result = registry.get_exchanger(AuthCredentialTypes.API_KEY)
    assert result is None

  def test_internal_dictionary_structure(self):
    """Test the internal structure of the registry."""
    registry = CredentialExchangerRegistry()
    mock_exchanger = MockCredentialExchanger()

    registry.register(AuthCredentialTypes.OAUTH2, mock_exchanger)

    # Verify internal dictionary structure
    assert AuthCredentialTypes.OAUTH2 in registry._exchangers
    assert registry._exchangers[AuthCredentialTypes.OAUTH2] is mock_exchanger
    assert len(registry._exchangers) == 1
