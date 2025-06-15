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

"""Tests for CredentialRefresherRegistry."""

from unittest.mock import Mock

from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.refresher.base_credential_refresher import BaseCredentialRefresher
from google.adk.auth.refresher.credential_refresher_registry import CredentialRefresherRegistry


class TestCredentialRefresherRegistry:
  """Tests for the CredentialRefresherRegistry class."""

  def test_init(self):
    """Test that registry initializes with empty refreshers dictionary."""
    registry = CredentialRefresherRegistry()
    assert registry._refreshers == {}

  def test_register_refresher(self):
    """Test registering a refresher instance for a credential type."""
    registry = CredentialRefresherRegistry()
    mock_refresher = Mock(spec=BaseCredentialRefresher)

    registry.register(AuthCredentialTypes.OAUTH2, mock_refresher)

    assert registry._refreshers[AuthCredentialTypes.OAUTH2] == mock_refresher

  def test_register_multiple_refreshers(self):
    """Test registering multiple refresher instances for different credential types."""
    registry = CredentialRefresherRegistry()
    mock_oauth2_refresher = Mock(spec=BaseCredentialRefresher)
    mock_openid_refresher = Mock(spec=BaseCredentialRefresher)
    mock_service_account_refresher = Mock(spec=BaseCredentialRefresher)

    registry.register(AuthCredentialTypes.OAUTH2, mock_oauth2_refresher)
    registry.register(
        AuthCredentialTypes.OPEN_ID_CONNECT, mock_openid_refresher
    )
    registry.register(
        AuthCredentialTypes.SERVICE_ACCOUNT, mock_service_account_refresher
    )

    assert (
        registry._refreshers[AuthCredentialTypes.OAUTH2]
        == mock_oauth2_refresher
    )
    assert (
        registry._refreshers[AuthCredentialTypes.OPEN_ID_CONNECT]
        == mock_openid_refresher
    )
    assert (
        registry._refreshers[AuthCredentialTypes.SERVICE_ACCOUNT]
        == mock_service_account_refresher
    )

  def test_register_overwrite_existing_refresher(self):
    """Test that registering a refresher overwrites an existing one for the same credential type."""
    registry = CredentialRefresherRegistry()
    mock_refresher_1 = Mock(spec=BaseCredentialRefresher)
    mock_refresher_2 = Mock(spec=BaseCredentialRefresher)

    # Register first refresher
    registry.register(AuthCredentialTypes.OAUTH2, mock_refresher_1)
    assert registry._refreshers[AuthCredentialTypes.OAUTH2] == mock_refresher_1

    # Register second refresher for same credential type
    registry.register(AuthCredentialTypes.OAUTH2, mock_refresher_2)
    assert registry._refreshers[AuthCredentialTypes.OAUTH2] == mock_refresher_2

  def test_get_refresher_existing(self):
    """Test getting a refresher instance for a registered credential type."""
    registry = CredentialRefresherRegistry()
    mock_refresher = Mock(spec=BaseCredentialRefresher)

    registry.register(AuthCredentialTypes.OAUTH2, mock_refresher)
    result = registry.get_refresher(AuthCredentialTypes.OAUTH2)

    assert result == mock_refresher

  def test_get_refresher_non_existing(self):
    """Test getting a refresher instance for a non-registered credential type returns None."""
    registry = CredentialRefresherRegistry()

    result = registry.get_refresher(AuthCredentialTypes.OAUTH2)

    assert result is None

  def test_get_refresher_after_registration(self):
    """Test getting refresher instances for multiple credential types."""
    registry = CredentialRefresherRegistry()
    mock_oauth2_refresher = Mock(spec=BaseCredentialRefresher)
    mock_api_key_refresher = Mock(spec=BaseCredentialRefresher)

    registry.register(AuthCredentialTypes.OAUTH2, mock_oauth2_refresher)
    registry.register(AuthCredentialTypes.API_KEY, mock_api_key_refresher)

    # Get registered refreshers
    oauth2_result = registry.get_refresher(AuthCredentialTypes.OAUTH2)
    api_key_result = registry.get_refresher(AuthCredentialTypes.API_KEY)

    assert oauth2_result == mock_oauth2_refresher
    assert api_key_result == mock_api_key_refresher

    # Get non-registered refresher
    http_result = registry.get_refresher(AuthCredentialTypes.HTTP)
    assert http_result is None

  def test_register_all_credential_types(self):
    """Test registering refreshers for all available credential types."""
    registry = CredentialRefresherRegistry()

    refreshers = {}
    for credential_type in AuthCredentialTypes:
      mock_refresher = Mock(spec=BaseCredentialRefresher)
      refreshers[credential_type] = mock_refresher
      registry.register(credential_type, mock_refresher)

    # Verify all refreshers are registered correctly
    for credential_type in AuthCredentialTypes:
      result = registry.get_refresher(credential_type)
      assert result == refreshers[credential_type]

  def test_empty_registry_get_refresher(self):
    """Test getting refresher from empty registry returns None for any credential type."""
    registry = CredentialRefresherRegistry()

    for credential_type in AuthCredentialTypes:
      result = registry.get_refresher(credential_type)
      assert result is None

  def test_registry_independence(self):
    """Test that multiple registry instances are independent."""
    registry1 = CredentialRefresherRegistry()
    registry2 = CredentialRefresherRegistry()

    mock_refresher1 = Mock(spec=BaseCredentialRefresher)
    mock_refresher2 = Mock(spec=BaseCredentialRefresher)

    registry1.register(AuthCredentialTypes.OAUTH2, mock_refresher1)
    registry2.register(AuthCredentialTypes.OAUTH2, mock_refresher2)

    # Verify registries are independent
    assert (
        registry1.get_refresher(AuthCredentialTypes.OAUTH2) == mock_refresher1
    )
    assert (
        registry2.get_refresher(AuthCredentialTypes.OAUTH2) == mock_refresher2
    )
    assert registry1.get_refresher(
        AuthCredentialTypes.OAUTH2
    ) != registry2.get_refresher(AuthCredentialTypes.OAUTH2)

  def test_register_with_none_refresher(self):
    """Test registering None as a refresher instance."""
    registry = CredentialRefresherRegistry()

    # This should technically work as the registry accepts any value
    registry.register(AuthCredentialTypes.OAUTH2, None)
    result = registry.get_refresher(AuthCredentialTypes.OAUTH2)

    assert result is None
