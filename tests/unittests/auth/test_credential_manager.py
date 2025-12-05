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

from unittest.mock import ANY
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlowImplicit
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import ExtendedOAuth2
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
from google.adk.auth.credential_manager import ServiceAccountCredentialExchanger
from google.adk.auth.oauth2_discovery import AuthorizationServerMetadata
import pytest


class TestCredentialManager:
  """Test suite for CredentialManager."""

  def test_init(self):
    """Test CredentialManager initialization."""
    auth_config = Mock(spec=AuthConfig)
    manager = CredentialManager(auth_config)
    assert manager._auth_config == auth_config

  @pytest.mark.asyncio
  async def test_request_credential(self):
    """Test request_credential method."""
    auth_config = Mock(spec=AuthConfig)
    callback_context = Mock()
    callback_context.request_credential = Mock()

    manager = CredentialManager(auth_config)
    await manager.request_credential(callback_context)

    callback_context.request_credential.assert_called_once_with(auth_config)

  @pytest.mark.asyncio
  async def test_load_auth_credentials_success(self):
    """Test load_auth_credential with successful flow."""
    # Create mocks
    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    # Mock the credential that will be returned
    mock_credential = Mock(spec=AuthCredential)
    mock_credential.auth_type = AuthCredentialTypes.API_KEY

    callback_context = Mock()

    manager = CredentialManager(auth_config)

    # Mock the private methods
    manager._validate_credential = AsyncMock()
    manager._is_credential_ready = Mock(return_value=False)
    manager._load_existing_credential = AsyncMock(return_value=None)
    manager._load_from_auth_response = AsyncMock(return_value=mock_credential)
    manager._exchange_credential = AsyncMock(
        return_value=(mock_credential, False)
    )
    manager._refresh_credential = AsyncMock(
        return_value=(mock_credential, False)
    )
    manager._save_credential = AsyncMock()

    result = await manager.get_auth_credential(callback_context)

    # Verify all methods were called
    manager._validate_credential.assert_called_once()
    manager._is_credential_ready.assert_called_once()
    manager._load_existing_credential.assert_called_once_with(callback_context)
    manager._load_from_auth_response.assert_called_once_with(callback_context)
    manager._exchange_credential.assert_called_once_with(mock_credential)
    manager._refresh_credential.assert_called_once_with(mock_credential)
    manager._save_credential.assert_called_once_with(
        callback_context, mock_credential
    )

    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_auth_credentials_no_credential(self):
    """Test load_auth_credential when no credential is available."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None
    # Add auth_scheme for the _is_client_credentials_flow method
    auth_config.auth_scheme = Mock()
    auth_config.auth_scheme.flows = None

    callback_context = Mock()

    manager = CredentialManager(auth_config)

    # Mock the private methods
    manager._validate_credential = AsyncMock()
    manager._is_credential_ready = Mock(return_value=False)
    manager._load_existing_credential = AsyncMock(return_value=None)
    manager._load_from_auth_response = AsyncMock(return_value=None)

    result = await manager.get_auth_credential(callback_context)

    # Verify methods were called but no credential returned
    manager._validate_credential.assert_called_once()
    manager._is_credential_ready.assert_called_once()
    manager._load_existing_credential.assert_called_once_with(callback_context)
    manager._load_from_auth_response.assert_called_once_with(callback_context)

    assert result is None

  @pytest.mark.asyncio
  async def test_load_existing_credential_already_exchanged(self):
    """Test _load_existing_credential when credential is already exchanged."""
    auth_config = Mock(spec=AuthConfig)
    mock_credential = Mock(spec=AuthCredential)
    auth_config.exchanged_auth_credential = mock_credential

    callback_context = Mock()

    manager = CredentialManager(auth_config)
    manager._load_from_credential_service = AsyncMock(return_value=None)

    result = await manager._load_existing_credential(callback_context)

    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_existing_credential_with_credential_service(self):
    """Test _load_existing_credential with credential service."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.exchanged_auth_credential = None

    mock_credential = Mock(spec=AuthCredential)

    callback_context = Mock()

    manager = CredentialManager(auth_config)
    manager._load_from_credential_service = AsyncMock(
        return_value=mock_credential
    )

    result = await manager._load_existing_credential(callback_context)

    manager._load_from_credential_service.assert_called_once_with(
        callback_context
    )
    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_from_credential_service_with_service(self):
    """Test _load_from_credential_service from callback context when credential service is available."""
    auth_config = Mock(spec=AuthConfig)

    mock_credential = Mock(spec=AuthCredential)

    # Mock credential service
    credential_service = Mock()

    # Mock invocation context
    invocation_context = Mock()
    invocation_context.credential_service = credential_service

    callback_context = Mock()
    callback_context._invocation_context = invocation_context
    callback_context.load_credential = AsyncMock(return_value=mock_credential)

    manager = CredentialManager(auth_config)
    result = await manager._load_from_credential_service(callback_context)

    callback_context.load_credential.assert_called_once_with(auth_config)
    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_from_credential_service_no_service(self):
    """Test _load_from_credential_service when no credential service is available."""
    auth_config = Mock(spec=AuthConfig)

    # Mock invocation context with no credential service
    invocation_context = Mock()
    invocation_context.credential_service = None

    callback_context = Mock()
    callback_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    result = await manager._load_from_credential_service(callback_context)

    assert result is None

  @pytest.mark.asyncio
  async def test_save_credential_with_service(self):
    """Test _save_credential with credential service."""
    auth_config = Mock(spec=AuthConfig)
    mock_credential = Mock(spec=AuthCredential)

    # Mock credential service
    credential_service = AsyncMock()

    # Mock invocation context
    invocation_context = Mock()
    invocation_context.credential_service = credential_service

    callback_context = Mock()
    callback_context._invocation_context = invocation_context
    callback_context.save_credential = AsyncMock()

    manager = CredentialManager(auth_config)
    await manager._save_credential(callback_context, mock_credential)

    callback_context.save_credential.assert_called_once_with(auth_config)
    assert auth_config.exchanged_auth_credential == mock_credential

  @pytest.mark.asyncio
  async def test_save_credential_no_service(self):
    """Test _save_credential when no credential service is available."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.exchanged_auth_credential = None
    mock_credential = Mock(spec=AuthCredential)

    # Mock invocation context with no credential service
    invocation_context = Mock()
    invocation_context.credential_service = None

    callback_context = Mock()
    callback_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    await manager._save_credential(callback_context, mock_credential)

    # Should not raise an error, and credential should be set in auth_config
    # even when there's no credential service (config is updated regardless)
    assert auth_config.exchanged_auth_credential == mock_credential

  @pytest.mark.asyncio
  async def test_refresh_credential_oauth2(self):
    """Test _refresh_credential with OAuth2 credential."""
    mock_oauth2_auth = Mock(spec=OAuth2Auth)

    mock_credential = Mock(spec=AuthCredential)
    mock_credential.auth_type = AuthCredentialTypes.OAUTH2

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = Mock()

    # Mock refresher
    mock_refresher = Mock()
    mock_refresher.is_refresh_needed = AsyncMock(return_value=True)
    mock_refresher.refresh = AsyncMock(return_value=mock_credential)

    auth_config.raw_auth_credential = mock_credential

    manager = CredentialManager(auth_config)

    # Mock the refresher registry to return our mock refresher
    with patch.object(
        manager._refresher_registry,
        "get_refresher",
        return_value=mock_refresher,
    ):
      result, was_refreshed = await manager._refresh_credential(mock_credential)

    mock_refresher.is_refresh_needed.assert_called_once_with(
        mock_credential, auth_config.auth_scheme
    )
    mock_refresher.refresh.assert_called_once_with(
        mock_credential, auth_config.auth_scheme
    )
    assert result == mock_credential
    assert was_refreshed is True

  @pytest.mark.asyncio
  async def test_refresh_credential_no_refresher(self):
    """Test _refresh_credential with credential that has no refresher."""
    mock_credential = Mock(spec=AuthCredential)
    mock_credential.auth_type = AuthCredentialTypes.API_KEY

    auth_config = Mock(spec=AuthConfig)

    manager = CredentialManager(auth_config)

    # Mock the refresher registry to return None (no refresher available)
    with patch.object(
        manager._refresher_registry,
        "get_refresher",
        return_value=None,
    ):
      result, was_refreshed = await manager._refresh_credential(mock_credential)

    assert result == mock_credential
    assert was_refreshed is False

  @pytest.mark.asyncio
  async def test_is_credential_ready_api_key(self):
    """Test _is_credential_ready with API key credential."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.API_KEY

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = mock_raw_credential

    manager = CredentialManager(auth_config)
    result = manager._is_credential_ready()

    assert result is True

  @pytest.mark.asyncio
  async def test_is_credential_ready_oauth2(self):
    """Test _is_credential_ready with OAuth2 credential (needs processing)."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.OAUTH2

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = mock_raw_credential

    manager = CredentialManager(auth_config)
    result = manager._is_credential_ready()

    assert result is False

  @pytest.mark.asyncio
  async def test_validate_credential_no_raw_credential_oauth2(self):
    """Test _validate_credential with no raw credential for OAuth2."""
    auth_scheme = Mock()
    auth_scheme.type_ = AuthSchemeType.oauth2

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.auth_scheme = auth_scheme

    manager = CredentialManager(auth_config)

    with pytest.raises(ValueError, match="raw_auth_credential is required"):
      await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_validate_credential_no_raw_credential_openid(self):
    """Test _validate_credential with no raw credential for OpenID Connect."""
    auth_scheme = Mock()
    auth_scheme.type_ = AuthSchemeType.openIdConnect

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.auth_scheme = auth_scheme

    manager = CredentialManager(auth_config)

    with pytest.raises(ValueError, match="raw_auth_credential is required"):
      await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_validate_credential_no_raw_credential_other_scheme(self):
    """Test _validate_credential with no raw credential for other schemes."""
    auth_scheme = Mock()
    auth_scheme.type_ = AuthSchemeType.apiKey

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.auth_scheme = auth_scheme

    manager = CredentialManager(auth_config)

    # Should not raise an error for non-OAuth schemes
    await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_validate_credential_oauth2_missing_oauth2_field(self):
    """Test _validate_credential with OAuth2 credential missing oauth2 field."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.OAUTH2
    mock_raw_credential.oauth2 = None

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = mock_raw_credential
    auth_config.auth_scheme = Mock()

    manager = CredentialManager(auth_config)

    with pytest.raises(ValueError, match="oauth2 required for credential type"):
      await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_validate_credential_oauth2_missing_scheme_info(
      self, extended_oauth2_scheme
  ):
    """Test _validate_credential with OAuth2 missing scheme info."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.OAUTH2
    mock_raw_credential.oauth2 = Mock(spec=OAuth2Auth)

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = mock_raw_credential
    auth_config.auth_scheme = extended_oauth2_scheme

    manager = CredentialManager(auth_config)

    with patch.object(
        manager,
        "_populate_auth_scheme",
        return_value=False,
    ) and pytest.raises(ValueError, match="OAuth scheme info is missing"):
      await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_exchange_credentials_service_account(
      self, service_account_credential, oauth2_auth_scheme
  ):
    """Test _exchange_credential with service account credential."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = oauth2_auth_scheme

    exchanged_credential = Mock(spec=AuthCredential)

    manager = CredentialManager(auth_config)

    with patch.object(
        ServiceAccountCredentialExchanger,
        "exchange_credential",
        return_value=exchanged_credential,
        autospec=True,
    ) as mock_exchange_credential:
      result, was_exchanged = await manager._exchange_credential(
          service_account_credential
      )

      mock_exchange_credential.assert_called_once_with(
          ANY, oauth2_auth_scheme, service_account_credential
      )
      assert result == exchanged_credential
      assert was_exchanged is True

  @pytest.mark.asyncio
  async def test_exchange_credential_no_exchanger(self):
    """Test _exchange_credential with credential that has no exchanger."""
    mock_credential = Mock(spec=AuthCredential)
    mock_credential.auth_type = AuthCredentialTypes.API_KEY

    auth_config = Mock(spec=AuthConfig)

    manager = CredentialManager(auth_config)

    # Mock the exchanger registry to return None (no exchanger available)
    with patch.object(
        manager._exchanger_registry,
        "get_exchanger",
        return_value=None,
    ):
      result, was_exchanged = await manager._exchange_credential(
          mock_credential
      )

    assert result == mock_credential
    assert was_exchanged is False

  @pytest.fixture
  def auth_server_metadata(self):
    """Create AuthorizationServerMetadata object."""
    return AuthorizationServerMetadata(
        issuer="https://auth.example.com",
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        scopes_supported=["read", "write"],
    )

  @pytest.fixture
  def extended_oauth2_scheme(self):
    """Create ExtendedOAuth2 object with empty endpoints."""
    return ExtendedOAuth2(
        issuer_url="https://auth.example.com",
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="",
                tokenUrl="",
            )
        ),
    )

  @pytest.fixture
  def implicit_oauth2_scheme(self):
    """Create OAuth2 object with implicit flow."""
    return OAuth2(
        flows=OAuthFlows(
            implicit=OAuthFlowImplicit(
                authorizationUrl="https://auth.example.com/authorize"
            )
        )
    )

  @pytest.mark.asyncio
  async def test_populate_auth_scheme_success(
      self, auth_server_metadata, extended_oauth2_scheme
  ):
    """Test _populate_auth_scheme successfully populates missing info."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = extended_oauth2_scheme

    manager = CredentialManager(auth_config)
    with patch.object(
        manager._discovery_manager,
        "discover_auth_server_metadata",
        return_value=auth_server_metadata,
    ):
      assert await manager._populate_auth_scheme()

    assert (
        manager._auth_config.auth_scheme.flows.authorizationCode.authorizationUrl
        == "https://auth.example.com/authorize"
    )
    assert (
        manager._auth_config.auth_scheme.flows.authorizationCode.tokenUrl
        == "https://auth.example.com/token"
    )

  @pytest.mark.asyncio
  async def test_populate_auth_scheme_fail(self, extended_oauth2_scheme):
    """Test _populate_auth_scheme when auto-discovery fails."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = extended_oauth2_scheme

    manager = CredentialManager(auth_config)
    with patch.object(
        manager._discovery_manager,
        "discover_auth_server_metadata",
        return_value=None,
    ):
      assert not await manager._populate_auth_scheme()

    assert (
        not manager._auth_config.auth_scheme.flows.authorizationCode.authorizationUrl
    )
    assert not manager._auth_config.auth_scheme.flows.authorizationCode.tokenUrl

  @pytest.mark.asyncio
  async def test_populate_auth_scheme_noop(self, implicit_oauth2_scheme):
    """Test _populate_auth_scheme when auth scheme info not missing."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = implicit_oauth2_scheme

    manager = CredentialManager(auth_config)
    assert not await manager._populate_auth_scheme()  # no-op

    assert manager._auth_config.auth_scheme == implicit_oauth2_scheme

  def test_is_client_credentials_flow_oauth2_with_client_credentials(self):
    """Test _is_client_credentials_flow returns True for OAuth2 with client credentials."""
    from fastapi.openapi.models import OAuth2
    from fastapi.openapi.models import OAuthFlowClientCredentials
    from fastapi.openapi.models import OAuthFlows

    # Create OAuth2 scheme with client credentials flow
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            clientCredentials=OAuthFlowClientCredentials(
                tokenUrl="https://example.com/token"
            )
        )
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = auth_scheme
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    manager = CredentialManager(auth_config)

    assert manager._is_client_credentials_flow() is True

  def test_is_client_credentials_flow_oauth2_without_client_credentials(self):
    """Test _is_client_credentials_flow returns False for OAuth2 without client credentials."""
    from fastapi.openapi.models import OAuth2
    from fastapi.openapi.models import OAuthFlowAuthorizationCode
    from fastapi.openapi.models import OAuthFlows

    # Create OAuth2 scheme with authorization code flow only
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://example.com/auth",
                tokenUrl="https://example.com/token",
            )
        )
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = auth_scheme
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    manager = CredentialManager(auth_config)

    assert manager._is_client_credentials_flow() is False

  def test_is_client_credentials_flow_oidc_with_client_credentials(self):
    """Test _is_client_credentials_flow returns True for OIDC with client credentials."""
    from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

    # Create OIDC scheme with client credentials support
    auth_scheme = OpenIdConnectWithConfig(
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        grant_types_supported=["authorization_code", "client_credentials"],
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = auth_scheme
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    manager = CredentialManager(auth_config)

    assert manager._is_client_credentials_flow() is True

  def test_is_client_credentials_flow_oidc_without_client_credentials(self):
    """Test _is_client_credentials_flow returns False for OIDC without client credentials."""
    from google.adk.auth.auth_schemes import OpenIdConnectWithConfig

    # Create OIDC scheme without client credentials support
    auth_scheme = OpenIdConnectWithConfig(
        authorization_endpoint="https://example.com/auth",
        token_endpoint="https://example.com/token",
        grant_types_supported=["authorization_code"],
    )

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = auth_scheme
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    manager = CredentialManager(auth_config)

    assert manager._is_client_credentials_flow() is False

  def test_is_client_credentials_flow_other_scheme(self):
    """Test _is_client_credentials_flow returns False for other auth schemes."""
    # Create a non-OAuth2/OIDC scheme
    auth_scheme = Mock()

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = auth_scheme
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    manager = CredentialManager(auth_config)

    assert manager._is_client_credentials_flow() is False


@pytest.fixture
def oauth2_auth_scheme():
  """OAuth2 auth scheme for testing."""
  auth_scheme = Mock(spec=AuthScheme)
  auth_scheme.type_ = AuthSchemeType.oauth2
  return auth_scheme


@pytest.fixture
def openid_auth_scheme():
  """OpenID Connect auth scheme for testing."""
  auth_scheme = Mock(spec=AuthScheme)
  auth_scheme.type_ = AuthSchemeType.openIdConnect
  return auth_scheme


@pytest.fixture
def bearer_auth_scheme():
  """Bearer auth scheme for testing."""
  auth_scheme = Mock(spec=AuthScheme)
  auth_scheme.type_ = AuthSchemeType.http
  return auth_scheme


@pytest.fixture
def oauth2_credential():
  """OAuth2 credential for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test_client_id",
          client_secret="test_client_secret",
          redirect_uri="https://example.com/callback",
      ),
  )


@pytest.fixture
def service_account_credential():
  """Service account credential for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=ServiceAccountCredential(
              type_="service_account",
              project_id="test_project",
              private_key_id="test_key_id",
              private_key=(
                  "-----BEGIN PRIVATE KEY-----\ntest_key\n-----END PRIVATE"
                  " KEY-----\n"
              ),
              client_email="test@test.iam.gserviceaccount.com",
              client_id="test_client_id",
              auth_uri="https://accounts.google.com/o/oauth2/auth",
              token_uri="https://oauth2.googleapis.com/token",
              auth_provider_x509_cert_url=(
                  "https://www.googleapis.com/oauth2/v1/certs"
              ),
              client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test.iam.gserviceaccount.com",
              universe_domain="googleapis.com",
          ),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      ),
  )


@pytest.fixture
def api_key_credential():
  """API key credential for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      api_key="test_api_key",
  )


@pytest.fixture
def http_bearer_credential():
  """HTTP bearer credential for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=Mock(),
  )
