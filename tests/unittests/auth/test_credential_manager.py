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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.credential_manager import CredentialManager
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
    tool_context = Mock()
    tool_context.request_credential = Mock()

    manager = CredentialManager(auth_config)
    await manager.request_credential(tool_context)

    tool_context.request_credential.assert_called_once_with(auth_config)

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

    tool_context = Mock()

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

    result = await manager.get_auth_credential(tool_context)

    # Verify all methods were called
    manager._validate_credential.assert_called_once()
    manager._is_credential_ready.assert_called_once()
    manager._load_existing_credential.assert_called_once_with(tool_context)
    manager._load_from_auth_response.assert_called_once_with(tool_context)
    manager._exchange_credential.assert_called_once_with(mock_credential)
    manager._refresh_credential.assert_called_once_with(mock_credential)
    manager._save_credential.assert_called_once_with(
        tool_context, mock_credential
    )

    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_auth_credentials_no_credential(self):
    """Test load_auth_credential when no credential is available."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = None
    auth_config.exchanged_auth_credential = None

    tool_context = Mock()

    manager = CredentialManager(auth_config)

    # Mock the private methods
    manager._validate_credential = AsyncMock()
    manager._is_credential_ready = Mock(return_value=False)
    manager._load_existing_credential = AsyncMock(return_value=None)
    manager._load_from_auth_response = AsyncMock(return_value=None)
    manager._exchange_credential = AsyncMock()
    manager._refresh_credential = AsyncMock()
    manager._save_credential = AsyncMock()

    result = await manager.get_auth_credential(tool_context)

    # Verify methods were called but no credential returned
    manager._validate_credential.assert_called_once()
    manager._is_credential_ready.assert_called_once()
    manager._load_existing_credential.assert_called_once_with(tool_context)
    manager._load_from_auth_response.assert_called_once_with(tool_context)
    manager._exchange_credential.assert_not_called()
    manager._refresh_credential.assert_not_called()
    manager._save_credential.assert_not_called()

    assert result is None

  @pytest.mark.asyncio
  async def test_load_existing_credential_already_exchanged(self):
    """Test _load_existing_credential when credential is already exchanged."""
    auth_config = Mock(spec=AuthConfig)
    mock_credential = Mock(spec=AuthCredential)
    auth_config.exchanged_auth_credential = mock_credential

    tool_context = Mock()

    manager = CredentialManager(auth_config)
    manager._load_from_credential_service = AsyncMock(return_value=None)

    result = await manager._load_existing_credential(tool_context)

    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_existing_credential_with_credential_service(self):
    """Test _load_existing_credential with credential service."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.exchanged_auth_credential = None

    mock_credential = Mock(spec=AuthCredential)

    tool_context = Mock()

    manager = CredentialManager(auth_config)
    manager._load_from_credential_service = AsyncMock(
        return_value=mock_credential
    )

    result = await manager._load_existing_credential(tool_context)

    manager._load_from_credential_service.assert_called_once_with(tool_context)
    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_from_credential_service_with_service(self):
    """Test _load_from_credential_service from tool context when credential service is available."""
    auth_config = Mock(spec=AuthConfig)

    mock_credential = Mock(spec=AuthCredential)

    # Mock credential service
    credential_service = Mock()
    credential_service.load_credential = AsyncMock(return_value=mock_credential)

    # Mock invocation context
    invocation_context = Mock()
    invocation_context.credential_service = credential_service

    tool_context = Mock()
    tool_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    result = await manager._load_from_credential_service(tool_context)

    credential_service.load_credential.assert_called_once_with(
        auth_config, tool_context
    )
    assert result == mock_credential

  @pytest.mark.asyncio
  async def test_load_from_credential_service_no_service(self):
    """Test _load_from_credential_service when no credential service is available."""
    auth_config = Mock(spec=AuthConfig)

    # Mock invocation context with no credential service
    invocation_context = Mock()
    invocation_context.credential_service = None

    tool_context = Mock()
    tool_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    result = await manager._load_from_credential_service(tool_context)

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

    tool_context = Mock()
    tool_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    await manager._save_credential(tool_context, mock_credential)

    credential_service.save_credential.assert_called_once_with(
        auth_config, tool_context
    )
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

    tool_context = Mock()
    tool_context._invocation_context = invocation_context

    manager = CredentialManager(auth_config)
    await manager._save_credential(tool_context, mock_credential)

    # Should not raise an error, and credential should not be set in auth_config
    # when there's no credential service (according to implementation)
    assert auth_config.exchanged_auth_credential is None

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
    await manager._validate_credential()

    # Should return without error for non-OAuth2/OpenID schemes

  @pytest.mark.asyncio
  async def test_validate_credential_oauth2_missing_oauth2_field(self):
    """Test _validate_credential with OAuth2 credential missing oauth2 field."""
    auth_scheme = Mock()
    auth_scheme.type_ = AuthSchemeType.oauth2

    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.OAUTH2
    mock_raw_credential.oauth2 = None

    auth_config = Mock(spec=AuthConfig)
    auth_config.raw_auth_credential = mock_raw_credential
    auth_config.auth_scheme = auth_scheme

    manager = CredentialManager(auth_config)

    with pytest.raises(
        ValueError, match="auth_config.raw_credential.oauth2 required"
    ):
      await manager._validate_credential()

  @pytest.mark.asyncio
  async def test_exchange_credentials_service_account(self):
    """Test _exchange_credential with service account credential (no exchanger available)."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.SERVICE_ACCOUNT

    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = Mock()

    manager = CredentialManager(auth_config)

    # Mock the exchanger registry to return None (no exchanger available)
    with patch.object(
        manager._exchanger_registry, "get_exchanger", return_value=None
    ):
      result, was_exchanged = await manager._exchange_credential(
          mock_raw_credential
      )

      assert result == mock_raw_credential
      assert was_exchanged is False

  @pytest.mark.asyncio
  async def test_exchange_credential_no_exchanger(self):
    """Test _exchange_credential with credential that has no exchanger."""
    mock_raw_credential = Mock(spec=AuthCredential)
    mock_raw_credential.auth_type = AuthCredentialTypes.API_KEY

    auth_config = Mock(spec=AuthConfig)

    manager = CredentialManager(auth_config)

    # Mock the exchanger registry to return None (no exchanger available)
    with patch.object(
        manager._exchanger_registry, "get_exchanger", return_value=None
    ):
      result, was_exchanged = await manager._exchange_credential(
          mock_raw_credential
      )

      assert result == mock_raw_credential
      assert was_exchanged is False


# Test fixtures
@pytest.fixture
def oauth2_auth_scheme():
  """Create an OAuth2 auth scheme for testing."""
  flows = OAuthFlows(
      authorizationCode=OAuthFlowAuthorizationCode(
          authorizationUrl="https://example.com/oauth2/authorize",
          tokenUrl="https://example.com/oauth2/token",
          scopes={"read": "Read access", "write": "Write access"},
      )
  )
  return OAuth2(flows=flows)


@pytest.fixture
def openid_auth_scheme():
  """Create an OpenID Connect auth scheme for testing."""
  return OpenIdConnectWithConfig(
      type_="openIdConnect",
      authorization_endpoint="https://example.com/auth",
      token_endpoint="https://example.com/token",
      scopes=["openid", "profile"],
  )


@pytest.fixture
def bearer_auth_scheme():
  """Create a Bearer auth scheme for testing."""
  return HTTPBearer(bearerFormat="JWT")


@pytest.fixture
def oauth2_credential():
  """Create OAuth2 credentials for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="mock_client_id",
          client_secret="mock_client_secret",
          redirect_uri="https://example.com/callback",
      ),
  )


@pytest.fixture
def service_account_credential():
  """Create service account credentials for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=ServiceAccountCredential(
              type="service_account",
              project_id="test-project",
              private_key_id="key-id",
              private_key=(
                  "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE"
                  " KEY-----\n"
              ),
              client_email="test@test-project.iam.gserviceaccount.com",
              client_id="123456789",
              auth_uri="https://accounts.google.com/o/oauth2/auth",
              token_uri="https://oauth2.googleapis.com/token",
              auth_provider_x509_cert_url=(
                  "https://www.googleapis.com/oauth2/v1/certs"
              ),
              client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com",
          ),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      ),
  )


@pytest.fixture
def api_key_credential():
  """Create API key credentials for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      api_key="test-api-key",
  )


@pytest.fixture
def http_bearer_credential():
  """Create HTTP Bearer credentials for testing."""
  return AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer",
          credentials=HttpCredentials(token="bearer-token"),
      ),
  )
