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
"""Unit tests for the GcpAuthProvider class."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.integrations.agent_identity import GcpAuthProvider
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
import pytest


@pytest.fixture
def auth_config():
  config = Mock(spec=AuthConfig)
  config.auth_scheme = Mock(spec=GcpAuthProviderScheme)
  return config


@pytest.fixture
def context():
  context = Mock(spec=CallbackContext)
  context.user_id = "user"
  return context


@pytest.fixture
def gcp_auth_provider():
  return GcpAuthProvider()


def test_supported_auth_schemes(gcp_auth_provider):
  """Verify the provider supports the correct auth scheme."""
  assert GcpAuthProviderScheme in gcp_auth_provider.supported_auth_schemes


async def test_get_auth_credential_raises_error_for_invalid_auth_scheme(
    context,
):
  """Test get_auth_credential raises ValueError for invalid auth scheme."""
  provider = GcpAuthProvider()
  invalid_auth_config = Mock(spec=AuthConfig)
  invalid_auth_config.auth_scheme = Mock()  # Not GcpAuthProviderScheme

  with pytest.raises(ValueError, match="Expected GcpAuthProviderScheme, got"):
    await provider.get_auth_credential(invalid_auth_config, context)


@patch(
    "google.adk.integrations.agent_identity.gcp_auth_provider._IamConnectorCredentialsProvider"
)
async def test_get_auth_credential_routes_to_iam_connector_service_provider(
    mock_iam_cls, auth_config, context
):
  """Test routing to IAM Connector Credentials service for legacy auth provider resource names."""
  auth_config.auth_scheme.name = (
      "projects/test-project/locations/test-location/connectors/test-connector"
  )
  provider = GcpAuthProvider()

  mock_credential = Mock(spec=AuthCredential)
  mock_iam_provider = mock_iam_cls.return_value
  mock_iam_provider.get_auth_credential = AsyncMock(
      return_value=mock_credential
  )

  result = await provider.get_auth_credential(auth_config, context)

  assert result == mock_credential
  mock_iam_provider.get_auth_credential.assert_awaited_once_with(
      auth_scheme=auth_config.auth_scheme, context=context
  )


@patch(
    "google.adk.integrations.agent_identity.gcp_auth_provider._AgentIdentityCredentialsProvider"
)
async def test_get_auth_credential_routes_to_agent_identity_service_provider(
    mock_agent_cls, auth_config, context
):
  """Test routing to Agent Identity Credentials service for new auth provider resource names."""
  auth_config.auth_scheme.name = "projects/test-project/locations/test-location/authProviders/test-provider"
  provider = GcpAuthProvider()

  mock_credential = Mock(spec=AuthCredential)
  mock_agent_provider = mock_agent_cls.return_value
  mock_agent_provider.get_auth_credential = AsyncMock(
      return_value=mock_credential
  )

  result = await provider.get_auth_credential(auth_config, context)

  assert result == mock_credential
  mock_agent_provider.get_auth_credential.assert_awaited_once_with(
      auth_scheme=auth_config.auth_scheme, context=context
  )
