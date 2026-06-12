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

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.integrations.agent_identity import GcpAuthProvider
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
from google.adk.integrations.agent_identity._iam_connector_credentials_provider import _IamConnectorCredentialsProvider
import pytest


@pytest.fixture
def auth_config():
  scheme = GcpAuthProviderScheme(
      name="projects/test-project/locations/global/connectors/test-connector",
      scopes=["test-scope"],
      continue_uri="https://example.com/continue",
  )
  return Mock(spec=AuthConfig, auth_scheme=scheme)


@pytest.fixture
def context():
  context = Mock(spec=CallbackContext)
  context.user_id = "user"
  return context
@pytest.fixture
def provider():
  return GcpAuthProvider()


def test_supported_auth_schemes(provider):
  """Verify the provider supports the correct auth scheme."""
  assert GcpAuthProviderScheme in provider.supported_auth_schemes


@patch("google.adk.integrations.agent_identity.gcp_auth_provider._IamConnectorCredentialsProvider")
async def test_gcp_auth_provider_delegates_get_auth_credential(mock_provider_class, auth_config, context):
  """Test that get_auth_credential delegates to the internal provider."""
  provider = GcpAuthProvider()

  mock_credential = Mock(spec=AuthCredential)
  mock_provider_instance = mock_provider_class.return_value
  mock_provider_instance.get_auth_credential = AsyncMock(return_value=mock_credential)

  result = await provider.get_auth_credential(auth_config, context)

  assert result == mock_credential
  mock_provider_instance.get_auth_credential.assert_awaited_once_with(
      auth_scheme=auth_config.auth_scheme, context=context
  )


async def test_get_auth_credential_raises_error_for_invalid_auth_scheme(context):
  """Test get_auth_credential raises ValueError for invalid auth scheme."""
  provider = GcpAuthProvider()
  invalid_auth_config = Mock(spec=AuthConfig)
  invalid_auth_config.auth_scheme = Mock()  # Not GcpAuthProviderScheme

  with pytest.raises(ValueError, match="Expected GcpAuthProviderScheme, got"):
    await provider.get_auth_credential(invalid_auth_config, context)
