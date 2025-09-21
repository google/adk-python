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

from unittest import mock

from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.tools.google_api_tool.google_api_tool import GoogleApiTool
from google.adk.tools.openapi_tool import RestApiTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import FunctionDeclaration
import pytest


@pytest.fixture
def mock_rest_api_tool():
  """Fixture for a mock RestApiTool."""
  mock_tool = mock.MagicMock(spec=RestApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  mock_tool.is_long_running = False
  mock_tool._get_declaration.return_value = FunctionDeclaration(
      name="test_function", description="Test function description"
  )
  mock_tool.run_async.return_value = {"result": "success"}
  return mock_tool


@pytest.fixture
def mock_tool_context():
  """Fixture for a mock ToolContext."""
  return mock.MagicMock(spec=ToolContext)


class TestGoogleApiTool:
  """Test suite for the GoogleApiTool class."""

  def test_init(self, mock_rest_api_tool):
    """Test GoogleApiTool initialization."""
    tool = GoogleApiTool(mock_rest_api_tool)

    assert tool.name == "test_tool"
    assert tool.description == "Test Tool Description"
    assert tool.is_long_running is False
    assert tool._rest_api_tool == mock_rest_api_tool

  def test_get_declaration(self, mock_rest_api_tool):
    """Test _get_declaration method."""
    tool = GoogleApiTool(mock_rest_api_tool)

    declaration = tool._get_declaration()

    assert isinstance(declaration, FunctionDeclaration)
    assert declaration.name == "test_function"
    assert declaration.description == "Test function description"
    mock_rest_api_tool._get_declaration.assert_called_once()

  @pytest.mark.asyncio
  async def test_run_async(self, mock_rest_api_tool, mock_tool_context):
    """Test run_async method."""
    tool = GoogleApiTool(mock_rest_api_tool)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=mock_tool_context)

    assert result == {"result": "success"}
    mock_rest_api_tool.run_async.assert_called_once_with(
        args=args, tool_context=mock_tool_context
    )

  def test_configure_auth(self, mock_rest_api_tool):
    """Test configure_auth method."""
    tool = GoogleApiTool(mock_rest_api_tool)
    client_id = "test_client_id"
    client_secret = "test_client_secret"

    tool.configure_auth(client_id=client_id, client_secret=client_secret)

    # Check that auth_credential was set correctly on the rest_api_tool
    assert mock_rest_api_tool.auth_credential is not None
    assert (
        mock_rest_api_tool.auth_credential.auth_type
        == AuthCredentialTypes.OPEN_ID_CONNECT
    )
    assert mock_rest_api_tool.auth_credential.oauth2.client_id == client_id
    assert (
        mock_rest_api_tool.auth_credential.oauth2.client_secret == client_secret
    )

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool.service_account_scheme_credential"
  )
  def test_configure_sa_auth(
      self, mock_service_account_scheme_credential, mock_rest_api_tool
  ):
    """Test configure_sa_auth method."""
    # Setup mock return values
    mock_auth_scheme = mock.MagicMock()
    mock_auth_credential = mock.MagicMock()
    mock_service_account_scheme_credential.return_value = (
        mock_auth_scheme,
        mock_auth_credential,
    )

    service_account = ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type="service_account",
            project_id="project_id",
            private_key_id="private_key_id",
            private_key="private_key",
            client_email="client_email",
            client_id="client_id",
            auth_uri="auth_uri",
            token_uri="token_uri",
            auth_provider_x509_cert_url="auth_provider_x509_cert_url",
            client_x509_cert_url="client_x509_cert_url",
            universe_domain="universe_domain",
        ),
        scopes=["scope1", "scope2"],
    )

    # Create tool and call method
    tool = GoogleApiTool(mock_rest_api_tool)
    tool.configure_sa_auth(service_account=service_account)

    # Verify service_account_scheme_credential was called correctly
    mock_service_account_scheme_credential.assert_called_once_with(
        service_account
    )

    # Verify auth_scheme and auth_credential were set correctly on the rest_api_tool
    assert mock_rest_api_tool.auth_scheme == mock_auth_scheme
    assert mock_rest_api_tool.auth_credential == mock_auth_credential
