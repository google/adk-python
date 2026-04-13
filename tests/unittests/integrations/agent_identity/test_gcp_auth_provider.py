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

from unittest.mock import Mock
from unittest.mock import patch

import pytest

pytest.importorskip(
    "google.cloud.iamconnectorcredentials_v1alpha",
    reason="Requires google-cloud-iamconnectorcredentials",
)

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.flows.llm_flows.functions import REQUEST_EUC_FUNCTION_CALL_NAME
from google.adk.integrations.agent_identity import gcp_auth_provider
from google.adk.integrations.agent_identity import GcpAuthProvider
from google.adk.integrations.agent_identity import GcpAuthProviderScheme
from google.adk.sessions.session import Session
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsMetadata
from google.cloud.iamconnectorcredentials_v1alpha import RetrieveCredentialsResponse
from google.longrunning.operations_pb2 import Operation
from google.protobuf.any_pb2 import Any
from google.rpc.status_pb2 import Status


@pytest.fixture
def mock_client():
  return Mock(spec=gcp_auth_provider.Client)


@pytest.fixture
def provider(mock_client):
  return GcpAuthProvider(client=mock_client)


@pytest.fixture
def auth_config():
  scheme = GcpAuthProviderScheme(
      name="projects/test-project/locations/global/connectors/test-connector",
      scopes=["test-scope"],
      continue_uri="https://example.com/continue",
  )
  return Mock(spec=AuthConfig, auth_scheme=scheme)


@pytest.fixture
def mock_operation(mocker, mock_client):
  op = Operation(done=True)

  class DummyCall:

    def __init__(self, operation):
      self.operation = operation

  mock_client.retrieve_credentials.return_value = DummyCall(op)
  return op


@pytest.fixture
def context():
  context = Mock(spec=CallbackContext)
  context.user_id = "user"
  context.function_call_id = "call_123"
  session = Mock(spec=Session)
  session.events = []
  context.session = session

  return context


@patch.dict(gcp_auth_provider.os.environ, clear=True)
@patch.object(gcp_auth_provider, "Client")
def test_get_client_uses_rest_transport(mock_client_class):
  provider = GcpAuthProvider()
  provider._get_client()

  mock_client_class.assert_called_once()
  _, kwargs = mock_client_class.call_args
  assert kwargs.get("transport") == "rest"


@patch.dict(
    gcp_auth_provider.os.environ,
    {"IAM_CONNECTOR_CREDENTIALS_TARGET_HOST": "some-host"},
)
@patch.object(gcp_auth_provider, "Client")
@patch.object(gcp_auth_provider, "ClientOptions")
def test_get_client_with_env_var(mock_client_options_class, mock_client_class):
  provider = GcpAuthProvider()
  client = provider._get_client()

  assert client == mock_client_class.return_value
  mock_client_options_class.assert_called_once_with(api_endpoint="some-host")
  mock_client_class.assert_called_once_with(
      client_options=mock_client_options_class.return_value, transport="rest"
  )


# ==============================================================================
# Non-interactive auth flows (API key and 2-legged OAuth)
# ==============================================================================


async def test_get_auth_credential_raises_error_for_invalid_auth_scheme(
    provider, context
):
  """Test get_auth_credential raises ValueError for invalid auth scheme."""
  invalid_auth_config = Mock(spec=AuthConfig)
  invalid_auth_config.auth_scheme = Mock()  # Not GcpAuthProviderScheme

  with pytest.raises(ValueError, match="Expected GcpAuthProviderScheme, got"):
    await provider.get_auth_credential(invalid_auth_config, context)


async def test_get_auth_credential_raises_error_if_context_is_missing(
    provider, auth_config
):
  """Test get_auth_credential raises ValueError if context is missing."""
  with pytest.raises(
      ValueError,
      match="GcpAuthProvider requires a context with a valid user_id",
  ):
    await provider.get_auth_credential(auth_config, context=None)


async def test_get_auth_credential_raises_error_if_user_id_is_missing(
    provider, auth_config
):
  """Test get_auth_credential raises ValueError if user_id is missing."""
  context = Mock(spec=CallbackContext)
  context.user_id = None
  with pytest.raises(
      ValueError,
      match="GcpAuthProvider requires a context with a valid user_id",
  ):
    await provider.get_auth_credential(auth_config, context=context)


async def test_get_auth_credential_returns_credential_if_available_immediately(
    mock_client,
    mock_operation,
    auth_config,
    context,
    provider,
):
  """Test get_auth_credential returns credential if available immediately."""
  mock_credential = RetrieveCredentialsResponse(
      header="Authorization: Bearer", token="test-token"
  )
  mock_operation.response.value = RetrieveCredentialsResponse.serialize(
      mock_credential
  )

  auth_credential = await provider.get_auth_credential(auth_config, context)

  assert auth_credential.auth_type == AuthCredentialTypes.HTTP
  assert auth_credential.http.scheme == "bearer"
  assert auth_credential.http.credentials.token == "test-token"
  mock_client.retrieve_credentials.assert_called_once()


async def test_get_auth_credential_raises_error_if_upstream_returns_empty_header(
    mock_operation,
    auth_config,
    context,
    provider,
):
  """Test get_auth_credential raises RuntimeError for empty header."""
  mock_credential = RetrieveCredentialsResponse(header="", token="test-token")
  mock_operation.response.value = RetrieveCredentialsResponse.serialize(
      mock_credential
  )

  with pytest.raises(
      ValueError,
      match=(
          "Received either empty header or token from Agent Identity"
          " Credentials service."
      ),
  ):
    await provider.get_auth_credential(auth_config, context)


async def test_get_auth_credential_raises_error_if_upstream_returns_empty_token(
    mock_operation,
    auth_config,
    context,
    provider,
):
  """Test get_auth_credential raises RuntimeError for empty token."""
  mock_credential = RetrieveCredentialsResponse(
      header="Authorization: Bearer", token=""
  )
  mock_operation.response.value = RetrieveCredentialsResponse.serialize(
      mock_credential
  )

  with pytest.raises(
      ValueError,
      match=(
          "Received either empty header or token from Agent Identity"
          " Credentials service."
      ),
  ):
    await provider.get_auth_credential(auth_config, context)


async def test_get_auth_credential_returns_credential_if_upstream_returns_custom_header(
    mock_operation,
    auth_config,
    context,
    provider,
):
  """Test get_auth_credential returns valid credential for custom header and sets X-GOOG-API-KEY header."""
  mock_credential = RetrieveCredentialsResponse(
      header="some-x-api-key", token="test-token"
  )
  mock_operation.response.value = RetrieveCredentialsResponse.serialize(
      mock_credential
  )

  auth_credential = await provider.get_auth_credential(auth_config, context)

  assert auth_credential.auth_type == AuthCredentialTypes.HTTP
  assert not auth_credential.http.scheme
  assert auth_credential.http.credentials.token is None
  assert auth_credential.http.additional_headers == {
      "some-x-api-key": "test-token",
      "X-GOOG-API-KEY": "test-token",
  }


async def test_get_auth_credential_raises_error_if_upstream_operation_errors(
    mock_operation, auth_config, context, provider
):
  """Test get_auth_credential raises RuntimeError for failed operations."""
  mock_operation.error.message = "OAuth server error"
  mock_operation.done = False

  with pytest.raises(
      RuntimeError, match="Operation failed: OAuth server error"
  ):
    await provider.get_auth_credential(auth_config, context)


async def test_get_auth_credential_raises_error_if_upstream_call_fails(
    mock_client, auth_config, context, provider
):
  """Test get_auth_credential raises RuntimeError for failed calls."""
  mock_client.retrieve_credentials.side_effect = Exception(
      "API Quota Exhausted"
  )

  with pytest.raises(
      RuntimeError,
      match="Failed to retrieve credential for user 'user' on connector",
  ) as exc_info:
    await provider.get_auth_credential(auth_config, context)

  # Assert that the original Exception is the chained cause!
  assert str(exc_info.value.__cause__) == "API Quota Exhausted"


@patch.object(gcp_auth_provider.time, "time")
async def test_get_auth_credential_raises_error_if_polling_times_out(
    mock_time,
    mock_operation,
    auth_config,
    context,
    provider,
):
  """Test get_auth_credential raises RuntimeError if polling times out."""

  # Force the operation into the polling loop state
  meta_pb = RetrieveCredentialsMetadata.pb()()
  meta_pb.consent_pending.SetInParent()
  meta = RetrieveCredentialsMetadata.deserialize(meta_pb.SerializeToString())
  mock_operation.metadata.value = RetrieveCredentialsMetadata.serialize(meta)

  # First call sets start_time=0.0, second call checks time > timeout
  # (20.0 > 10.0)
  mock_time.side_effect = [0.0, 20.0]

  mock_metadata = Mock(spec=RetrieveCredentialsMetadata)
  mock_metadata.consent_pending = True
  mock_metadata.uri_consent_required = False
  mock_operation.done = True
  mock_operation.ClearField("error")
  mock_client = Mock(spec=gcp_auth_provider.Client)
  mock_client.retrieve_credentials.side_effect = Exception(
      "Timeout waiting for credentials."
  )
  provider._client = mock_client

  with pytest.raises(
      RuntimeError,
      match="Failed to retrieve credential for user 'user' on connector",
  ) as exc_info:
    await provider.get_auth_credential(auth_config, context)

  assert "Timeout waiting for credentials." in str(exc_info.value.__cause__)


# ==============================================================================
# Interactive Auth Flows (3-legged OAuth for User Consents)
# ==============================================================================


async def test_get_auth_credential_initiates_user_consent(
    mock_operation, auth_config, context, provider
):
  # Explicitly set the mock behavior for this test
  expected_uri = "https://example.com/auth"
  expected_nonce = "sample-nonce-123"
  meta = RetrieveCredentialsMetadata({
      "uri_consent_required": {
          "authorization_uri": expected_uri,
          "consent_nonce": expected_nonce,
      }
  })
  mock_operation.metadata.value = RetrieveCredentialsMetadata.serialize(meta)
  mock_operation.done = False
  # Assert that there is no prior user consent completion event
  assert not context.session.events

  credential = await provider.get_auth_credential(auth_config, context)

  assert credential is not None
  assert credential.auth_type == AuthCredentialTypes.OAUTH2
  assert credential.oauth2.auth_uri == expected_uri
  assert credential.oauth2.nonce == expected_nonce


async def test_get_auth_credential_returns_fresh_auth_uri_for_repeated_requests(
    mock_client, mock_operation, auth_config, context, provider
):
  """Test that repeated calls fetch fresh auth URIs if consent is still pending."""
  # Arrange: Explicit initial URI
  initial_uri = "https://example.com/auth"
  initial_nonce = "initial-nonce-123"
  meta1 = RetrieveCredentialsMetadata({
      "uri_consent_required": {
          "authorization_uri": initial_uri,
          "consent_nonce": initial_nonce,
      }
  })
  mock_operation.metadata.value = RetrieveCredentialsMetadata.serialize(meta1)
  mock_operation.done = False

  credential1 = await provider.get_auth_credential(auth_config, context)
  assert credential1.oauth2.auth_uri == initial_uri
  assert credential1.oauth2.nonce == initial_nonce

  # Arrange: Explicit new URI for the second call
  fresh_auth_uri = "https://example.com/auth_new"
  fresh_nonce = "fresh-nonce-456"
  meta2 = RetrieveCredentialsMetadata({
      "uri_consent_required": {
          "authorization_uri": fresh_auth_uri,
          "consent_nonce": fresh_nonce,
      }
  })
  mock_operation.metadata.value = RetrieveCredentialsMetadata.serialize(meta2)

  credential2 = await provider.get_auth_credential(auth_config, context)

  assert mock_client.retrieve_credentials.call_count == 2
  assert credential2.oauth2.auth_uri == fresh_auth_uri
  assert credential2.oauth2.nonce == fresh_nonce


async def test_get_auth_credential_returns_token_if_consent_was_completed(
    mock_operation, auth_config, context, provider
):
  # Setup mock credential for successful credential retrieval
  mock_credential = RetrieveCredentialsResponse(
      header="Authorization: Bearer", token="test-token"
  )
  mock_operation.response.value = RetrieveCredentialsResponse.serialize(
      mock_credential
  )

  # Create mock events
  # 1. FunctionCall event for adk_request_credential
  function_call = Mock()
  function_call.id = "auth-req-1"
  function_call.name = REQUEST_EUC_FUNCTION_CALL_NAME
  function_call.args = {"function_call_id": "call-123"}

  event1 = Mock()
  event1.get_function_calls.return_value = [function_call]
  event1.get_function_responses.return_value = []

  # 2. FunctionResponse event for adk_request_credential
  function_response = Mock()
  function_response.id = "auth-req-1"
  function_response.name = REQUEST_EUC_FUNCTION_CALL_NAME

  event2 = Mock()
  event2.get_function_calls.return_value = []
  event2.get_function_responses.return_value = [function_response]

  # Setup tool context and event history (order of events matters)
  context.session.events = [event1, event2]
  context.function_call_id = "call-123"

  # Also set uri_consent_required to True-ish so it enters the check block
  meta = RetrieveCredentialsMetadata(
      uri_consent_required=RetrieveCredentialsMetadata.UriConsentRequired()
  )
  mock_operation.metadata.value = RetrieveCredentialsMetadata.serialize(meta)

  # Execute
  auth_credential = await provider.get_auth_credential(auth_config, context)

  # Verify
  assert auth_credential is not None
  assert auth_credential.auth_type == AuthCredentialTypes.HTTP
  assert auth_credential.http.scheme == "bearer"
  assert auth_credential.http.credentials.token == "test-token"
