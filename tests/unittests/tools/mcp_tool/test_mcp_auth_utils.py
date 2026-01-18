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

import base64
from unittest.mock import patch

from fastapi.openapi import models as openapi_models
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.tools.mcp_tool import mcp_auth_utils
import pytest


def test_get_mcp_auth_headers_no_credential():
  """Test header generation with no credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="bearer")
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=None
  )
  assert headers is None


def test_get_mcp_auth_headers_no_auth_scheme():
  """Test header generation with no auth_scheme."""
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(access_token="test_token"),
  )
  with patch.object(mcp_auth_utils, "logger") as mock_logger:
    headers = mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=None, credential=credential
    )
    assert headers == {"Authorization": "Bearer test_token"}


def test_get_mcp_auth_headers_oauth2():
  """Test header generation for OAuth2 credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="bearer")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(access_token="test_token"),
  )
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=credential
  )
  assert headers == {"Authorization": "Bearer test_token"}


def test_get_mcp_auth_headers_http_bearer():
  """Test header generation for HTTP Bearer credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="bearer")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="bearer_token")
      ),
  )
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=credential
  )
  assert headers == {"Authorization": "Bearer bearer_token"}


def test_get_mcp_auth_headers_http_basic():
  """Test header generation for HTTP Basic credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="basic")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password="pass"),
      ),
  )
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=credential
  )
  expected_encoded = base64.b64encode(b"user:pass").decode()
  assert headers == {"Authorization": f"Basic {expected_encoded}"}


def test_get_mcp_auth_headers_http_basic_missing_credentials():
  """Test header generation for HTTP Basic with missing credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="basic")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password=None),
      ),
  )
  with patch.object(mcp_auth_utils, "logger") as mock_logger:
    headers = mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )
    assert headers is None
    mock_logger.warning.assert_called_once_with(
        "Basic auth scheme missing username or password."
    )


def test_get_mcp_auth_headers_http_custom_scheme():
  """Test header generation for custom HTTP scheme."""
  auth_scheme = openapi_models.HTTPBase(scheme="custom")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="custom", credentials=HttpCredentials(token="custom_token")
      ),
  )
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=credential
  )
  assert headers == {"Authorization": "custom custom_token"}


def test_get_mcp_auth_headers_http_cred_wrong_scheme():
  """Test HTTP credential with non-HTTPBase auth scheme."""
  auth_scheme = openapi_models.APIKey(**{
      "type": AuthSchemeType.apiKey,
      "in": openapi_models.APIKeyIn.header,
      "name": "X-API-Key",
  })
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="bearer_token")
      ),
  )
  with patch.object(mcp_auth_utils, "logger") as mock_logger:
    headers = mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )
    assert headers is None
    mock_logger.warning.assert_called_once_with(
        "HTTP credential provided, but auth_scheme is missing or not HTTPBase."
    )


def test_get_mcp_auth_headers_api_key_header():
  """Test header generation for API Key in header."""
  auth_scheme = openapi_models.APIKey(**{
      "type": AuthSchemeType.apiKey,
      "in": openapi_models.APIKeyIn.header,
      "name": "X-Custom-API-Key",
  })
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="my_api_key"
  )
  headers = mcp_auth_utils.get_mcp_auth_headers(
      auth_scheme=auth_scheme, credential=credential
  )
  assert headers == {"X-Custom-API-Key": "my_api_key"}


def test_get_mcp_auth_headers_api_key_query_raises_error():
  """Test API Key in query raises ValueError."""
  auth_scheme = openapi_models.APIKey(**{
      "type": AuthSchemeType.apiKey,
      "in": openapi_models.APIKeyIn.query,
      "name": "api_key",
  })
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="my_api_key"
  )
  with pytest.raises(
      ValueError,
      match="MCP tools only support header-based API key authentication.",
  ):
    mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )


def test_get_mcp_auth_headers_api_key_cookie_raises_error():
  """Test API Key in cookie raises ValueError."""
  auth_scheme = openapi_models.APIKey(**{
      "type": AuthSchemeType.apiKey,
      "in": openapi_models.APIKeyIn.cookie,
      "name": "session_id",
  })
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="my_api_key"
  )
  with pytest.raises(
      ValueError,
      match="MCP tools only support header-based API key authentication.",
  ):
    mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )


def test_get_mcp_auth_headers_api_key_cred_wrong_scheme():
  """Test API key credential with non-APIKey auth scheme."""
  auth_scheme = openapi_models.HTTPBase(scheme="bearer")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="my_api_key"
  )
  with patch.object(mcp_auth_utils, "logger") as mock_logger:
    headers = mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )
    assert headers is None
    mock_logger.warning.assert_called_once_with(
        "API key credential provided, but auth_scheme is missing or not APIKey."
    )


def test_get_mcp_auth_headers_service_account():
  """Test header generation for service account credentials."""
  auth_scheme = openapi_models.HTTPBase(scheme="bearer")
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(scopes=["test"]),
  )
  with patch.object(mcp_auth_utils, "logger") as mock_logger:
    headers = mcp_auth_utils.get_mcp_auth_headers(
        auth_scheme=auth_scheme, credential=credential
    )
    assert headers is None
    mock_logger.warning.assert_called_once_with(
        "Service account credentials should be exchanged for an access "
        "token before calling get_mcp_auth_headers."
    )
