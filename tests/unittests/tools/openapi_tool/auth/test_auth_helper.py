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

from unittest.mock import patch

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import HTTPBase
from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OpenIdConnect
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.openapi_tool.auth.auth_helpers import credential_to_param
from google.adk.tools.openapi_tool.auth.auth_helpers import dict_to_auth_scheme
from google.adk.tools.openapi_tool.auth.auth_helpers import INTERNAL_AUTH_PREFIX
from google.adk.tools.openapi_tool.auth.auth_helpers import openid_dict_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import openid_url_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import service_account_dict_to_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import service_account_scheme_credential
from google.adk.tools.openapi_tool.auth.auth_helpers import token_to_scheme_credential
import pytest
import requests


def test_token_to_scheme_credential_api_key_header():
  scheme, credential = token_to_scheme_credential(
      "apikey", "header", "X-API-Key", "test_key"
  )

  assert isinstance(scheme, APIKey)
  assert scheme.type_ == AuthSchemeType.apiKey
  assert scheme.in_ == APIKeyIn.header
  assert scheme.name == "X-API-Key"
  assert credential == AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )


def test_token_to_scheme_credential_api_key_query():
  scheme, credential = token_to_scheme_credential(
      "apikey", "query", "api_key", "test_key"
  )

  assert isinstance(scheme, APIKey)
  assert scheme.type_ == AuthSchemeType.apiKey
  assert scheme.in_ == APIKeyIn.query
  assert scheme.name == "api_key"
  assert credential == AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )


def test_token_to_scheme_credential_api_key_cookie():
  scheme, credential = token_to_scheme_credential(
      "apikey", "cookie", "session_id", "test_key"
  )

  assert isinstance(scheme, APIKey)
  assert scheme.type_ == AuthSchemeType.apiKey
  assert scheme.in_ == APIKeyIn.cookie
  assert scheme.name == "session_id"
  assert credential == AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )


def test_token_to_scheme_credential_api_key_no_credential():
  scheme, credential = token_to_scheme_credential(
      "apikey", "cookie", "session_id"
  )

  assert isinstance(scheme, APIKey)
  assert credential is None


def test_token_to_scheme_credential_oauth2_token():
  scheme, credential = token_to_scheme_credential(
      "oauth2Token", "header", "Authorization", "test_token"
  )

  assert isinstance(scheme, HTTPBearer)
  assert scheme.bearerFormat == "JWT"
  assert credential == AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="test_token")
      ),
  )


def test_token_to_scheme_credential_oauth2_no_credential():
  scheme, credential = token_to_scheme_credential(
      "oauth2Token", "header", "Authorization"
  )

  assert isinstance(scheme, HTTPBearer)
  assert credential is None


def test_service_account_dict_to_scheme_credential():
  config = {
      "type": "service_account",
      "project_id": "project_id",
      "private_key_id": "private_key_id",
      "private_key": "private_key",
      "client_email": "client_email",
      "client_id": "client_id",
      "auth_uri": "auth_uri",
      "token_uri": "token_uri",
      "auth_provider_x509_cert_url": "auth_provider_x509_cert_url",
      "client_x509_cert_url": "client_x509_cert_url",
      "universe_domain": "universe_domain",
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = service_account_dict_to_scheme_credential(config, scopes)

  assert isinstance(scheme, HTTPBearer)
  assert scheme.bearerFormat == "JWT"
  assert credential.auth_type == AuthCredentialTypes.SERVICE_ACCOUNT
  assert credential.service_account.scopes == scopes
  assert (
      credential.service_account.service_account_credential.project_id
      == "project_id"
  )


def test_service_account_scheme_credential():
  config = ServiceAccount(
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

  scheme, credential = service_account_scheme_credential(config)

  assert isinstance(scheme, HTTPBearer)
  assert scheme.bearerFormat == "JWT"
  assert credential.auth_type == AuthCredentialTypes.SERVICE_ACCOUNT
  assert credential.service_account == config


def test_openid_dict_to_scheme_credential():
  config_dict = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
      "openIdConnectUrl": "openid_url",
  }
  credential_dict = {
      "client_id": "client_id",
      "client_secret": "client_secret",
      "redirect_uri": "redirect_uri",
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = openid_dict_to_scheme_credential(
      config_dict, scopes, credential_dict
  )

  assert isinstance(scheme, OpenIdConnectWithConfig)
  assert scheme.authorization_endpoint == "auth_url"
  assert scheme.token_endpoint == "token_url"
  assert scheme.scopes == scopes
  assert credential.auth_type == AuthCredentialTypes.OPEN_ID_CONNECT
  assert credential.oauth2.client_id == "client_id"
  assert credential.oauth2.client_secret == "client_secret"
  assert credential.oauth2.redirect_uri == "redirect_uri"


def test_openid_dict_to_scheme_credential_no_openid_url():
  config_dict = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
  }
  credential_dict = {
      "client_id": "client_id",
      "client_secret": "client_secret",
      "redirect_uri": "redirect_uri",
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = openid_dict_to_scheme_credential(
      config_dict, scopes, credential_dict
  )

  assert scheme.openIdConnectUrl == ""


def test_openid_dict_to_scheme_credential_google_oauth_credential():
  config_dict = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
      "openIdConnectUrl": "openid_url",
  }
  credential_dict = {
      "web": {
          "client_id": "client_id",
          "client_secret": "client_secret",
          "redirect_uri": "redirect_uri",
      }
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = openid_dict_to_scheme_credential(
      config_dict, scopes, credential_dict
  )

  assert isinstance(scheme, OpenIdConnectWithConfig)
  assert credential.auth_type == AuthCredentialTypes.OPEN_ID_CONNECT
  assert credential.oauth2.client_id == "client_id"
  assert credential.oauth2.client_secret == "client_secret"
  assert credential.oauth2.redirect_uri == "redirect_uri"


def test_openid_dict_to_scheme_credential_invalid_config():
  config_dict = {
      "invalid_field": "value",
  }
  credential_dict = {
      "client_id": "client_id",
      "client_secret": "client_secret",
  }
  scopes = ["scope1", "scope2"]

  with pytest.raises(ValueError, match="Invalid OpenID Connect configuration"):
    openid_dict_to_scheme_credential(config_dict, scopes, credential_dict)


def test_openid_dict_to_scheme_credential_missing_credential_fields():
  config_dict = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
  }
  credential_dict = {
      "client_id": "client_id",
  }
  scopes = ["scope1", "scope2"]

  with pytest.raises(
      ValueError,
      match="Missing required fields in credential_dict: client_secret",
  ):
    openid_dict_to_scheme_credential(config_dict, scopes, credential_dict)


@patch("requests.get")
def test_openid_url_to_scheme_credential(mock_get):
  mock_response = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
      "userinfo_endpoint": "userinfo_url",
  }
  mock_get.return_value.json.return_value = mock_response
  mock_get.return_value.raise_for_status.return_value = None
  credential_dict = {
      "client_id": "client_id",
      "client_secret": "client_secret",
      "redirect_uri": "redirect_uri",
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = openid_url_to_scheme_credential(
      "openid_url", scopes, credential_dict
  )

  assert isinstance(scheme, OpenIdConnectWithConfig)
  assert scheme.authorization_endpoint == "auth_url"
  assert scheme.token_endpoint == "token_url"
  assert scheme.scopes == scopes
  assert credential.auth_type == AuthCredentialTypes.OPEN_ID_CONNECT
  assert credential.oauth2.client_id == "client_id"
  assert credential.oauth2.client_secret == "client_secret"
  assert credential.oauth2.redirect_uri == "redirect_uri"
  mock_get.assert_called_once_with("openid_url", timeout=10)


@patch("requests.get")
def test_openid_url_to_scheme_credential_no_openid_url(mock_get):
  mock_response = {
      "authorization_endpoint": "auth_url",
      "token_endpoint": "token_url",
      "userinfo_endpoint": "userinfo_url",
  }
  mock_get.return_value.json.return_value = mock_response
  mock_get.return_value.raise_for_status.return_value = None
  credential_dict = {
      "client_id": "client_id",
      "client_secret": "client_secret",
      "redirect_uri": "redirect_uri",
  }
  scopes = ["scope1", "scope2"]

  scheme, credential = openid_url_to_scheme_credential(
      "openid_url", scopes, credential_dict
  )

  assert scheme.openIdConnectUrl == "openid_url"


@patch("requests.get")
def test_openid_url_to_scheme_credential_request_exception(mock_get):
  mock_get.side_effect = requests.exceptions.RequestException("Test Error")
  credential_dict = {"client_id": "client_id", "client_secret": "client_secret"}

  with pytest.raises(
      ValueError, match="Failed to fetch OpenID configuration from openid_url"
  ):
    openid_url_to_scheme_credential("openid_url", [], credential_dict)


@patch("requests.get")
def test_openid_url_to_scheme_credential_invalid_json(mock_get):
  mock_get.return_value.json.side_effect = ValueError("Invalid JSON")
  mock_get.return_value.raise_for_status.return_value = None
  credential_dict = {"client_id": "client_id", "client_secret": "client_secret"}

  with pytest.raises(
      ValueError,
      match=(
          "Invalid JSON response from OpenID configuration endpoint openid_url"
      ),
  ):
    openid_url_to_scheme_credential("openid_url", [], credential_dict)


def test_credential_to_param_api_key_header():
  auth_scheme = APIKey(
      **{"type": "apiKey", "in": "header", "name": "X-API-Key"}
  )
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "X-API-Key"
  assert param.param_location == "header"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "X-API-Key": "test_key"}


def test_credential_to_param_api_key_query():
  auth_scheme = APIKey(**{"type": "apiKey", "in": "query", "name": "api_key"})
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "api_key"
  assert param.param_location == "query"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "api_key": "test_key"}


def test_credential_to_param_api_key_cookie():
  auth_scheme = APIKey(
      **{"type": "apiKey", "in": "cookie", "name": "session_id"}
  )
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="test_key"
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "session_id"
  assert param.param_location == "cookie"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "session_id": "test_key"}


def test_credential_to_param_http_bearer():
  auth_scheme = HTTPBearer(bearerFormat="JWT")
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="test_token")
      ),
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "Authorization"
  assert param.param_location == "header"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "Authorization": "Bearer test_token"}


def test_credential_to_param_http_basic_not_supported():
  auth_scheme = HTTPBase(scheme="basic")
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password="password"),
      ),
  )

  with pytest.raises(
      NotImplementedError, match="Basic Authentication is not supported."
  ):
    credential_to_param(auth_scheme, auth_credential)


def test_credential_to_param_http_invalid_credentials_no_http():
  auth_scheme = HTTPBase(scheme="basic")
  auth_credential = AuthCredential(auth_type=AuthCredentialTypes.HTTP)

  with pytest.raises(ValueError, match="Invalid HTTP auth credentials"):
    credential_to_param(auth_scheme, auth_credential)


def test_credential_to_param_oauth2():
  auth_scheme = OAuth2(flows={})
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="test_token")
      ),
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "Authorization"
  assert param.param_location == "header"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "Authorization": "Bearer test_token"}


def test_credential_to_param_openid_connect():
  auth_scheme = OpenIdConnect(openIdConnectUrl="openid_url")
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer", credentials=HttpCredentials(token="test_token")
      ),
  )

  param, kwargs = credential_to_param(auth_scheme, auth_credential)

  assert param.original_name == "Authorization"
  assert param.param_location == "header"
  assert kwargs == {INTERNAL_AUTH_PREFIX + "Authorization": "Bearer test_token"}


def test_credential_to_param_openid_no_credential():
  auth_scheme = OpenIdConnect(openIdConnectUrl="openid_url")

  param, kwargs = credential_to_param(auth_scheme, None)

  assert param == None
  assert kwargs == None


def test_credential_to_param_oauth2_no_credential():
  auth_scheme = OAuth2(flows={})

  param, kwargs = credential_to_param(auth_scheme, None)

  assert param == None
  assert kwargs == None


def test_dict_to_auth_scheme_api_key():
  data = {"type": "apiKey", "in": "header", "name": "X-API-Key"}

  scheme = dict_to_auth_scheme(data)

  assert isinstance(scheme, APIKey)
  assert scheme.type_ == AuthSchemeType.apiKey
  assert scheme.in_ == APIKeyIn.header
  assert scheme.name == "X-API-Key"


def test_dict_to_auth_scheme_http_bearer():
  data = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}

  scheme = dict_to_auth_scheme(data)

  assert isinstance(scheme, HTTPBearer)
  assert scheme.scheme == "bearer"
  assert scheme.bearerFormat == "JWT"


def test_dict_to_auth_scheme_http_base():
  data = {"type": "http", "scheme": "basic"}

  scheme = dict_to_auth_scheme(data)

  assert isinstance(scheme, HTTPBase)
  assert scheme.scheme == "basic"


def test_dict_to_auth_scheme_oauth2():
  data = {
      "type": "oauth2",
      "flows": {
          "authorizationCode": {
              "authorizationUrl": "https://example.com/auth",
              "tokenUrl": "https://example.com/token",
          }
      },
  }

  scheme = dict_to_auth_scheme(data)

  assert isinstance(scheme, OAuth2)
  assert hasattr(scheme.flows, "authorizationCode")


def test_dict_to_auth_scheme_openid_connect():
  data = {
      "type": "openIdConnect",
      "openIdConnectUrl": (
          "https://example.com/.well-known/openid-configuration"
      ),
  }

  scheme = dict_to_auth_scheme(data)

  assert isinstance(scheme, OpenIdConnect)
  assert (
      scheme.openIdConnectUrl
      == "https://example.com/.well-known/openid-configuration"
  )


def test_dict_to_auth_scheme_missing_type():
  data = {"in": "header", "name": "X-API-Key"}
  with pytest.raises(
      ValueError, match="Missing 'type' field in security scheme dictionary."
  ):
    dict_to_auth_scheme(data)


def test_dict_to_auth_scheme_invalid_type():
  data = {"type": "invalid", "in": "header", "name": "X-API-Key"}
  with pytest.raises(ValueError, match="Invalid security scheme type: invalid"):
    dict_to_auth_scheme(data)


def test_dict_to_auth_scheme_invalid_data():
  data = {"type": "apiKey", "in": "header"}  # Missing 'name'
  with pytest.raises(ValueError, match="Invalid security scheme data"):
    dict_to_auth_scheme(data)


if __name__ == "__main__":
  pytest.main([__file__])
