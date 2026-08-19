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

"""Tests for building HTTP headers from a resolved credential."""

import base64
import logging

from fastapi.openapi.models import APIKey as APIKeyScheme
from fastapi.openapi.models import APIKeyIn
from google.adk.auth._auth_headers import build_auth_headers
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth


def test_returns_none_without_credential():
  assert build_auth_headers(None) is None


def test_oauth2_access_token():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(access_token="access-token"),
  )

  assert build_auth_headers(credential) == {
      "Authorization": "Bearer access-token"
  }


def test_http_bearer_token():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer",
          credentials=HttpCredentials(token="bearer-token"),
      ),
  )

  assert build_auth_headers(credential) == {
      "Authorization": "Bearer bearer-token"
  }


def test_http_basic_auth():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password="pass"),
      ),
  )

  encoded = base64.b64encode(b"user:pass").decode()
  assert build_auth_headers(credential) == {"Authorization": f"Basic {encoded}"}


def test_http_custom_scheme_uses_scheme_name():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="Custom",
          credentials=HttpCredentials(token="custom-token"),
      ),
  )

  assert build_auth_headers(credential) == {
      "Authorization": "Custom custom-token"
  }


def test_http_additional_headers_without_token():
  """Agent Identity returns custom headers with no usable scheme or token."""
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="",
          credentials=HttpCredentials(),
          additional_headers={"X-Goog-Api-Key": "api-key"},
      ),
  )

  assert build_auth_headers(credential) == {"X-Goog-Api-Key": "api-key"}


def test_api_key_in_header():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="api-key"
  )
  auth_scheme = APIKeyScheme(**{"in": APIKeyIn.header, "name": "X-API-Key"})

  assert build_auth_headers(credential, auth_scheme) == {"X-API-Key": "api-key"}


def test_api_key_outside_header_is_unsupported(caplog):
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="api-key"
  )
  auth_scheme = APIKeyScheme(**{"in": APIKeyIn.query, "name": "api_key"})

  with caplog.at_level(logging.WARNING, logger="google_adk"):
    assert build_auth_headers(credential, auth_scheme) is None

  assert "Only header-based API key authentication is supported" in caplog.text


def test_api_key_without_scheme_is_unsupported():
  credential = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY, api_key="api-key"
  )

  assert build_auth_headers(credential) is None
