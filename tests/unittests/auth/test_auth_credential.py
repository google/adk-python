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

"""Tests for the auth credential models and their shared base model."""

from __future__ import annotations

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import BaseModelWithConfig
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccountCredential
import pydantic
import pytest


class _Sample(BaseModelWithConfig):
  access_token: str


def test_base_model_with_config_accepts_camel_case_alias():
  """Credentials arrive as JSON using the camelCase wire names."""
  model = _Sample.model_validate({'accessToken': 'abc'})
  assert model.access_token == 'abc'


def test_base_model_with_config_accepts_the_python_field_name():
  """Python callers construct with the snake_case field name."""
  model = _Sample(access_token='abc')
  assert model.access_token == 'abc'


def test_base_model_with_config_keeps_unknown_fields():
  # Provider-specific keys are not modelled here, but dropping them would
  # lose data on a load/dump round trip.
  model = _Sample.model_validate({'accessToken': 'abc', 'tenantId': 'xyz'})
  assert model.model_dump()['tenantId'] == 'xyz'


def test_base_model_with_config_dumps_camel_case_only_when_asked():
  model = _Sample(access_token='abc')
  assert model.model_dump()['access_token'] == 'abc'
  assert model.model_dump(by_alias=True)['accessToken'] == 'abc'


def test_api_key_redacted_in_repr_and_str():
  """An API key is not rendered, but is still readable on the model."""
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      api_key='sk-live-secret-api-key-12345',
  )
  repr_str = repr(cred)
  str_str = str(cred)
  assert 'sk-live-secret-api-key-12345' not in repr_str
  assert 'sk-live-secret-api-key-12345' not in str_str
  # Only the rendering is redacted; the value itself is untouched.
  assert cred.api_key == 'sk-live-secret-api-key-12345'


def test_http_credentials_redacted_in_repr_and_str():
  """HTTP passwords, tokens and auth headers are not rendered."""
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme='basic',
          credentials=HttpCredentials(
              username='my_user',
              password='secret_password_999',
              token='secret_token_abc',
          ),
          additional_headers={'Authorization': 'Bearer secret_bearer_token'},
      ),
  )
  repr_str = repr(cred)
  str_str = str(cred)
  assert 'secret_password_999' not in repr_str
  assert 'secret_token_abc' not in repr_str
  assert 'secret_bearer_token' not in repr_str
  assert 'secret_password_999' not in str_str
  assert 'secret_token_abc' not in str_str


def test_oauth2_credentials_redacted_in_repr_and_str():
  """OAuth2 secrets, tokens and the auth response URI are not rendered."""
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id='my_client_id',
          client_secret='top_secret_client_secret',
          access_token='secret_access_token',
          refresh_token='secret_refresh_token',
          id_token='secret_id_token',
          auth_code='secret_auth_code',
          auth_response_uri=(
              'https://example.com/callback?code=secret_response_code'
          ),
          code_verifier='secret_code_verifier',
      ),
  )
  repr_str = repr(cred)
  str_str = str(cred)
  assert 'top_secret_client_secret' not in repr_str
  assert 'secret_access_token' not in repr_str
  assert 'secret_refresh_token' not in repr_str
  assert 'secret_id_token' not in repr_str
  assert 'secret_auth_code' not in repr_str
  assert 'secret_response_code' not in repr_str
  assert 'secret_code_verifier' not in repr_str
  assert 'top_secret_client_secret' not in str_str
  assert 'secret_response_code' not in str_str


def test_service_account_redacted_in_repr_and_str():
  """A service account private key and its ID are not rendered."""
  sa_cred = ServiceAccountCredential(
      type_='service_account',
      project_id='test_project',
      private_key_id='secret_private_key_id',
      private_key=(
          '-----BEGIN PRIVATE KEY-----\nsecret_key_data\n-----END PRIVATE'
          ' KEY-----'
      ),
      client_email='test@iam.gserviceaccount.com',
      client_id='12345',
      auth_uri='https://example.com/o/oauth2/auth',
      token_uri='https://example.com/token',
      auth_provider_x509_cert_url='https://example.com/oauth2/v1/certs',
      client_x509_cert_url='https://example.com/robot/v1/metadata/x509/test',
      universe_domain='example.com',
  )
  repr_str = repr(sa_cred)
  str_str = str(sa_cred)
  assert 'secret_key_data' not in repr_str
  assert 'secret_private_key_id' not in repr_str
  assert 'secret_key_data' not in str_str
  assert 'secret_private_key_id' not in str_str


def test_extra_fields_redacted_in_repr_and_str():
  """A secret under an undeclared key is redacted, not rendered."""
  # `extra="allow"` means a secret can arrive under a key the model does not
  # declare, which pydantic would otherwise render in repr unconditionally.
  cred = AuthCredential.model_validate({
      'auth_type': AuthCredentialTypes.API_KEY,
      'undeclared_secret': 'secret_extra_value',
  })
  repr_str = repr(cred)
  str_str = str(cred)
  assert 'secret_extra_value' not in repr_str
  assert 'secret_extra_value' not in str_str
  # The key is still surfaced so the redaction is visible when debugging, and
  # the value remains readable programmatically.
  assert 'undeclared_secret' in repr_str
  assert cred.undeclared_secret == 'secret_extra_value'


def test_nested_extra_fields_redacted_in_repr_and_str():
  """Undeclared keys on a nested credential model are redacted too."""
  # Mirrors an OAuth2 provider returning a non-standard token field.
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth.model_validate({
          'client_id': 'my_client_id',
          'unexpected_token': 'secret_unexpected_token',
      }),
  )
  repr_str = repr(cred)
  str_str = str(cred)
  assert 'secret_unexpected_token' not in repr_str
  assert 'secret_unexpected_token' not in str_str
  assert 'my_client_id' in repr_str


def test_validation_error_does_not_echo_secret_value():
  """A rejected value is not echoed back in the ValidationError text."""
  # Pydantic reports the rejected value as `input_value=...` by default, which
  # would put the secret into the error string surfaced to the LLM.
  with pytest.raises(pydantic.ValidationError) as exc_info:
    AuthCredential.model_validate({
        'auth_type': AuthCredentialTypes.API_KEY,
        'api_key': ['sk-live-secret-api-key-12345'],
    })
  message = str(exc_info.value)
  assert 'sk-live-secret-api-key-12345' not in message
  # The field and the reason are still reported.
  assert 'api_key' in message
