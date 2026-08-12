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

"""Unit tests for the service account credential exchanger."""

from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from google.adk.tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger
import google.auth
from google.auth import exceptions as google_auth_exceptions
import pytest

_ACCESS_TOKEN_MONKEYPATCH_TARGET = (
    "google.adk.tools.openapi_tool.auth.credential_exchangers."
    "service_account_exchanger.service_account.Credentials."
    "from_service_account_info"
)

_ID_TOKEN_MONKEYPATCH_TARGET = (
    "google.adk.tools.openapi_tool.auth.credential_exchangers."
    "service_account_exchanger.service_account.IDTokenCredentials."
    "from_service_account_info"
)

_FETCH_ID_TOKEN_MONKEYPATCH_TARGET = "google.oauth2.id_token.fetch_id_token"


@pytest.fixture
def service_account_exchanger():
  return ServiceAccountCredentialExchanger()


@pytest.fixture
def auth_scheme():
  scheme = MagicMock(spec=AuthScheme)
  scheme.type_ = AuthSchemeType.oauth2
  scheme.description = "Google Service Account"
  return scheme


@pytest.fixture
def sa_credential():
  """A minimal valid ServiceAccountCredential for testing."""
  return ServiceAccountCredential(
      type_="service_account",
      project_id="test_project_id",
      private_key_id="test_private_key_id",
      private_key="-----BEGIN PRIVATE KEY-----...",
      client_email="test@test.iam.gserviceaccount.com",
      client_id="test_client_id",
      auth_uri="https://accounts.google.com/o/oauth2/auth",
      token_uri="https://oauth2.googleapis.com/token",
      auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
      client_x509_cert_url=(
          "https://www.googleapis.com/robot/v1/metadata/x509/test"
      ),
      universe_domain="googleapis.com",
  )


_DEFAULT_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


# --- Access token exchange tests ---


def test_exchange_access_token_with_explicit_credentials(
    service_account_exchanger, auth_scheme, sa_credential, monkeypatch
):
  mock_credentials = MagicMock()
  mock_credentials.token = "mock_access_token"
  mock_from_sa_info = MagicMock(return_value=mock_credentials)
  monkeypatch.setattr(_ACCESS_TOKEN_MONKEYPATCH_TARGET, mock_from_sa_info)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=sa_credential,
          scopes=_DEFAULT_SCOPES,
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_access_token"
  mock_from_sa_info.assert_called_once()
  mock_credentials.refresh.assert_called_once()


@pytest.mark.parametrize(
    "cred_quota_project_id, adc_project_id, expected_quota_project_id",
    [
        ("test_project", "another_project", "test_project"),
        (None, "adc_project", "adc_project"),
        (None, None, None),
    ],
)
def test_exchange_access_token_with_adc_sets_quota_project(
    service_account_exchanger,
    auth_scheme,
    monkeypatch,
    cred_quota_project_id,
    adc_project_id,
    expected_quota_project_id,
):
  mock_credentials = MagicMock()
  mock_credentials.token = "mock_access_token"
  mock_credentials.quota_project_id = cred_quota_project_id
  mock_google_auth_default = MagicMock(
      return_value=(mock_credentials, adc_project_id)
  )
  monkeypatch.setattr(google.auth, "default", mock_google_auth_default)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          use_default_credential=True,
          scopes=["https://www.googleapis.com/auth/bigquery"],
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_access_token"
  if expected_quota_project_id:
    assert (
        result.http.additional_headers["x-goog-user-project"]
        == expected_quota_project_id
    )
  else:
    assert not result.http.additional_headers
  mock_google_auth_default.assert_called_once_with(
      scopes=["https://www.googleapis.com/auth/bigquery"]
  )
  mock_credentials.refresh.assert_called_once()


def test_exchange_access_token_with_adc_defaults_to_cloud_platform_scope(
    service_account_exchanger, auth_scheme, monkeypatch
):
  mock_credentials = MagicMock()
  mock_credentials.token = "mock_access_token"
  mock_credentials.quota_project_id = None
  mock_google_auth_default = MagicMock(return_value=(mock_credentials, None))
  monkeypatch.setattr(google.auth, "default", mock_google_auth_default)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          use_default_credential=True,
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_access_token"
  mock_google_auth_default.assert_called_once_with(scopes=_DEFAULT_SCOPES)


def test_exchange_raises_when_auth_credential_is_none(
    service_account_exchanger, auth_scheme
):
  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, None)
  assert "Service account credentials are missing" in str(exc_info.value)


def test_exchange_raises_when_service_account_is_none(
    service_account_exchanger, auth_scheme
):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
  )
  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Service account credentials are missing" in str(exc_info.value)


def test_exchange_wraps_google_auth_error_as_missing_error(
    service_account_exchanger, auth_scheme, sa_credential, monkeypatch
):
  mock_from_sa_info = MagicMock(
      side_effect=ValueError("Failed to load credentials")
  )
  monkeypatch.setattr(_ACCESS_TOKEN_MONKEYPATCH_TARGET, mock_from_sa_info)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=sa_credential,
          scopes=_DEFAULT_SCOPES,
      ),
  )

  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Failed to exchange service account token" in str(exc_info.value)
  mock_from_sa_info.assert_called_once()


def test_exchange_raises_when_explicit_credentials_have_no_scopes(
    service_account_exchanger, auth_scheme, sa_credential
):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=sa_credential,
      ),
  )

  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "scopes are required" in str(exc_info.value)


# --- ID token exchange tests ---


def test_exchange_id_token_with_explicit_credentials(
    service_account_exchanger, auth_scheme, sa_credential, monkeypatch
):
  mock_id_credentials = MagicMock()
  mock_id_credentials.token = "mock_id_token"
  mock_from_sa_info = MagicMock(return_value=mock_id_credentials)
  monkeypatch.setattr(_ID_TOKEN_MONKEYPATCH_TARGET, mock_from_sa_info)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=sa_credential,
          scopes=_DEFAULT_SCOPES,
          use_id_token=True,
          audience="https://my-service.run.app",
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_id_token"
  assert result.http.additional_headers is None
  mock_from_sa_info.assert_called_once()
  assert (
      mock_from_sa_info.call_args[1]["target_audience"]
      == "https://my-service.run.app"
  )
  mock_id_credentials.refresh.assert_called_once()


def test_exchange_id_token_with_adc(
    service_account_exchanger, auth_scheme, monkeypatch
):
  mock_fetch_id_token = MagicMock(return_value="mock_adc_id_token")
  monkeypatch.setattr(_FETCH_ID_TOKEN_MONKEYPATCH_TARGET, mock_fetch_id_token)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          use_default_credential=True,
          scopes=_DEFAULT_SCOPES,
          use_id_token=True,
          audience="https://my-service.run.app",
      ),
  )

  result = service_account_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert result.auth_type == AuthCredentialTypes.HTTP
  assert result.http.scheme == "bearer"
  assert result.http.credentials.token == "mock_adc_id_token"
  assert result.http.additional_headers is None
  mock_fetch_id_token.assert_called_once()
  assert mock_fetch_id_token.call_args[0][1] == "https://my-service.run.app"


def test_id_token_requires_audience():
  with pytest.raises(
      ValueError, match="audience is required when use_id_token is True"
  ):
    ServiceAccount(
        use_default_credential=True,
        use_id_token=True,
    )


def test_exchange_id_token_wraps_error_with_explicit_credentials(
    service_account_exchanger, auth_scheme, sa_credential, monkeypatch
):
  mock_from_sa_info = MagicMock(
      side_effect=ValueError("Failed to create ID token credentials")
  )
  monkeypatch.setattr(_ID_TOKEN_MONKEYPATCH_TARGET, mock_from_sa_info)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          service_account_credential=sa_credential,
          scopes=_DEFAULT_SCOPES,
          use_id_token=True,
          audience="https://my-service.run.app",
      ),
  )

  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Failed to exchange service account for ID token" in str(
      exc_info.value
  )


def test_exchange_id_token_wraps_error_with_adc(
    service_account_exchanger, auth_scheme, monkeypatch
):
  mock_fetch_id_token = MagicMock(
      side_effect=google_auth_exceptions.DefaultCredentialsError(
          "Metadata service unavailable"
      )
  )
  monkeypatch.setattr(_FETCH_ID_TOKEN_MONKEYPATCH_TARGET, mock_fetch_id_token)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
      service_account=ServiceAccount(
          use_default_credential=True,
          scopes=_DEFAULT_SCOPES,
          use_id_token=True,
          audience="https://my-service.run.app",
      ),
  )

  with pytest.raises(AuthCredentialMissingError) as exc_info:
    service_account_exchanger.exchange_credential(auth_scheme, auth_credential)
  assert "Failed to exchange service account for ID token" in str(
      exc_info.value
  )


# --- Model validator tests ---


def test_model_validator_rejects_missing_credential_without_adc():
  with pytest.raises(
      ValueError,
      match="service_account_credential is required",
  ):
    ServiceAccount(
        use_default_credential=False,
        scopes=_DEFAULT_SCOPES,
    )


def test_model_validator_allows_adc_without_explicit_credential():
  sa = ServiceAccount(
      use_default_credential=True,
      scopes=_DEFAULT_SCOPES,
  )
  assert sa.service_account_credential is None
  assert sa.use_default_credential is True
