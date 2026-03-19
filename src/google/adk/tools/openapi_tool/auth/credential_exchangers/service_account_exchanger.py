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

"""Credential fetcher for Google Service Account."""

from __future__ import annotations

from typing import Optional

import google.auth
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import google.oauth2.credentials

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_credential import ServiceAccount
from .....auth.auth_schemes import AuthScheme
from .base_credential_exchanger import AuthCredentialMissingError
from .base_credential_exchanger import BaseAuthCredentialExchanger


class ServiceAccountCredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for Google Service Account.

  Uses the default service credential if `use_default_credential = True`.
  Otherwise, uses the service account credential provided in the auth
  credential.

  Supports exchanging for either an access token (default) or an ID token
  when ``ServiceAccount.use_id_token`` is True. ID tokens are required for
  service-to-service authentication with Cloud Run, Cloud Functions, and
  other services that verify caller identity.
  """

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the service account auth credential for a token.

    If auth_credential contains a service account credential, it will be used
    to fetch a token. Otherwise, the default service credential will be
    used for fetching a token.

    When ``service_account.use_id_token`` is True, an ID token is fetched
    using the configured ``audience``. This is required for authenticating
    to Cloud Run, Cloud Functions, and similar services.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential in HTTPBearer format, containing the token.
    """
    if auth_credential is None or auth_credential.service_account is None:
      raise AuthCredentialMissingError(
          "Service account credentials are missing. Please provide them, or"
          " set `use_default_credential = True` to use application default"
          " credential in a hosted service like Cloud Run."
      )

    sa_config = auth_credential.service_account

    if sa_config.use_id_token:
      return self._exchange_for_id_token(sa_config)

    return self._exchange_for_access_token(sa_config)

  def _exchange_for_id_token(self, sa_config: ServiceAccount) -> AuthCredential:
    """Exchanges the service account credential for an ID token.

    Args:
        sa_config: The service account configuration.

    Returns:
        An AuthCredential in HTTPBearer format containing the ID token.

    Raises:
        AuthCredentialMissingError: If token exchange fails.
    """
    # audience and credential presence are validated by the ServiceAccount
    # model_validator at construction time.
    try:
      if sa_config.use_default_credential:
        from google.oauth2 import id_token as oauth2_id_token

        request = Request()
        token = oauth2_id_token.fetch_id_token(request, sa_config.audience)
      else:
        # Guaranteed non-None by ServiceAccount model_validator.
        assert sa_config.service_account_credential is not None
        credentials = (
            service_account.IDTokenCredentials.from_service_account_info(
                sa_config.service_account_credential.model_dump(),
                target_audience=sa_config.audience,
            )
        )
        credentials.refresh(Request())
        token = credentials.token

      return AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=token),
          ),
      )

    # ValueError is raised by google-auth when service account JSON is
    # missing required fields (e.g. client_email, private_key), or when
    # fetch_id_token cannot determine credentials from the environment.
    except (google_auth_exceptions.GoogleAuthError, ValueError) as e:
      raise AuthCredentialMissingError(
          f"Failed to exchange service account for ID token: {e}"
      ) from e

  def _exchange_for_access_token(
      self, sa_config: ServiceAccount
  ) -> AuthCredential:
    """Exchanges the service account credential for an access token.

    Args:
        sa_config: The service account configuration.

    Returns:
        An AuthCredential in HTTPBearer format containing the access token.

    Raises:
        AuthCredentialMissingError: If scopes are missing for explicit
            credentials or token exchange fails.
    """
    if not sa_config.use_default_credential and not sa_config.scopes:
      raise AuthCredentialMissingError(
          "scopes are required when using explicit service account credentials"
          " for access token exchange."
      )

    try:
      if sa_config.use_default_credential:
        scopes = (
            sa_config.scopes
            if sa_config.scopes
            else ["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials, project_id = google.auth.default(
            scopes=scopes,
        )
        quota_project_id = credentials.quota_project_id or project_id
      else:
        # Guaranteed non-None by ServiceAccount model_validator.
        assert sa_config.service_account_credential is not None
        credentials = service_account.Credentials.from_service_account_info(
            sa_config.service_account_credential.model_dump(),
            scopes=sa_config.scopes,
        )
        quota_project_id = None

      credentials.refresh(Request())

      return AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=credentials.token),
              additional_headers={
                  "x-goog-user-project": quota_project_id,
              }
              if quota_project_id
              else None,
          ),
      )

    # ValueError is raised by google-auth when service account JSON is
    # missing required fields (e.g. client_email, private_key).
    except (google_auth_exceptions.GoogleAuthError, ValueError) as e:
      raise AuthCredentialMissingError(
          f"Failed to exchange service account token: {e}"
      ) from e
