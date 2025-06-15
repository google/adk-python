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

"""Credential fetcher for Google Service Account."""

from __future__ import annotations

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_credential import HttpAuth
from .auth_credential import HttpCredentials


@experimental
class ServiceAccountCredentialExchanger:
  """Exchanges Google Service Account credentials for an access token.

  Uses the default service credential if `use_default_credential = True`.
  Otherwise, uses the service account credential provided in the auth
  credential.
  """

  def __init__(self, credential: AuthCredential):
    if credential.auth_type != AuthCredentialTypes.SERVICE_ACCOUNT:
      raise ValueError("Credential is not a service account credential.")
    self._credential = credential

  def exchange(self) -> AuthCredential:
    """Exchanges the service account auth credential for an access token.

    If the AuthCredential contains a service account credential, it will be used
    to exchange for an access token. Otherwise, if use_default_credential is True,
    the default application credential will be used for exchanging an access token.

    Returns:
        An AuthCredential in HTTP Bearer format, containing the access token.

    Raises:
        ValueError: If service account credentials are missing or invalid.
        Exception: If credential exchange or refresh fails.
    """
    if (
        self._credential is None
        or self._credential.service_account is None
        or (
            self._credential.service_account.service_account_credential is None
            and not self._credential.service_account.use_default_credential
        )
    ):
      raise ValueError(
          "Service account credentials are missing. Please provide them, or set"
          " `use_default_credential = True` to use application default"
          " credential in a hosted service like Google Cloud Run."
      )

    try:
      if self._credential.service_account.use_default_credential:
        credentials, _ = google.auth.default()
      else:
        config = self._credential.service_account
        credentials = service_account.Credentials.from_service_account_info(
            config.service_account_credential.model_dump(), scopes=config.scopes
        )

      # Refresh credentials to ensure we have a valid access token
      credentials.refresh(Request())

      return AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=credentials.token),
          ),
      )
    except Exception as e:
      raise ValueError(f"Failed to exchange service account token: {e}") from e
