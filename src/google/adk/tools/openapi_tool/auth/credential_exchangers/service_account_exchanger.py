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
from google.auth.transport.requests import Request
from google.oauth2 import id_token as google_id_token
from google.oauth2 import service_account
import google.oauth2.credentials

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .base_credential_exchanger import AuthCredentialMissingError
from .base_credential_exchanger import BaseAuthCredentialExchanger


class ServiceAccountCredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for Google Service Account.

  Uses the default service credential if `use_default_credential = True`.
  Otherwise, uses the service account credential provided in the auth
  credential.
  """

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the service account auth credential for an access token.

    If auth_credential contains a service account credential, it will be used
    to fetch an access token. Otherwise, the default service credential will be
    used for fetching an access token.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential in HTTPBearer format, containing the access token.
    """
    if (
        auth_credential is None
        or auth_credential.service_account is None
        or (
            auth_credential.service_account.service_account_credential is None
            and not auth_credential.service_account.use_default_credential
        )
    ):
      raise AuthCredentialMissingError(
          "Service account credentials are missing. Please provide them, or set"
          " `use_default_credential = True` to use application default"
          " credential in a hosted service like Cloud Run."
      )

    try:
      config = auth_credential.service_account
      token_kind = getattr(config, "token_kind", "access_token")
      request = Request()

      quota_project_id = None
      token = None

      if token_kind == "id_token":
        audience = getattr(config, "audience", None)
        if config.use_default_credential:
          token = google_id_token.fetch_id_token(request, audience)
        else:
          if config.service_account_credential is None:
            raise ValueError("service_account_credential is required when use_default_credential is False")
          
          id_creds = (
              service_account.IDTokenCredentials.from_service_account_info(
                  config.service_account_credential.model_dump(),
                  target_audience=audience,
              )
          )
          id_creds.refresh(request)
          token = id_creds.token
      else:
        if config.use_default_credential:
          credentials, project_id = google.auth.default(
              scopes=["https://www.googleapis.com/auth/cloud-platform"],
          )
          quota_project_id = (
              getattr(credentials, "quota_project_id", None) or project_id
          )
        else:
          if config.service_account_credential is None:
            raise ValueError("service_account_credential is required when use_default_credential is False")

          credentials = service_account.Credentials.from_service_account_info(
              config.service_account_credential.model_dump(),
              scopes=config.scopes,
          )
          quota_project_id = None

        credentials.refresh(Request())
        token = credentials.token

      updated_credential = AuthCredential(
          auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
          http=HttpAuth(
              scheme="bearer",
              credentials=HttpCredentials(token=token),
              additional_headers={
                  "x-goog-user-project": quota_project_id,
              }
              if quota_project_id
              else None,
          ),
      )
      return updated_credential

    except Exception as e:
      raise AuthCredentialMissingError(
          f"Failed to exchange service account token: {e}"
      ) from e
