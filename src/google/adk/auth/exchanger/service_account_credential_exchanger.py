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

from typing import Optional

import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from typing_extensions import override

from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredential
from ..auth_credential import AuthCredentialTypes
from ..auth_schemes import AuthScheme
from .base_credential_exchanger import BaseCredentialExchanger


@experimental
class ServiceAccountCredentialExchanger(BaseCredentialExchanger):
  """Exchanges Google Service Account credentials for an access token.

  Uses the default service credential if `use_default_credential = True`.
  Otherwise, uses the service account credential provided in the auth
  credential.
  """

  @override
  async def exchange(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Exchanges the service account auth credential for an access token.

    If the AuthCredential contains a service account credential, it will be used
    to exchange for an access token. Otherwise, if use_default_credential is True,
    the default application credential will be used for exchanging an access token.

    Args:
        auth_scheme: The authentication scheme.
        auth_credential: The credential to exchange.

    Returns:
        An AuthCredential in OAUTH2 format, containing the exchanged credential JSON.

    Raises:
        ValueError: If service account credentials are missing or invalid.
        Exception: If credential exchange or refresh fails.
    """
    if auth_credential is None:
      raise ValueError("Credential cannot be None.")

    if auth_credential.auth_type != AuthCredentialTypes.SERVICE_ACCOUNT:
      raise ValueError("Credential is not a service account credential.")

    if auth_credential.service_account is None:
      raise ValueError(
          "Service account credentials are missing. Please provide them."
      )

    if (
        auth_credential.service_account.service_account_credential is None
        and not auth_credential.service_account.use_default_credential
    ):
      raise ValueError(
          "Service account credentials are invalid. Please set the"
          " service_account_credential field or set `use_default_credential ="
          " True` to use application default credential in a hosted service"
          " like Google Cloud Run."
      )

    try:
      if auth_credential.service_account.use_default_credential:
        credentials, _ = google.auth.default()
      else:
        config = auth_credential.service_account
        credentials = service_account.Credentials.from_service_account_info(
            config.service_account_credential.model_dump(), scopes=config.scopes
        )

      # Refresh credentials to ensure we have a valid access token
      credentials.refresh(Request())

      return AuthCredential(
          auth_type=AuthCredentialTypes.OAUTH2,
          google_oauth2_json=credentials.to_json(),
      )
    except Exception as e:
      raise ValueError(f"Failed to exchange service account token: {e}") from e
