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

from __future__ import annotations

import json
from typing import Optional

import google.auth
from google.auth import default as default_service_credential
import google.auth.transport.requests
from google.cloud import secretmanager
from google.oauth2 import service_account


class SecretManagerClient:
  """A client for interacting with Google Cloud Secret Manager.

  This class provides a simplified interface for retrieving secrets from
  Secret Manager, handling authentication using either a service account
  JSON keyfile (passed as a string) or a pre-existing authorization token.

  Attributes:
      _credentials:  Google Cloud credentials object (ServiceAccountCredentials
        or Credentials).
      _client: Secret Manager client instance.
  """

  def __init__(
      self,
      service_account_json: Optional[str] = None,
      auth_token: Optional[str] = None,
  ):
    """Initializes the SecretManagerClient.

    Args:
        service_account_json:  The content of a service account JSON keyfile (as
          a string), not the file path.  Must be valid JSON.
        auth_token: An existing Google Cloud authorization token.

    Raises:
        ValueError: If neither `service_account_json` nor `auth_token` is
        provided,
            or if both are provided.  Also raised if the service_account_json
            is not valid JSON.
        google.auth.exceptions.GoogleAuthError: If authentication fails.
    """
    if service_account_json:
      try:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(service_account_json)
        )
      except json.JSONDecodeError as e:
        raise ValueError(f"Invalid service account JSON: {e}") from e
    elif auth_token:
      credentials = google.auth.credentials.Credentials(
          token=auth_token,
          refresh_token=None,
          token_uri=None,
          client_id=None,
          client_secret=None,
      )
      request = google.auth.transport.requests.Request()
      credentials.refresh(request)
    else:
      try:
        credentials, _ = default_service_credential(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
      except Exception as e:
        raise ValueError(
            "'service_account_json' or 'auth_token' are both missing, and"
            f" error occurred while trying to use default credentials: {e}"
        ) from e

    if not credentials:
      raise ValueError(
          "Must provide either 'service_account_json' or 'auth_token', not both"
          " or neither."
      )

    self._credentials = credentials
    self._client = secretmanager.SecretManagerServiceClient(
        credentials=self._credentials
    )

  def get_secret(self, resource_name: str) -> str:
    """Retrieves a secret from Google Cloud Secret Manager.

    Args:
        resource_name: The full resource name of the secret, in the format
          "projects/*/secrets/*/versions/*".  Usually you want the "latest"
          version, e.g.,
          "projects/my-project/secrets/my-secret/versions/latest".

    Returns:
        The secret payload as a string.

    Raises:
        google.api_core.exceptions.GoogleAPIError: If the Secret Manager API
            returns an error (e.g., secret not found, permission denied).
        Exception: For other unexpected errors.
    """
    try:
      response = self._client.access_secret_version(name=resource_name)
      return response.payload.data.decode("UTF-8")
    except Exception as e:
      raise e  # Re-raise the exception to allow for handling by the caller
      # Consider logging the exception here before re-raising.
