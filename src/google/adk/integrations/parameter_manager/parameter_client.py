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

from __future__ import annotations

import json
from typing import cast
from typing import Optional

import google.auth
from google.auth import default as default_service_credential
import google.auth.transport.requests
from google.cloud import parametermanager_v1
from google.oauth2 import service_account


class ParameterManagerClient:
  """A client for interacting with Google Cloud Parameter Manager.

  This class provides a simplified interface for retrieving parameters from
  Parameter Manager, handling authentication using either a service account
  JSON keyfile (passed as a string), a preexisting authorization token, or
  default credentials.

  Attributes:
      _credentials:  Google Cloud credentials object (ServiceAccountCredentials
        or Credentials).
      _client: Parameter Manager client instance.
  """

  def __init__(
      self,
      service_account_json: Optional[str] = None,
      auth_token: Optional[str] = None,
      location: Optional[str] = None,
  ):
    """Initializes the ParameterManagerClient.

    If neither `service_account_json` nor `auth_token` is provided, default
    credentials are used.

    Args:
        service_account_json: The content of a service account JSON keyfile (as
          a string), not the file path. Must be valid JSON.
        auth_token: An existing Google Cloud authorization token.
        location: The Google Cloud location (region) to use for the Parameter
          Manager service. If not provided, the global endpoint is used.

    Raises:
        ValueError: If both 'service_account_json' and 'auth_token' are
        provided. Also raised if the 'service_account_json' is not valid JSON.
        google.auth.exceptions.GoogleAuthError: If authentication fails.
    """
    if service_account_json and auth_token:
      raise ValueError(
          "Must provide either 'service_account_json' or 'auth_token', not"
          " both."
      )

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
            " error occurred while trying to use default credentials: {e}"
        ) from e

    if not credentials:
      raise ValueError(
          "Failed to obtain credentials. Provide either 'service_account_json'"
          " or 'auth_token', not both. If neither is provided, default"
          " credentials are used."
      )

    self._credentials = credentials

    client_options = None
    if location:
      client_options = {
          "api_endpoint": f"parametermanager.{location}.rep.googleapis.com"
      }

    self._client = parametermanager_v1.ParameterManagerClient(
        credentials=self._credentials, client_options=client_options
    )

  def get_parameter(self, resource_name: str) -> str:
    """Retrieves a rendered parameter value from Google Cloud Parameter Manager.

    Args:
        resource_name: The full resource name of the parameter version, in the
          format "projects/*/locations/*/parameters/*/versions/*". Usually you
          want the "latest" version, e.g.,
          "projects/my-project/locations/global/parameters/my-param/versions/latest".

    Returns:
        The rendered parameter value as a string.

    Raises:
        google.api_core.exceptions.GoogleAPIError: If the Parameter Manager API
            returns an error (e.g., parameter not found, permission denied).
    """
    request = parametermanager_v1.RenderParameterVersionRequest(
        name=resource_name
    )
    response = self._client.render_parameter_version(request=request)
    return cast(str, response.rendered_payload.decode("UTF-8"))
