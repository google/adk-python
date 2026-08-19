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

"""Conversion of a resolved `AuthCredential` into HTTP headers."""

from __future__ import annotations

import base64
import logging

from fastapi.openapi.models import APIKeyIn

from .auth_credential import AuthCredential
from .auth_schemes import AuthScheme

logger = logging.getLogger("google_adk." + __name__)


def build_auth_headers(
    credential: AuthCredential | None,
    auth_scheme: AuthScheme | None = None,
) -> dict[str, str] | None:
  """Builds the HTTP headers that carry an exchanged credential.

  Args:
    credential: The resolved credential. May be None.
    auth_scheme: The scheme the credential was resolved for. Only used to name
      the header for API key credentials.

  Returns:
    The headers to add to the outgoing request, or None if the credential
    cannot be expressed as headers.
  """
  if not credential:
    return None

  headers: dict[str, str] | None = None

  if credential.oauth2:
    headers = {"Authorization": f"Bearer {credential.oauth2.access_token}"}
  elif credential.http:
    # Handle HTTP authentication schemes
    if (
        credential.http.scheme.lower() == "bearer"
        and credential.http.credentials
        and credential.http.credentials.token
    ):
      headers = {"Authorization": f"Bearer {credential.http.credentials.token}"}
    elif credential.http.scheme.lower() == "basic":
      # Handle basic auth
      if (
          credential.http.credentials
          and credential.http.credentials.username
          and credential.http.credentials.password
      ):
        credentials_str = (
            f"{credential.http.credentials.username}"
            f":{credential.http.credentials.password}"
        )
        encoded_credentials = base64.b64encode(
            credentials_str.encode()
        ).decode()
        headers = {"Authorization": f"Basic {encoded_credentials}"}
    elif credential.http.credentials and credential.http.credentials.token:
      # Handle other HTTP schemes with token
      headers = {
          "Authorization": (
              f"{credential.http.scheme} {credential.http.credentials.token}"
          )
      }

    if credential.http.additional_headers:
      headers = headers or {}
      headers.update(credential.http.additional_headers)
  elif credential.api_key:
    # For API key, use the auth scheme to determine header name.
    # `AuthScheme` is a union and only `APIKey` declares `name`, so both reads
    # below are unchecked. That is the behaviour this function was extracted
    # from: an API key paired with a scheme that has no `name` raised before
    # and still raises. The ignores keep the move type-check neutral rather
    # than quietly turning that into a silent no-op.
    if auth_scheme:
      if hasattr(auth_scheme, "in_"):
        if auth_scheme.in_ == APIKeyIn.header:
          headers = {auth_scheme.name: credential.api_key}  # type: ignore[union-attr]
        else:
          logger.warning(
              "Only header-based API key authentication is supported."
              " Configured location: %s",
              auth_scheme.in_,
          )
      else:
        # Default to using scheme name as header
        headers = {auth_scheme.name: credential.api_key}  # type: ignore[union-attr]

  return headers
