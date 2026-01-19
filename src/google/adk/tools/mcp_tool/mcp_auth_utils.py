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

"""Utility functions for MCP tool authentication."""

from __future__ import annotations

import base64
import logging
from typing import Dict
from typing import Optional

from fastapi.openapi import models as openapi_models
from fastapi.openapi.models import APIKey
from fastapi.openapi.models import HTTPBase

from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme

logger = logging.getLogger("google_adk." + __name__)


def get_mcp_auth_headers(
    auth_scheme: Optional[AuthScheme], credential: Optional[AuthCredential]
) -> Optional[Dict[str, str]]:
  """Generates HTTP authentication headers for MCP calls.

  Args:
      auth_scheme: The authentication scheme.
      credential: The resolved authentication credential.

  Returns:
      A dictionary of headers, or None if no auth is applicable.

  Raises:
      ValueError: If the auth scheme is unsupported or misconfigured.
  """
  if not credential:
    return None

  headers: Optional[Dict[str, str]] = None

  if credential.oauth2:
    headers = {"Authorization": f"Bearer {credential.oauth2.access_token}"}
  elif credential.http:
    if not auth_scheme or not isinstance(auth_scheme, HTTPBase):
      logger.warning(
          "HTTP credential provided, but auth_scheme is missing or not"
          " HTTPBase."
      )
      return None

    scheme = auth_scheme.scheme.lower()
    if scheme == "bearer" and credential.http.credentials.token:
      headers = {"Authorization": f"Bearer {credential.http.credentials.token}"}
    elif scheme == "basic":
      if (
          credential.http.credentials.username
          and credential.http.credentials.password
      ):
        creds = f"{credential.http.credentials.username}:{credential.http.credentials.password}"
        encoded_creds = base64.b64encode(creds.encode()).decode()
        headers = {"Authorization": f"Basic {encoded_creds}"}
      else:
        logger.warning("Basic auth scheme missing username or password.")
    elif credential.http.credentials.token:
      # Handle other HTTP schemes like Digest, etc. if token is present
      headers = {
          "Authorization": (
              f"{auth_scheme.scheme} {credential.http.credentials.token}"
          )
      }
    else:
      logger.warning(f"Unsupported or incomplete HTTP auth scheme '{scheme}'.")
  elif credential.api_key:
    if not auth_scheme or not isinstance(auth_scheme, APIKey):
      logger.warning(
          "API key credential provided, but auth_scheme is missing or not"
          " APIKey."
      )
      return None

    if auth_scheme.in_ != openapi_models.APIKeyIn.header:
      error_msg = (
          "MCP tools only support header-based API key authentication. "
          f"Configured location: {auth_scheme.in_}"
      )
      logger.error(error_msg)
      raise ValueError(error_msg)
    headers = {auth_scheme.name: credential.api_key}
  elif credential.service_account:
    logger.warning(
        "Service account credentials should be exchanged for an access token "
        "before calling get_mcp_auth_headers."
    )
  else:
    logger.warning(f"Unsupported credential type: {type(credential)}")

  return headers
