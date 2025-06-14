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
from typing import List
from typing import Optional

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlows
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from pydantic import BaseModel
from pydantic import model_validator

from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes
from ...auth.auth_credential import OAuth2Auth
from ...auth.auth_tool import AuthConfig
from ...utils.feature_decorator import experimental
from ..tool_context import ToolContext

BIGQUERY_TOKEN_CACHE_KEY = "bigquery_token_cache"
BIGQUERY_DEFAULT_SCOPE = ["https://www.googleapis.com/auth/bigquery"]


@experimental
class BigQueryCredentialsConfig(BaseModel):
  """Configuration for Google API tools. (Experimental)"""

  # Configure the model to allow arbitrary types like Credentials
  model_config = {"arbitrary_types_allowed": True}

  credentials: Optional[Credentials] = None
  """the existing oauth credentials to use. If set,this credential will be used
  for every end user, end users don't need to be involved in the oauthflow. This
  field is mutually exclusive with client_id, client_secret and scopes.
  Don't set this field unless you are sure this credential has the permission to
  access every end user's data.

  Example usage: when the agent is deployed in Google Cloud environment and
  the service account (used as application default credentials) has access to
  all the required BigQuery resource. Setting this credential to allow user to
  access the BigQuery resource without end users going through oauth flow.

  To get application default credential: `google.auth.default(...)`. See more
  details in https://cloud.google.com/docs/authentication/application-default-credentials.

  When the deployed environment cannot provide a pre-existing credential,
  consider setting below client_id, client_secret and scope for end users to go
  through oauth flow, so that agent can access the user data.
  """
  client_id: Optional[str] = None
  """the oauth client ID to use."""
  client_secret: Optional[str] = None
  """the oauth client secret to use."""
  scopes: Optional[List[str]] = None
  """the scopes to use."""

  @model_validator(mode="after")
  def __post_init__(self) -> BigQueryCredentialsConfig:
    """Validate that either credentials or client ID/secret are provided."""
    if not self.credentials and (not self.client_id or not self.client_secret):
      raise ValueError(
          "Must provide either credentials or client_id and client_secret pair."
      )
    if self.credentials and (
        self.client_id or self.client_secret or self.scopes
    ):
      raise ValueError(
          "Cannot provide both existing credentials and"
          " client_id/client_secret/scopes."
      )

    if self.credentials:
      self.client_id = self.credentials.client_id
      self.client_secret = self.credentials.client_secret
      self.scopes = self.credentials.scopes

    if not self.scopes:
      self.scopes = BIGQUERY_DEFAULT_SCOPE

    return self


class BigQueryCredentialsManager:
  """Manages Google API credentials with automatic refresh and OAuth flow handling.

  This class centralizes credential management so multiple tools can share
  the same authenticated session without duplicating OAuth logic.
  """

  def __init__(self, credentials_config: BigQueryCredentialsConfig):
    """Initialize the credential manager.

    Args:
        credentials_config: Credentials containing client id and client secrete
        or default credentials
    """
    self.credentials_config = credentials_config

  async def get_valid_credentials(
      self, tool_context: ToolContext
  ) -> Optional[Credentials]:
    """Get valid credentials, handling refresh and OAuth flow as needed.

    Args:
        tool_context: The tool context for OAuth flow and state management

    Returns:
        Valid Credentials object, or None if OAuth flow is needed
    """
    # First, try to get credentials from the tool context
    creds_json = tool_context.state.get(BIGQUERY_TOKEN_CACHE_KEY, None)
    creds = (
        Credentials.from_authorized_user_info(
            json.loads(creds_json), self.credentials_config.scopes
        )
        if creds_json
        else None
    )

    # If credentails are empty use the default credential
    if not creds:
      creds = self.credentials_config.credentials

    # Check if we have valid credentials
    if creds and creds.valid:
      return creds

    # Try to refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
      try:
        creds.refresh(Request())
        if creds.valid:
          # Cache the refreshed credentials
          tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] = creds.to_json()
          return creds
      except RefreshError:
        # Refresh failed, need to re-authenticate
        pass

    # Need to perform OAuth flow
    return await self._perform_oauth_flow(tool_context)

  async def _perform_oauth_flow(
      self, tool_context: ToolContext
  ) -> Optional[Credentials]:
    """Perform OAuth flow to get new credentials.

    Args:
        tool_context: The tool context for OAuth flow
        required_scopes: Set of required OAuth scopes

    Returns:
        New Credentials object, or None if flow is in progress
    """

    # Create OAuth configuration
    auth_scheme = OAuth2(
        flows=OAuthFlows(
            authorizationCode=OAuthFlowAuthorizationCode(
                authorizationUrl="https://accounts.google.com/o/oauth2/auth",
                tokenUrl="https://oauth2.googleapis.com/token",
                scopes={
                    scope: f"Access to {scope}"
                    for scope in self.credentials_config.scopes
                },
            )
        )
    )

    auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OAUTH2,
        oauth2=OAuth2Auth(
            client_id=self.credentials_config.client_id,
            client_secret=self.credentials_config.client_secret,
        ),
    )

    # Check if OAuth response is available
    auth_response = tool_context.get_auth_response(
        AuthConfig(auth_scheme=auth_scheme, raw_auth_credential=auth_credential)
    )

    if auth_response:
      # OAuth flow completed, create credentials
      creds = Credentials(
          token=auth_response.oauth2.access_token,
          refresh_token=auth_response.oauth2.refresh_token,
          token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
          client_id=self.credentials_config.client_id,
          client_secret=self.credentials_config.client_secret,
          scopes=list(self.credentials_config.scopes),
      )

      # Cache the new credentials
      tool_context.state[BIGQUERY_TOKEN_CACHE_KEY] = creds.to_json()

      return creds
    else:
      # Request OAuth flow
      tool_context.request_credential(
          AuthConfig(
              auth_scheme=auth_scheme,
              raw_auth_credential=auth_credential,
          )
      )
      return None
