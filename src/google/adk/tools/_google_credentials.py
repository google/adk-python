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
import google.auth.credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
import google.oauth2.credentials
from pydantic import BaseModel
from pydantic import model_validator

from ..auth.auth_credential import AuthCredential
from ..auth.auth_credential import AuthCredentialTypes
from ..auth.auth_credential import OAuth2Auth
from ..auth.auth_tool import AuthConfig
from ..utils.feature_decorator import experimental
from .tool_context import ToolContext


@experimental
class BaseGoogleCredentialsConfig(BaseModel):
  """Base Google Credentials Configuration for Google API tools (Experimental).

  Please do not use this in production, as it may be deprecated later.
  """

  # Configure the model to allow arbitrary types like Credentials
  model_config = {"arbitrary_types_allowed": True}

  credentials: Optional[google.auth.credentials.Credentials] = None
  """The existing auth credentials to use. If set, this credential will be used
  for every end user, end users don't need to be involved in the oauthflow. This
  field is mutually exclusive with client_id, client_secret and scopes.
  Don't set this field unless you are sure this credential has the permission to
  access every end user's data.

  Example usage 1: When the agent is deployed in Google Cloud environment and
  the service account (used as application default credentials) has access to
  all the required Google Cloud resource. Setting this credential to allow user
  to access the Google Cloud resource without end users going through oauth
  flow.

  To get application default credential, use: `google.auth.default(...)`. See
  more details in
  https://cloud.google.com/docs/authentication/application-default-credentials.

  Example usage 2: When the agent wants to access the user's Google Cloud
  resources using the service account key credentials.

  To load service account key credentials, use:
  `google.auth.load_credentials_from_file(...)`. See more details in
  https://cloud.google.com/iam/docs/service-account-creds#user-managed-keys.

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

  _token_cache_key: Optional[str] = None
  """The key to cache the token in the tool context."""

  @model_validator(mode="after")
  def __post_init__(self) -> BaseGoogleCredentialsConfig:
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

    if self.credentials and isinstance(
        self.credentials, google.oauth2.credentials.Credentials
    ):
      self.client_id = self.credentials.client_id
      self.client_secret = self.credentials.client_secret
      self.scopes = self.credentials.scopes

    return self


class GoogleCredentialsManager:
  """Manages Google API credentials with automatic refresh and OAuth flow handling.

  This class centralizes credential management so multiple tools can share
  the same authenticated session without duplicating OAuth logic.
  """

  def __init__(
      self,
      credentials_config: BaseGoogleCredentialsConfig,
  ):
    """Initialize the credential manager.

    Args:
        credentials_config: Credentials containing client id and client secrete
          or default credentials
    """
    self.credentials_config = credentials_config

  async def get_valid_credentials(
      self, tool_context: ToolContext
  ) -> Optional[google.auth.credentials.Credentials]:
    """Get valid credentials, handling refresh and OAuth flow as needed.

    Args:
        tool_context: The tool context for OAuth flow and state management

    Returns:
        Valid Credentials object, or None if OAuth flow is needed
    """
    # First, try to get credentials from the tool context
    creds_json = (
        tool_context.state.get(self.credentials_config._token_cache_key, None)
        if self.credentials_config._token_cache_key
        else None
    )
    creds = (
        google.oauth2.credentials.Credentials.from_authorized_user_info(
            json.loads(creds_json), self.credentials_config.scopes
        )
        if creds_json
        else None
    )

    # If credentails are empty use the default credential
    if not creds:
      creds = self.credentials_config.credentials

    # If non-oauth credentials are provided then use them as is. This helps
    # in flows such as service account keys
    if creds and not isinstance(creds, google.oauth2.credentials.Credentials):
      return creds

    # Check if we have valid credentials
    if creds and creds.valid:
      return creds

    # Try to refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
      try:
        creds.refresh(Request())
        if creds.valid:
          # Cache the refreshed credentials if token cache key is set
          if self.credentials_config._token_cache_key:
            tool_context.state[self.credentials_config._token_cache_key] = (
                creds.to_json()
            )
          return creds
      except RefreshError:
        # Refresh failed, need to re-authenticate
        pass

    # Need to perform OAuth flow
    return await self._perform_oauth_flow(tool_context)

  async def _perform_oauth_flow(
      self, tool_context: ToolContext
  ) -> Optional[google.oauth2.credentials.Credentials]:
    """Perform OAuth flow to get new credentials.

    Args:
        tool_context: The tool context for OAuth flow

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
      creds = google.oauth2.credentials.Credentials(
          token=auth_response.oauth2.access_token,
          refresh_token=auth_response.oauth2.refresh_token,
          token_uri=auth_scheme.flows.authorizationCode.tokenUrl,
          client_id=self.credentials_config.client_id,
          client_secret=self.credentials_config.client_secret,
          scopes=list(self.credentials_config.scopes),
      )

      # Cache the new credentials if token cache key is set
      if self.credentials_config._token_cache_key:
        tool_context.state[self.credentials_config._token_cache_key] = (
            creds.to_json()
        )

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
