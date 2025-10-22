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

from enum import Enum
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlows
from fastapi.openapi.models import SecurityBase
from fastapi.openapi.models import SecurityScheme
from fastapi.openapi.models import SecuritySchemeType
from pydantic import Field

from ..utils.feature_decorator import experimental


class OpenIdConnectWithConfig(SecurityBase):
  type_: SecuritySchemeType = Field(
      default=SecuritySchemeType.openIdConnect, alias="type"
  )
  authorization_endpoint: str
  token_endpoint: str
  userinfo_endpoint: Optional[str] = None
  revocation_endpoint: Optional[str] = None
  token_endpoint_auth_methods_supported: Optional[List[str]] = None
  grant_types_supported: Optional[List[str]] = None
  scopes: Optional[List[str]] = None


# AuthSchemes contains SecuritySchemes from OpenAPI 3.0 and an extra flattened OpenIdConnectWithConfig.
AuthScheme = Union[SecurityScheme, OpenIdConnectWithConfig]


class OAuthGrantType(str, Enum):
  """Represents the OAuth2 flow (or grant type)."""

  CLIENT_CREDENTIALS = "client_credentials"
  AUTHORIZATION_CODE = "authorization_code"
  IMPLICIT = "implicit"
  PASSWORD = "password"

  @staticmethod
  def from_flow(flow: OAuthFlows) -> "OAuthGrantType":
    """Converts an OAuthFlows object to a OAuthGrantType."""
    if flow.clientCredentials:
      return OAuthGrantType.CLIENT_CREDENTIALS
    if flow.authorizationCode:
      return OAuthGrantType.AUTHORIZATION_CODE
    if flow.implicit:
      return OAuthGrantType.IMPLICIT
    if flow.password:
      return OAuthGrantType.PASSWORD
    return None


# AuthSchemeType re-exports SecuritySchemeType from OpenAPI 3.0.
AuthSchemeType = SecuritySchemeType


@experimental
class ExtendedOAuth2(OAuth2):
  """OAuth2 scheme that incorporates auto-discovery for endpoints."""

  issuer_url: Optional[str] = None  # Used for endpoint-discovery
