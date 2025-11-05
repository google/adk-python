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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import alias_generators
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class BaseModelWithConfig(BaseModel):
  model_config = ConfigDict(
      extra="allow",
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
  )
  """The pydantic model config."""


class HttpCredentials(BaseModelWithConfig):
  """Represents the secret token value for HTTP authentication, like user name, password, oauth token, etc."""

  username: Optional[str] = None
  password: Optional[str] = None
  token: Optional[str] = None

  @classmethod
  def model_validate(cls, data: Dict[str, Any]) -> "HttpCredentials":
    return cls(
        username=data.get("username"),
        password=data.get("password"),
        token=data.get("token"),
    )


class HttpAuth(BaseModelWithConfig):
  """The credentials and metadata for HTTP authentication."""

  # The name of the HTTP Authorization scheme to be used in the Authorization
  # header as defined in RFC7235. The values used SHOULD be registered in the
  # IANA Authentication Scheme registry.
  # Examples: 'basic', 'bearer'
  scheme: str
  credentials: HttpCredentials


class OAuth2Auth(BaseModelWithConfig):
  """Represents credential value and its metadata for a OAuth2 credential."""

  client_id: Optional[str] = None
  client_secret: Optional[str] = None
  # tool or adk can generate the auth_uri with the state info thus client
  # can verify the state
  auth_uri: Optional[str] = None
  state: Optional[str] = None
  # tool or adk can decide the redirect_uri if they don't want client to decide
  redirect_uri: Optional[str] = None
  auth_response_uri: Optional[str] = None
  auth_code: Optional[str] = None
  access_token: Optional[str] = None
  refresh_token: Optional[str] = None
  expires_at: Optional[int] = None
  expires_in: Optional[int] = None
  audience: Optional[str] = None


class ServiceAccountCredential(BaseModelWithConfig):
  """Represents Google Service Account configuration.

  Attributes:
    type: The type should be "service_account".
    project_id: The project ID.
    private_key_id: The ID of the private key.
    private_key: The private key.
    client_email: The client email.
    client_id: The client ID.
    auth_uri: The authorization URI.
    token_uri: The token URI.
    auth_provider_x509_cert_url: URL for auth provider's X.509 cert.
    client_x509_cert_url: URL for the client's X.509 cert.
    universe_domain: The universe domain.

  Example:

      config = ServiceAccountCredential(
          type_="service_account",
          project_id="your_project_id",
          private_key_id="your_private_key_id",
          private_key="-----BEGIN PRIVATE KEY-----...",
          client_email="...@....iam.gserviceaccount.com",
          client_id="your_client_id",
          auth_uri="https://accounts.google.com/o/oauth2/auth",
          token_uri="https://oauth2.googleapis.com/token",
          auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
          client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/...",
          universe_domain="googleapis.com"
      )


      config = ServiceAccountConfig.model_construct(**{
          ...service account config dict
      })
  """

  type_: str = Field("", alias="type")
  project_id: str
  private_key_id: str
  private_key: str
  client_email: str
  client_id: str
  auth_uri: str
  token_uri: str
  auth_provider_x509_cert_url: str
  client_x509_cert_url: str
  universe_domain: str


class ServiceAccount(BaseModelWithConfig):
  """Represents Google Service Account configuration."""

  service_account_credential: Optional[ServiceAccountCredential] = None
  scopes: List[str]
  use_default_credential: Optional[bool] = False


class AuthCredentialTypes(str, Enum):
  """Represents the type of authentication credential."""

  # API Key credential:
  # https://swagger.io/docs/specification/v3_0/authentication/api-keys/
  API_KEY = "apiKey"

  # Credentials for HTTP Auth schemes:
  # https://www.iana.org/assignments/http-authschemes/http-authschemes.xhtml
  HTTP = "http"

  # OAuth2 credentials:
  # https://swagger.io/docs/specification/v3_0/authentication/oauth2/
  OAUTH2 = "oauth2"

  # OpenID Connect credentials:
  # https://swagger.io/docs/specification/v3_0/authentication/openid-connect-discovery/
  OPEN_ID_CONNECT = "openIdConnect"

  # Service Account credentials:
  # https://cloud.google.com/iam/docs/service-account-creds
  SERVICE_ACCOUNT = "serviceAccount"


class AuthCredential(BaseModelWithConfig):
  """Data class representing an authentication credential.

  To exchange for the actual credential, please use
  CredentialExchanger.exchange_credential().

  Examples: API Key Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      api_key="1234",
  )

  Example: HTTP Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(username="user", password="password"),
      ),
  )

  Example: OAuth2 Bearer Token in HTTP Header
  AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer",
          credentials=HttpCredentials(token="eyAkaknabna...."),
      ),
  )

  Example: OAuth2 Auth with Authorization Code Flow
  AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="1234",
          client_secret="secret",
      ),
  )

  Example: OpenID Connect Auth
  AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id="1234",
          client_secret="secret",
          redirect_uri="https://example.com",
          scopes=["scope1", "scope2"],
      ),
  )

  Example: Auth with resource reference
  AuthCredential(
      auth_type=AuthCredentialTypes.API_KEY,
      resource_ref="projects/1234/locations/us-central1/resources/resource1",
  )
  """

  auth_type: AuthCredentialTypes
  # Resource reference for the credential.
  # This will be supported in the future.
  resource_ref: Optional[str] = None

  api_key: Optional[str] = None
  http: Optional[HttpAuth] = None
  service_account: Optional[ServiceAccount] = None
  oauth2: Optional[OAuth2Auth] = None
