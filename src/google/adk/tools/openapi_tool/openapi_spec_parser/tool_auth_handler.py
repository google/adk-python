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


import logging
from typing import Literal
from typing import Optional

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_credential import AuthCredentialTypes
from ....auth.auth_schemes import AuthScheme
from ....auth.auth_schemes import AuthSchemeType
from ....auth.auth_tool import AuthConfig
from ...tool_context import ToolContext
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from ..auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger

logger = logging.getLogger("google_adk." + __name__)

AuthPreparationState = Literal["pending", "done"]


class AuthPreparationResult(BaseModel):
  """Result of the credential preparation process."""

  state: AuthPreparationState
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None


class ToolContextCredentialStore:
  """Handles storage and retrieval of credentials within a ToolContext."""

  def __init__(self, tool_context: ToolContext):
    self.tool_context = tool_context

  def get_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    """Generates a unique key for the given auth scheme and credential."""
    scheme_name = (
        f"{auth_scheme.type_.name}_{hash(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{hash(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )
    # no need to prepend temp: namespace, session state is a copy, changes to
    # it won't be persisted , only changes in event_action.state_delta will be
    # persisted. temp: namespace will be cleared after current run. but tool
    # want access token to be there stored across runs

    return f"{scheme_name}_{credential_name}_existing_exchanged_credential"

  def get_credential(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> Optional[AuthCredential]:
    if not self.tool_context:
      return None

    token_key = self.get_credential_key(auth_scheme, auth_credential)
    # TODO try not to use session state, this looks a hacky way, depend on
    # session implementation, we don't want session to persist the token,
    # meanwhile we want the token shared across runs.
    serialized_credential = self.tool_context.state.get(token_key)
    if not serialized_credential:
      return None
    return AuthCredential.model_validate(serialized_credential)

  def store_credential(
      self,
      key: str,
      auth_credential: Optional[AuthCredential],
  ):
    if self.tool_context:
      serializable_credential = jsonable_encoder(
          auth_credential, exclude_none=True
      )
      self.tool_context.state[key] = serializable_credential

  def remove_credential(self, key: str):
    del self.tool_context.state[key]


class ToolAuthHandler:
  """Handles the preparation and exchange of authentication credentials for tools."""

  def __init__(
      self,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      credential_store: Optional["ToolContextCredentialStore"] = None,
  ):
    self.tool_context = tool_context
    self.auth_scheme = (
        auth_scheme.model_copy(deep=True) if auth_scheme else None
    )
    self.auth_credential = (
        auth_credential.model_copy(deep=True) if auth_credential else None
    )
    self.credential_exchanger = (
        credential_exchanger or AutoAuthCredentialExchanger()
    )
    self.credential_store = credential_store
    self.should_store_credential = True

  @classmethod
  def from_tool_context(
      cls,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
  ) -> "ToolAuthHandler":
    """Creates a ToolAuthHandler instance from a ToolContext."""
    credential_store = ToolContextCredentialStore(tool_context)
    return cls(
        tool_context,
        auth_scheme,
        auth_credential,
        credential_exchanger,
        credential_store,
    )

  def _handle_existing_credential(
      self,
  ) -> Optional[AuthPreparationResult]:
    """Checks for and returns an existing, exchanged credential."""
    if self.credential_store:
      existing_credential = self.credential_store.get_credential(
          self.auth_scheme, self.auth_credential
      )
      if existing_credential:
        return AuthPreparationResult(
            state="done",
            auth_scheme=self.auth_scheme,
            auth_credential=existing_credential,
        )
    return None

  def _exchange_credential(
      self, auth_credential: AuthCredential
  ) -> Optional[AuthPreparationResult]:
    """Handles an OpenID Connect authorization response."""

    exchanged_credential = None
    try:
      exchanged_credential = self.credential_exchanger.exchange_credential(
          self.auth_scheme, auth_credential
      )
    except Exception as e:
      logger.error("Failed to exchange credential: %s", e)
    return exchanged_credential

  def _store_credential(self, auth_credential: AuthCredential) -> None:
    """stores the auth_credential."""

    if self.credential_store:
      key = self.credential_store.get_credential_key(
          self.auth_scheme, self.auth_credential
      )
      self.credential_store.store_credential(key, auth_credential)

  def _reqeust_credential(self) -> None:
    """Handles the case where an OpenID Connect or OAuth2 authentication request is needed."""
    if self.auth_scheme.type_ in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      if not self.auth_credential or not self.auth_credential.oauth2:
        raise ValueError(
            f"auth_credential is empty for scheme {self.auth_scheme.type_}."
            "Please create AuthCredential using OAuth2Auth."
        )

      if not self.auth_credential.oauth2.client_id:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_id is missing."
        )

      if not self.auth_credential.oauth2.client_secret:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_secret is missing."
        )

    self.tool_context.request_credential(
        AuthConfig(
            auth_scheme=self.auth_scheme,
            raw_auth_credential=self.auth_credential,
        )
    )
    return None

  def _get_auth_response(self) -> AuthCredential:
    return self.tool_context.get_auth_response(
        AuthConfig(
            auth_scheme=self.auth_scheme,
            raw_auth_credential=self.auth_credential,
        )
    )

  def _request_credential(self, auth_config: AuthConfig):
    if not self.tool_context:
      return
    self.tool_context.request_credential(auth_config)

  def prepare_auth_credentials(
      self,
  ) -> AuthPreparationResult:
    """Prepares authentication credentials, handling exchange and user interaction."""

    # no auth is needed
    if not self.auth_scheme:
      return AuthPreparationResult(state="done")

    # Check for existing credential.
    existing_result = self._handle_existing_credential()
    if existing_result:
      return existing_result

    # fetch credential from adk framework
    # Some auth scheme like OAuth2 AuthCode & OpenIDConnect may require
    # multi-step exchange:
    # client_id , client_secret -> auth_uri -> auth_code -> access_token
    # -> bearer token
    # adk framework supports exchange access_token already
    fetched_credential = self._get_auth_response() or self.auth_credential

    exchanged_credential = self._exchange_credential(fetched_credential)

    if exchanged_credential:
      self._store_credential(exchanged_credential)
      return AuthPreparationResult(
          state="done",
          auth_scheme=self.auth_scheme,
          auth_credential=exchanged_credential,
      )
    else:
      self._reqeust_credential()
      return AuthPreparationResult(
          state="pending",
          auth_scheme=self.auth_scheme,
          auth_credential=self.auth_credential,
      )
