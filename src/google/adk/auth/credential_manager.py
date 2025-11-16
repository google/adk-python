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

import logging
from typing import Optional

from fastapi.openapi.models import OAuth2

from ..agents.callback_context import CallbackContext
from ..tools.openapi_tool.auth.credential_exchangers.service_account_exchanger import ServiceAccountCredentialExchanger
from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_schemes import AuthSchemeType
from .auth_schemes import ExtendedOAuth2
from .auth_schemes import OpenIdConnectWithConfig
from .auth_tool import AuthConfig
from .exchanger.base_credential_exchanger import BaseCredentialExchanger
from .exchanger.credential_exchanger_registry import CredentialExchangerRegistry
from .oauth2_discovery import OAuth2DiscoveryManager
from .refresher.credential_refresher_registry import CredentialRefresherRegistry

logger = logging.getLogger("google_adk." + __name__)


@experimental
class CredentialManager:
  """Manages authentication credentials through a structured workflow.

  The CredentialManager orchestrates the complete lifecycle of authentication
  credentials, from initial loading to final preparation for use. It provides
  a centralized interface for handling various credential types and authentication
  schemes while maintaining proper credential hygiene (refresh, exchange, caching).

  This class is only for use by Agent Development Kit.

  Args:
      auth_config: Configuration containing authentication scheme and credentials

  Example:
      ```python
      auth_config = AuthConfig(
          auth_scheme=oauth2_scheme,
          raw_auth_credential=service_account_credential
      )
      manager = CredentialManager(auth_config)

      # Register custom exchanger if needed
      manager.register_credential_exchanger(
          AuthCredentialTypes.CUSTOM_TYPE,
          CustomCredentialExchanger()
      )

      # Register custom refresher if needed
      manager.register_credential_refresher(
          AuthCredentialTypes.CUSTOM_TYPE,
          CustomCredentialRefresher()
      )

      # Load and prepare credential
      credential = await manager.load_auth_credential(callback_context)
      ```
  """

  def __init__(
      self,
      auth_config: AuthConfig,
  ):
    self._auth_config = auth_config
    self._exchanger_registry = CredentialExchangerRegistry()
    self._refresher_registry = CredentialRefresherRegistry()
    self._discovery_manager = OAuth2DiscoveryManager()

    # Register default exchangers and refreshers
    from .exchanger.oauth2_credential_exchanger import OAuth2CredentialExchanger
    from .refresher.oauth2_credential_refresher import OAuth2CredentialRefresher

    oauth2_exchanger = OAuth2CredentialExchanger()
    self._exchanger_registry.register(
        AuthCredentialTypes.OAUTH2, oauth2_exchanger
    )
    self._exchanger_registry.register(
        AuthCredentialTypes.OPEN_ID_CONNECT, oauth2_exchanger
    )

    # TODO: Move ServiceAccountCredentialExchanger to the auth module
    self._exchanger_registry.register(
        AuthCredentialTypes.SERVICE_ACCOUNT,
        ServiceAccountCredentialExchanger(),
    )

    oauth2_refresher = OAuth2CredentialRefresher()
    self._refresher_registry.register(
        AuthCredentialTypes.OAUTH2, oauth2_refresher
    )
    self._refresher_registry.register(
        AuthCredentialTypes.OPEN_ID_CONNECT, oauth2_refresher
    )

  def register_credential_exchanger(
      self,
      credential_type: AuthCredentialTypes,
      exchanger_instance: BaseCredentialExchanger,
  ) -> None:
    """Register a credential exchanger for a credential type.

    Args:
        credential_type: The credential type to register for.
        exchanger_instance: The exchanger instance to register.
    """
    self._exchanger_registry.register(credential_type, exchanger_instance)

  async def request_credential(self, callback_context: CallbackContext) -> None:
    callback_context.request_credential(self._auth_config)

  async def get_auth_credential(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load and prepare authentication credential through a structured workflow."""

    # Step 1: Validate credential configuration
    await self._validate_credential()

    # Step 2: Check if credential is already ready (no processing needed)
    if self._is_credential_ready():
      return self._auth_config.raw_auth_credential

    # Step 3: Try to load existing processed credential
    credential = await self._load_existing_credential(callback_context)

    # Step 4: If no existing credential, load from auth response
    # TODO instead of load from auth response, we can store auth response in
    # credential service.
    was_from_auth_response = False
    if not credential:
      credential = await self._load_from_auth_response(callback_context)
      was_from_auth_response = True

    # Step 5: If still no credential available, check if client credentials
    if not credential:
      # For client credentials flow, use raw credentials directly
      if self._is_client_credentials_flow():
        credential = self._auth_config.raw_auth_credential
      else:
        # For authorization code flow, return None to trigger user authorization
        return None

    # Step 6: Exchange credential if needed (e.g., service account to access token)
    credential, was_exchanged = await self._exchange_credential(credential)

    # Step 7: Refresh credential if expired
    was_refreshed = False
    if not was_exchanged:
      credential, was_refreshed = await self._refresh_credential(credential)

    # Step 8: Save credential if it was modified
    if was_from_auth_response or was_exchanged or was_refreshed:
      await self._save_credential(callback_context, credential)

    return credential

  async def _load_existing_credential(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load existing credential from credential service or cached exchanged credential."""

    # Try loading from credential service first
    credential = await self._load_from_credential_service(callback_context)
    if credential:
      return credential

    # Check if we have a cached exchanged credential
    if self._auth_config.exchanged_auth_credential:
      return self._auth_config.exchanged_auth_credential

    return None

  async def _load_from_credential_service(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load credential from credential service if available."""
    credential_service = callback_context._invocation_context.credential_service
    if credential_service:
      # Note: This should be made async in a future refactor
      # For now, assuming synchronous operation
      return await callback_context.load_credential(self._auth_config)
    return None

  async def _load_from_auth_response(
      self, callback_context: CallbackContext
  ) -> Optional[AuthCredential]:
    """Load credential from auth response in callback context."""
    return callback_context.get_auth_response(self._auth_config)

  async def _exchange_credential(
      self, credential: AuthCredential
  ) -> tuple[AuthCredential, bool]:
    """Exchange credential if needed and return the credential and whether it was exchanged."""
    exchanger = self._exchanger_registry.get_exchanger(credential.auth_type)
    if not exchanger:
      return credential, False

    if isinstance(exchanger, ServiceAccountCredentialExchanger):
      exchanged_credential = exchanger.exchange_credential(
          self._auth_config.auth_scheme, credential
      )
    else:
      exchanged_credential = await exchanger.exchange(
          credential, self._auth_config.auth_scheme
      )

    return exchanged_credential, True

  async def _refresh_credential(
      self, credential: AuthCredential
  ) -> tuple[AuthCredential, bool]:
    """Refresh credential if expired and return the credential and whether it was refreshed."""
    refresher = self._refresher_registry.get_refresher(credential.auth_type)
    if not refresher:
      return credential, False

    if await refresher.is_refresh_needed(
        credential, self._auth_config.auth_scheme
    ):
      refreshed_credential = await refresher.refresh(
          credential, self._auth_config.auth_scheme
      )
      return refreshed_credential, True

    return credential, False

  def _is_credential_ready(self) -> bool:
    """Check if credential is ready to use without further processing."""
    raw_credential = self._auth_config.raw_auth_credential
    if not raw_credential:
      return False

    # Simple credentials that don't need exchange or refresh
    return raw_credential.auth_type in (
        AuthCredentialTypes.API_KEY,
        AuthCredentialTypes.HTTP,
        # Add other simple auth types as needed
    )

  async def _validate_credential(self) -> None:
    """Validate credential configuration and raise errors if invalid."""
    if not self._auth_config.raw_auth_credential:
      if self._auth_config.auth_scheme.type_ in (
          AuthSchemeType.oauth2,
          AuthSchemeType.openIdConnect,
      ):
        raise ValueError(
            "raw_auth_credential is required for auth_scheme type "
            f"{self._auth_config.auth_scheme.type_}"
        )

    raw_credential = self._auth_config.raw_auth_credential
    if raw_credential:
      if (
          raw_credential.auth_type
          in (
              AuthCredentialTypes.OAUTH2,
              AuthCredentialTypes.OPEN_ID_CONNECT,
          )
          and not raw_credential.oauth2
      ):
        raise ValueError(
            "auth_config.raw_credential.oauth2 required for credential type "
            f"{raw_credential.auth_type}"
        )

    if self._missing_oauth_info() and not await self._populate_auth_scheme():
      raise ValueError(
          "OAuth scheme info is missing, and auto-discovery has failed to fill"
          " them in."
      )

    # Additional validation can be added here

  async def _save_credential(
      self, callback_context: CallbackContext, credential: AuthCredential
  ) -> None:
    """Save credential to credential service if available."""
    # Update the exchanged credential in config
    self._auth_config.exchanged_auth_credential = credential

    credential_service = callback_context._invocation_context.credential_service
    if credential_service:
      await callback_context.save_credential(self._auth_config)

  async def _populate_auth_scheme(self) -> bool:
    """Auto-discover server metadata and populate missing auth scheme info.

    Returns:
      True if auto-discovery was successful, False otherwise.
    """
    auth_scheme = self._auth_config.auth_scheme
    if (
        not isinstance(auth_scheme, ExtendedOAuth2)
        or not auth_scheme.issuer_url
    ):
      logger.warning("No issuer_url was provided for auto-discovery.")
      return False

    metadata = await self._discovery_manager.discover_auth_server_metadata(
        auth_scheme.issuer_url
    )
    if not metadata:
      logger.warning("Auto-discovery has failed to populate OAuth scheme info.")
      return False

    flows = auth_scheme.flows

    if flows.implicit and not flows.implicit.authorizationUrl:
      flows.implicit.authorizationUrl = metadata.authorization_endpoint
    if flows.password and not flows.password.tokenUrl:
      flows.password.tokenUrl = metadata.token_endpoint
    if flows.clientCredentials and not flows.clientCredentials.tokenUrl:
      flows.clientCredentials.tokenUrl = metadata.token_endpoint
    if flows.authorizationCode and not flows.authorizationCode.authorizationUrl:
      flows.authorizationCode.authorizationUrl = metadata.authorization_endpoint
    if flows.authorizationCode and not flows.authorizationCode.tokenUrl:
      flows.authorizationCode.tokenUrl = metadata.token_endpoint
    return True

  def _missing_oauth_info(self) -> bool:
    """Checks if we are missing auth/token URLs needed for OAuth."""
    auth_scheme = self._auth_config.auth_scheme
    if isinstance(auth_scheme, OAuth2):
      flows = auth_scheme.flows
      return (
          flows.implicit
          and not flows.implicit.authorizationUrl
          or flows.password
          and not flows.password.tokenUrl
          or flows.clientCredentials
          and not flows.clientCredentials.tokenUrl
          or flows.authorizationCode
          and not flows.authorizationCode.authorizationUrl
          or flows.authorizationCode
          and not flows.authorizationCode.tokenUrl
      )
    return False

  def _is_client_credentials_flow(self) -> bool:
    """Check if the auth scheme uses client credentials flow.

    Supports both OAuth2 and OIDC schemes.

    Returns:
      True if using client credentials flow, False otherwise.
    """
    auth_scheme = self._auth_config.auth_scheme

    # Check OAuth2 schemes
    if isinstance(auth_scheme, OAuth2) and auth_scheme.flows:
      return auth_scheme.flows.clientCredentials is not None

    # Check OIDC schemes
    if isinstance(auth_scheme, OpenIdConnectWithConfig):
      return (
          auth_scheme.grant_types_supported is not None
          and "client_credentials" in auth_scheme.grant_types_supported
      )

    return False
