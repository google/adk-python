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

from typing import Optional

from ..tools.tool_context import ToolContext
from ..utils.feature_decorator import experimental
from .auth_credential import AuthCredential
from .auth_credential import AuthCredentialTypes
from .auth_schemes import AuthSchemeType
from .auth_tool import AuthConfig
from .exchanger.base_credential_exchanger import BaseCredentialExchanger
from .exchanger.credential_exchanger_registry import CredentialExchangerRegistry
from .refresher.base_credential_refresher import BaseCredentialRefresher
from .refresher.credential_refresher_registry import CredentialRefresherRegistry


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
      credential = await manager.load_auth_credential(tool_context)
      ```
  """

  def __init__(
      self,
      auth_config: AuthConfig,
  ):
    self._auth_config = auth_config
    self._exchanger_registry = CredentialExchangerRegistry()
    self._refresher_registry = CredentialRefresherRegistry()

    # Register default exchangers and refreshers
    # TODO: support service account credential exchanger
    from .refresher.oauth2_credential_refresher import OAuth2CredentialRefresher

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

  async def request_credential(self, tool_context: ToolContext) -> None:
    tool_context.request_credential(self._auth_config)

  async def get_auth_credential(
      self, tool_context: ToolContext
  ) -> Optional[AuthCredential]:
    """Load and prepare authentication credential through a structured workflow."""

    # Step 1: Validate credential configuration
    await self._validate_credential()

    # Step 2: Check if credential is already ready (no processing needed)
    if self._is_credential_ready():
      return self._auth_config.raw_auth_credential

    # Step 3: Try to load existing processed credential
    credential = await self._load_existing_credential(tool_context)

    # Step 4: If no existing credential, load from auth response
    # TODO instead of load from auth response, we can store auth response in
    # credential service.
    was_from_auth_response = False
    if not credential:
      credential = await self._load_from_auth_response(tool_context)
      was_from_auth_response = True

    # Step 5: If still no credential available, return None
    if not credential:
      return None

    # Step 6: Exchange credential if needed (e.g., service account to access token)
    credential, was_exchanged = await self._exchange_credential(credential)

    # Step 7: Refresh credential if expired
    if not was_exchanged:
      credential, was_refreshed = await self._refresh_credential(credential)

    # Step 8: Save credential if it was modified
    if was_from_auth_response or was_exchanged or was_refreshed:
      await self._save_credential(tool_context, credential)

    return credential

  async def _load_existing_credential(
      self, tool_context: ToolContext
  ) -> Optional[AuthCredential]:
    """Load existing credential from credential service or cached exchanged credential."""

    # Try loading from credential service first
    credential = await self._load_from_credential_service(tool_context)
    if credential:
      return credential

    # Check if we have a cached exchanged credential
    if self._auth_config.exchanged_auth_credential:
      return self._auth_config.exchanged_auth_credential

    return None

  async def _load_from_credential_service(
      self, tool_context: ToolContext
  ) -> Optional[AuthCredential]:
    """Load credential from credential service if available."""
    credential_service = tool_context._invocation_context.credential_service
    if credential_service:
      # Note: This should be made async in a future refactor
      # For now, assuming synchronous operation
      return await credential_service.load_credential(
          self._auth_config, tool_context
      )
    return None

  async def _load_from_auth_response(
      self, tool_context: ToolContext
  ) -> Optional[AuthCredential]:
    """Load credential from auth response in tool context."""
    return tool_context.get_auth_response(self._auth_config)

  async def _exchange_credential(
      self, credential: AuthCredential
  ) -> tuple[AuthCredential, bool]:
    """Exchange credential if needed and return the credential and whether it was exchanged."""
    exchanger = self._exchanger_registry.get_exchanger(credential.auth_type)
    if not exchanger:
      return credential, False

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
        # Additional validation can be added here

  async def _save_credential(
      self, tool_context: ToolContext, credential: AuthCredential
  ) -> None:
    """Save credential to credential service if available."""
    credential_service = tool_context._invocation_context.credential_service
    if credential_service:
      # Update the exchanged credential in config
      self._auth_config.exchanged_auth_credential = credential
      await credential_service.save_credential(self._auth_config, tool_context)
