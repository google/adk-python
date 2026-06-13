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

"""Authentication provider using Google Cloud Agent Identity Credentials service."""

from __future__ import annotations

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_tool import AuthConfig
from google.adk.auth.base_auth_provider import BaseAuthProvider
from typing_extensions import override

from ._agent_identity_credentials_provider import _AgentIdentityCredentialsProvider
from ._iam_connector_credentials_provider import _IamConnectorCredentialsProvider
from .gcp_auth_provider_scheme import GcpAuthProviderScheme


class GcpAuthProvider(BaseAuthProvider):
  """An auth provider that uses Credentials service to generate access tokens."""

  def __init__(self):
    self._iam_connector_provider = _IamConnectorCredentialsProvider()
    self._agent_identity_provider = _AgentIdentityCredentialsProvider()

  @property
  @override
  def supported_auth_schemes(self) -> tuple[type[GcpAuthProviderScheme], ...]:
    return (GcpAuthProviderScheme,)

  @override
  async def get_auth_credential(
      self,
      auth_config: AuthConfig,
      context: CallbackContext | None = None,
  ) -> AuthCredential:
    """Retrieves credentials using the Credentials service.

    Args:
      auth_config: The authentication configuration.
      context: Optional context for the callback.

    Returns:
      An AuthCredential instance.

    Raises:
      ValueError: If auth_scheme is not a GcpAuthProviderScheme.
    """
    auth_scheme = auth_config.auth_scheme
    if not isinstance(auth_scheme, GcpAuthProviderScheme):
      raise ValueError(
          f"Expected GcpAuthProviderScheme, got {type(auth_scheme)}"
      )

    if re.match(
        r"^projects/[^/]+/locations/[^/]+/connectors/[^/]+$", auth_scheme.name
    ):
      return await self._iam_connector_provider.get_auth_credential(
          auth_scheme=auth_scheme, context=context
      )

    return await self._agent_identity_provider.get_auth_credential(
        auth_scheme=auth_scheme, context=context
    )
