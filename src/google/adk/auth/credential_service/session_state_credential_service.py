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

from __future__ import annotations

import logging
from typing import Optional

from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredential
from ..auth_tool import AuthConfig
from ..oauth2_credential_util import _credential_without_client_secret
from ..oauth2_credential_util import _with_configured_client_secret
from .base_credential_service import BaseCredentialService

logger = logging.getLogger("google_adk." + __name__)


@experimental
class SessionStateCredentialService(BaseCredentialService):
  """Class for implementation of credential service using session state as the
  store.

  Note: store credential in session may not be secure, use at your own risk.
  Session state is returned verbatim by the session-read endpoints, so whatever
  is kept here is readable by anything that can reach them. The agent's OAuth2
  client secret is stripped before storage and put back from the auth config on
  load; the end user's own access and refresh tokens are what this store exists
  to hold and are kept in clear text.
  """

  @override
  async def load_credential(
      self,
      auth_config: AuthConfig,
      callback_context: CallbackContext,
  ) -> Optional[AuthCredential]:
    """
    Loads the credential by auth config and current callback context from the
    backend credential store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to load the credential.

        callback_context: The context of the current invocation when the tool is
        trying to load the credential.

    Returns:
        Optional[AuthCredential]: the credential saved in the store.

    """
    stored = callback_context.state.get(auth_config.credential_key)
    # A session service that persists state as JSON (DatabaseSessionService,
    # SqliteSessionService) hands back what `save_credential` wrote as a plain
    # dict, not as the model it was written as.
    if isinstance(stored, dict):
      stored = AuthCredential.model_validate(stored)
    elif stored is not None and not isinstance(stored, AuthCredential):
      # Something other than `save_credential` put this here. `AuthHandler`
      # accepts a bare token string under the same key, but this store's
      # contract is a credential or nothing.
      logger.warning(
          "Ignoring %s stored under the credential key: this store only reads"
          " credentials written by save_credential.",
          type(stored).__name__,
      )
      return None

    # `save_credential` kept no client secret, so put the configured one back
    # here rather than in any one caller: a credential without it cannot
    # exchange or refresh.
    return _with_configured_client_secret(
        credential=stored,
        raw_credential=auth_config.raw_auth_credential,
    )

  @override
  async def save_credential(
      self,
      auth_config: AuthConfig,
      callback_context: CallbackContext,
  ) -> None:
    """
    Saves the exchanged_auth_credential in auth config to the backend credential
    store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to save the credential.

        callback_context: The context of the current invocation when the tool is
        trying to save the credential.

    Returns:
        None
    """

    # The client secret belongs to the agent's OAuth2 client, not to the end
    # user whose session this is, and session state is readable by anything
    # that can read the session. `load_credential` re-attaches it from the
    # auth config, so it is not needed here.
    callback_context.state[auth_config.credential_key] = (
        _credential_without_client_secret(auth_config.exchanged_auth_credential)
    )
