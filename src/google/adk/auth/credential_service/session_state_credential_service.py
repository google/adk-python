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

from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredential
from ..auth_tool import AuthConfig
from .base_credential_service import BaseCredentialService


@experimental
class SessionStateCredentialService(BaseCredentialService):
  """Class for implementation of credential service using session state as the
  store.
  Note: store credential in session may not be secure, use at your own risk.
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
    return callback_context.state.get(auth_config.credential_key)

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

    callback_context.state[auth_config.credential_key] = (
        auth_config.exchanged_auth_credential
    )
