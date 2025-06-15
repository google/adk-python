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

from ...tools.tool_context import ToolContext
from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredential
from ..auth_tool import AuthConfig
from .base_credential_service import BaseCredentialService


@experimental
class InMemoryCredentialService(BaseCredentialService):
  """Class for in memory implementation of credential service(Experimental)"""

  def __init__(self):
    super().__init__()
    self._store: dict[str, AuthCredential] = {}

  @override
  async def load_credential(
      self,
      auth_config: AuthConfig,
      tool_context: ToolContext,
  ) -> Optional[AuthCredential]:
    """
    Loads the credential by auth config and current tool context from the
    backend credential store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to load the credential.

        tool_context: The context of the current invocation when the tool is
        trying to load the credential.

    Returns:
        Optional[AuthCredential]: the credential saved in the store.

    """
    storage = self._get_storage_for_current_context(tool_context)
    return storage.get(auth_config.credential_key)

  @override
  async def save_credential(
      self,
      auth_config: AuthConfig,
      tool_context: ToolContext,
  ) -> None:
    """
    Saves the exchanged_auth_credential in auth config to the backend credential
    store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to save the credential.

        tool_context: The context of the current invocation when the tool is
        trying to save the credential.

    Returns:
        None
    """
    storage = self._get_storage_for_current_context(tool_context)
    storage[auth_config.credential_key] = auth_config.exchanged_auth_credential

  def _get_storage_for_current_context(self, tool_context: ToolContext) -> str:
    app_name = tool_context._invocation_context.app_name
    user_id = tool_context._invocation_context.user_id

    if app_name not in self._store:
      self._store[app_name] = {}
    if user_id not in self._store[app_name]:
      self._store[app_name][user_id] = {}
    return self._store[app_name][user_id]
