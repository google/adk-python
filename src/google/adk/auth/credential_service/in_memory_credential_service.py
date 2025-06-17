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
    self._credentials = {}

  @override
  async def load_credential(
      self,
      auth_config: AuthConfig,
      tool_context: ToolContext,
  ) -> Optional[AuthCredential]:
    credential_bucket = self._get_bucket_for_current_context(tool_context)
    return credential_bucket.get(auth_config.credential_key)

  @override
  async def save_credential(
      self,
      auth_config: AuthConfig,
      tool_context: ToolContext,
  ) -> None:
    credential_bucket = self._get_bucket_for_current_context(tool_context)
    credential_bucket[auth_config.credential_key] = (
        auth_config.exchanged_auth_credential
    )

  def _get_bucket_for_current_context(self, tool_context: ToolContext) -> str:
    app_name = tool_context._invocation_context.app_name
    user_id = tool_context._invocation_context.user_id

    if app_name not in self._credentials:
      self._credentials[app_name] = {}
    if user_id not in self._credentials[app_name]:
      self._credentials[app_name][user_id] = {}
    return self._credentials[app_name][user_id]
