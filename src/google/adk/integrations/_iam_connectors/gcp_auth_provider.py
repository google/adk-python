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

from typing import Optional
from typing import TYPE_CHECKING

from google.adk.auth.base_auth_provider import BaseAuthProvider
from google.adk.features import experimental
from google.adk.features import FeatureName

if TYPE_CHECKING:
  from google.adk.agents.callback_context import CallbackContext
  from google.adk.auth.auth_credential import AuthCredential
  from google.adk.auth.auth_tool import AuthConfig


@experimental(FeatureName.GCP_IAM_CONNECTOR_AUTH)
class GcpAuthProvider(BaseAuthProvider):
  """Manages auth credentials using GCP IAM Connector Credential service."""

  async def get_auth_credential(
      self,
      auth_config: AuthConfig,
      context: Optional[CallbackContext] = None,
  ) -> AuthCredential:
    """Retrieves credentials (Not Implemented)."""
    raise NotImplementedError("GcpAuthProvider is not yet implemented.")
