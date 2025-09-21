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

"""Credential refresher registry."""

from __future__ import annotations

from typing import Dict
from typing import Optional

from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.utils.feature_decorator import experimental

from .base_credential_refresher import BaseCredentialRefresher


@experimental
class CredentialRefresherRegistry:
  """Registry for credential refresher instances."""

  def __init__(self):
    self._refreshers: Dict[AuthCredentialTypes, BaseCredentialRefresher] = {}

  def register(
      self,
      credential_type: AuthCredentialTypes,
      refresher_instance: BaseCredentialRefresher,
  ) -> None:
    """Register a refresher instance for a credential type.

    Args:
        credential_type: The credential type to register for.
        refresher_instance: The refresher instance to register.
    """
    self._refreshers[credential_type] = refresher_instance

  def get_refresher(
      self, credential_type: AuthCredentialTypes
  ) -> Optional[BaseCredentialRefresher]:
    """Get the refresher instance for a credential type.

    Args:
        credential_type: The credential type to get refresher for.

    Returns:
        The refresher instance if registered, None otherwise.
    """
    return self._refreshers.get(credential_type)
