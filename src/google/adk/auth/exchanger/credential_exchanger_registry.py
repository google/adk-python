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

"""Credential exchanger registry."""

from __future__ import annotations

from typing import Dict
from typing import Optional

from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredentialTypes
from .base_credential_exchanger import BaseCredentialExchanger


@experimental
class CredentialExchangerRegistry:
  """Registry for credential exchanger instances."""

  def __init__(self):
    self._exchangers: Dict[AuthCredentialTypes, BaseCredentialExchanger] = {}

  def register(
      self,
      credential_type: AuthCredentialTypes,
      exchanger_instance: BaseCredentialExchanger,
  ) -> None:
    """Register an exchanger instance for a credential type.

    Args:
        credential_type: The credential type to register for.
        exchanger_instance: The exchanger instance to register.
    """
    self._exchangers[credential_type] = exchanger_instance

  def get_exchanger(
      self, credential_type: AuthCredentialTypes
  ) -> Optional[BaseCredentialExchanger]:
    """Get the exchanger instance for a credential type.

    Args:
        credential_type: The credential type to get exchanger for.

    Returns:
        The exchanger instance if registered, None otherwise.
    """
    return self._exchangers.get(credential_type)
