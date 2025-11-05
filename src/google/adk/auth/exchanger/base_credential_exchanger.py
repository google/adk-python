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

"""Base credential exchanger interface."""

from __future__ import annotations

import abc
from typing import Optional

from ...utils.feature_decorator import experimental
from ..auth_credential import AuthCredential
from ..auth_schemes import AuthScheme


class CredentialExchangeError(Exception):
  """Base exception for credential exchange errors."""


@experimental
class BaseCredentialExchanger(abc.ABC):
  """Base interface for credential exchangers.

  Credential exchangers are responsible for exchanging credentials from
  one format or scheme to another.
  """

  @abc.abstractmethod
  async def exchange(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Exchange credential if needed.

    Args:
        auth_credential: The credential to exchange.
        auth_scheme: The authentication scheme (optional, some exchangers don't need it).

    Returns:
        The exchanged credential.

    Raises:
        CredentialExchangeError: If credential exchange fails.
    """
    pass
