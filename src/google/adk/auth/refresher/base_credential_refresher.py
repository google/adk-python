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

"""Base credential refresher interface."""

from __future__ import annotations

import abc
from typing import Optional

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.utils.feature_decorator import experimental


@experimental
class BaseCredentialRefresher(abc.ABC):
  """Base interface for credential refreshers."""

  @abc.abstractmethod
  async def needs_refresh(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> bool:
    """Check if credential needs to be refreshed.

    Args:
        auth_credential: The credential to check.
        auth_scheme: The authentication scheme (optional, some refreshers don't need it).

    Returns:
        True if the credential needs to be refreshed, False otherwise.
    """
    pass

  @abc.abstractmethod
  async def refresh(
      self,
      auth_credential: AuthCredential,
      auth_scheme: Optional[AuthScheme] = None,
  ) -> AuthCredential:
    """Refresh credential if needed.

    Args:
        auth_credential: The credential to refresh.
        auth_scheme: The authentication scheme (optional, some refreshers don't need it).

    Returns:
        The refreshed credential.

    Raises:
        ValueError: If credential refresh fails.
    """
    pass
