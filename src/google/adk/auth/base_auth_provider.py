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

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .auth_schemes import AuthScheme

from ..agents.callback_context import CallbackContext
from ..features import experimental
from ..features import FeatureName
from .auth_credential import AuthCredential
from .auth_tool import AuthConfig


@experimental(FeatureName.PLUGGABLE_AUTH)
class BaseAuthProvider(ABC):
  """Abstract base class for custom authentication providers."""

  @property
  def supported_auth_schemes(self) -> tuple[type[AuthScheme], ...]:
    """The AuthScheme types supported by this provider.

    Subclasses can override this to return a tuple of scheme types, enabling
    1-parameter registration.
    """
    return ()

  @abstractmethod
  async def get_auth_credential(
      self, auth_config: AuthConfig, context: CallbackContext
  ) -> AuthCredential | None:
    """Provide an AuthCredential asynchronously.

    Args:
       auth_config: The current authentication configuration.
       context: The current callback context.

    Returns:
       The retrieved AuthCredential, or None if unavailable.
    """
