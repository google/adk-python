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

"""Auth provider registry."""

from __future__ import annotations

from ..features import experimental
from ..features import FeatureName
from .auth_schemes import AuthScheme
from .base_auth_provider import BaseAuthProvider


@experimental(FeatureName.PLUGGABLE_AUTH)
class AuthProviderRegistry:
  """Registry for auth provider instances."""

  def __init__(self):
    self._providers: dict[type[AuthScheme], BaseAuthProvider] = {}

  def register(
      self,
      auth_scheme_type: type[AuthScheme],
      provider_instance: BaseAuthProvider,
  ) -> None:
    """Register a provider instance for an auth scheme type.

    Args:
        auth_scheme_type: The auth scheme type to register for.
        provider_instance: The provider instance to register.
    """
    self._providers[auth_scheme_type] = provider_instance

  def get_provider(
      self, auth_scheme: AuthScheme | type[AuthScheme]
  ) -> BaseAuthProvider | None:
    """Get the provider instance for an auth scheme.

    Args:
        auth_scheme: The auth scheme or the auth scheme type to get the provider
            for.

    Returns:
        The provider instance if registered, None otherwise.
    """
    if isinstance(auth_scheme, type):
      return self._providers.get(auth_scheme)
    return self._providers.get(type(auth_scheme))
