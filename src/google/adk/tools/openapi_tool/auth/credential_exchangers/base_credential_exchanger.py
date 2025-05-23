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

import abc
from typing import Optional

from .....auth.auth_credential import AuthCredential
from .....auth.auth_schemes import AuthScheme


class AuthCredentialMissingError(Exception):
  """Exception raised when required authentication credentials are missing."""

  def __init__(self, message: str):
    super().__init__(message)
    self.message = message


class BaseAuthCredentialExchanger:
  """Base class for authentication credential exchangers."""

  @abc.abstractmethod
  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the provided authentication credential for a usable token/credential.

    Args:
        auth_scheme: The security scheme.
        auth_credential: The authentication credential.

    Returns:
        An updated AuthCredential object containing the fetched credential.
        For simple schemes like API key, it may return the original credential
        if no exchange is needed.

    Raises:
        NotImplementedError: If the method is not implemented by a subclass.
    """
    raise NotImplementedError("Subclasses must implement exchange_credential.")
