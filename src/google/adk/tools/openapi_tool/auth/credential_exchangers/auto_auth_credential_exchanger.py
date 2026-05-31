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

from typing import Dict
from typing import Optional
from typing import Type

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_schemes import AuthScheme
from .base_credential_exchanger import BaseAuthCredentialExchanger
from .oauth2_exchanger import OAuth2CredentialExchanger
from .service_account_exchanger import ServiceAccountCredentialExchanger


class AutoAuthCredentialExchanger(BaseAuthCredentialExchanger):
  """Automatically selects the appropriate credential exchanger based on the auth scheme.

  Optionally, an override can be provided to use a specific exchanger for a
  given auth scheme.

  Example (common case):
  ```
  exchanger = AutoAuthCredentialExchanger()
  auth_credential = exchanger.exchange_credential(
      auth_scheme=service_account_scheme,
      auth_credential=service_account_credential,
  )
  # Returns an oauth token in the form of a bearer token.
  ```

  Example (use CustomAuthExchanger for OAuth2):
  ```
  exchanger = AutoAuthCredentialExchanger(
      custom_exchangers={
          AuthScheme.OAUTH2: CustomAuthExchanger,
      }
  )
  ```

  Attributes:
    exchangers: A dictionary mapping auth scheme to credential exchanger class.
  """

  def __init__(
      self,
      custom_exchangers: Optional[
          Dict[str, Type[BaseAuthCredentialExchanger]]
      ] = None,
  ):
    """Initializes the AutoAuthCredentialExchanger.

    Args:
      custom_exchangers: Optional dictionary for adding or overriding auth
        exchangers. The key is the auth scheme, and the value is the credential
        exchanger class.
    """
    self.exchangers = {
        AuthCredentialTypes.OAUTH2: OAuth2CredentialExchanger,
        AuthCredentialTypes.OPEN_ID_CONNECT: OAuth2CredentialExchanger,
        AuthCredentialTypes.SERVICE_ACCOUNT: ServiceAccountCredentialExchanger,
    }

    if custom_exchangers:
      self.exchangers.update(custom_exchangers)

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> Optional[AuthCredential]:
    """Automatically exchanges for the credential uses the appropriate credential exchanger.

    Args:
        auth_scheme (AuthScheme): The security scheme.
        auth_credential (AuthCredential): Optional. The authentication
          credential.

    Returns: (AuthCredential)
        A new AuthCredential object containing the exchanged credential.

    """
    if not auth_credential:
      return None

    exchanger_class = self.exchangers.get(
        auth_credential.auth_type if auth_credential else None
    )

    if not exchanger_class:
      return auth_credential

    exchanger = exchanger_class()
    return exchanger.exchange_credential(auth_scheme, auth_credential)
