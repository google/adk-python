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

import logging
import os
from typing import Any
from typing import Optional

from typing_extensions import override

from ...agents.callback_context import CallbackContext
from ...utils.feature_decorator import experimental
from .._kms_encryptor import decrypt_value
from .._kms_encryptor import encrypt_value
from ..auth_credential import AuthCredential
from ..auth_tool import AuthConfig
from .base_credential_service import BaseCredentialService

logger = logging.getLogger("google_adk." + __name__)


def _encrypt_auth_credential(
    kms_key: str, cred: AuthCredential
) -> dict[str, Any]:
  data = cred.model_dump(by_alias=True)
  data["kmsKeyName"] = kms_key

  try:
    if cred.api_key and not cred.api_key.startswith("kms:"):
      data["apiKey"] = "kms:" + encrypt_value(kms_key, cred.api_key)

    if cred.http and cred.http.credentials:
      if (
          cred.http.credentials.password
          and not cred.http.credentials.password.startswith("kms:")
      ):
        data["http"]["credentials"]["password"] = "kms:" + encrypt_value(
            kms_key, cred.http.credentials.password
        )
      if (
          cred.http.credentials.token
          and not cred.http.credentials.token.startswith("kms:")
      ):
        data["http"]["credentials"]["token"] = "kms:" + encrypt_value(
            kms_key, cred.http.credentials.token
        )

    if cred.oauth2:
      if cred.oauth2.access_token and not cred.oauth2.access_token.startswith(
          "kms:"
      ):
        data["oauth2"]["accessToken"] = "kms:" + encrypt_value(
            kms_key, cred.oauth2.access_token
        )
      if (
          cred.oauth2.refresh_token
          and not cred.oauth2.refresh_token.startswith("kms:")
      ):
        data["oauth2"]["refreshToken"] = "kms:" + encrypt_value(
            kms_key, cred.oauth2.refresh_token
        )
      if (
          cred.oauth2.client_secret
          and not cred.oauth2.client_secret.startswith("kms:")
      ):
        data["oauth2"]["clientSecret"] = "kms:" + encrypt_value(
            kms_key, cred.oauth2.client_secret
        )

    if cred.service_account and cred.service_account.service_account_credential:
      pk = cred.service_account.service_account_credential.private_key
      if pk and not pk.startswith("kms:"):
        sa_dict = data.get("serviceAccount") or data.get("service_account")
        if sa_dict:
          sac_key = (
              "serviceAccountCredential"
              if "serviceAccountCredential" in sa_dict
              else "service_account_credential"
          )
          sac_dict = sa_dict.get(sac_key)
          if sac_dict:
            sac_dict["privateKey"] = "kms:" + encrypt_value(kms_key, pk)
  except Exception as e:
    logger.error("Failed to encrypt AuthCredential with KMS: %s", e)
    raise e

  return data


def _decrypt_auth_credential(
    val: Any, default_kms_key: str | None
) -> AuthCredential | None:
  if isinstance(val, dict):
    cred_dict = dict(val)
  elif isinstance(val, AuthCredential):
    cred_dict = val.model_dump(by_alias=True)
  else:
    return None

  kms_key = (
      cred_dict.get("kms_key_name")
      or cred_dict.get("kmsKeyName")
      or default_kms_key
  )

  try:
    for key in ("api_key", "apiKey"):
      if cred_dict.get(key) and str(cred_dict[key]).startswith("kms:"):
        if not kms_key:
          logger.warning(
              "Encrypted api_key found but no KMS key provided. Falling back to"
              " re-auth."
          )
          return None
        cred_dict[key] = decrypt_value(kms_key, str(cred_dict[key])[4:])

    if "http" in cred_dict and isinstance(cred_dict["http"], dict):
      http_creds = cred_dict["http"].get("credentials")
      if http_creds and isinstance(http_creds, dict):
        for pwd_key in ("password",):
          if http_creds.get(pwd_key) and str(http_creds[pwd_key]).startswith(
              "kms:"
          ):
            if not kms_key:
              logger.warning(
                  "Encrypted password found but no KMS key provided. Falling"
                  " back to re-auth."
              )
              return None
            http_creds[pwd_key] = decrypt_value(
                kms_key, str(http_creds[pwd_key])[4:]
            )
        for tok_key in ("token",):
          if http_creds.get(tok_key) and str(http_creds[tok_key]).startswith(
              "kms:"
          ):
            if not kms_key:
              logger.warning(
                  "Encrypted token found but no KMS key provided. Falling back"
                  " to re-auth."
              )
              return None
            http_creds[tok_key] = decrypt_value(
                kms_key, str(http_creds[tok_key])[4:]
            )

    if "oauth2" in cred_dict and isinstance(cred_dict["oauth2"], dict):
      oa = cred_dict["oauth2"]
      for secret_key in ("client_secret", "clientSecret"):
        if oa.get(secret_key) and str(oa[secret_key]).startswith("kms:"):
          if not kms_key:
            return None
          oa[secret_key] = decrypt_value(kms_key, str(oa[secret_key])[4:])
      for token_key in ("access_token", "accessToken"):
        if oa.get(token_key) and str(oa[token_key]).startswith("kms:"):
          if not kms_key:
            return None
          oa[token_key] = decrypt_value(kms_key, str(oa[token_key])[4:])
      for refresh_key in ("refresh_token", "refreshToken"):
        if oa.get(refresh_key) and str(oa[refresh_key]).startswith("kms:"):
          if not kms_key:
            return None
          oa[refresh_key] = decrypt_value(kms_key, str(oa[refresh_key])[4:])

    sa_dict = cred_dict.get("serviceAccount") or cred_dict.get(
        "service_account"
    )
    if sa_dict and isinstance(sa_dict, dict):
      sac_dict = sa_dict.get("serviceAccountCredential") or sa_dict.get(
          "service_account_credential"
      )
      if sac_dict and isinstance(sac_dict, dict):
        for pk_key in ("private_key", "privateKey"):
          if sac_dict.get(pk_key) and str(sac_dict[pk_key]).startswith("kms:"):
            if not kms_key:
              return None
            sac_dict[pk_key] = decrypt_value(kms_key, str(sac_dict[pk_key])[4:])

    return AuthCredential.model_validate(cred_dict)
  except Exception as e:
    logger.warning(
        "Failed to decrypt AuthCredential from session state: %s. Falling back"
        " to re-authentication.",
        e,
    )
    return None


@experimental
class SessionStateCredentialService(BaseCredentialService):
  """Class for implementation of credential service using session state as the
  store.
  Note: store credential in session may not be secure, use at your own risk.
  """

  @override
  async def load_credential(
      self,
      auth_config: AuthConfig,
      callback_context: CallbackContext,
  ) -> Optional[AuthCredential]:
    """
    Loads the credential by auth config and current callback context from the
    backend credential store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to load the credential.

        callback_context: The context of the current invocation when the tool is
        trying to load the credential.

    Returns:
        Optional[AuthCredential]: the credential saved in the store.

    """
    val = callback_context.state.get(auth_config.credential_key)
    if not val:
      return None

    kms_key = getattr(auth_config, "kms_key_name", None) or os.environ.get(
        "GOOGLE_CREDENTIAL_KMS_KEY"
    )
    return _decrypt_auth_credential(val, kms_key)

  @override
  async def save_credential(
      self,
      auth_config: AuthConfig,
      callback_context: CallbackContext,
  ) -> None:
    """
    Saves the exchanged_auth_credential in auth config to the backend credential
    store.

    Args:
        auth_config: The auth config which contains the auth scheme and auth
        credential information. auth_config.get_credential_key will be used to
        build the key to save the credential.

        callback_context: The context of the current invocation when the tool is
        trying to save the credential.

    Returns:
        None
    """
    cred = auth_config.exchanged_auth_credential
    if not cred:
      return

    kms_key = (
        getattr(auth_config, "kms_key_name", None)
        or (cred.kms_key_name if hasattr(cred, "kms_key_name") else None)
        or os.environ.get("GOOGLE_CREDENTIAL_KMS_KEY")
    )

    if kms_key:
      callback_context.state[auth_config.credential_key] = (
          _encrypt_auth_credential(kms_key, cred)
      )
    else:
      callback_context.state[auth_config.credential_key] = cred
