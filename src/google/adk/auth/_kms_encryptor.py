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

import base64
import logging
from typing import Dict
from typing import Tuple

from cryptography.fernet import Fernet

logger = logging.getLogger("google_adk." + __name__)

# Cache mapping KMS key name -> tuple of (plaintext_dek: bytes, wrapped_dek: str)
_KMS_KEY_DEK_CACHE: Dict[str, Tuple[bytes, str]] = {}

# Cache mapping wrapped_dek (str) -> Fernet instance
_DEK_FERNET_CACHE: Dict[str, Fernet] = {}

# KMS Client cache
_KMS_CLIENT_CACHE: Dict[str, any] = {}


def _get_kms_client(kms_key_name: str):
  """Gets or creates a cached Google Cloud KMS client."""
  if kms_key_name not in _KMS_CLIENT_CACHE:
    from google.cloud import kms

    _KMS_CLIENT_CACHE[kms_key_name] = kms.KeyManagementServiceClient()
  return _KMS_CLIENT_CACHE[kms_key_name]


def _get_or_create_dek(kms_key_name: str) -> Tuple[bytes, str]:
  """Gets the cached DEK for a KMS key, or generates and wraps a new one."""
  if kms_key_name not in _KMS_KEY_DEK_CACHE:
    try:
      # Generate a new 32-byte Fernet key
      plaintext_dek = Fernet.generate_key()

      # Wrap (encrypt) the DEK using Cloud KMS
      client = _get_kms_client(kms_key_name)
      response = client.encrypt(
          request={
              "name": kms_key_name,
              "plaintext": plaintext_dek,
          }
      )
      wrapped_dek = base64.b64encode(response.ciphertext).decode("utf-8")
      _KMS_KEY_DEK_CACHE[kms_key_name] = (plaintext_dek, wrapped_dek)
      # Also populate the Fernet cache for this wrapped DEK
      _DEK_FERNET_CACHE[wrapped_dek] = Fernet(plaintext_dek)
    except Exception as e:
      logger.error(
          "Failed to generate and wrap DEK using KMS key %s: %s",
          kms_key_name,
          e,
      )
      raise e

  return _KMS_KEY_DEK_CACHE[kms_key_name]


def _get_crypto_key_name(kms_key_name: str) -> str:
  """Returns the CryptoKey resource name by stripping any version suffix if present."""
  if "/cryptoKeyVersions/" in kms_key_name:
    return kms_key_name.split("/cryptoKeyVersions/")[0]
  return kms_key_name


def _get_fernet_for_wrapped_dek(kms_key_name: str, wrapped_dek: str) -> Fernet:
  """Gets the cached Fernet instance for a wrapped DEK, unwrapping it with KMS if needed."""
  if wrapped_dek not in _DEK_FERNET_CACHE:
    try:
      # Unwrap (decrypt) the DEK using Cloud KMS (decrypt requires CryptoKey name, not version)
      client = _get_kms_client(kms_key_name)
      ciphertext_bytes = base64.b64decode(wrapped_dek.encode("utf-8"))
      crypto_key_name = _get_crypto_key_name(kms_key_name)
      response = client.decrypt(
          request={
              "name": crypto_key_name,
              "ciphertext": ciphertext_bytes,
          }
      )
      plaintext_dek = response.plaintext
      _DEK_FERNET_CACHE[wrapped_dek] = Fernet(plaintext_dek)
    except Exception as e:
      logger.error("Failed to unwrap DEK using KMS key %s: %s", kms_key_name, e)
      raise e

  return _DEK_FERNET_CACHE[wrapped_dek]


def encrypt_credentials(
    kms_key_name: str,
    token: str | None,
    refresh_token: str | None,
    client_secret: str | None,
) -> Tuple[str | None, str | None, str | None, str | None]:
  """Encrypts the sensitive credential fields using envelope encryption.

  Returns a tuple of (encrypted_token, encrypted_refresh_token, encrypted_client_secret, wrapped_dek).
  """
  try:
    plaintext_dek, wrapped_dek = _get_or_create_dek(kms_key_name)
    fernet = _DEK_FERNET_CACHE[wrapped_dek]

    enc_token = (
        fernet.encrypt(token.encode("utf-8")).decode("utf-8") if token else None
    )
    enc_refresh = (
        fernet.encrypt(refresh_token.encode("utf-8")).decode("utf-8")
        if refresh_token
        else None
    )
    enc_secret = (
        fernet.encrypt(client_secret.encode("utf-8")).decode("utf-8")
        if client_secret
        else None
    )

    return enc_token, enc_refresh, enc_secret, wrapped_dek
  except Exception as e:
    logger.error("Failed to encrypt credentials: %s", e)
    raise e


def decrypt_credentials(
    kms_key_name: str,
    encrypted_token: str | None,
    encrypted_refresh_token: str | None,
    encrypted_client_secret: str | None,
    wrapped_dek: str | None,
) -> Tuple[str | None, str | None, str | None]:
  """Decrypts the sensitive credential fields using the wrapped DEK."""
  if not wrapped_dek:
    # Backward compatibility
    return encrypted_token, encrypted_refresh_token, encrypted_client_secret

  try:
    fernet = _get_fernet_for_wrapped_dek(kms_key_name, wrapped_dek)

    dec_token = (
        fernet.decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
        if encrypted_token
        else None
    )
    dec_refresh = (
        fernet.decrypt(encrypted_refresh_token.encode("utf-8")).decode("utf-8")
        if encrypted_refresh_token
        else None
    )
    dec_secret = (
        fernet.decrypt(encrypted_client_secret.encode("utf-8")).decode("utf-8")
        if encrypted_client_secret
        else None
    )

    return dec_token, dec_refresh, dec_secret
  except Exception as e:
    logger.error("Failed to decrypt credentials: %s", e)
    raise e


def encrypt_value(kms_key_name: str, plaintext: str) -> str:
  """Fallback/Direct encryption helper."""
  try:
    client = _get_kms_client(kms_key_name)
    response = client.encrypt(
        request={
            "name": kms_key_name,
            "plaintext": plaintext.encode("utf-8"),
        }
    )
    return base64.b64encode(response.ciphertext).decode("utf-8")
  except Exception as e:
    logger.error("Failed to encrypt value: %s", e)
    raise e


def decrypt_value(kms_key_name: str, ciphertext: str) -> str:
  """Fallback/Direct decryption helper."""
  try:
    client = _get_kms_client(kms_key_name)
    ciphertext_bytes = base64.b64decode(ciphertext.encode("utf-8"))
    crypto_key_name = _get_crypto_key_name(kms_key_name)
    response = client.decrypt(
        request={
            "name": crypto_key_name,
            "ciphertext": ciphertext_bytes,
        }
    )
    return response.plaintext.decode("utf-8")
  except Exception as e:
    logger.error("Failed to decrypt value: %s", e)
    raise e
