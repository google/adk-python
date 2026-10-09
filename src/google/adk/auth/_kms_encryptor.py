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
from collections.abc import Mapping
from collections.abc import Sequence
import datetime
import json
import threading
from typing import Any
from typing import cast

from cryptography.fernet import Fernet
import google.oauth2.credentials

_MAX_CACHE_SIZE = 1024
_CACHE_LOCK = threading.Lock()

# Cache mapping KMS key name -> tuple of (plaintext_dek: bytes, wrapped_dek: str)
_KMS_KEY_DEK_CACHE: dict[str, tuple[bytes, str]] = {}

# Cache mapping (kms_key_name, wrapped_dek) -> Fernet instance
_DEK_FERNET_CACHE: dict[tuple[str, str], Fernet] = {}

# KMS Client cache
_KMS_CLIENT_CACHE: dict[str, Any] = {}


def _cache_put(cache: dict[Any, Any], key: Any, value: Any) -> None:
  """Adds an item to a bounded cache, evicting the oldest entry if full."""
  with _CACHE_LOCK:
    if key in cache:
      cache[key] = value
      return
    if len(cache) >= _MAX_CACHE_SIZE:
      oldest_key = next(iter(cache))
      cache.pop(oldest_key, None)
    cache[key] = value


def _get_kms_client() -> Any:
  """Gets or creates a cached Google Cloud KMS client."""
  if "client" in _KMS_CLIENT_CACHE:
    return _KMS_CLIENT_CACHE["client"]

  try:
    from google.cloud import kms
  except ImportError as e:
    from ..utils._dependency import missing_extra

    raise missing_extra("google-cloud-kms", "gcp") from e

  client = kms.KeyManagementServiceClient()
  with _CACHE_LOCK:
    if "client" not in _KMS_CLIENT_CACHE:
      _KMS_CLIENT_CACHE["client"] = client
    else:
      client.close()
    return _KMS_CLIENT_CACHE["client"]


def _get_or_create_dek(kms_key_name: str) -> tuple[bytes, str]:
  """Gets the cached DEK for a KMS key, or generates and wraps a new one."""
  with _CACHE_LOCK:
    if kms_key_name in _KMS_KEY_DEK_CACHE:
      return _KMS_KEY_DEK_CACHE[kms_key_name]

  # Generate a new 32-byte Fernet key
  plaintext_dek = Fernet.generate_key()

  # Wrap (encrypt) the DEK using Cloud KMS
  client = _get_kms_client()
  response = client.encrypt(
      request={
          "name": kms_key_name,
          "plaintext": plaintext_dek,
      }
  )
  wrapped_dek = base64.b64encode(response.ciphertext).decode("utf-8")
  _cache_put(_KMS_KEY_DEK_CACHE, kms_key_name, (plaintext_dek, wrapped_dek))
  # Also populate the Fernet cache for this (kms_key_name, wrapped_dek)
  _cache_put(
      _DEK_FERNET_CACHE, (kms_key_name, wrapped_dek), Fernet(plaintext_dek)
  )

  return plaintext_dek, wrapped_dek


def _get_fernet_for_wrapped_dek(
    *, kms_key_name: str, wrapped_dek: str
) -> Fernet:
  """Gets the cached Fernet instance for a wrapped DEK, unwrapping it with KMS if needed."""
  cache_key = (kms_key_name, wrapped_dek)
  with _CACHE_LOCK:
    if cache_key in _DEK_FERNET_CACHE:
      return _DEK_FERNET_CACHE[cache_key]

  # Unwrap (decrypt) the DEK using Cloud KMS.
  # Cloud KMS Decrypt requires a CryptoKey name, not a CryptoKeyVersion.
  client = _get_kms_client()
  ciphertext_bytes = base64.b64decode(wrapped_dek.encode("utf-8"))
  key_name = kms_key_name.split("/cryptoKeyVersions/")[0]
  response = client.decrypt(
      request={
          "name": key_name,
          "ciphertext": ciphertext_bytes,
      }
  )
  plaintext_dek = response.plaintext
  fernet = Fernet(plaintext_dek)
  _cache_put(_DEK_FERNET_CACHE, cache_key, fernet)

  return fernet


def _check_kms_encrypted_fields(info: Mapping[str, Any]) -> None:
  """Raises ValueError if any sensitive field contains KMS-encrypted ciphertext."""
  for field in ("token", "refresh_token", "client_secret", "rapt_token"):
    value = info.get(field)
    if isinstance(value, str) and value.startswith("kms:"):
      raise ValueError(
          f"{field} is KMS-encrypted but no usable kms_key_name/wrapped_dek"
          " was found; cannot decrypt."
      )


def encrypt_credentials(
    *,
    kms_key_name: str,
    token: str | None = None,
    refresh_token: str | None = None,
    client_secret: str | None = None,
    rapt_token: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
  """Encrypts the sensitive credential fields using envelope encryption.

  Returns a tuple of (encrypted_token, encrypted_refresh_token, encrypted_client_secret, encrypted_rapt_token, wrapped_dek).
  """
  plaintext_dek, wrapped_dek = _get_or_create_dek(kms_key_name)
  cache_key = (kms_key_name, wrapped_dek)
  with _CACHE_LOCK:
    fernet = _DEK_FERNET_CACHE.get(cache_key)
  if fernet is None:
    fernet = Fernet(plaintext_dek)
    _cache_put(_DEK_FERNET_CACHE, cache_key, fernet)

  enc_token = (
      fernet.encrypt(token.encode("utf-8")).decode("utf-8")
      if token is not None
      else None
  )
  enc_refresh = (
      fernet.encrypt(refresh_token.encode("utf-8")).decode("utf-8")
      if refresh_token is not None
      else None
  )
  enc_secret = (
      fernet.encrypt(client_secret.encode("utf-8")).decode("utf-8")
      if client_secret is not None
      else None
  )
  enc_rapt = (
      fernet.encrypt(rapt_token.encode("utf-8")).decode("utf-8")
      if rapt_token is not None
      else None
  )

  return enc_token, enc_refresh, enc_secret, enc_rapt, wrapped_dek


def decrypt_credentials(
    *,
    kms_key_name: str,
    encrypted_token: str | None = None,
    encrypted_refresh_token: str | None = None,
    encrypted_client_secret: str | None = None,
    encrypted_rapt_token: str | None = None,
    wrapped_dek: str | None = None,
) -> tuple[str | None, str | None, str | None, str | None]:
  """Decrypts the sensitive credential fields using the wrapped DEK."""
  if not wrapped_dek:
    # Backward compatibility
    return (
        encrypted_token,
        encrypted_refresh_token,
        encrypted_client_secret,
        encrypted_rapt_token,
    )

  fernet = _get_fernet_for_wrapped_dek(
      kms_key_name=kms_key_name, wrapped_dek=wrapped_dek
  )

  dec_token = (
      fernet.decrypt(encrypted_token.encode("utf-8")).decode("utf-8")
      if encrypted_token is not None
      else None
  )
  dec_refresh = (
      fernet.decrypt(encrypted_refresh_token.encode("utf-8")).decode("utf-8")
      if encrypted_refresh_token is not None
      else None
  )
  dec_secret = (
      fernet.decrypt(encrypted_client_secret.encode("utf-8")).decode("utf-8")
      if encrypted_client_secret is not None
      else None
  )
  dec_rapt = (
      fernet.decrypt(encrypted_rapt_token.encode("utf-8")).decode("utf-8")
      if encrypted_rapt_token is not None
      else None
  )

  return dec_token, dec_refresh, dec_secret, dec_rapt


class KmsEncryptedCredentials(google.oauth2.credentials.Credentials):
  """Subclass of Google Credentials that supports encrypting sensitive fields using KMS."""

  def __init__(
      self,
      token: str | None = None,
      *,
      refresh_token: str | None = None,
      id_token: str | None = None,
      token_uri: str | None = None,
      client_id: str | None = None,
      client_secret: str | None = None,
      scopes: Sequence[str] | None = None,
      default_scopes: Sequence[str] | None = None,
      quota_project_id: str | None = None,
      expiry: datetime.datetime | None = None,
      rapt_token: str | None = None,
      refresh_handler: Any = None,
      enable_reauth_refresh: bool = False,
      granted_scopes: Sequence[str] | None = None,
      universe_domain: str | None = None,
      account: str | None = None,
      trust_boundary: Any = None,
      kms_key_name: str | None = None,
      **kwargs: Any,
  ) -> None:
    kwargs_to_pass = {
        "token": token,
        "refresh_token": refresh_token,
        "id_token": id_token,
        "token_uri": token_uri,
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": scopes,
        "default_scopes": default_scopes,
        "quota_project_id": quota_project_id,
        "expiry": expiry,
        "rapt_token": rapt_token,
        "refresh_handler": refresh_handler,
        "enable_reauth_refresh": enable_reauth_refresh,
        "granted_scopes": granted_scopes,
        "account": account,
        "trust_boundary": trust_boundary,
    }
    if universe_domain is not None:
      kwargs_to_pass["universe_domain"] = universe_domain
    kwargs_to_pass.update(kwargs)
    super().__init__(**kwargs_to_pass)  # type: ignore[no-untyped-call]
    self.kms_key_name = kms_key_name

  def _make_copy(self) -> KmsEncryptedCredentials:
    copy = cast(
        KmsEncryptedCredentials,
        super()._make_copy(),  # type: ignore[no-untyped-call]
    )
    copy.kms_key_name = self.kms_key_name
    return copy

  def __setstate__(self, d: dict[str, Any]) -> None:
    super().__setstate__(d)  # type: ignore[no-untyped-call]
    self.kms_key_name = d.get("kms_key_name")

  def to_json(self, strip: Sequence[str] | None = None) -> str:
    """Serialize credentials to JSON, encrypting sensitive fields if kms_key_name is present."""
    serialized_json = super().to_json(strip=strip)  # type: ignore[no-untyped-call]
    data = json.loads(serialized_json)

    if getattr(self, "quota_project_id", None) and (
        strip is None or "quota_project_id" not in strip
    ):
      data["quota_project_id"] = self.quota_project_id
    if getattr(self, "_trust_boundary", None) and (
        strip is None or "trust_boundary" not in strip
    ):
      data["trust_boundary"] = self._trust_boundary
    if getattr(self, "default_scopes", None) and (
        strip is None or "default_scopes" not in strip
    ):
      data["default_scopes"] = list(self.default_scopes)
    if getattr(self, "granted_scopes", None) and (
        strip is None or "granted_scopes" not in strip
    ):
      data["granted_scopes"] = list(self.granted_scopes)

    if self.kms_key_name:
      token = data.get("token")
      refresh_token = data.get("refresh_token")
      client_secret = data.get("client_secret")
      rapt_token = data.get("rapt_token")

      enc_token, enc_refresh, enc_secret, enc_rapt, wrapped_dek = (
          encrypt_credentials(
              kms_key_name=self.kms_key_name,
              token=token,
              refresh_token=refresh_token,
              client_secret=client_secret,
              rapt_token=rapt_token,
          )
      )

      if enc_token is not None:
        data["token"] = "kms:" + enc_token
      if enc_refresh is not None:
        data["refresh_token"] = "kms:" + enc_refresh
      if enc_secret is not None:
        data["client_secret"] = "kms:" + enc_secret
      if enc_rapt is not None:
        data["rapt_token"] = "kms:" + enc_rapt

      if wrapped_dek:
        data["wrapped_dek"] = wrapped_dek
      data["kms_key_name"] = self.kms_key_name

    if strip is not None:
      data = {k: v for k, v in data.items() if k not in strip}

    return json.dumps(data)

  @classmethod
  def from_authorized_user_info(
      cls,
      info: Mapping[str, Any],
      scopes: Sequence[str] | None = None,
      *,
      kms_key_name: str | None = None,
  ) -> KmsEncryptedCredentials:
    """Deserialize credentials from user info, decrypting sensitive fields if encrypted."""
    if kms_key_name is None:
      kms_key_name = info.get("kms_key_name")
    wrapped_dek = info.get("wrapped_dek")
    info_copy = dict(info)

    if not (kms_key_name and wrapped_dek):
      _check_kms_encrypted_fields(info_copy)

    if kms_key_name and wrapped_dek:
      token = info_copy.get("token")
      refresh_token = info_copy.get("refresh_token")
      client_secret = info_copy.get("client_secret")
      rapt_token = info_copy.get("rapt_token")

      def _unwrap(value: Any) -> tuple[str | None, bool]:
        if isinstance(value, str) and value.startswith("kms:"):
          return value[4:], True
        return value, False

      enc_token, token_enc = _unwrap(token)
      enc_refresh, refresh_enc = _unwrap(refresh_token)
      enc_secret, secret_enc = _unwrap(client_secret)
      enc_rapt, rapt_enc = _unwrap(rapt_token)

      dec_token, dec_refresh, dec_secret, dec_rapt = decrypt_credentials(
          kms_key_name=kms_key_name,
          encrypted_token=enc_token if token_enc else None,
          encrypted_refresh_token=enc_refresh if refresh_enc else None,
          encrypted_client_secret=enc_secret if secret_enc else None,
          encrypted_rapt_token=enc_rapt if rapt_enc else None,
          wrapped_dek=wrapped_dek,
      )

      if token_enc:
        info_copy["token"] = dec_token
      if refresh_enc:
        info_copy["refresh_token"] = dec_refresh
      if secret_enc:
        info_copy["client_secret"] = dec_secret
      if rapt_enc:
        info_copy["rapt_token"] = dec_rapt

    creds = google.oauth2.credentials.Credentials.from_authorized_user_info(
        info_copy, scopes=scopes
    )  # type: ignore[no-untyped-call]

    return cls(
        token=creds.token,
        refresh_token=creds.refresh_token,
        id_token=creds.id_token,
        token_uri=creds.token_uri,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        scopes=creds.scopes,
        default_scopes=info_copy.get("default_scopes")
        or getattr(creds, "default_scopes", None),
        expiry=creds.expiry,
        quota_project_id=creds.quota_project_id,
        rapt_token=getattr(creds, "rapt_token", None),
        refresh_handler=getattr(creds, "refresh_handler", None),
        enable_reauth_refresh=getattr(creds, "_enable_reauth_refresh", False),
        universe_domain=getattr(creds, "universe_domain", None),
        account=getattr(creds, "account", None),
        trust_boundary=info_copy.get("trust_boundary")
        or getattr(creds, "_trust_boundary", None),
        granted_scopes=info_copy.get("granted_scopes")
        or getattr(creds, "granted_scopes", None),
        kms_key_name=kms_key_name,
    )
