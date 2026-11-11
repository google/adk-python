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

import base64
import datetime
import json
import logging
from unittest.mock import Mock

from google.adk.auth._kms_encryptor import _DEK_FERNET_CACHE
from google.adk.auth._kms_encryptor import _get_kms_client as _real_get_kms_client
from google.adk.auth._kms_encryptor import _KMS_CLIENT_CACHE
from google.adk.auth._kms_encryptor import _KMS_KEY_DEK_CACHE
from google.adk.auth._kms_encryptor import decrypt_credentials
from google.adk.auth._kms_encryptor import encrypt_credentials
from google.adk.auth._kms_encryptor import KmsEncryptedCredentials
from google.adk.tools._google_credentials import BaseGoogleCredentialsConfig
from google.adk.tools._google_credentials import GoogleCredentialsManager
from google.adk.tools.tool_context import ToolContext
import pytest


@pytest.fixture(autouse=True)
def mock_kms_client(monkeypatch):
  """Mock the Google Cloud KMS client for testing."""

  class MockKmsClient:

    def encrypt(self, request):
      # Wrap the plaintext DEK by adding a prefix
      ct_val = b"mock_wrapped_" + request["plaintext"]
      return Mock(ciphertext=ct_val)

    def decrypt(self, request):
      # Unwrap the ciphertext to retrieve original DEK
      ct_val = request["ciphertext"]
      assert ct_val.startswith(b"mock_wrapped_")
      pt_val = ct_val[13:]
      return Mock(plaintext=pt_val)

  import google.adk.auth._kms_encryptor

  monkeypatch.setattr(
      google.adk.auth._kms_encryptor,
      "_get_kms_client",
      lambda: MockKmsClient(),
  )


def test_kms_missing_extra_raises_import_error(monkeypatch):
  """Test that _get_kms_client raises missing_extra when google.cloud.kms is not installed."""
  import builtins

  _KMS_CLIENT_CACHE.clear()

  real_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    if "google.cloud.kms" in name or name == "google.cloud":
      raise ImportError("No module named 'google.cloud.kms'")
    return real_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", mock_import)

  with pytest.raises(ImportError, match=r"google-cloud-kms.*google-adk\[gcp\]"):
    _real_get_kms_client()


def test_kms_envelope_encryption_caching_and_crypto():
  """Test that kms_encryptor properly encrypts, decrypts, and uses in-memory DEK caching."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )

  # Reset caches
  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  token = "secret_access_token"
  refresh_token = "secret_refresh_token"

  # Encrypt credentials using envelope encryption
  enc_token, enc_refresh, _, _, wrapped_dek = encrypt_credentials(
      kms_key_name=key_name,
      token=token,
      refresh_token=refresh_token,
      client_secret=None,
  )

  assert enc_token != token
  assert enc_refresh != refresh_token
  assert wrapped_dek is not None

  # Verify the DEK is cached
  assert key_name in _KMS_KEY_DEK_CACHE
  assert (key_name, wrapped_dek) in _DEK_FERNET_CACHE

  # Decrypt credentials
  dec_token, dec_refresh, _, _ = decrypt_credentials(
      kms_key_name=key_name,
      encrypted_token=enc_token,
      encrypted_refresh_token=enc_refresh,
      encrypted_client_secret=None,
      wrapped_dek=wrapped_dek,
  )

  assert dec_token == token
  assert dec_refresh == refresh_token


def test_kms_encrypted_credentials_serialization():
  """Test that KmsEncryptedCredentials properly serializes to JSON with envelope encryption and deserializes back."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )

  creds = KmsEncryptedCredentials(
      token="secret_access_token",
      refresh_token="secret_refresh_token",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      kms_key_name=key_name,
  )

  # Serialize to JSON (envelope encryption)
  serialized = creds.to_json()
  data = json.loads(serialized)

  # Ensure sensitive values are prefixed and wrapped DEK is stored
  assert data["token"].startswith("kms:")
  assert data["refresh_token"].startswith("kms:")
  assert data["client_secret"].startswith("kms:")
  assert "wrapped_dek" in data
  assert data["kms_key_name"] == key_name
  assert data["client_id"] == "my_client_id"

  # Deserialize back
  deserialized = KmsEncryptedCredentials.from_authorized_user_info(data)

  assert deserialized.token == "secret_access_token"
  assert deserialized.refresh_token == "secret_refresh_token"
  assert deserialized.client_secret == "secret_client_secret"
  assert deserialized.client_id == "my_client_id"
  assert deserialized.kms_key_name == key_name


def test_kms_env_var_detection(monkeypatch):
  """Test that BaseGoogleCredentialsConfig automatically detects GOOGLE_CREDENTIAL_KMS_KEY."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  monkeypatch.setenv("GOOGLE_CREDENTIAL_KMS_KEY", key_name)

  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
  )

  assert config.kms_key_name == key_name


def test_kms_credentials_backward_compatibility():
  """Test that loading a non-encrypted credentials json works normally without crashing."""
  info = {
      "token": "plaintext_token",
      "refresh_token": "plaintext_refresh",
      "client_id": "my_client_id",
      "client_secret": "plaintext_secret",
  }

  # Load via KmsEncryptedCredentials but without kms_key_name in info
  creds = KmsEncryptedCredentials.from_authorized_user_info(info)

  assert creds.token == "plaintext_token"
  assert creds.refresh_token == "plaintext_refresh"
  assert creds.client_secret == "plaintext_secret"
  assert creds.kms_key_name is None

  # to_json should not encrypt when kms_key_name is not set
  serialized = creds.to_json()
  data = json.loads(serialized)
  assert data["token"] == "plaintext_token"
  assert "kms_key_name" not in data


def test_kms_credentials_quota_project_id():
  """Test that quota_project_id is preserved when deserializing."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_access_token",
      refresh_token="secret_refresh_token",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      quota_project_id="my_quota_project",
      kms_key_name=key_name,
  )
  serialized = creds.to_json()
  data = json.loads(serialized)
  assert data["quota_project_id"] == "my_quota_project"

  deserialized = KmsEncryptedCredentials.from_authorized_user_info(data)
  assert deserialized.quota_project_id == "my_quota_project"


def test_kms_credentials_from_authorized_user_info_with_key():
  """Test that from_authorized_user_info accepts kms_key_name."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  info = {
      "token": "plaintext_token",
      "refresh_token": "plaintext_refresh",
      "client_id": "my_client_id",
      "client_secret": "plaintext_secret",
  }

  creds = KmsEncryptedCredentials.from_authorized_user_info(
      info, kms_key_name=key_name
  )
  assert creds.kms_key_name == key_name

  serialized = creds.to_json()
  data = json.loads(serialized)
  assert data["token"].startswith("kms:")
  assert data["kms_key_name"] == key_name


def test_kms_credentials_deserialization_with_unencrypted_field():
  """Test that from_authorized_user_info handles unencrypted fields when wrapped_dek is present."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      kms_key_name=key_name,
  )
  data = json.loads(creds.to_json())
  data["token"] = "new_plaintext_token"

  deserialized = KmsEncryptedCredentials.from_authorized_user_info(data)
  assert deserialized.token == "new_plaintext_token"
  assert deserialized.refresh_token == "secret_refresh"
  assert deserialized.client_secret == "secret_client_secret"


def test_kms_credentials_copy_and_with_methods():
  """Test that with_* methods work properly and preserve kms_key_name."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      kms_key_name=key_name,
  )

  new_creds = creds.with_quota_project("new_quota_project")
  assert isinstance(new_creds, KmsEncryptedCredentials)
  assert new_creds.quota_project_id == "new_quota_project"
  assert new_creds.kms_key_name == key_name
  assert new_creds.token == "secret_token"

  new_creds2 = creds.with_token_uri("https://example.com/custom_token")
  assert isinstance(new_creds2, KmsEncryptedCredentials)
  assert new_creds2.token_uri == "https://example.com/custom_token"
  assert new_creds2.kms_key_name == key_name

  new_creds3 = creds.with_universe_domain("custom.universe.domain")
  assert isinstance(new_creds3, KmsEncryptedCredentials)
  assert new_creds3.universe_domain == "custom.universe.domain"
  assert new_creds3.kms_key_name == key_name

  new_creds4 = creds.with_account("user@example.com")
  assert isinstance(new_creds4, KmsEncryptedCredentials)
  assert new_creds4.account == "user@example.com"
  assert new_creds4.kms_key_name == key_name


def test_kms_cache_bounded(monkeypatch):
  """Test that DEK and Fernet caches are bounded and evict oldest entries."""
  import google.adk.auth._kms_encryptor as kms_enc

  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  monkeypatch.setattr(kms_enc, "_MAX_CACHE_SIZE", 3)

  for i in range(5):
    encrypt_credentials(
        kms_key_name=f"key_{i}",
        token=f"token_{i}",
    )

  assert len(_KMS_KEY_DEK_CACHE) <= 3
  assert len(_DEK_FERNET_CACHE) <= 3
  # Oldest entries should have been evicted
  assert "key_0" not in _KMS_KEY_DEK_CACHE
  assert "key_1" not in _KMS_KEY_DEK_CACHE
  assert "key_4" in _KMS_KEY_DEK_CACHE


@pytest.mark.asyncio
async def test_google_credentials_manager_refresh_token_encryption(monkeypatch):
  """Test that GoogleCredentialsManager re-wraps and encrypts refreshed credentials in session state."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  creds_to_expire = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      expiry=expired_dt,
      kms_key_name=key_name,
  )
  data = json.loads(creds_to_expire.to_json())

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": json.dumps(data)}

  import google.oauth2.credentials

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "new_refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert creds.token == "new_refreshed_access_token"

  cached_state = json.loads(tool_context.state["user_token"])
  assert cached_state["token"].startswith("kms:")
  assert cached_state["kms_key_name"] == key_name
  assert "wrapped_dek" in cached_state


@pytest.mark.asyncio
async def test_google_credentials_manager_configured_credentials_refresh_not_cached_in_state(
    monkeypatch,
):
  """Test that plain OAuth2 Credentials passed via credentials_config.credentials is refreshed without persisting to session state."""
  import google.oauth2.credentials

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  plain_oauth_creds = google.oauth2.credentials.Credentials(
      token="plain_secret_token",
      refresh_token="plain_secret_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      token_uri="https://example.com/token",
      expiry=expired_dt,
  )

  config = BaseGoogleCredentialsConfig(
      credentials=plain_oauth_creds,
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "new_refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert creds.token == "new_refreshed_access_token"
  assert "user_token" not in tool_context.state


@pytest.mark.asyncio
async def test_google_credentials_manager_oauth_completion_encryption():
  """Test that completing OAuth flow creates KmsEncryptedCredentials and caches encrypted JSON."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      scopes=["scope1", "scope2"],
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  auth_response = Mock()
  auth_response.oauth2 = Mock(
      access_token="oauth_access_token",
      refresh_token="oauth_refresh_token",
  )
  tool_context.get_auth_response = Mock(return_value=auth_response)

  creds = await manager.get_valid_credentials(tool_context)
  assert isinstance(creds, KmsEncryptedCredentials)
  assert creds.token == "oauth_access_token"
  assert creds.refresh_token == "oauth_refresh_token"
  assert creds.kms_key_name == key_name

  cached_json = tool_context.state.get("user_token")
  assert cached_json is not None
  data = json.loads(cached_json)
  assert data["token"].startswith("kms:")
  assert data["refresh_token"].startswith("kms:")
  assert data["kms_key_name"] == key_name
  assert "wrapped_dek" in data


@pytest.mark.asyncio
async def test_google_credentials_manager_oauth_completion_without_scopes():
  """Test that OAuth completion works when scopes is None (default)."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  assert config.scopes is None
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  auth_response = Mock()
  auth_response.oauth2 = Mock(
      access_token="oauth_access_token",
      refresh_token="oauth_refresh_token",
  )
  tool_context.get_auth_response = Mock(return_value=auth_response)

  creds = await manager.get_valid_credentials(tool_context)
  assert isinstance(creds, KmsEncryptedCredentials)
  assert creds.scopes is None


@pytest.mark.asyncio
async def test_google_credentials_manager_decryption_failure_fallback():
  """Test that decryption failures gracefully discard invalid state and fall back to OAuth flow."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  # State containing corrupted/un-decryptable ciphertext
  corrupted_creds = {
      "token": "kms:corrupted_ciphertext",
      "wrapped_dek": "invalid_wrapped_dek",
      "kms_key_name": key_name,
  }
  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": json.dumps(corrupted_creds)}
  tool_context.get_auth_response = Mock(return_value=None)
  tool_context.request_credential = Mock()

  # Should not raise exception; instead requests new credential via OAuth
  result = await manager.get_valid_credentials(tool_context)
  assert result is None
  tool_context.request_credential.assert_called_once()


@pytest.mark.asyncio
async def test_google_credentials_manager_operator_key_precedence():
  """Test that operator-configured kms_key_name takes precedence over state-stored key."""
  operator_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/operator_key/cryptoKeyVersions/1"
  state_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/attacker_key/cryptoKeyVersions/1"

  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=operator_key,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  creds_to_save = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      expiry=future_dt,
      kms_key_name=operator_key,
  )
  serialized = creds_to_save.to_json()
  data = json.loads(serialized)
  data["kms_key_name"] = state_key  # state claims different key

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": json.dumps(data)}

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert creds.kms_key_name == operator_key


def test_encrypt_credentials_after_fernet_cache_eviction(monkeypatch):
  """Test that encrypt_credentials succeeds even if wrapped_dek was evicted from _DEK_FERNET_CACHE."""
  from cryptography.fernet import Fernet
  import google.adk.auth._kms_encryptor as kms_enc

  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  monkeypatch.setattr(kms_enc, "_MAX_CACHE_SIZE", 2)

  try:
    key_name = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
    *_, wrapped_dek = encrypt_credentials(
        kms_key_name=key_name,
        token="token1",
    )

    # Evict wrapped_dek from _DEK_FERNET_CACHE by decrypting with other wrapped DEKs
    # while key_name remains in _KMS_KEY_DEK_CACHE
    fake_wrapped_1 = base64.b64encode(
        b"mock_wrapped_" + Fernet.generate_key()
    ).decode("utf-8")
    fake_wrapped_2 = base64.b64encode(
        b"mock_wrapped_" + Fernet.generate_key()
    ).decode("utf-8")
    decrypt_credentials(
        kms_key_name="other_key_1",
        encrypted_token=None,
        wrapped_dek=fake_wrapped_1,
    )
    decrypt_credentials(
        kms_key_name="other_key_2",
        encrypted_token=None,
        wrapped_dek=fake_wrapped_2,
    )

    assert (key_name, wrapped_dek) not in _DEK_FERNET_CACHE
    assert key_name in _KMS_KEY_DEK_CACHE

    # Encrypting again with key_name should not raise KeyError
    enc_token, *_, _ = encrypt_credentials(
        kms_key_name=key_name,
        token="token2",
    )
    assert enc_token is not None
  finally:
    _KMS_KEY_DEK_CACHE.clear()
    _DEK_FERNET_CACHE.clear()


def test_kms_encrypted_credentials_to_json_respects_strip():
  """Test that to_json respects the strip argument for quota_project_id."""
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      quota_project_id="my_quota_project",
      kms_key_name="projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1",
  )
  data = json.loads(creds.to_json(strip=["quota_project_id"]))
  assert "quota_project_id" not in data


def test_kms_encrypted_field_without_key_raises_value_error():
  """Test that KMS-prefixed fields raise ValueError when no usable key/wrapped_dek is provided."""
  info = {
      "token": "kms:encrypted_token",
      "client_id": "my_client_id",
      "client_secret": "my_client_secret",
  }
  with pytest.raises(ValueError, match=r"token is KMS-encrypted but no usable"):
    KmsEncryptedCredentials.from_authorized_user_info(info)


@pytest.mark.asyncio
async def test_google_credentials_manager_unconfigured_kms_rejects_encrypted_state():
  """Test that when KMS is not configured, state-provided kms_key_name is ignored and encrypted tokens fall back to re-auth."""
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  future_expiry = (
      datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
  ).strftime("%Y-%m-%dT%H:%M:%SZ")

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {
      "user_token": json.dumps({
          "token": "kms:some_token",
          "refresh_token": "my_refresh_token",
          "expiry": future_expiry,
          "kms_key_name": "projects/p/locations/l/keyRings/kr/cryptoKeys/k",
          "wrapped_dek": "some_dek",
          "client_id": "my_client_id",
          "client_secret": "my_client_secret",
      })
  }
  tool_context.get_auth_response = Mock(return_value=None)
  tool_context.request_credential = Mock()

  result = await manager.get_valid_credentials(tool_context)
  assert result is None
  tool_context.request_credential.assert_called_once()


@pytest.mark.asyncio
async def test_google_credentials_manager_decryption_failure_logs_warning(
    caplog,
):
  """Test that decryption failure logs a warning before requesting re-authentication."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  corrupted_creds = {
      "token": "kms:corrupted_ciphertext",
      "wrapped_dek": "invalid_wrapped_dek",
      "kms_key_name": key_name,
  }
  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": json.dumps(corrupted_creds)}
  tool_context.get_auth_response = Mock(return_value=None)
  tool_context.request_credential = Mock()

  with caplog.at_level(logging.WARNING):
    result = await manager.get_valid_credentials(tool_context)
    assert result is None
    assert any(
        "Failed to decrypt or deserialize cached credentials" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_google_credentials_manager_invalid_json_payload_fallback():
  """Test that invalid (non-JSON) payload in state gracefully falls back to OAuth flow."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": "{not-valid-json: xyz"}
  tool_context.get_auth_response = Mock(return_value=None)
  tool_context.request_credential = Mock()

  result = await manager.get_valid_credentials(tool_context)
  assert result is None
  tool_context.request_credential.assert_called_once()


@pytest.mark.asyncio
async def test_google_credentials_manager_invalid_json_payload_logs_warning(
    caplog,
):
  """Test that invalid JSON payload logs a warning before requesting re-authentication."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": "corrupt_non_json_string"}
  tool_context.get_auth_response = Mock(return_value=None)
  tool_context.request_credential = Mock()

  with caplog.at_level(logging.WARNING):
    result = await manager.get_valid_credentials(tool_context)
    assert result is None
    assert any(
        "Failed to decrypt or deserialize cached credentials" in record.message
        for record in caplog.records
    )


def test_kms_client_losing_race_closes_client(monkeypatch):
  """Test that if a second thread constructs a client while another already cached one, the loser is closed."""
  import sys
  from unittest.mock import MagicMock

  _KMS_CLIENT_CACHE.clear()

  mock_existing_client = MagicMock()
  mock_new_client = MagicMock()

  mock_kms_module = MagicMock()

  def mock_constructor():
    # Simulate another thread populating the cache right before constructor returns
    _KMS_CLIENT_CACHE["client"] = mock_existing_client
    return mock_new_client

  mock_kms_module.KeyManagementServiceClient.side_effect = mock_constructor

  monkeypatch.setitem(sys.modules, "google.cloud.kms", mock_kms_module)
  monkeypatch.setitem(
      sys.modules, "google.cloud", MagicMock(kms=mock_kms_module)
  )

  try:
    client = _real_get_kms_client()
    assert client is mock_existing_client
    mock_new_client.close.assert_called_once()
  finally:
    _KMS_CLIENT_CACHE.clear()


def test_kms_concurrent_encryption_and_decryption(monkeypatch):
  """Test concurrent encryption and decryption across multiple threads."""
  import concurrent.futures

  import google.adk.auth._kms_encryptor as kms_enc

  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  monkeypatch.setattr(kms_enc, "_MAX_CACHE_SIZE", 2)

  def worker(i: int) -> tuple[str, str]:
    key_name = f"projects/p1/locations/l1/keyRings/kr1/cryptoKeys/key_{i}/cryptoKeyVersions/1"
    token = f"token_{i}"
    enc_token, *_, wrapped_dek = encrypt_credentials(
        kms_key_name=key_name,
        token=token,
    )
    assert enc_token is not None
    assert wrapped_dek is not None
    dec_token, *_, _ = decrypt_credentials(
        kms_key_name=key_name,
        encrypted_token=enc_token,
        wrapped_dek=wrapped_dek,
    )
    assert dec_token == token
    return token, dec_token

  try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(worker, i) for i in range(20)]
      for future in concurrent.futures.as_completed(futures):
        orig, dec = future.result()
        assert orig == dec
  finally:
    _KMS_KEY_DEK_CACHE.clear()
    _DEK_FERNET_CACHE.clear()


def test_kms_credentials_deepcopy_and_pickle():
  """Test that copy/deepcopy and pickle preserve kms_key_name and to_json works."""
  import copy
  import pickle

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      kms_key_name=key_name,
  )

  # deepcopy
  copied = copy.deepcopy(creds)
  assert isinstance(copied, KmsEncryptedCredentials)
  assert copied.kms_key_name == key_name
  data = json.loads(copied.to_json())
  assert data["kms_key_name"] == key_name
  assert data["token"].startswith("kms:")

  # pickle
  pickled = pickle.loads(pickle.dumps(creds))
  assert isinstance(pickled, KmsEncryptedCredentials)
  assert pickled.kms_key_name == key_name
  pickled_data = json.loads(pickled.to_json())
  assert pickled_data["kms_key_name"] == key_name
  assert pickled_data["token"].startswith("kms:")


def test_kms_key_rotation_cache_isolation(monkeypatch):
  """Test that key rotation does not hit Fernet cache for old key and invokes KMS with new key."""
  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  old_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/old_k/cryptoKeyVersions/1"
  new_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/new_k/cryptoKeyVersions/1"

  enc_token, *_, wrapped_dek = encrypt_credentials(
      kms_key_name=old_key,
      token="secret_token",
  )

  assert (old_key, wrapped_dek) in _DEK_FERNET_CACHE
  assert (new_key, wrapped_dek) not in _DEK_FERNET_CACHE

  # Mock KMS client to reject decrypting ciphertext wrapped with old_key when requested with new_key
  class KeyRotationMockKmsClient:

    def decrypt(self, request):
      if request["name"] == new_key.split("/cryptoKeyVersions/")[0]:
        raise ValueError("KMS decryption failed: key mismatch")
      ct_val = request["ciphertext"]
      assert ct_val.startswith(b"mock_wrapped_")
      return Mock(plaintext=ct_val[13:])

  import google.adk.auth._kms_encryptor as kms_enc

  monkeypatch.setattr(
      kms_enc,
      "_get_kms_client",
      lambda: KeyRotationMockKmsClient(),
  )

  with pytest.raises(ValueError, match=r"KMS decryption failed: key mismatch"):
    decrypt_credentials(
        kms_key_name=new_key,
        encrypted_token=enc_token,
        wrapped_dek=wrapped_dek,
    )


def test_kms_decrypt_strips_crypto_key_version(monkeypatch):
  """Test that decrypt_credentials strips /cryptoKeyVersions/N before calling Cloud KMS decrypt."""
  _KMS_KEY_DEK_CACHE.clear()
  _DEK_FERNET_CACHE.clear()

  key_with_version = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/2"
  )
  expected_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1"

  enc_token, *_, wrapped_dek = encrypt_credentials(
      kms_key_name=key_with_version,
      token="secret_token",
  )

  _DEK_FERNET_CACHE.clear()

  decrypted_key_name = None

  class VersionCheckMockKmsClient:

    def decrypt(self, request):
      nonlocal decrypted_key_name
      decrypted_key_name = request["name"]
      ct_val = request["ciphertext"]
      assert ct_val.startswith(b"mock_wrapped_")
      return Mock(plaintext=ct_val[13:])

  import google.adk.auth._kms_encryptor as kms_enc

  monkeypatch.setattr(
      kms_enc,
      "_get_kms_client",
      lambda: VersionCheckMockKmsClient(),
  )

  dec_token, *_, _ = decrypt_credentials(
      kms_key_name=key_with_version,
      encrypted_token=enc_token,
      wrapped_dek=wrapped_dek,
  )
  assert dec_token == "secret_token"
  assert decrypted_key_name == expected_key


def test_kms_encrypted_credentials_rapt_token():
  """Test that rapt_token is encrypted by to_json and restored by from_authorized_user_info."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      rapt_token="secret_rapt_token",
      kms_key_name=key_name,
  )

  serialized = creds.to_json()
  data = json.loads(serialized)
  assert data["rapt_token"].startswith("kms:")
  assert data["rapt_token"] != "secret_rapt_token"

  deserialized = KmsEncryptedCredentials.from_authorized_user_info(data)
  assert deserialized.rapt_token == "secret_rapt_token"


def test_kms_encrypted_rapt_token_without_key_raises_value_error():
  """Test that KMS-prefixed rapt_token raises ValueError when no usable key/wrapped_dek is provided."""
  info = {
      "token": "plaintext_token",
      "refresh_token": "plaintext_refresh",
      "rapt_token": "kms:encrypted_rapt",
      "client_id": "my_client_id",
      "client_secret": "my_client_secret",
  }
  with pytest.raises(
      ValueError, match=r"rapt_token is KMS-encrypted but no usable"
  ):
    KmsEncryptedCredentials.from_authorized_user_info(info)


def test_kms_empty_string_credentials_fields():
  """Test that empty string credentials fields are encrypted rather than omitted or treated as None."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="",
      refresh_token="",
      client_id="my_client_id",
      client_secret="",
      rapt_token="",
      kms_key_name=key_name,
  )

  serialized = creds.to_json()
  data = json.loads(serialized)
  assert data["token"].startswith("kms:")
  assert data["refresh_token"].startswith("kms:")
  assert data["client_secret"].startswith("kms:")
  assert data["rapt_token"].startswith("kms:")

  deserialized = KmsEncryptedCredentials.from_authorized_user_info(data)
  assert deserialized.token == ""
  assert deserialized.refresh_token == ""
  assert deserialized.client_secret == ""
  assert deserialized.rapt_token == ""


@pytest.mark.asyncio
async def test_google_credentials_manager_key_rotation_on_refreshed_credentials(
    monkeypatch,
):
  """Test that refreshed KmsEncryptedCredentials re-encrypts with new key when key is rotated."""
  import google.oauth2.credentials

  old_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/old_k/cryptoKeyVersions/1"
  new_key = "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/new_k/cryptoKeyVersions/1"

  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  old_creds = KmsEncryptedCredentials(
      token="old_access_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      expiry=expired_dt,
      kms_key_name=old_key,
  )

  config = BaseGoogleCredentialsConfig(
      client_id="my_client_id",
      client_secret="my_client_secret",
      kms_key_name=new_key,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {"user_token": old_creds.to_json()}

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert isinstance(creds, KmsEncryptedCredentials)
  assert creds.kms_key_name == new_key
  assert creds.token == "refreshed_access_token"

  cached_state = json.loads(tool_context.state["user_token"])
  assert cached_state["kms_key_name"] == new_key
  assert cached_state["token"].startswith("kms:")


def test_kms_credentials_from_authorized_user_info_preserves_trust_boundary():
  """Test that trust_boundary is preserved by from_authorized_user_info."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  info = {
      "token": "secret_token",
      "refresh_token": "secret_refresh",
      "client_id": "my_client_id",
      "client_secret": "secret_client_secret",
      "trust_boundary": "custom_trust_boundary",
      "kms_key_name": key_name,
  }
  creds = KmsEncryptedCredentials.from_authorized_user_info(info)
  assert creds._trust_boundary == "custom_trust_boundary"


@pytest.mark.asyncio
async def test_google_credentials_manager_refresh_preserves_default_scopes(
    monkeypatch,
):
  """Test that default_scopes is preserved when plain Credentials is refreshed and re-wrapped."""
  import google.oauth2.credentials

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  plain_oauth_creds = google.oauth2.credentials.Credentials(
      token="plain_token",
      refresh_token="plain_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      token_uri="https://example.com/token",
      expiry=expired_dt,
      default_scopes=["https://www.googleapis.com/auth/userinfo.email"],
      trust_boundary="custom_trust_boundary",
  )

  config = BaseGoogleCredentialsConfig(
      credentials=plain_oauth_creds,
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "new_refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert creds.default_scopes == [
      "https://www.googleapis.com/auth/userinfo.email"
  ]
  assert creds._trust_boundary == "custom_trust_boundary"


def test_kms_credentials_to_json_and_from_authorized_user_info_round_trip_preserves_attributes():
  """Test that default_scopes, granted_scopes, and trust_boundary survive to_json and from_authorized_user_info while token_uri is hardened against untrusted input."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      token_uri="https://custom.oauth.endpoint/token",
      default_scopes=["https://www.googleapis.com/auth/userinfo.email"],
      granted_scopes=["https://www.googleapis.com/auth/calendar.readonly"],
      trust_boundary="custom_trust_boundary",
      quota_project_id="test_quota_proj",
      kms_key_name=key_name,
  )

  json_str = creds.to_json()
  data = json.loads(json_str)

  assert data["token_uri"] == "https://custom.oauth.endpoint/token"
  assert data["default_scopes"] == [
      "https://www.googleapis.com/auth/userinfo.email"
  ]
  assert data["granted_scopes"] == [
      "https://www.googleapis.com/auth/calendar.readonly"
  ]
  assert data["trust_boundary"] == "custom_trust_boundary"
  assert data["quota_project_id"] == "test_quota_proj"

  import google.oauth2.credentials

  restored = KmsEncryptedCredentials.from_authorized_user_info(data)
  assert (
      restored.token_uri
      == google.oauth2.credentials._GOOGLE_OAUTH2_TOKEN_ENDPOINT
  )
  assert restored.token_uri != "https://custom.oauth.endpoint/token"
  assert restored.default_scopes == [
      "https://www.googleapis.com/auth/userinfo.email"
  ]
  assert restored.granted_scopes == [
      "https://www.googleapis.com/auth/calendar.readonly"
  ]
  assert restored._trust_boundary == "custom_trust_boundary"
  assert restored.quota_project_id == "test_quota_proj"
  assert restored.token == "secret_token"
  assert restored.refresh_token == "secret_refresh"
  assert restored.client_id == "my_client_id"
  assert restored.client_secret == "secret_client_secret"
  assert restored.kms_key_name == key_name


def test_kms_credentials_to_json_respects_strip_for_extended_attributes():
  """Test that to_json strips trust_boundary, default_scopes, and granted_scopes when requested."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  creds = KmsEncryptedCredentials(
      token="secret_token",
      refresh_token="secret_refresh",
      client_id="my_client_id",
      client_secret="secret_client_secret",
      default_scopes=["scope1"],
      granted_scopes=["scope2"],
      trust_boundary="boundary",
      kms_key_name=key_name,
  )
  json_str = creds.to_json(
      strip=["default_scopes", "granted_scopes", "trust_boundary"]
  )
  data = json.loads(json_str)
  assert "default_scopes" not in data
  assert "granted_scopes" not in data
  assert "trust_boundary" not in data


@pytest.mark.asyncio
async def test_google_credentials_manager_valid_configured_credentials_not_cached_in_state():
  """Test that initially valid plain Credentials passed via credentials_config.credentials is wrapped in KmsEncryptedCredentials without persisting to session state."""
  import google.oauth2.credentials

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  mock_refresh_handler = Mock()
  plain_creds = google.oauth2.credentials.Credentials(
      token="valid_token",
      refresh_token="valid_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      token_uri="https://custom.oauth.endpoint/token",
      expiry=future_dt,
      default_scopes=["https://www.googleapis.com/auth/userinfo.email"],
      granted_scopes=["https://www.googleapis.com/auth/calendar.readonly"],
      trust_boundary="custom_trust_boundary",
      refresh_handler=mock_refresh_handler,
      enable_reauth_refresh=True,
  )

  config = BaseGoogleCredentialsConfig(
      credentials=plain_creds,
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  creds = await manager.get_valid_credentials(tool_context)
  assert isinstance(creds, KmsEncryptedCredentials)
  assert creds.kms_key_name == key_name
  assert creds.token == "valid_token"
  assert creds.refresh_token == "valid_refresh"
  assert creds.token_uri == "https://custom.oauth.endpoint/token"
  assert creds.default_scopes == [
      "https://www.googleapis.com/auth/userinfo.email"
  ]
  assert creds.granted_scopes == [
      "https://www.googleapis.com/auth/calendar.readonly"
  ]
  assert creds._trust_boundary == "custom_trust_boundary"
  assert creds.refresh_handler == mock_refresh_handler
  assert creds._enable_reauth_refresh is True

  assert "user_token" not in tool_context.state


@pytest.mark.asyncio
async def test_google_credentials_manager_refresh_preserves_refresh_handler_and_reauth(
    monkeypatch,
):
  """Test that refresh_handler and enable_reauth_refresh are preserved when credentials refresh."""
  import google.oauth2.credentials

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  mock_refresh_handler = Mock()
  plain_oauth_creds = google.oauth2.credentials.Credentials(
      token="plain_token",
      refresh_token="plain_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      token_uri="https://example.com/token",
      expiry=expired_dt,
      refresh_handler=mock_refresh_handler,
      enable_reauth_refresh=True,
  )

  config = BaseGoogleCredentialsConfig(
      credentials=plain_oauth_creds,
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "new_refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert creds.refresh_handler == mock_refresh_handler
  assert creds._enable_reauth_refresh is True


@pytest.mark.asyncio
async def test_google_credentials_manager_refresh_wraps_plain_credentials_in_kms_encrypted_credentials(
    monkeypatch,
):
  """Test that refreshed plain Credentials is re-wrapped into KmsEncryptedCredentials when kms_key_name is configured."""
  import google.oauth2.credentials

  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  expired_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) - datetime.timedelta(hours=1)

  plain_oauth_creds = google.oauth2.credentials.Credentials(
      token="plain_token",
      refresh_token="plain_refresh",
      client_id="my_client_id",
      client_secret="my_client_secret",
      token_uri="https://example.com/token",
      expiry=expired_dt,
  )

  config = BaseGoogleCredentialsConfig(
      credentials=plain_oauth_creds,
      kms_key_name=key_name,
  )
  config._token_cache_key = "user_token"
  manager = GoogleCredentialsManager(config)

  tool_context = Mock(spec=ToolContext)
  tool_context.state = {}

  future_dt = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(hours=1)

  def mock_refresh(self, request):
    self.token = "new_refreshed_access_token"
    self.expiry = future_dt

  monkeypatch.setattr(
      google.oauth2.credentials.Credentials, "refresh", mock_refresh
  )

  creds = await manager.get_valid_credentials(tool_context)
  assert creds is not None
  assert isinstance(creds, KmsEncryptedCredentials)

  # Subsequent call when plain_oauth_creds is now valid in-place should also return KmsEncryptedCredentials
  creds_second = await manager.get_valid_credentials(tool_context)
  assert isinstance(creds_second, KmsEncryptedCredentials)
  assert creds_second.kms_key_name == key_name
