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
import json
import os
from unittest.mock import Mock

from google.adk.auth._kms_encryptor import _DEK_FERNET_CACHE
from google.adk.auth._kms_encryptor import _KMS_KEY_DEK_CACHE
from google.adk.auth._kms_encryptor import decrypt_credentials
from google.adk.auth._kms_encryptor import encrypt_credentials
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import HttpAuth
from google.adk.auth.auth_credential import HttpCredentials
from google.adk.auth.auth_credential import KmsEncryptedCredentials
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.credential_service.session_state_credential_service import SessionStateCredentialService
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
      lambda kms_key_name: MockKmsClient(),
  )


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
  enc_token, enc_refresh, _, wrapped_dek = encrypt_credentials(
      key_name, token, refresh_token, None
  )

  assert enc_token != token
  assert enc_refresh != refresh_token
  assert wrapped_dek is not None

  # Verify the DEK is cached
  assert key_name in _KMS_KEY_DEK_CACHE
  assert wrapped_dek in _DEK_FERNET_CACHE

  # Decrypt credentials
  dec_token, dec_refresh, _ = decrypt_credentials(
      key_name, enc_token, enc_refresh, None, wrapped_dek
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


def test_session_state_credential_service_kms_encryption(monkeypatch):
  """Test that SessionStateCredentialService encrypts credentials on save and decrypts on load."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  monkeypatch.setenv("GOOGLE_CREDENTIAL_KMS_KEY", key_name)

  # 1. Create a model with plaintext fields
  cred = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="basic",
          credentials=HttpCredentials(
              username="bob",
              password="secretpassword",
              token="tokensecret",
          ),
      ),
      api_key="api_key_secret",
  )

  service = SessionStateCredentialService()

  class MockConfig:
    credential_key = "test_cred_key"
    exchanged_auth_credential = cred

  callback_ctx = Mock()
  callback_ctx.state = {}

  import asyncio

  # Save credential (should encrypt sensitive fields in state)
  asyncio.run(service.save_credential(MockConfig(), callback_ctx))

  saved_data = callback_ctx.state["test_cred_key"]
  assert saved_data["apiKey"].startswith("kms:")
  assert saved_data["http"]["credentials"]["password"].startswith("kms:")
  assert saved_data["http"]["credentials"]["token"].startswith("kms:")

  # Load credential (should decrypt back to plaintext)
  loaded = asyncio.run(service.load_credential(MockConfig(), callback_ctx))
  assert loaded.api_key == "api_key_secret"
  assert loaded.http.credentials.password == "secretpassword"
  assert loaded.http.credentials.token == "tokensecret"


def test_kms_decryption_failure_fallback(monkeypatch):
  """Test that decryption failures (e.g. key destroyed) trigger fallback by returning None instead of crashing."""
  key_name = (
      "projects/p1/locations/l1/keyRings/kr1/cryptoKeys/k1/cryptoKeyVersions/1"
  )
  monkeypatch.setenv("GOOGLE_CREDENTIAL_KMS_KEY", key_name)

  # Mock KMS client to raise an exception on decrypt (simulating destroyed key or permission failure)
  class FailedMockKmsClient:

    def encrypt(self, request):
      return Mock(ciphertext=b"mock_wrapped_" + request["plaintext"])

    def decrypt(self, request):
      raise RuntimeError("KMS key has been destroyed or IAM permission denied")

  import google.adk.auth._kms_encryptor

  monkeypatch.setattr(
      google.adk.auth._kms_encryptor,
      "_get_kms_client",
      lambda kms_key_name: FailedMockKmsClient(),
  )

  # Encrypted dict representation of a credential
  encrypted_data = {
      "auth_type": "apiKey",
      "api_key": "kms:some_ciphertext_base64",
  }

  # Loading this via SessionStateCredentialService should return None instead of crashing the runner
  service = SessionStateCredentialService()

  class MockConfig:
    credential_key = "test_cred_key"

  # Create a mock callback context with the encrypted state
  callback_ctx = Mock()
  callback_ctx.state = {"test_cred_key": encrypted_data}

  import asyncio

  res = asyncio.run(service.load_credential(MockConfig(), callback_ctx))
  assert res is None
