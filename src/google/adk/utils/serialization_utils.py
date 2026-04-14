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

"""Utilities for secure serialization and deserialization."""

import hashlib
import hmac
import os
import pickle
from typing import Any


class SecurityError(Exception):
  """Raised when a security validation fails during deserialization."""

  pass


def _get_secret_key() -> bytes:
  """Retrieves the secret key used for HMAC signing.

  In a production environment, this should be fetched from a secure secret
  manager or KMS. For this remediation, we default to an environment variable.
  """
  secret = os.environ.get("ADK_SECURITY_SECRET")
  if not secret:
    # Fallback for demonstration/local development only.
    # WARNING: This should be replaced with mandatory secret fetching in prod.
    return b"default_insecure_development_secret"
  return secret.encode("utf-8")


def secure_dumps(obj: Any) -> bytes:
  """Serializes an object using pickle and appends an HMAC signature.

  Args:
    obj: The Python object to serialize.

  Returns:
    The signed binary blob.
  """
  serialized = pickle.dumps(obj)
  key = _get_secret_key()
  signature = hmac.new(key, serialized, hashlib.sha256).digest()
  return signature + serialized


def secure_loads(data: bytes) -> Any:
  """Verifies the HMAC signature and deserializes a binary blob.

  Args:
    data: The signed binary blob.

  Returns:
    The deserialized Python object.

  Raises:
    SecurityError: If the signature is invalid or missing.
  """
  if len(data) < 32:
    raise SecurityError("Data too short to contain a valid signature.")

  signature = data[:32]
  serialized = data[32:]

  key = _get_secret_key()
  expected_signature = hmac.new(key, serialized, hashlib.sha256).digest()

  if not hmac.compare_digest(signature, expected_signature):
    raise SecurityError(
        "Invalid signature detected during deserialization. "
        "The data may have been tampered with or originated from an "
        "untrusted source."
    )

  return pickle.loads(serialized)
