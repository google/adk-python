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

import os
import pickle
import unittest
from unittest import mock

from google.adk.utils import serialization_utils


class TestPickleRCE(unittest.TestCase):

  def test_secure_serialization_roundtrip(self):
    """Verifies that an object can be securely serialized and deserialized."""
    test_obj = {"key": "value", "list": [1, 2, 3]}
    signed_data = serialization_utils.secure_dumps(test_obj)

    # Verification
    decoded_obj = serialization_utils.secure_loads(signed_data)
    self.assertEqual(test_obj, decoded_obj)

  def test_secure_loads_rejects_unsigned_data(self):
    """Verifies that legacy unsigned pickle data is rejected."""
    raw_pickle = pickle.dumps({"malicious": "payload"})

    with self.assertRaises(serialization_utils.SecurityError) as cm:
      serialization_utils.secure_loads(raw_pickle)

    self.assertIn("Invalid signature detected", str(cm.exception))

  def test_secure_loads_rejects_tampered_data(self):
    """Verifies that tampered data (wrong signature) is rejected."""
    test_obj = "safe_data"
    signed_data = serialization_utils.secure_dumps(test_obj)

    # Tamper with the data (change the payload but keep the signature length)
    tampered_data = signed_data[:32] + b"tampered_payload"

    with self.assertRaises(serialization_utils.SecurityError):
      serialization_utils.secure_loads(tampered_data)

  def test_secure_loads_rejects_wrong_key(self):
    """Verifies that data signed with a different key is rejected."""
    test_obj = "secret_data"

    with mock.patch.dict(os.environ, {"ADK_SECURITY_SECRET": "key_one"}):
      signed_data = serialization_utils.secure_dumps(test_obj)

    with mock.patch.dict(os.environ, {"ADK_SECURITY_SECRET": "key_two"}):
      with self.assertRaises(serialization_utils.SecurityError):
        serialization_utils.secure_loads(signed_data)

  def test_secure_loads_too_short(self):
    """Verifies that data shorter than the HMAC signature is rejected."""
    with self.assertRaises(serialization_utils.SecurityError) as cm:
      serialization_utils.secure_loads(b"short")
    self.assertEqual(
        str(cm.exception), "Data too short to contain a valid signature."
    )


if __name__ == "__main__":
  unittest.main()
