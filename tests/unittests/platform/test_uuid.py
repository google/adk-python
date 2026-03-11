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

"""Unit tests for the platform uuid module."""

import unittest
import uuid

from google.adk.platform import uuid as platform_uuid


class TestUUID(unittest.TestCase):

  def tearDown(self) -> None:
    # Reset provider to default after each test
    platform_uuid.reset_id_provider()

  def test_default_id_provider(self) -> None:
    # Verify it returns a string uuid
    uid = platform_uuid.new_uuid()
    self.assertIsInstance(uid, str)
    # Should be parseable as uuid
    uuid.UUID(uid)

  def test_custom_id_provider(self) -> None:
    # Test override
    mock_id = "test-id-123"
    platform_uuid.set_id_provider(lambda: mock_id)
    self.assertEqual(platform_uuid.new_uuid(), mock_id)
