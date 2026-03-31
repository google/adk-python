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

"""Unit tests for the platform time module."""

import time
import unittest

from google.adk.platform import time as platform_time


class TestTime(unittest.TestCase):

  def tearDown(self) -> None:
    # Reset provider to default after each test
    platform_time.reset_time_provider()

  def test_default_time_provider(self) -> None:
    # Verify it returns a float that is close to now
    now = time.time()
    rt_time = platform_time.get_time()
    self.assertIsInstance(rt_time, float)
    self.assertAlmostEqual(rt_time, now, delta=1.0)

  def test_custom_time_provider(self) -> None:
    # Test override
    mock_time = 123456789.0
    platform_time.set_time_provider(lambda: mock_time)
    self.assertEqual(platform_time.get_time(), mock_time)
