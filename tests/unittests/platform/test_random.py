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

"""Unit tests for the platform random module."""

import random
import unittest

from google.adk import platform as adk_platform


class TestRandom(unittest.TestCase):

  def tearDown(self) -> None:
    # Reset provider to default after each test
    adk_platform.reset_random_provider()

  def test_default_random_provider(self) -> None:
    # Verify it returns a random.Random instance producing values in range
    rng = adk_platform.get_random()
    self.assertIsInstance(rng, random.Random)
    value = rng.uniform(0.0, 1.0)
    self.assertGreaterEqual(value, 0.0)
    self.assertLessEqual(value, 1.0)

  def test_custom_random_provider(self) -> None:
    # Test override
    seeded = random.Random(42)
    adk_platform.set_random_provider(lambda: seeded)
    self.assertIs(adk_platform.get_random(), seeded)
    expected = random.Random(42).uniform(0.0, 1.0)
    self.assertEqual(adk_platform.get_random().uniform(0.0, 1.0), expected)

  def test_reset_random_provider(self) -> None:
    seeded = random.Random(42)
    adk_platform.set_random_provider(lambda: seeded)
    adk_platform.reset_random_provider()
    self.assertIsNot(adk_platform.get_random(), seeded)
    # Default provider returns a stable module-level instance
    self.assertIs(adk_platform.get_random(), adk_platform.get_random())
