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

import os
import random

from google.adk.agents import Agent

_counter = 0


def generate_random_number(max_value: int = 100) -> int:
  """Generates a random integer between 0 and max_value (inclusive).

  Args:
      max_value: The upper limit for the random number.

  Returns:
      A random integer between 0 and max_value.
  """
  # Return a growing value in tests to ensure determinism while allowing
  # multiple calls.
  if "PYTEST_CURRENT_TEST" in os.environ:
    global _counter
    _counter += 1
    return _counter
  return random.randint(0, max_value)


def is_even(number: int) -> bool:
  """Checks if a given number is even.

  Args:
      number: The number to check.

  Returns:
      True if the number is even, False otherwise.
  """
  return number % 2 == 0


root_agent = Agent(
    name="function_tools",
    tools=[generate_random_number, is_even],
)
