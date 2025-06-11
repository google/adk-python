# Copyright 2025 Google LLC
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

import pytest
from google.adk.utils.instructions_utils import _is_valid_state_name


@pytest.mark.parametrize(
    "var_name,expected",
    [
        ("valid_name", True),
        ("app:valid_name", True),
        ("user:valid_name", True),
        ("temp:valid_name", True),
        ("invalid-name", False),
        ("app:invalid-name", False),
        ("app::name", False),
        ("app:", False),
        (":name", False),
        ("", False),
        ("123_name", False),  # Corrected: should be False as it starts with a digit
        ("app:123_name", False),  # Corrected: should be False as it starts with a digit
        ("app:valid name", False),  # Corrected: should be False because of space
        ("app:valid.name", False),  # Corrected: should be False because of dot
        ("long_valid_name_with_numbers_123", True),
        ("app:long_valid_name_with_numbers_123", True),
        ("app:valid_name_!@#", False),  # Corrected: should be False because of special characters
        ("app:valid_name_with_underscores", True), # Added a new test case for clarity
        ("app:valid-name-with-hyphens", False), # Added a new test case for clarity
        ("custom:valid_name", False),  # custom prefix is not valid
    ],
)
def test_is_valid_state_name(var_name, expected):
  assert _is_valid_state_name(var_name) == expected


