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

import pytest

from google.adk.integrations.redis._redis_session_service import _escape_redis_glob


@pytest.mark.parametrize(
    "value, expected",
    [
        # plain value is unchanged
        ("alice", "alice"),
        # a bare wildcard must not stay a wildcard (this is the injection)
        ("*", r"\*"),
        ("?", r"\?"),
        ("bob*", r"bob\*"),
        # character classes are neutralized
        ("[a-z]", r"\[a-z\]"),
        # backslash is escaped first so it cannot form an escape sequence
        ("\\", "\\\\"),
        # empty input round-trips
        ("", ""),
    ],
)
def test_escape_redis_glob_neutralizes_metacharacters(value, expected):
  assert _escape_redis_glob(value) == expected
