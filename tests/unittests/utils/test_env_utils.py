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

from google.adk.utils.env_utils import is_env_enabled
import pytest


@pytest.mark.parametrize(
    'env_value,expected',
    [
        ('true', True),
        ('TRUE', True),
        ('TrUe', True),
        ('1', True),
        ('false', False),
        ('FALSE', False),
        ('0', False),
        ('', False),
    ],
)
def test_is_env_enabled(monkeypatch, env_value, expected):
  """Test is_env_enabled with various environment variable values."""
  monkeypatch.setenv('TEST_FLAG', env_value)
  assert is_env_enabled('TEST_FLAG') is expected


@pytest.mark.parametrize(
    'default,expected',
    [
        ('0', False),
        ('1', True),
        ('true', True),
    ],
)
def test_is_env_enabled_with_defaults(monkeypatch, default, expected):
  """Test is_env_enabled when env var is not set with different defaults."""
  monkeypatch.delenv('TEST_FLAG', raising=False)
  assert is_env_enabled('TEST_FLAG', default=default) is expected
