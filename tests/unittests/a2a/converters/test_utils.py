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

from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.a2a.converters.utils import ADK_METADATA_KEY_PREFIX
import pytest


class TestUtilsFunctions:
  """Test suite for utils module functions."""

  def test_get_adk_metadata_key_success(self):
    """Test successful metadata key generation."""
    key = "test_key"
    result = _get_adk_metadata_key(key)
    assert result == f"{ADK_METADATA_KEY_PREFIX}{key}"

  def test_get_adk_metadata_key_empty_string(self):
    """Test metadata key generation with empty string."""
    with pytest.raises(
        ValueError, match="Metadata key cannot be empty or None"
    ):
      _get_adk_metadata_key("")

  def test_get_adk_metadata_key_none(self):
    """Test metadata key generation with None."""
    with pytest.raises(
        ValueError, match="Metadata key cannot be empty or None"
    ):
      _get_adk_metadata_key(None)

  def test_get_adk_metadata_key_whitespace(self):
    """Test metadata key generation with whitespace string."""
    key = "   "
    result = _get_adk_metadata_key(key)
    assert result == f"{ADK_METADATA_KEY_PREFIX}{key}"
