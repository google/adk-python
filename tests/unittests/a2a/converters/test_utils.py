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

import sys

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

from google.adk.a2a.converters.utils import _from_a2a_context_id
from google.adk.a2a.converters.utils import _get_adk_metadata_key
from google.adk.a2a.converters.utils import _to_a2a_context_id
from google.adk.a2a.converters.utils import ADK_CONTEXT_ID_PREFIX
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

  def test_to_a2a_context_id_success(self):
    """Test successful context ID generation."""
    app_name = "test-app"
    user_id = "test-user"
    session_id = "test-session"

    result = _to_a2a_context_id(app_name, user_id, session_id)

    expected = f"{ADK_CONTEXT_ID_PREFIX}/test-app/test-user/test-session"
    assert result == expected

  def test_to_a2a_context_id_empty_app_name(self):
    """Test context ID generation with empty app name."""
    with pytest.raises(
        ValueError,
        match=(
            "All parameters \\(app_name, user_id, session_id\\) must be"
            " non-empty"
        ),
    ):
      _to_a2a_context_id("", "user", "session")

  def test_to_a2a_context_id_empty_user_id(self):
    """Test context ID generation with empty user ID."""
    with pytest.raises(
        ValueError,
        match=(
            "All parameters \\(app_name, user_id, session_id\\) must be"
            " non-empty"
        ),
    ):
      _to_a2a_context_id("app", "", "session")

  def test_to_a2a_context_id_empty_session_id(self):
    """Test context ID generation with empty session ID."""
    with pytest.raises(
        ValueError,
        match=(
            "All parameters \\(app_name, user_id, session_id\\) must be"
            " non-empty"
        ),
    ):
      _to_a2a_context_id("app", "user", "")

  def test_to_a2a_context_id_none_values(self):
    """Test context ID generation with None values."""
    with pytest.raises(
        ValueError,
        match=(
            "All parameters \\(app_name, user_id, session_id\\) must be"
            " non-empty"
        ),
    ):
      _to_a2a_context_id(None, "user", "session")

  def test_to_a2a_context_id_special_characters(self):
    """Test context ID generation with special characters."""
    app_name = "test-app@2024"
    user_id = "user_123"
    session_id = "session-456"

    result = _to_a2a_context_id(app_name, user_id, session_id)

    expected = f"{ADK_CONTEXT_ID_PREFIX}/test-app@2024/user_123/session-456"
    assert result == expected

  def test_from_a2a_context_id_success(self):
    """Test successful context ID parsing."""
    context_id = f"{ADK_CONTEXT_ID_PREFIX}/test-app/test-user/test-session"

    app_name, user_id, session_id = _from_a2a_context_id(context_id)

    assert app_name == "test-app"
    assert user_id == "test-user"
    assert session_id == "test-session"

  def test_from_a2a_context_id_none_input(self):
    """Test context ID parsing with None input."""
    result = _from_a2a_context_id(None)
    assert result == (None, None, None)

  def test_from_a2a_context_id_empty_string(self):
    """Test context ID parsing with empty string."""
    result = _from_a2a_context_id("")
    assert result == (None, None, None)

  def test_from_a2a_context_id_invalid_prefix(self):
    """Test context ID parsing with invalid prefix."""
    context_id = "INVALID/test-app/test-user/test-session"

    result = _from_a2a_context_id(context_id)

    assert result == (None, None, None)

  def test_from_a2a_context_id_too_few_parts(self):
    """Test context ID parsing with too few parts."""
    context_id = f"{ADK_CONTEXT_ID_PREFIX}/test-app/test-user"

    result = _from_a2a_context_id(context_id)

    assert result == (None, None, None)

  def test_from_a2a_context_id_too_many_parts(self):
    """Test context ID parsing with too many parts."""
    context_id = (
        f"{ADK_CONTEXT_ID_PREFIX}/test-app/test-user/test-session/extra"
    )

    result = _from_a2a_context_id(context_id)

    assert result == (None, None, None)

  def test_from_a2a_context_id_empty_components(self):
    """Test context ID parsing with empty components."""
    context_id = f"{ADK_CONTEXT_ID_PREFIX}//test-user/test-session"

    result = _from_a2a_context_id(context_id)

    assert result == (None, None, None)

  def test_from_a2a_context_id_no_dollar_separator(self):
    """Test context ID parsing without dollar separators."""
    context_id = f"{ADK_CONTEXT_ID_PREFIX}-test-app-test-user-test-session"

    result = _from_a2a_context_id(context_id)

    assert result == (None, None, None)

  def test_roundtrip_context_id(self):
    """Test roundtrip conversion: to -> from."""
    app_name = "test-app"
    user_id = "test-user"
    session_id = "test-session"

    # Convert to context ID
    context_id = _to_a2a_context_id(app_name, user_id, session_id)

    # Convert back
    parsed_app, parsed_user, parsed_session = _from_a2a_context_id(context_id)

    assert parsed_app == app_name
    assert parsed_user == user_id
    assert parsed_session == session_id

  def test_from_a2a_context_id_special_characters(self):
    """Test context ID parsing with special characters."""
    context_id = f"{ADK_CONTEXT_ID_PREFIX}/test-app@2024/user_123/session-456"

    app_name, user_id, session_id = _from_a2a_context_id(context_id)

    assert app_name == "test-app@2024"
    assert user_id == "user_123"
    assert session_id == "session-456"
