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
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A tool requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.server.agent_execution import RequestContext
  from google.adk.a2a.converters.request_converter import _get_user_id
  from google.adk.a2a.converters.request_converter import convert_a2a_request_to_adk_run_args
  from google.adk.runners import RunConfig
  from google.genai import types as genai_types
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    a2a_types = DummyTypes()
    genai_types = DummyTypes()
    RequestContext = DummyTypes()
    RunConfig = DummyTypes()
    _get_user_id = lambda x: None
    convert_a2a_request_to_adk_run_args = lambda x: None
  else:
    raise e


class TestGetUserId:
  """Test cases for _get_user_id function."""

  def test_get_user_id_from_call_context(self):
    """Test getting user ID from call context when auth is enabled."""
    # Arrange
    mock_user = Mock()
    mock_user.user_name = "authenticated_user"

    mock_call_context = Mock()
    mock_call_context.user = mock_user

    request = Mock(spec=RequestContext)
    request.call_context = mock_call_context
    request.context_id = "test_context"

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "authenticated_user"

  def test_get_user_id_from_context_when_no_call_context(self):
    """Test getting user ID from context when call context is not available."""
    # Arrange
    request = Mock(spec=RequestContext)
    request.call_context = None
    request.context_id = "test_context"

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "A2A_USER_test_context"

  def test_get_user_id_from_context_when_call_context_has_no_user(self):
    """Test getting user ID from context when call context has no user."""
    # Arrange
    mock_call_context = Mock()
    mock_call_context.user = None

    request = Mock(spec=RequestContext)
    request.call_context = mock_call_context
    request.context_id = "test_context"

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "A2A_USER_test_context"

  def test_get_user_id_with_empty_user_name(self):
    """Test getting user ID when user exists but user_name is empty."""
    # Arrange
    mock_user = Mock()
    mock_user.user_name = ""

    mock_call_context = Mock()
    mock_call_context.user = mock_user

    request = Mock(spec=RequestContext)
    request.call_context = mock_call_context
    request.context_id = "test_context"

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "A2A_USER_test_context"

  def test_get_user_id_with_none_user_name(self):
    """Test getting user ID when user exists but user_name is None."""
    # Arrange
    mock_user = Mock()
    mock_user.user_name = None

    mock_call_context = Mock()
    mock_call_context.user = mock_user

    request = Mock(spec=RequestContext)
    request.call_context = mock_call_context
    request.context_id = "test_context"

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "A2A_USER_test_context"

  def test_get_user_id_with_none_context_id(self):
    """Test getting user ID when context_id is None."""
    # Arrange
    request = Mock(spec=RequestContext)
    request.call_context = None
    request.context_id = None

    # Act
    result = _get_user_id(request)

    # Assert
    assert result == "A2A_USER_None"


class TestConvertA2aRequestToAdkRunArgs:
  """Test cases for convert_a2a_request_to_adk_run_args function."""

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_request_basic(self, mock_convert_part):
    """Test basic conversion of A2A request to ADK run args."""
    # Arrange
    mock_part1 = Mock()
    mock_part2 = Mock()

    mock_message = Mock()
    mock_message.parts = [mock_part1, mock_part2]

    mock_user = Mock()
    mock_user.user_name = "test_user"

    mock_call_context = Mock()
    mock_call_context.user = mock_user

    request = Mock(spec=RequestContext)
    request.message = mock_message
    request.context_id = "test_context_123"
    request.call_context = mock_call_context

    # Create proper genai_types.Part objects instead of mocks
    mock_genai_part1 = genai_types.Part(text="test part 1")
    mock_genai_part2 = genai_types.Part(text="test part 2")
    mock_convert_part.side_effect = [mock_genai_part1, mock_genai_part2]

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert result["user_id"] == "test_user"
    assert result["session_id"] == "test_context_123"
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == [mock_genai_part1, mock_genai_part2]
    assert isinstance(result["run_config"], RunConfig)

    # Verify calls
    assert mock_convert_part.call_count == 2
    mock_convert_part.assert_any_call(mock_part1)
    mock_convert_part.assert_any_call(mock_part2)

  def test_convert_a2a_request_no_message_raises_error(self):
    """Test that conversion raises ValueError when message is None."""
    # Arrange
    request = Mock(spec=RequestContext)
    request.message = None

    # Act & Assert
    with pytest.raises(ValueError, match="Request message cannot be None"):
      convert_a2a_request_to_adk_run_args(request)

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_request_empty_parts(self, mock_convert_part):
    """Test conversion with empty parts list."""
    # Arrange
    mock_message = Mock()
    mock_message.parts = []

    request = Mock(spec=RequestContext)
    request.message = mock_message
    request.context_id = "test_context_123"
    request.call_context = None

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert result["user_id"] == "A2A_USER_test_context_123"
    assert result["session_id"] == "test_context_123"
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == []
    assert isinstance(result["run_config"], RunConfig)

    # Verify convert_part wasn't called
    mock_convert_part.assert_not_called()

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_request_none_context_id(self, mock_convert_part):
    """Test conversion when context_id is None."""
    # Arrange
    mock_part = Mock()
    mock_message = Mock()
    mock_message.parts = [mock_part]

    request = Mock(spec=RequestContext)
    request.message = mock_message
    request.context_id = None
    request.call_context = None

    # Create proper genai_types.Part object instead of mock
    mock_genai_part = genai_types.Part(text="test part")
    mock_convert_part.return_value = mock_genai_part

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert result["user_id"] == "A2A_USER_None"
    assert result["session_id"] is None
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == [mock_genai_part]
    assert isinstance(result["run_config"], RunConfig)

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_convert_a2a_request_no_auth(self, mock_convert_part):
    """Test conversion when no authentication is available."""
    # Arrange
    mock_part = Mock()
    mock_message = Mock()
    mock_message.parts = [mock_part]

    request = Mock(spec=RequestContext)
    request.message = mock_message
    request.context_id = "session_123"
    request.call_context = None

    # Create proper genai_types.Part object instead of mock
    mock_genai_part = genai_types.Part(text="test part")
    mock_convert_part.return_value = mock_genai_part

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert result["user_id"] == "A2A_USER_session_123"
    assert result["session_id"] == "session_123"
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == [mock_genai_part]
    assert isinstance(result["run_config"], RunConfig)


class TestIntegration:
  """Integration test cases combining both functions."""

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_end_to_end_conversion_with_auth_user(self, mock_convert_part):
    """Test end-to-end conversion with authenticated user."""
    # Arrange
    mock_user = Mock()
    mock_user.user_name = "auth_user"

    mock_call_context = Mock()
    mock_call_context.user = mock_user

    mock_part = Mock()
    mock_message = Mock()
    mock_message.parts = [mock_part]

    request = Mock(spec=RequestContext)
    request.call_context = mock_call_context
    request.message = mock_message
    request.context_id = "mysession"

    # Create proper genai_types.Part object instead of mock
    mock_genai_part = genai_types.Part(text="test part")
    mock_convert_part.return_value = mock_genai_part

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert result["user_id"] == "auth_user"  # Should use authenticated user
    assert result["session_id"] == "mysession"
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == [mock_genai_part]
    assert isinstance(result["run_config"], RunConfig)

  @patch(
      "google.adk.a2a.converters.request_converter.convert_a2a_part_to_genai_part"
  )
  def test_end_to_end_conversion_with_fallback_user(self, mock_convert_part):
    """Test end-to-end conversion with fallback user ID."""
    # Arrange
    mock_part = Mock()
    mock_message = Mock()
    mock_message.parts = [mock_part]

    request = Mock(spec=RequestContext)
    request.call_context = None
    request.message = mock_message
    request.context_id = "test_session_456"

    # Create proper genai_types.Part object instead of mock
    mock_genai_part = genai_types.Part(text="test part")
    mock_convert_part.return_value = mock_genai_part

    # Act
    result = convert_a2a_request_to_adk_run_args(request)

    # Assert
    assert result is not None
    assert (
        result["user_id"] == "A2A_USER_test_session_456"
    )  # Should fallback to context ID
    assert result["session_id"] == "test_session_456"
    assert isinstance(result["new_message"], genai_types.Content)
    assert result["new_message"].role == "user"
    assert result["new_message"].parts == [mock_genai_part]
    assert isinstance(result["run_config"], RunConfig)
