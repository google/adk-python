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

from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.base_authenticated_tool import BaseAuthenticatedTool
from google.adk.tools.tool_context import ToolContext
import pytest


class _TestAuthenticatedTool(BaseAuthenticatedTool):
  """Test implementation of BaseAuthenticatedTool for testing purposes."""

  def __init__(
      self,
      name="test_auth_tool",
      description="Test authenticated tool",
      auth_config=None,
      unauthenticated_response=None,
  ):
    super().__init__(
        name=name,
        description=description,
        auth_config=auth_config,
        response_for_auth_required=unauthenticated_response,
    )
    self.run_impl_called = False
    self.run_impl_result = "test_result"

  async def _run_async_impl(self, *, args, tool_context, credential):
    """Test implementation of the abstract method."""
    self.run_impl_called = True
    self.last_args = args
    self.last_tool_context = tool_context
    self.last_credential = credential
    return self.run_impl_result


def _create_mock_auth_config():
  """Creates a mock AuthConfig with proper structure."""
  auth_scheme = Mock(spec=AuthScheme)
  auth_scheme.type_ = AuthSchemeType.oauth2

  auth_config = Mock(spec=AuthConfig)
  auth_config.auth_scheme = auth_scheme

  return auth_config


def _create_mock_auth_credential():
  """Creates a mock AuthCredential."""
  credential = Mock(spec=AuthCredential)
  credential.auth_type = AuthCredentialTypes.OAUTH2
  return credential


class TestBaseAuthenticatedTool:
  """Test suite for BaseAuthenticatedTool."""

  def test_init_with_auth_config(self):
    """Test initialization with auth_config."""
    auth_config = _create_mock_auth_config()
    unauthenticated_response = {"error": "Not authenticated"}

    tool = _TestAuthenticatedTool(
        name="test_tool",
        description="Test description",
        auth_config=auth_config,
        unauthenticated_response=unauthenticated_response,
    )

    assert tool.name == "test_tool"
    assert tool.description == "Test description"
    assert tool._credentials_manager is not None
    assert tool._response_for_auth_required == unauthenticated_response

  def test_init_with_no_auth_config(self):
    """Test initialization without auth_config."""
    tool = _TestAuthenticatedTool()

    assert tool.name == "test_auth_tool"
    assert tool.description == "Test authenticated tool"
    assert tool._credentials_manager is None
    assert tool._response_for_auth_required is None

  def test_init_with_empty_auth_scheme(self):
    """Test initialization with auth_config but no auth_scheme."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = None

    tool = _TestAuthenticatedTool(auth_config=auth_config)

    assert tool._credentials_manager is None

  def test_init_with_default_unauthenticated_response(self):
    """Test initialization with default unauthenticated response."""
    auth_config = _create_mock_auth_config()

    tool = _TestAuthenticatedTool(auth_config=auth_config)

    assert tool._response_for_auth_required is None

  @pytest.mark.asyncio
  async def test_run_async_no_credentials_manager(self):
    """Test run_async when no credentials manager is configured."""
    tool = _TestAuthenticatedTool()
    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "test_result"
    assert tool.run_impl_called
    assert tool.last_args == args
    assert tool.last_tool_context == tool_context
    assert tool.last_credential is None

  @pytest.mark.asyncio
  async def test_run_async_with_valid_credential(self):
    """Test run_async when valid credential is available."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = _TestAuthenticatedTool(auth_config=auth_config)
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "test_result"
    assert tool.run_impl_called
    assert tool.last_args == args
    assert tool.last_tool_context == tool_context
    assert tool.last_credential == credential
    mock_credentials_manager.get_auth_credential.assert_called_once_with(
        tool_context
    )

  @pytest.mark.asyncio
  async def test_run_async_no_credential_available(self):
    """Test run_async when no credential is available."""
    auth_config = _create_mock_auth_config()

    # Mock the credentials manager to return None
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(return_value=None)
    mock_credentials_manager.request_credential = AsyncMock()

    tool = _TestAuthenticatedTool(auth_config=auth_config)
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "Pending User Authorization."
    assert not tool.run_impl_called
    mock_credentials_manager.get_auth_credential.assert_called_once_with(
        tool_context
    )
    mock_credentials_manager.request_credential.assert_called_once_with(
        tool_context
    )

  @pytest.mark.asyncio
  async def test_run_async_no_credential_with_custom_response(self):
    """Test run_async when no credential is available with custom response."""
    auth_config = _create_mock_auth_config()
    custom_response = {
        "status": "authentication_required",
        "message": "Please login",
    }

    # Mock the credentials manager to return None
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(return_value=None)
    mock_credentials_manager.request_credential = AsyncMock()

    tool = _TestAuthenticatedTool(
        auth_config=auth_config, unauthenticated_response=custom_response
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == custom_response
    assert not tool.run_impl_called
    mock_credentials_manager.get_auth_credential.assert_called_once_with(
        tool_context
    )
    mock_credentials_manager.request_credential.assert_called_once_with(
        tool_context
    )

  @pytest.mark.asyncio
  async def test_run_async_no_credential_with_string_response(self):
    """Test run_async when no credential is available with string response."""
    auth_config = _create_mock_auth_config()
    custom_response = "Custom authentication required message"

    # Mock the credentials manager to return None
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(return_value=None)
    mock_credentials_manager.request_credential = AsyncMock()

    tool = _TestAuthenticatedTool(
        auth_config=auth_config, unauthenticated_response=custom_response
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == custom_response
    assert not tool.run_impl_called

  @pytest.mark.asyncio
  async def test_run_async_propagates_impl_exception(self):
    """Test that run_async propagates exceptions from _run_async_impl."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = _TestAuthenticatedTool(auth_config=auth_config)
    tool._credentials_manager = mock_credentials_manager

    # Make the implementation raise an exception
    async def failing_impl(*, args, tool_context, credential):
      raise ValueError("Implementation failed")

    tool._run_async_impl = failing_impl

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    with pytest.raises(ValueError, match="Implementation failed"):
      await tool.run_async(args=args, tool_context=tool_context)

  @pytest.mark.asyncio
  async def test_run_async_with_different_args_types(self):
    """Test run_async with different argument types."""
    tool = _TestAuthenticatedTool()
    tool_context = Mock(spec=ToolContext)

    # Test with empty args
    result = await tool.run_async(args={}, tool_context=tool_context)
    assert result == "test_result"
    assert tool.last_args == {}

    # Test with complex args
    complex_args = {
        "string_param": "test",
        "number_param": 42,
        "list_param": [1, 2, 3],
        "dict_param": {"nested": "value"},
    }
    result = await tool.run_async(args=complex_args, tool_context=tool_context)
    assert result == "test_result"
    assert tool.last_args == complex_args

  @pytest.mark.asyncio
  async def test_run_async_credentials_manager_exception(self):
    """Test run_async when credentials manager raises an exception."""
    auth_config = _create_mock_auth_config()

    # Mock the credentials manager to raise an exception
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        side_effect=RuntimeError("Credential service error")
    )

    tool = _TestAuthenticatedTool(auth_config=auth_config)
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"param1": "value1"}

    with pytest.raises(RuntimeError, match="Credential service error"):
      await tool.run_async(args=args, tool_context=tool_context)

  def test_abstract_nature(self):
    """Test that BaseAuthenticatedTool cannot be instantiated directly."""
    with pytest.raises(TypeError):
      # This should fail because _run_async_impl is abstract
      BaseAuthenticatedTool(name="test", description="test")

  @pytest.mark.asyncio
  async def test_run_async_return_values(self):
    """Test run_async with different return value types."""
    tool = _TestAuthenticatedTool()
    tool_context = Mock(spec=ToolContext)
    args = {}

    # Test with None return
    tool.run_impl_result = None
    result = await tool.run_async(args=args, tool_context=tool_context)
    assert result is None

    # Test with dict return
    tool.run_impl_result = {"key": "value"}
    result = await tool.run_async(args=args, tool_context=tool_context)
    assert result == {"key": "value"}

    # Test with list return
    tool.run_impl_result = [1, 2, 3]
    result = await tool.run_async(args=args, tool_context=tool_context)
    assert result == [1, 2, 3]
