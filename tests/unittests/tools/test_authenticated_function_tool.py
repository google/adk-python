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

import inspect
from unittest.mock import AsyncMock
from unittest.mock import Mock

from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_tool import AuthConfig
from google.adk.tools.authenticated_function_tool import AuthenticatedFunctionTool
from google.adk.tools.tool_context import ToolContext
import pytest

# Test functions for different scenarios


def sync_function_no_credential(arg1: str, arg2: int) -> str:
  """Test sync function without credential parameter."""
  return f"sync_result_{arg1}_{arg2}"


async def async_function_no_credential(arg1: str, arg2: int) -> str:
  """Test async function without credential parameter."""
  return f"async_result_{arg1}_{arg2}"


def sync_function_with_credential(arg1: str, credential: AuthCredential) -> str:
  """Test sync function with credential parameter."""
  return f"sync_cred_result_{arg1}_{credential.auth_type.value}"


async def async_function_with_credential(
    arg1: str, credential: AuthCredential
) -> str:
  """Test async function with credential parameter."""
  return f"async_cred_result_{arg1}_{credential.auth_type.value}"


def sync_function_with_tool_context(
    arg1: str, tool_context: ToolContext
) -> str:
  """Test sync function with tool_context parameter."""
  return f"sync_context_result_{arg1}"


async def async_function_with_both(
    arg1: str, tool_context: ToolContext, credential: AuthCredential
) -> str:
  """Test async function with both tool_context and credential parameters."""
  return f"async_both_result_{arg1}_{credential.auth_type.value}"


def function_with_optional_args(
    arg1: str, arg2: str = "default", credential: AuthCredential = None
) -> str:
  """Test function with optional arguments."""
  cred_type = credential.auth_type.value if credential else "none"
  return f"optional_result_{arg1}_{arg2}_{cred_type}"


class MockCallable:
  """Test callable class for testing."""

  def __init__(self):
    self.__name__ = "MockCallable"
    self.__doc__ = "Test callable documentation"

  def __call__(self, arg1: str, credential: AuthCredential) -> str:
    return f"callable_result_{arg1}_{credential.auth_type.value}"


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
  # Create a mock auth_type that returns the expected value
  mock_auth_type = Mock()
  mock_auth_type.value = "oauth2"
  credential.auth_type = mock_auth_type
  return credential


class TestAuthenticatedFunctionTool:
  """Test suite for AuthenticatedFunctionTool."""

  def test_init_with_sync_function(self):
    """Test initialization with synchronous function."""
    auth_config = _create_mock_auth_config()

    tool = AuthenticatedFunctionTool(
        func=sync_function_no_credential,
        auth_config=auth_config,
        response_for_auth_required="Please authenticate",
    )

    assert tool.name == "sync_function_no_credential"
    assert (
        tool.description == "Test sync function without credential parameter."
    )
    assert tool.func == sync_function_no_credential
    assert tool._credentials_manager is not None
    assert tool._response_for_auth_required == "Please authenticate"
    assert "credential" in tool._ignore_params

  def test_init_with_async_function(self):
    """Test initialization with asynchronous function."""
    auth_config = _create_mock_auth_config()

    tool = AuthenticatedFunctionTool(
        func=async_function_no_credential, auth_config=auth_config
    )

    assert tool.name == "async_function_no_credential"
    assert (
        tool.description == "Test async function without credential parameter."
    )
    assert tool.func == async_function_no_credential
    assert tool._response_for_auth_required is None

  def test_init_with_callable(self):
    """Test initialization with callable object."""
    auth_config = _create_mock_auth_config()
    test_callable = MockCallable()

    tool = AuthenticatedFunctionTool(
        func=test_callable, auth_config=auth_config
    )

    assert tool.name == "MockCallable"
    assert tool.description == "Test callable documentation"
    assert tool.func == test_callable

  def test_init_no_auth_config(self):
    """Test initialization without auth_config."""
    tool = AuthenticatedFunctionTool(func=sync_function_no_credential)

    assert tool._credentials_manager is None
    assert tool._response_for_auth_required is None

  def test_init_with_empty_auth_scheme(self):
    """Test initialization with auth_config but no auth_scheme."""
    auth_config = Mock(spec=AuthConfig)
    auth_config.auth_scheme = None

    tool = AuthenticatedFunctionTool(
        func=sync_function_no_credential, auth_config=auth_config
    )

    assert tool._credentials_manager is None

  @pytest.mark.asyncio
  async def test_run_async_sync_function_no_credential_manager(self):
    """Test run_async with sync function when no credential manager is configured."""
    tool = AuthenticatedFunctionTool(func=sync_function_no_credential)
    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test", "arg2": 42}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "sync_result_test_42"

  @pytest.mark.asyncio
  async def test_run_async_async_function_no_credential_manager(self):
    """Test run_async with async function when no credential manager is configured."""
    tool = AuthenticatedFunctionTool(func=async_function_no_credential)
    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test", "arg2": 42}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "async_result_test_42"

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

    tool = AuthenticatedFunctionTool(
        func=sync_function_with_credential, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == f"sync_cred_result_test_{credential.auth_type.value}"
    mock_credentials_manager.get_auth_credential.assert_called_once_with(
        tool_context
    )

  @pytest.mark.asyncio
  async def test_run_async_async_function_with_credential(self):
    """Test run_async with async function that expects credential."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=async_function_with_credential, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == f"async_cred_result_test_{credential.auth_type.value}"

  @pytest.mark.asyncio
  async def test_run_async_no_credential_available(self):
    """Test run_async when no credential is available."""
    auth_config = _create_mock_auth_config()

    # Mock the credentials manager to return None
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(return_value=None)
    mock_credentials_manager.request_credential = AsyncMock()

    tool = AuthenticatedFunctionTool(
        func=sync_function_with_credential,
        auth_config=auth_config,
        response_for_auth_required="Custom auth required",
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "Custom auth required"
    mock_credentials_manager.get_auth_credential.assert_called_once_with(
        tool_context
    )
    mock_credentials_manager.request_credential.assert_called_once_with(
        tool_context
    )

  @pytest.mark.asyncio
  async def test_run_async_no_credential_default_message(self):
    """Test run_async when no credential is available with default message."""
    auth_config = _create_mock_auth_config()

    # Mock the credentials manager to return None
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(return_value=None)
    mock_credentials_manager.request_credential = AsyncMock()

    tool = AuthenticatedFunctionTool(
        func=sync_function_with_credential, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "Pending User Authorization."

  @pytest.mark.asyncio
  async def test_run_async_function_without_credential_param(self):
    """Test run_async with function that doesn't have credential parameter."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=sync_function_no_credential, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test", "arg2": 42}

    result = await tool.run_async(args=args, tool_context=tool_context)

    # Credential should not be passed to function since it doesn't have the parameter
    assert result == "sync_result_test_42"

  @pytest.mark.asyncio
  async def test_run_async_function_with_tool_context(self):
    """Test run_async with function that has tool_context parameter."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=sync_function_with_tool_context, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "sync_context_result_test"

  @pytest.mark.asyncio
  async def test_run_async_function_with_both_params(self):
    """Test run_async with function that has both tool_context and credential parameters."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=async_function_with_both, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == f"async_both_result_test_{credential.auth_type.value}"

  @pytest.mark.asyncio
  async def test_run_async_function_with_optional_credential(self):
    """Test run_async with function that has optional credential parameter."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=function_with_optional_args, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert (
        result == f"optional_result_test_default_{credential.auth_type.value}"
    )

  @pytest.mark.asyncio
  async def test_run_async_callable_object(self):
    """Test run_async with callable object."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()
    test_callable = MockCallable()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=test_callable, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == f"callable_result_test_{credential.auth_type.value}"

  @pytest.mark.asyncio
  async def test_run_async_propagates_function_exception(self):
    """Test that run_async propagates exceptions from the wrapped function."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    def failing_function(arg1: str, credential: AuthCredential) -> str:
      raise ValueError("Function failed")

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    tool = AuthenticatedFunctionTool(
        func=failing_function, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    with pytest.raises(ValueError, match="Function failed"):
      await tool.run_async(args=args, tool_context=tool_context)

  @pytest.mark.asyncio
  async def test_run_async_missing_required_args(self):
    """Test run_async with missing required arguments."""
    tool = AuthenticatedFunctionTool(func=sync_function_no_credential)
    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}  # Missing arg2

    result = await tool.run_async(args=args, tool_context=tool_context)

    # Should return error dict indicating missing parameters
    assert isinstance(result, dict)
    assert "error" in result
    assert "arg2" in result["error"]

  @pytest.mark.asyncio
  async def test_run_async_credentials_manager_exception(self):
    """Test run_async when credentials manager raises an exception."""
    auth_config = _create_mock_auth_config()

    # Mock the credentials manager to raise an exception
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        side_effect=RuntimeError("Credential service error")
    )

    tool = AuthenticatedFunctionTool(
        func=sync_function_with_credential, auth_config=auth_config
    )
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    with pytest.raises(RuntimeError, match="Credential service error"):
      await tool.run_async(args=args, tool_context=tool_context)

  def test_credential_in_ignore_params(self):
    """Test that 'credential' is added to ignore_params during initialization."""
    tool = AuthenticatedFunctionTool(func=sync_function_with_credential)

    assert "credential" in tool._ignore_params

  @pytest.mark.asyncio
  async def test_run_async_with_none_credential(self):
    """Test run_async when credential is None but function expects it."""
    tool = AuthenticatedFunctionTool(func=function_with_optional_args)
    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "optional_result_test_default_none"

  def test_signature_inspection(self):
    """Test that the tool correctly inspects function signatures."""
    tool = AuthenticatedFunctionTool(func=sync_function_with_credential)

    signature = inspect.signature(tool.func)
    assert "credential" in signature.parameters
    assert "arg1" in signature.parameters

  @pytest.mark.asyncio
  async def test_args_to_call_modification(self):
    """Test that args_to_call is properly modified with credential."""
    auth_config = _create_mock_auth_config()
    credential = _create_mock_auth_credential()

    # Mock the credentials manager
    mock_credentials_manager = AsyncMock()
    mock_credentials_manager.get_auth_credential = AsyncMock(
        return_value=credential
    )

    # Create a spy function to check what arguments are passed
    original_args = {}

    def spy_function(arg1: str, credential: AuthCredential) -> str:
      nonlocal original_args
      original_args = {"arg1": arg1, "credential": credential}
      return "spy_result"

    tool = AuthenticatedFunctionTool(func=spy_function, auth_config=auth_config)
    tool._credentials_manager = mock_credentials_manager

    tool_context = Mock(spec=ToolContext)
    args = {"arg1": "test"}

    result = await tool.run_async(args=args, tool_context=tool_context)

    assert result == "spy_result"
    assert original_args is not None
    assert original_args["arg1"] == "test"
    assert original_args["credential"] == credential
