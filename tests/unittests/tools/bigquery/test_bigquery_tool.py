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


from unittest.mock import Mock
from unittest.mock import patch

from google.adk.tools import ToolContext
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsConfig
from google.adk.tools.bigquery.bigquery_credentials import BigQueryCredentialsManager
from google.adk.tools.bigquery.bigquery_tool import BigQueryTool
# Mock the Google OAuth and API dependencies
from google.oauth2.credentials import Credentials
import pytest


class TestBigQueryTool:
  """Test suite for BigQueryTool OAuth integration and execution.

  This class tests the high-level tool execution logic that combines
  credential management with actual function execution.
  """

  @pytest.fixture
  def mock_tool_context(self):
    """Create a mock ToolContext for testing tool execution."""
    context = Mock(spec=ToolContext)
    context.get_auth_response = Mock(return_value=None)
    context.request_credential = Mock()
    return context

  @pytest.fixture
  def sample_function(self):
    """Create a sample function that accepts credentials for testing.

    This simulates a real Google API tool function that needs
    authenticated credentials to perform its work.
    """

    def sample_func(param1: str, credentials: Credentials = None) -> dict:
      """Sample function that uses Google API credentials."""
      if credentials:
        return {"result": f"Success with {param1}", "authenticated": True}
      else:
        return {"result": f"Success with {param1}", "authenticated": False}

    return sample_func

  @pytest.fixture
  def async_sample_function(self):
    """Create an async sample function for testing async execution paths."""

    async def async_sample_func(
        param1: str, credentials: Credentials = None
    ) -> dict:
      """Async sample function that uses Google API credentials."""
      if credentials:
        return {"result": f"Async success with {param1}", "authenticated": True}
      else:
        return {
            "result": f"Async success with {param1}",
            "authenticated": False,
        }

    return async_sample_func

  @pytest.fixture
  def credentials_config(self):
    """Create credentials configuration for testing."""
    return BigQueryCredentialsConfig(
        client_id="test_client_id",
        client_secret="test_client_secret",
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )

  def test_tool_initialization_with_credentials(
      self, sample_function, credentials_config
  ):
    """Test that BigQueryTool initializes correctly with credentials.

    The tool should properly inherit from FunctionTool while adding
    Google API specific credential management capabilities.
    """
    tool = BigQueryTool(func=sample_function, credentials=credentials_config)

    assert tool.func == sample_function
    assert tool.credentials_manager is not None
    assert isinstance(tool.credentials_manager, BigQueryCredentialsManager)
    # Verify that 'credentials' parameter is ignored in function signature analysis
    assert "credentials" in tool._ignore_params

  def test_tool_initialization_without_credentials(self, sample_function):
    """Test tool initialization when no credential management is needed.

    Some tools might handle authentication externally or use service
    accounts, so credential management should be optional.
    """
    tool = BigQueryTool(func=sample_function, credentials=None)

    assert tool.func == sample_function
    assert tool.credentials_manager is None

  @pytest.mark.asyncio
  async def test_run_async_with_valid_credentials(
      self, sample_function, credentials_config, mock_tool_context
  ):
    """Test successful tool execution with valid credentials.

    This tests the main happy path where credentials are available
    and the underlying function executes successfully.
    """
    tool = BigQueryTool(func=sample_function, credentials=credentials_config)

    # Mock the credentials manager to return valid credentials
    mock_creds = Mock(spec=Credentials)
    with patch.object(
        tool.credentials_manager,
        "get_valid_credentials",
        return_value=mock_creds,
    ) as mock_get_creds:

      result = await tool.run_async(
          args={"param1": "test_value"}, tool_context=mock_tool_context
      )

      mock_get_creds.assert_called_once_with(mock_tool_context)
      assert result["result"] == "Success with test_value"
      assert result["authenticated"] is True

  @pytest.mark.asyncio
  async def test_run_async_oauth_flow_in_progress(
      self, sample_function, credentials_config, mock_tool_context
  ):
    """Test tool behavior when OAuth flow is in progress.

    When credentials aren't available and OAuth flow is needed,
    the tool should return a user-friendly message rather than failing.
    """
    tool = BigQueryTool(func=sample_function, credentials=credentials_config)

    # Mock credentials manager to return None (OAuth flow in progress)
    with patch.object(
        tool.credentials_manager, "get_valid_credentials", return_value=None
    ) as mock_get_creds:

      result = await tool.run_async(
          args={"param1": "test_value"}, tool_context=mock_tool_context
      )

      mock_get_creds.assert_called_once_with(mock_tool_context)
      assert "authorization is required" in result.lower()
      assert tool.name in result

  @pytest.mark.asyncio
  async def test_run_async_without_credentials_manager(
      self, sample_function, mock_tool_context
  ):
    """Test tool execution when no credential management is configured.

    Tools without credential managers should execute normally,
    passing None for credentials if the function accepts them.
    """
    tool = BigQueryTool(func=sample_function, credentials=None)

    result = await tool.run_async(
        args={"param1": "test_value"}, tool_context=mock_tool_context
    )

    assert result["result"] == "Success with test_value"
    assert result["authenticated"] is False

  @pytest.mark.asyncio
  async def test_run_async_with_async_function(
      self, async_sample_function, credentials_config, mock_tool_context
  ):
    """Test that async functions are properly handled.

    The tool should correctly detect and execute async functions,
    which is important for tools that make async API calls.
    """
    tool = BigQueryTool(
        func=async_sample_function, credentials=credentials_config
    )

    mock_creds = Mock(spec=Credentials)
    with patch.object(
        tool.credentials_manager,
        "get_valid_credentials",
        return_value=mock_creds,
    ):

      result = await tool.run_async(
          args={"param1": "test_value"}, tool_context=mock_tool_context
      )

      assert result["result"] == "Async success with test_value"
      assert result["authenticated"] is True

  @pytest.mark.asyncio
  async def test_run_async_exception_handling(
      self, credentials_config, mock_tool_context
  ):
    """Test that exceptions in tool execution are properly handled.

    Tools should gracefully handle errors and return structured
    error responses rather than letting exceptions propagate.
    """

    def failing_function(param1: str, credentials: Credentials = None) -> dict:
      raise ValueError("Something went wrong")

    tool = BigQueryTool(func=failing_function, credentials=credentials_config)

    mock_creds = Mock(spec=Credentials)
    with patch.object(
        tool.credentials_manager,
        "get_valid_credentials",
        return_value=mock_creds,
    ):

      result = await tool.run_async(
          args={"param1": "test_value"}, tool_context=mock_tool_context
      )

      assert result["status"] == "ERROR"
      assert "Something went wrong" in result["error_details"]

  def test_function_signature_analysis(self, credentials_config):
    """Test that function signature analysis correctly handles credentials parameter.

    The tool should properly identify and handle the credentials parameter
    while preserving other parameter analysis for LLM function calling.
    """

    def complex_function(
        required_param: str,
        optional_param: str = "default",
        credentials: Credentials = None,
    ) -> dict:
      return {"success": True}

    tool = BigQueryTool(func=complex_function, credentials=credentials_config)

    # The 'credentials' parameter should be ignored in mandatory args analysis
    mandatory_args = tool._get_mandatory_args()
    assert "required_param" in mandatory_args
    assert "credentials" not in mandatory_args
    assert "optional_param" not in mandatory_args
