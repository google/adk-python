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

from unittest import mock

import pytest

from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.tools.google_api_tool.google_api_tool import GoogleApiTool
from google.adk.tools.google_api_tool.google_api_tool_set import GoogleApiToolSet
from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import GoogleApiToOpenApiConverter
from google.adk.tools.openapi_tool import OpenAPIToolset
from google.adk.tools.openapi_tool import RestApiTool


@pytest.fixture
def mock_rest_api_tool():
  """Fixture for a mock RestApiTool."""
  mock_tool = mock.MagicMock(spec=RestApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  return mock_tool


@pytest.fixture
def mock_google_api_tool(mock_rest_api_tool):
  """Fixture for a mock GoogleApiTool."""
  mock_tool = mock.MagicMock(spec=GoogleApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  mock_tool.rest_api_tool = mock_rest_api_tool
  return mock_tool


@pytest.fixture
def mock_rest_api_tools():
  """Fixture for a list of mock RestApiTools."""
  tools = []
  for i in range(3):
    mock_tool = mock.MagicMock(spec=RestApiTool)
    mock_tool.name = f"test_tool_{i}"
    mock_tool.description = f"Test Tool Description {i}"
    tools.append(mock_tool)
  return tools


@pytest.fixture
def mock_openapi_toolset():
  """Fixture for a mock OpenAPIToolset."""
  mock_toolset = mock.MagicMock(spec=OpenAPIToolset)
  return mock_toolset


@pytest.fixture
def mock_converter():
  """Fixture for a mock GoogleApiToOpenApiConverter."""
  mock_conv = mock.MagicMock(spec=GoogleApiToOpenApiConverter)
  mock_conv.convert.return_value = {
      "components": {
          "securitySchemes": {
              "oauth2": {
                  "flows": {
                      "authorizationCode": {
                          "scopes": {
                              "https://www.googleapis.com/auth/calendar": (
                                  "Full access to Google Calendar"
                              )
                          }
                      }
                  }
              }
          }
      }
  }
  return mock_conv


class TestGoogleApiToolSet:
  """Test suite for the GoogleApiToolSet class."""

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiTool"
  )
  def test_init(self, mock_google_api_tool_class, mock_rest_api_tools):
    """Test GoogleApiToolSet initialization."""
    # Setup mock GoogleApiTool instances
    mock_google_api_tools = []
    for i in range(len(mock_rest_api_tools)):
      mock_tool = mock.MagicMock()
      mock_tool.name = f"test_tool_{i}"
      mock_google_api_tools.append(mock_tool)

    # Make the GoogleApiTool constructor return our mock instances
    mock_google_api_tool_class.side_effect = mock_google_api_tools

    tool_set = GoogleApiToolSet(mock_rest_api_tools)

    assert len(tool_set.tools) == 3
    for i, tool in enumerate(tool_set.tools):
      assert tool.name == f"test_tool_{i}"
      # Verify GoogleApiTool was called with the correct RestApiTool
      mock_google_api_tool_class.assert_any_call(mock_rest_api_tools[i])

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiTool"
  )
  def test_get_tools(self, mock_google_api_tool_class, mock_rest_api_tools):
    """Test get_tools method."""
    # Setup mock GoogleApiTool instances
    mock_google_api_tools = []
    for i in range(len(mock_rest_api_tools)):
      mock_tool = mock.MagicMock()
      mock_tool.name = f"test_tool_{i}"
      mock_google_api_tools.append(mock_tool)

    # Make the GoogleApiTool constructor return our mock instances
    mock_google_api_tool_class.side_effect = mock_google_api_tools

    tool_set = GoogleApiToolSet(mock_rest_api_tools)
    tools = tool_set.get_tools()

    assert len(tools) == 3
    for i, tool in enumerate(tools):
      assert tool.name == f"test_tool_{i}"
      assert tool is mock_google_api_tools[i]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiTool"
  )
  def test_get_tool(self, mock_google_api_tool_class, mock_rest_api_tools):
    """Test get_tool method."""
    # Setup mock GoogleApiTool instances
    mock_google_api_tools = []
    for i in range(len(mock_rest_api_tools)):
      mock_tool = mock.MagicMock()
      mock_tool.name = f"test_tool_{i}"
      mock_google_api_tools.append(mock_tool)

    # Make the GoogleApiTool constructor return our mock instances
    mock_google_api_tool_class.side_effect = mock_google_api_tools

    tool_set = GoogleApiToolSet(mock_rest_api_tools)

    # Test getting an existing tool
    tool = tool_set.get_tool("test_tool_1")
    assert tool is not None
    assert tool is mock_google_api_tools[1]
    assert tool.name == "test_tool_1"

    # Test getting a non-existent tool
    tool = tool_set.get_tool("non_existent_tool")
    assert tool is None

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.OpenAPIToolset"
  )
  def test_load_tool_set_with_oidc_auth_spec_dict(
      self, mock_openapi_toolset_class
  ):
    """Test _load_tool_set_with_oidc_auth method with spec_dict."""
    # Setup mock
    mock_toolset_instance = mock.MagicMock()
    mock_openapi_toolset_class.return_value = mock_toolset_instance

    # Call method
    spec_dict = {"openapi": "3.0.0"}
    scopes = ["https://www.googleapis.com/auth/calendar"]
    result = GoogleApiToolSet._load_tool_set_with_oidc_auth(
        spec_dict=spec_dict, scopes=scopes
    )

    # Verify
    mock_openapi_toolset_class.assert_called_once()
    _, kwargs = mock_openapi_toolset_class.call_args
    assert kwargs["spec_dict"] == spec_dict
    assert kwargs["spec_str"] is None
    assert kwargs["spec_str_type"] == "yaml"
    assert "auth_scheme" in kwargs
    assert kwargs["auth_scheme"].scopes == scopes
    assert result == mock_toolset_instance

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.os.path.dirname"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.os.path.abspath"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.os.path.join"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.open",
      new_callable=mock.mock_open,
      read_data="test yaml content",
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.inspect.stack"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.OpenAPIToolset"
  )
  def test_load_tool_set_with_oidc_auth_spec_file(
      self,
      mock_openapi_toolset_class,
      mock_stack,
      mock_open,
      mock_join,
      mock_abspath,
      mock_dirname,
  ):
    """Test _load_tool_set_with_oidc_auth method with spec_file."""
    # Setup mocks
    mock_frame = mock.MagicMock()
    mock_frame.filename = "/path/to/caller.py"
    mock_stack.return_value = [None, mock_frame]
    mock_dirname.return_value = "/path/to"
    mock_abspath.return_value = "/path/to/caller.py"
    mock_join.return_value = "/path/to/spec.yaml"

    mock_toolset_instance = mock.MagicMock()
    mock_openapi_toolset_class.return_value = mock_toolset_instance

    # Call method
    spec_file = "spec.yaml"
    scopes = ["https://www.googleapis.com/auth/calendar"]
    result = GoogleApiToolSet._load_tool_set_with_oidc_auth(
        spec_file=spec_file, scopes=scopes
    )

    # Verify
    mock_open.assert_called_once_with(
        "/path/to/spec.yaml", "r", encoding="utf-8"
    )
    mock_openapi_toolset_class.assert_called_once()
    _, kwargs = mock_openapi_toolset_class.call_args
    assert kwargs["spec_dict"] is None
    assert kwargs["spec_str"] == "test yaml content"
    assert kwargs["spec_str_type"] == "yaml"
    assert "auth_scheme" in kwargs
    assert kwargs["auth_scheme"].scopes == scopes
    assert result == mock_toolset_instance

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiTool"
  )
  def test_configure_auth(
      self, mock_google_api_tool_class, mock_rest_api_tools
  ):
    """Test configure_auth method."""
    # Setup mock GoogleApiTool instances
    mock_google_api_tools = []
    for i in range(len(mock_rest_api_tools)):
      mock_tool = mock.MagicMock()
      mock_tool.name = f"test_tool_{i}"
      mock_google_api_tools.append(mock_tool)

    # Make the GoogleApiTool constructor return our mock instances
    mock_google_api_tool_class.side_effect = mock_google_api_tools

    tool_set = GoogleApiToolSet(mock_rest_api_tools)
    client_id = "test_client_id"
    client_secret = "test_client_secret"

    tool_set.configure_auth(client_id, client_secret)

    # Verify each tool had configure_auth called
    for tool in mock_google_api_tools:
      tool.configure_auth.assert_called_once_with(client_id, client_secret)

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiTool"
  )
  def test_configure_sa_auth(
      self, mock_google_api_tool_class, mock_rest_api_tools
  ):
    """Test configure_sa_auth method."""
    # Setup mock GoogleApiTool instances
    mock_google_api_tools = []
    for i in range(len(mock_rest_api_tools)):
      mock_tool = mock.MagicMock()
      mock_tool.name = f"test_tool_{i}"
      mock_google_api_tools.append(mock_tool)

    # Make the GoogleApiTool constructor return our mock instances
    mock_google_api_tool_class.side_effect = mock_google_api_tools

    tool_set = GoogleApiToolSet(mock_rest_api_tools)
    service_account = ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type="service_account",
            project_id="project_id",
            private_key_id="private_key_id",
            private_key="private_key",
            client_email="client_email",
            client_id="client_id",
            auth_uri="auth_uri",
            token_uri="token_uri",
            auth_provider_x509_cert_url="auth_provider_x509_cert_url",
            client_x509_cert_url="client_x509_cert_url",
            universe_domain="universe_domain",
        ),
        scopes=["scope1", "scope2"],
    )

    tool_set.configure_sa_auth(service_account)

    # Verify each tool had configure_sa_auth called
    for tool in mock_google_api_tools:
      tool.configure_sa_auth.assert_called_once_with(service_account)

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiToOpenApiConverter"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_tool_set.GoogleApiToolSet._load_tool_set_with_oidc_auth"
  )
  def test_load_tool_set(
      self, mock_load_tool_set, mock_converter_class, mock_rest_api_tools
  ):
    """Test load_tool_set class method."""
    # Setup mocks
    mock_converter_instance = mock.MagicMock()
    mock_converter_instance.convert.return_value = {
        "components": {
            "securitySchemes": {
                "oauth2": {
                    "flows": {
                        "authorizationCode": {
                            "scopes": {
                                "https://www.googleapis.com/auth/calendar": (
                                    "Full access to Google Calendar"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
    mock_converter_class.return_value = mock_converter_instance

    mock_toolset = mock.MagicMock()
    mock_toolset.get_tools.return_value = mock_rest_api_tools
    mock_load_tool_set.return_value = mock_toolset

    # Call method
    api_name = "calendar"
    api_version = "v3"
    result = GoogleApiToolSet.load_tool_set(api_name, api_version)

    # Verify
    mock_converter_class.assert_called_once_with(api_name, api_version)
    mock_converter_instance.convert.assert_called_once()

    spec_dict = mock_converter_instance.convert.return_value
    scope = "https://www.googleapis.com/auth/calendar"
    mock_load_tool_set.assert_called_once_with(
        spec_dict=spec_dict, scopes=[scope]
    )

    assert isinstance(result, GoogleApiToolSet)
