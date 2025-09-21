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
from typing import Optional
from unittest import mock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.google_api_tool.google_api_tool import GoogleApiTool
from google.adk.tools.google_api_tool.google_api_toolset import GoogleApiToolset
from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import GoogleApiToOpenApiConverter
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
import pytest

TEST_API_NAME = "calendar"
TEST_API_VERSION = "v3"
DEFAULT_SCOPE = "https://www.googleapis.com/auth/calendar"


@pytest.fixture
def mock_rest_api_tool():
  """Fixture for a mock RestApiTool."""
  mock_tool = mock.MagicMock(spec=RestApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  return mock_tool


@pytest.fixture
def mock_google_api_tool_instance(
    mock_rest_api_tool,
):  # Renamed from mock_google_api_tool
  """Fixture for a mock GoogleApiTool instance."""
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
    mock_tool = mock.MagicMock(
        spec=RestApiTool, description=f"Test Tool Description {i}"
    )
    mock_tool.name = f"test_tool_{i}"
    tools.append(mock_tool)
  return tools


@pytest.fixture
def mock_openapi_toolset_instance():  # Renamed from mock_openapi_toolset
  """Fixture for a mock OpenAPIToolset instance."""
  mock_toolset = mock.MagicMock(spec=OpenAPIToolset)
  # Mock async methods if they are called
  mock_toolset.get_tools = mock.AsyncMock(return_value=[])
  mock_toolset.close = mock.AsyncMock()
  return mock_toolset


@pytest.fixture
def mock_converter_instance():  # Renamed from mock_converter
  """Fixture for a mock GoogleApiToOpenApiConverter instance."""
  mock_conv = mock.MagicMock(spec=GoogleApiToOpenApiConverter)
  mock_conv.convert.return_value = {
      "components": {
          "securitySchemes": {
              "oauth2": {
                  "flows": {
                      "authorizationCode": {
                          "scopes": {
                              DEFAULT_SCOPE: "Full access to Google Calendar"
                          }
                      }
                  }
              }
          }
      }
  }
  return mock_conv


@pytest.fixture
def mock_readonly_context():
  """Fixture for a mock ReadonlyContext."""
  return mock.MagicMock(spec=ReadonlyContext)


class TestGoogleApiToolset:
  """Test suite for the GoogleApiToolset class."""

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    client_id = "test_client_id"
    client_secret = "test_client_secret"

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id=client_id,
        client_secret=client_secret,
    )

    assert tool_set.api_name == TEST_API_NAME
    assert tool_set.api_version == TEST_API_VERSION
    assert tool_set._client_id == client_id
    assert tool_set._client_secret == client_secret
    assert tool_set._service_account is None
    assert tool_set.tool_filter is None
    assert tool_set._openapi_toolset == mock_openapi_toolset_instance

    mock_converter_class.assert_called_once_with(
        TEST_API_NAME, TEST_API_VERSION
    )
    mock_converter_instance.convert.assert_called_once()
    spec_dict = mock_converter_instance.convert.return_value

    mock_openapi_toolset_class.assert_called_once()
    _, kwargs = mock_openapi_toolset_class.call_args
    assert kwargs["spec_dict"] == spec_dict
    assert kwargs["spec_str_type"] == "yaml"
    assert isinstance(kwargs["auth_scheme"], OpenIdConnectWithConfig)
    assert kwargs["auth_scheme"].scopes == [DEFAULT_SCOPE]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiTool"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_google_api_tool_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,
      mock_readonly_context,
  ):
    """Test get_tools method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    # Setup mock GoogleApiTool instances to be returned by the constructor
    mock_google_api_tool_instances = [
        mock.MagicMock(spec=GoogleApiTool, name=f"google_tool_{i}")
        for i in range(len(mock_rest_api_tools))
    ]
    mock_google_api_tool_class.side_effect = mock_google_api_tool_instances

    client_id = "cid"
    client_secret = "csecret"
    sa_mock = mock.MagicMock(spec=ServiceAccount)

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id=client_id,
        client_secret=client_secret,
        service_account=sa_mock,
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == len(mock_rest_api_tools)
    mock_openapi_toolset_instance.get_tools.assert_called_once_with(
        mock_readonly_context
    )

    for i, rest_tool in enumerate(mock_rest_api_tools):
      mock_google_api_tool_class.assert_any_call(
          rest_tool, client_id, client_secret, sa_mock
      )
      assert tools[i] is mock_google_api_tool_instances[i]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools_with_filter_list(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,  # Has test_tool_0, test_tool_1, test_tool_2
      mock_readonly_context,
      mock_converter_instance,
  ):
    """Test get_tools method with a list filter."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    tool_filter = ["test_tool_0", "test_tool_2"]
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_filter=tool_filter,
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == 2
    assert tools[0].name == "test_tool_0"
    assert tools[1].name == "test_tool_2"

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools_with_filter_predicate(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,  # Has test_tool_0, test_tool_1, test_tool_2
      mock_readonly_context,
  ):
    """Test get_tools method with a predicate filter."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    class MyPredicate(ToolPredicate):

      def __call__(
          self,
          tool: BaseTool,
          readonly_context: Optional[ReadonlyContext] = None,
      ) -> bool:
        return tool.name == "test_tool_1"

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_filter=MyPredicate(),
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == 1
    assert tools[0].name == "test_tool_1"

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_configure_auth(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test configure_auth method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    client_id = "test_client_id"
    client_secret = "test_client_secret"

    tool_set.configure_auth(client_id, client_secret)

    assert tool_set._client_id == client_id
    assert tool_set._client_secret == client_secret

    # To verify its effect, we would ideally call get_tools and check
    # how GoogleApiTool is instantiated. This is covered in test_get_tools.

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_configure_sa_auth(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test configure_sa_auth method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    service_account = ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type="service_account",
            project_id="project_id",
            private_key_id="private_key_id",
            private_key=(
                "-----BEGIN PRIVATE KEY-----\nprivate_key\n-----END PRIVATE"
                " KEY-----\n"
            ),
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
    assert tool_set._service_account == service_account
    # Effect verification is covered in test_get_tools.

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_close(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test close method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    await tool_set.close()

    mock_openapi_toolset_instance.close.assert_called_once()

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_set_tool_filter(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test set_tool_filter method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )

    assert tool_set.tool_filter is None

    new_filter_list = ["tool1", "tool2"]
    tool_set.set_tool_filter(new_filter_list)
    assert tool_set.tool_filter == new_filter_list

    def new_filter_predicate(
        tool_name: str,
        tool: RestApiTool,
        readonly_context: Optional[ReadonlyContext] = None,
    ) -> bool:
      return True

    tool_set.set_tool_filter(new_filter_predicate)
    assert tool_set.tool_filter == new_filter_predicate

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init_with_tool_name_prefix(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization with tool_name_prefix."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_name_prefix = "test_prefix"
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_name_prefix=tool_name_prefix,
    )

    assert tool_set.tool_name_prefix == tool_name_prefix
