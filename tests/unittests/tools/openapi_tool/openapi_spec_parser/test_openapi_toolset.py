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

import os
from typing import Any
from typing import Dict

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import MediaType
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import ParameterInType
from fastapi.openapi.models import SecuritySchemeType
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
import pytest
import yaml


def load_spec(file_path: str) -> Dict:
  """Loads the OpenAPI specification from a YAML file."""
  with open(file_path, "r", encoding="utf-8") as f:
    return yaml.safe_load(f)


@pytest.fixture
def openapi_spec() -> Dict:
  """Fixture to load the OpenAPI specification."""
  current_dir = os.path.dirname(os.path.abspath(__file__))
  # Join the directory path with the filename
  yaml_path = os.path.join(current_dir, "test.yaml")
  return load_spec(yaml_path)


def test_openapi_toolset_initialization_from_dict(openapi_spec: Dict):
  """Test initialization of OpenAPIToolset with a dictionary."""
  toolset = OpenAPIToolset(spec_dict=openapi_spec)
  assert isinstance(toolset._tools, list)
  assert len(toolset._tools) == 5
  assert all(isinstance(tool, RestApiTool) for tool in toolset._tools)


def test_openapi_toolset_initialization_from_yaml_string(openapi_spec: Dict):
  """Test initialization of OpenAPIToolset with a YAML string."""
  spec_str = yaml.dump(openapi_spec)
  toolset = OpenAPIToolset(spec_str=spec_str, spec_str_type="yaml")
  assert isinstance(toolset._tools, list)
  assert len(toolset._tools) == 5
  assert all(isinstance(tool, RestApiTool) for tool in toolset._tools)


def test_openapi_toolset_tool_existing(openapi_spec: Dict):
  """Test the tool() method for an existing tool."""
  toolset = OpenAPIToolset(spec_dict=openapi_spec)
  tool_name = "calendar_calendars_insert"  # Example operationId from the spec
  tool = toolset.get_tool(tool_name)
  assert isinstance(tool, RestApiTool)
  assert tool.name == tool_name
  assert tool.description == "Creates a secondary calendar."
  assert tool.endpoint.method == "post"
  assert tool.endpoint.base_url == "https://www.googleapis.com/calendar/v3"
  assert tool.endpoint.path == "/calendars"
  assert tool.is_long_running is False
  assert tool.operation.operationId == "calendar.calendars.insert"
  assert tool.operation.description == "Creates a secondary calendar."
  assert isinstance(
      tool.operation.requestBody.content["application/json"], MediaType
  )
  assert len(tool.operation.responses) == 1
  response = tool.operation.responses["200"]
  assert response.description == "Successful response"
  assert isinstance(response.content["application/json"], MediaType)
  assert isinstance(tool.auth_scheme, OAuth2)

  tool_name = "calendar_calendars_get"
  tool = toolset.get_tool(tool_name)
  assert isinstance(tool, RestApiTool)
  assert tool.name == tool_name
  assert tool.description == "Returns metadata for a calendar."
  assert tool.endpoint.method == "get"
  assert tool.endpoint.base_url == "https://www.googleapis.com/calendar/v3"
  assert tool.endpoint.path == "/calendars/{calendarId}"
  assert tool.is_long_running is False
  assert tool.operation.operationId == "calendar.calendars.get"
  assert tool.operation.description == "Returns metadata for a calendar."
  assert len(tool.operation.parameters) == 8
  assert tool.operation.parameters[0].name == "calendarId"
  assert tool.operation.parameters[0].in_ == ParameterInType.path
  assert tool.operation.parameters[0].required is True
  assert tool.operation.parameters[0].schema_.type == "string"
  assert (
      tool.operation.parameters[0].description
      == "Calendar identifier. To retrieve calendar IDs call the"
      " calendarList.list method. If you want to access the primary calendar"
      ' of the currently logged in user, use the "primary" keyword.'
  )
  assert isinstance(tool.auth_scheme, OAuth2)

  assert isinstance(toolset.get_tool("calendar_calendars_update"), RestApiTool)
  assert isinstance(toolset.get_tool("calendar_calendars_delete"), RestApiTool)
  assert isinstance(toolset.get_tool("calendar_calendars_patch"), RestApiTool)


def test_openapi_toolset_tool_non_existing(openapi_spec: Dict):
  """Test the tool() method for a non-existing tool."""
  toolset = OpenAPIToolset(spec_dict=openapi_spec)
  tool = toolset.get_tool("non_existent_tool")
  assert tool is None


def test_openapi_toolset_configure_auth_on_init(openapi_spec: Dict):
  """Test configuring auth during initialization."""

  auth_scheme = APIKey(**{
      "in": APIKeyIn.header,  # Use alias name in dict
      "name": "api_key",
      "type": SecuritySchemeType.http,
  })
  auth_credential = AuthCredential(auth_type=AuthCredentialTypes.API_KEY)
  toolset = OpenAPIToolset(
      spec_dict=openapi_spec,
      auth_scheme=auth_scheme,
      auth_credential=auth_credential,
  )
  assert all(tool.auth_scheme == auth_scheme for tool in toolset._tools)
  assert all(tool.auth_credential == auth_credential for tool in toolset._tools)


@pytest.mark.parametrize(
    "verify_value", ["/path/to/enterprise-ca-bundle.crt", False]
)
def test_openapi_toolset_verify_on_init(
    openapi_spec: Dict[str, Any], verify_value: str | bool
):
  """Test configuring verify during initialization."""
  toolset = OpenAPIToolset(
      spec_dict=openapi_spec,
      ssl_verify=verify_value,
  )
  assert all(tool._ssl_verify == verify_value for tool in toolset._tools)


def test_openapi_toolset_configure_verify_all(openapi_spec: Dict[str, Any]):
  """Test configure_verify_all method."""
  toolset = OpenAPIToolset(spec_dict=openapi_spec)

  # Initially verify should be None
  assert all(tool._ssl_verify is None for tool in toolset._tools)

  # Configure verify for all tools
  ca_bundle_path = "/path/to/custom-ca.crt"
  toolset.configure_ssl_verify_all(ca_bundle_path)

  assert all(tool._ssl_verify == ca_bundle_path for tool in toolset._tools)


async def test_openapi_toolset_tool_name_prefix(openapi_spec: Dict[str, Any]):
  """Test tool_name_prefix parameter prefixes tool names."""
  prefix = "my_api"
  toolset = OpenAPIToolset(spec_dict=openapi_spec, tool_name_prefix=prefix)

  # Verify the toolset has the prefix set
  assert toolset.tool_name_prefix == prefix

  prefixed_tools = await toolset.get_tools_with_prefix()
  assert len(prefixed_tools) == 5

  # Verify all tool names are prefixed
  assert all(tool.name.startswith(f"{prefix}_") for tool in prefixed_tools)

  # Verify specific tool name is prefixed
  expected_prefixed_name = "my_api_calendar_calendars_insert"
  prefixed_tool_names = [t.name for t in prefixed_tools]
  assert expected_prefixed_name in prefixed_tool_names


def test_openapi_toolset_header_provider(openapi_spec: Dict[str, Any]):
  """Test header_provider parameter is passed to tools."""

  def my_header_provider(context):
    return {"X-Custom-Header": "custom-value", "X-Request-ID": "12345"}

  toolset = OpenAPIToolset(
      spec_dict=openapi_spec,
      header_provider=my_header_provider,
  )

  # Verify the toolset has the header_provider set
  assert toolset._header_provider is my_header_provider

  # Verify all tools have the header_provider
  assert all(
      tool._header_provider is my_header_provider for tool in toolset._tools
  )


def test_openapi_toolset_header_provider_none_by_default(
    openapi_spec: Dict[str, Any],
):
  """Test that header_provider is None by default."""
  toolset = OpenAPIToolset(spec_dict=openapi_spec)

  # Verify the toolset has no header_provider by default
  assert toolset._header_provider is None

  # Verify all tools have no header_provider
  assert all(tool._header_provider is None for tool in toolset._tools)
