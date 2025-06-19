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

import os
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
  for tool in toolset._tools:
    assert tool.auth_scheme == auth_scheme
    assert tool.auth_credential == auth_credential
