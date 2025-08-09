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


import json
from unittest import mock

from fastapi.openapi.models import Operation
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.tools.application_integration_tool.application_integration_toolset import ApplicationIntegrationToolset
from google.adk.tools.application_integration_tool.integration_connector_tool import IntegrationConnectorTool
from google.adk.tools.openapi_tool.auth.auth_helpers import dict_to_auth_scheme
from google.adk.tools.openapi_tool.openapi_spec_parser import rest_api_tool
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_spec_parser import OperationEndpoint
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_spec_parser import ParsedOperation
import pytest


@pytest.fixture
def mock_integration_client():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.IntegrationClient"
  ) as mock_client:
    yield mock_client


@pytest.fixture
def mock_connections_client():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.ConnectionsClient"
  ) as mock_client:
    yield mock_client


@pytest.fixture
def mock_openapi_toolset():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.OpenAPIToolset"
  ) as mock_toolset:
    mock_toolset_instance = mock.MagicMock()
    mock_rest_api_tool = mock.MagicMock(spec=rest_api_tool.RestApiTool)
    mock_rest_api_tool.name = "Test Tool"

    # Create an async mock for the get_tools method
    async def mock_get_tools(context: ReadonlyContext = None):
      return [mock_rest_api_tool]

    # Assign the async mock function to get_tools
    mock_toolset_instance.get_tools = mock_get_tools

    mock_toolset.return_value = mock_toolset_instance
    yield mock_toolset


@pytest.fixture
def mock_openapi_toolset_with_multiple_tools_and_no_tools():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.OpenAPIToolset"
  ) as mock_toolset:
    mock_toolset_instance = mock.MagicMock()
    mock_rest_api_tool = mock.MagicMock(spec=rest_api_tool.RestApiTool)
    mock_rest_api_tool.name = "Test Tool"
    mock_rest_api_tool_2 = mock.MagicMock(spec=rest_api_tool.RestApiTool)
    mock_rest_api_tool_2.name = "Test Tool 2"

    # Create an async mock for the get_tools method
    async def mock_get_tools(context: ReadonlyContext = None):
      return [mock_rest_api_tool, mock_rest_api_tool_2]

    mock_toolset_instance.get_tools = mock_get_tools
    mock_toolset.return_value = mock_toolset_instance
    yield mock_toolset


def get_mocked_parsed_operation(operation_id, attributes):
  mock_openapi_spec_parser_instance = mock.MagicMock()
  mock_parsed_operation = mock.MagicMock(spec=ParsedOperation)
  mock_parsed_operation.name = "list_issues"
  mock_parsed_operation.description = "list_issues_description"
  mock_parsed_operation.endpoint = OperationEndpoint(
      base_url="http://localhost:8080",
      path="/v1/issues",
      method="GET",
  )
  mock_parsed_operation.auth_scheme = None
  mock_parsed_operation.auth_credential = None
  mock_parsed_operation.additional_context = {}
  mock_parsed_operation.parameters = []
  mock_operation = mock.MagicMock(spec=Operation)
  mock_operation.operationId = operation_id
  mock_operation.description = "list_issues_description"
  mock_operation.parameters = []
  mock_operation.requestBody = None
  mock_operation.responses = {}
  mock_operation.callbacks = {}
  for key, value in attributes.items():
    setattr(mock_operation, key, value)
  mock_parsed_operation.operation = mock_operation
  mock_openapi_spec_parser_instance.parse.return_value = [mock_parsed_operation]
  return mock_openapi_spec_parser_instance


@pytest.fixture
def mock_openapi_entity_spec_parser():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.OpenApiSpecParser"
  ) as mock_spec_parser:
    mock_openapi_spec_parser_instance = get_mocked_parsed_operation(
        "list_issues", {"x-entity": "Issues", "x-operation": "LIST_ENTITIES"}
    )
    mock_spec_parser.return_value = mock_openapi_spec_parser_instance
    yield mock_spec_parser


@pytest.fixture
def mock_openapi_action_spec_parser():
  with mock.patch(
      "google.adk.tools.application_integration_tool.application_integration_toolset.OpenApiSpecParser"
  ) as mock_spec_parser:
    mock_openapi_action_spec_parser_instance = get_mocked_parsed_operation(
        "list_issues_operation",
        {"x-action": "CustomAction", "x-operation": "EXECUTE_ACTION"},
    )
    mock_spec_parser.return_value = mock_openapi_action_spec_parser_instance
    yield mock_spec_parser


@pytest.fixture
def project():
  return "test-project"


@pytest.fixture
def location():
  return "us-central1"


@pytest.fixture
def integration_spec():
  return {"openapi": "3.0.0", "info": {"title": "Integration API"}}


@pytest.fixture
def connection_spec():
  return {"openapi": "3.0.0", "info": {"title": "Connection API"}}


@pytest.fixture
def connection_details():
  return {
      "serviceName": "test-service",
      "host": "test.host",
      "name": "test-connection",
  }


@pytest.fixture
def connection_details_auth_override_enabled():
  return {
      "serviceName": "test-service",
      "host": "test.host",
      "name": "test-connection",
      "authOverrideEnabled": True,
  }


@pytest.mark.asyncio
async def test_initialization_with_integration_and_trigger(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_toolset,
):
  integration_name = "test-integration"
  triggers = ["test-trigger"]
  toolset = ApplicationIntegrationToolset(
      project, location, integration=integration_name, triggers=triggers
  )
  mock_integration_client.assert_called_once_with(
      project, location, integration_name, triggers, None, None, None, None
  )
  mock_integration_client.return_value.get_openapi_spec_for_integration.assert_called_once()
  mock_connections_client.assert_not_called()
  mock_openapi_toolset.assert_called_once()
  tools = await toolset.get_tools()
  assert len(tools) == 1
  assert tools[0].name == "Test Tool"


@pytest.mark.asyncio
async def test_initialization_with_integration_and_list_of_triggers(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_toolset_with_multiple_tools_and_no_tools,
):
  integration_name = "test-integration"
  triggers = ["test-trigger1", "test-trigger2"]
  toolset = ApplicationIntegrationToolset(
      project, location, integration=integration_name, triggers=triggers
  )
  mock_integration_client.assert_called_once_with(
      project,
      location,
      integration_name,
      triggers,
      None,
      None,
      None,
      None,
  )
  mock_integration_client.return_value.get_openapi_spec_for_integration.assert_called_once()
  mock_connections_client.assert_not_called()
  mock_openapi_toolset_with_multiple_tools_and_no_tools.assert_called_once()
  tools = await toolset.get_tools()
  assert len(tools) == 2
  assert tools[0].name == "Test Tool"
  assert tools[1].name == "Test Tool 2"


@pytest.mark.asyncio
async def test_initialization_with_integration_and_empty_trigger_list(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_toolset_with_multiple_tools_and_no_tools,
):
  integration_name = "test-integration"
  toolset = ApplicationIntegrationToolset(
      project, location, integration=integration_name
  )
  mock_integration_client.assert_called_once_with(
      project, location, integration_name, None, None, None, None, None
  )
  mock_integration_client.return_value.get_openapi_spec_for_integration.assert_called_once()
  mock_connections_client.assert_not_called()
  mock_openapi_toolset_with_multiple_tools_and_no_tools.assert_called_once()
  tools = await toolset.get_tools()
  assert len(tools) == 2
  assert tools[0].name == "Test Tool"
  assert tools[1].name == "Test Tool 2"


@pytest.mark.asyncio
async def test_initialization_with_connection_and_entity_operations(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_entity_spec_parser,
    connection_details,
):
  connection_name = "test-connection"
  entity_operations_list = ["list", "get"]
  tool_name = "My Connection Tool"
  tool_instructions = "Use this tool to manage entities."
  mock_connections_client.return_value.get_connection_details.return_value = (
      connection_details
  )
  toolset = ApplicationIntegrationToolset(
      project,
      location,
      connection=connection_name,
      entity_operations=entity_operations_list,
      tool_name_prefix=tool_name,
      tool_instructions=tool_instructions,
  )
  mock_integration_client.assert_called_once_with(
      project,
      location,
      None,
      None,
      connection_name,
      entity_operations_list,
      None,
      None,
  )
  mock_connections_client.assert_called_once_with(
      project, location, connection_name, None
  )
  mock_openapi_entity_spec_parser.return_value.parse.assert_called_once()
  mock_connections_client.return_value.get_connection_details.assert_called_once()
  mock_integration_client.return_value.get_openapi_spec_for_connection.assert_called_once_with(
      tool_name,
      tool_instructions,
  )

  tools = await toolset.get_tools()
  assert len(tools) == 1
  assert tools[0].name == "list_issues"
  assert isinstance(tools[0], IntegrationConnectorTool)
  assert tools[0]._entity == "Issues"
  assert tools[0]._operation == "LIST_ENTITIES"


@pytest.mark.asyncio
async def test_initialization_with_connection_and_actions(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_action_spec_parser,
    connection_details,
):
  connection_name = "test-connection"
  actions_list = ["create", "delete"]
  tool_name = "My Actions Tool"
  tool_instructions = "Perform actions using this tool."
  mock_connections_client.return_value.get_connection_details.return_value = (
      connection_details
  )
  toolset = ApplicationIntegrationToolset(
      project,
      location,
      connection=connection_name,
      actions=actions_list,
      tool_name_prefix=tool_name,
      tool_instructions=tool_instructions,
  )
  mock_integration_client.assert_called_once_with(
      project, location, None, None, connection_name, None, actions_list, None
  )
  mock_connections_client.assert_called_once_with(
      project, location, connection_name, None
  )
  mock_connections_client.return_value.get_connection_details.assert_called_once()
  mock_integration_client.return_value.get_openapi_spec_for_connection.assert_called_once_with(
      tool_name, tool_instructions
  )
  mock_openapi_action_spec_parser.return_value.parse.assert_called_once()
  tools = await toolset.get_tools()
  assert len(tools) == 1
  assert tools[0].name == "list_issues_operation"
  assert isinstance(tools[0], IntegrationConnectorTool)
  assert tools[0]._action == "CustomAction"
  assert tools[0]._operation == "EXECUTE_ACTION"


def test_initialization_without_required_params(project, location):
  with pytest.raises(
      ValueError,
      match=(
          "Invalid request, Either integration or \\(connection and"
          " \\(entity_operations or actions\\)\\) should be provided."
      ),
  ):
    ApplicationIntegrationToolset(project, location)

  with pytest.raises(
      ValueError,
      match=(
          "Invalid request, Either integration or \\(connection and"
          " \\(entity_operations or actions\\)\\) should be provided."
      ),
  ):
    ApplicationIntegrationToolset(project, location, triggers=["test"])

  with pytest.raises(
      ValueError,
      match=(
          "Invalid request, Either integration or \\(connection and"
          " \\(entity_operations or actions\\)\\) should be provided."
      ),
  ):
    ApplicationIntegrationToolset(project, location, connection="test")


def test_initialization_with_service_account_credentials(
    project, location, mock_integration_client, mock_openapi_toolset
):
  service_account_json = json.dumps({
      "type": "service_account",
      "project_id": "dummy",
      "private_key_id": "dummy",
      "private_key": "dummy",
      "client_email": "test@example.com",
      "client_id": "131331543646416",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": (
          "https://www.googleapis.com/oauth2/v1/certs"
      ),
      "client_x509_cert_url": (
          "http://www.googleapis.com/robot/v1/metadata/x509/dummy%40dummy.com"
      ),
      "universe_domain": "googleapis.com",
  })
  integration_name = "test-integration"
  triggers = ["test-trigger"]
  toolset = ApplicationIntegrationToolset(
      project,
      location,
      integration=integration_name,
      triggers=triggers,
      service_account_json=service_account_json,
  )
  mock_integration_client.assert_called_once_with(
      project,
      location,
      integration_name,
      triggers,
      None,
      None,
      None,
      service_account_json,
  )
  mock_openapi_toolset.assert_called_once()
  _, kwargs = mock_openapi_toolset.call_args
  assert isinstance(kwargs["auth_credential"], AuthCredential)
  assert (
      kwargs[
          "auth_credential"
      ].service_account.service_account_credential.client_email
      == "test@example.com"
  )


def test_initialization_without_explicit_service_account_credentials(
    project, location, mock_integration_client, mock_openapi_toolset
):
  integration_name = "test-integration"
  triggers = "test-trigger"
  toolset = ApplicationIntegrationToolset(
      project, location, integration=integration_name, triggers=triggers
  )
  mock_integration_client.assert_called_once_with(
      project, location, integration_name, triggers, None, None, None, None
  )
  mock_openapi_toolset.assert_called_once()
  _, kwargs = mock_openapi_toolset.call_args
  assert isinstance(kwargs["auth_credential"], AuthCredential)
  assert kwargs["auth_credential"].service_account.use_default_credential


@pytest.mark.asyncio
async def test_get_tools(
    project, location, mock_integration_client, mock_openapi_toolset
):
  integration_name = "test-integration"
  triggers = ["test-trigger"]
  toolset = ApplicationIntegrationToolset(
      project, location, integration=integration_name, triggers=triggers
  )
  tools = await toolset.get_tools()
  assert len(tools) == 1
  assert isinstance(tools[0], rest_api_tool.RestApiTool)
  assert tools[0].name == "Test Tool"


def test_initialization_with_connection_details(
    project,
    location,
    mock_integration_client,
    mock_connections_client,
    mock_openapi_toolset,
):
  connection_name = "test-connection"
  entity_operations_list = ["list"]
  tool_name = "My Connection Tool"
  tool_instructions = "Use this tool."
  mock_connections_client.return_value.get_connection_details.return_value = {
      "serviceName": "custom-service",
      "host": "custom.host",
  }
  toolset = ApplicationIntegrationToolset(
      project,
      location,
      connection=connection_name,
      entity_operations=entity_operations_list,
      tool_name_prefix=tool_name,
      tool_instructions=tool_instructions,
  )
  mock_integration_client.return_value.get_openapi_spec_for_connection.assert_called_once_with(
      tool_name, tool_instructions
  )


@pytest.mark.asyncio
async def test_init_with_connection_and_custom_auth(
    mock_integration_client,
    mock_connections_client,
    mock_openapi_action_spec_parser,
    connection_details_auth_override_enabled,
):
  connection_name = "test-connection"
  actions_list = ["create", "delete"]
  tool_name = "My Actions Tool"
  tool_instructions = "Perform actions using this tool."
  mock_connections_client.return_value.get_connection_details.return_value = (
      connection_details_auth_override_enabled
  )

  oauth2_data_google_cloud = {
      "type": "oauth2",
      "flows": {
          "authorizationCode": {
              "authorizationUrl": "https://test-url/o/oauth2/auth",
              "tokenUrl": "https://test-url/token",
              "scopes": {
                  "https://test-url/auth/test-scope": "test scope",
                  "https://www.test-url.com/auth/test-scope2": "test scope 2",
              },
          }
      },
  }

  oauth2_scheme = dict_to_auth_scheme(oauth2_data_google_cloud)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test-client-id",
          client_secret="test-client-secret",
      ),
  )

  toolset = ApplicationIntegrationToolset(
      project,
      location,
      connection=connection_name,
      actions=actions_list,
      tool_name_prefix=tool_name,
      tool_instructions=tool_instructions,
      auth_scheme=oauth2_scheme,
      auth_credential=auth_credential,
  )
  mock_integration_client.assert_called_once_with(
      project, location, None, None, connection_name, None, actions_list, None
  )
  mock_connections_client.assert_called_once_with(
      project, location, connection_name, None
  )
  mock_connections_client.return_value.get_connection_details.assert_called_once()
  mock_integration_client.return_value.get_openapi_spec_for_connection.assert_called_once_with(
      tool_name, tool_instructions
  )
  mock_openapi_action_spec_parser.return_value.parse.assert_called_once()
  assert len(await toolset.get_tools()) == 1
  assert (await toolset.get_tools())[0].name == "list_issues_operation"
  assert isinstance((await toolset.get_tools())[0], IntegrationConnectorTool)
  assert (await toolset.get_tools())[0]._action == "CustomAction"
  assert (await toolset.get_tools())[0]._operation == "EXECUTE_ACTION"
  assert (await toolset.get_tools())[0]._auth_scheme == oauth2_scheme
  assert (await toolset.get_tools())[0]._auth_credential == auth_credential


@pytest.mark.asyncio
async def test_init_with_connection_with_auth_override_disabled_and_custom_auth(
    mock_integration_client,
    mock_connections_client,
    mock_openapi_action_spec_parser,
    connection_details,
):
  connection_name = "test-connection"
  actions_list = ["create", "delete"]
  tool_name = "My Actions Tool"
  tool_instructions = "Perform actions using this tool."
  mock_connections_client.return_value.get_connection_details.return_value = (
      connection_details
  )

  oauth2_data_google_cloud = {
      "type": "oauth2",
      "flows": {
          "authorizationCode": {
              "authorizationUrl": "https://test-url/o/oauth2/auth",
              "tokenUrl": "https://test-url/token",
              "scopes": {
                  "https://test-url/auth/test-scope": "test scope",
                  "https://www.test-url.com/auth/test-scope2": "test scope 2",
              },
          }
      },
  }

  oauth2_scheme = dict_to_auth_scheme(oauth2_data_google_cloud)

  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id="test-client-id",
          client_secret="test-client-secret",
      ),
  )

  toolset = ApplicationIntegrationToolset(
      project,
      location,
      connection=connection_name,
      actions=actions_list,
      tool_name_prefix=tool_name,
      tool_instructions=tool_instructions,
      auth_scheme=oauth2_scheme,
      auth_credential=auth_credential,
  )
  mock_integration_client.assert_called_once_with(
      project, location, None, None, connection_name, None, actions_list, None
  )
  mock_connections_client.assert_called_once_with(
      project, location, connection_name, None
  )
  mock_connections_client.return_value.get_connection_details.assert_called_once()
  mock_integration_client.return_value.get_openapi_spec_for_connection.assert_called_once_with(
      tool_name, tool_instructions
  )
  mock_openapi_action_spec_parser.return_value.parse.assert_called_once()
  assert len(await toolset.get_tools()) == 1
  assert (await toolset.get_tools())[0].name == "list_issues_operation"
  assert isinstance((await toolset.get_tools())[0], IntegrationConnectorTool)
  assert (await toolset.get_tools())[0]._action == "CustomAction"
  assert (await toolset.get_tools())[0]._operation == "EXECUTE_ACTION"
  assert not (await toolset.get_tools())[0]._auth_scheme
  assert not (await toolset.get_tools())[0]._auth_credential
