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

from google.adk.tools.application_integration_tool.clients.connections_client import ConnectionsClient
import google.auth
import pytest
import requests
from requests import exceptions


@pytest.fixture
def project():
  return "test-project"


@pytest.fixture
def location():
  return "us-central1"


@pytest.fixture
def connection_name():
  return "test-connection"


@pytest.fixture
def mock_credentials():
  creds = mock.create_autospec(google.auth.credentials.Credentials)
  creds.token = "test_token"
  creds.expired = False
  return creds


@pytest.fixture
def mock_auth_request():
  return mock.create_autospec(google.auth.transport.requests.Request)


class TestConnectionsClient:

  def test_initialization(self, project, location, connection_name):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(
        project, location, connection_name, json.dumps(credentials)
    )
    assert client.project == project
    assert client.location == location
    assert client.connection == connection_name
    assert client.connector_url == "https://connectors.googleapis.com"
    assert client.service_account_json == json.dumps(credentials)
    assert client.credential_cache is None

  def test_execute_api_call_success(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"data": "test"}

    with (
        mock.patch.object(
            client, "_get_access_token", return_value=mock_credentials.token
        ),
        mock.patch("requests.get", return_value=mock_response),
    ):
      response = client._execute_api_call("https://test.url")
      assert response.json() == {"data": "test"}
      requests.get.assert_called_once_with(
          "https://test.url",
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Bearer {mock_credentials.token}",
          },
      )

  def test_execute_api_call_credential_error(
      self, project, location, connection_name
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    with mock.patch.object(
        client,
        "_get_access_token",
        side_effect=google.auth.exceptions.DefaultCredentialsError("Test"),
    ):
      with pytest.raises(PermissionError, match="Credentials error: Test"):
        client._execute_api_call("https://test.url")

  @pytest.mark.parametrize(
      "status_code, response_text",
      [(404, "Not Found"), (400, "Bad Request")],
  )
  def test_execute_api_call_request_error_not_found_or_bad_request(
      self,
      project,
      location,
      connection_name,
      mock_credentials,
      status_code,
      response_text,
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = status_code
    mock_response.raise_for_status.side_effect = exceptions.HTTPError(
        f"HTTP error {status_code}: {response_text}"
    )

    with (
        mock.patch.object(
            client, "_get_access_token", return_value=mock_credentials.token
        ),
        mock.patch("requests.get", return_value=mock_response),
    ):
      with pytest.raises(
          ValueError, match="Invalid request. Please check the provided"
      ):
        client._execute_api_call("https://test.url")

  def test_execute_api_call_other_request_error(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = exceptions.HTTPError(
        "Internal Server Error"
    )

    with (
        mock.patch.object(
            client, "_get_access_token", return_value=mock_credentials.token
        ),
        mock.patch("requests.get", return_value=mock_response),
    ):
      with pytest.raises(ValueError, match="Request error: "):
        client._execute_api_call("https://test.url")

  def test_execute_api_call_unexpected_error(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    with (
        mock.patch.object(
            client, "_get_access_token", return_value=mock_credentials.token
        ),
        mock.patch(
            "requests.get", side_effect=Exception("Something went wrong")
        ),
    ):
      with pytest.raises(
          Exception, match="An unexpected error occurred: Something went wrong"
      ):
        client._execute_api_call("https://test.url")

  def test_get_connection_details_success_with_host(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "test-connection",
        "serviceDirectory": "test_service",
        "host": "test.host",
        "tlsServiceDirectory": "tls_test_service",
        "authOverrideEnabled": True,
    }

    with mock.patch.object(
        client, "_execute_api_call", return_value=mock_response
    ):
      details = client.get_connection_details()
      assert details == {
          "name": "test-connection",
          "serviceName": "tls_test_service",
          "host": "test.host",
          "authOverrideEnabled": True,
      }

  def test_get_connection_details_success_without_host(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "name": "test-connection",
        "serviceDirectory": "test_service",
        "authOverrideEnabled": False,
    }

    with mock.patch.object(
        client, "_execute_api_call", return_value=mock_response
    ):
      details = client.get_connection_details()
      assert details == {
          "name": "test-connection",
          "serviceName": "test_service",
          "host": "",
          "authOverrideEnabled": False,
      }

  def test_get_connection_details_without_name(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "serviceDirectory": "test_service",
        "authOverrideEnabled": False,
    }

    with mock.patch.object(
        client, "_execute_api_call", return_value=mock_response
    ):
      details = client.get_connection_details()
      assert details == {
          "name": "",
          "serviceName": "test_service",
          "host": "",
          "authOverrideEnabled": False,
      }

  def test_get_connection_details_error(
      self, project, location, connection_name
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    with mock.patch.object(
        client, "_execute_api_call", side_effect=ValueError("Request error")
    ):
      with pytest.raises(ValueError, match="Request error"):
        client.get_connection_details()

  def test_get_entity_schema_and_operations_success(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_execute_response_initial = mock.MagicMock()
    mock_execute_response_initial.status_code = 200
    mock_execute_response_initial.json.return_value = {
        "name": "operations/test_op"
    }

    mock_execute_response_poll_done = mock.MagicMock()
    mock_execute_response_poll_done.status_code = 200
    mock_execute_response_poll_done.json.return_value = {
        "done": True,
        "response": {
            "jsonSchema": {"type": "object"},
            "operations": ["LIST", "GET"],
        },
    }

    with mock.patch.object(
        client,
        "_execute_api_call",
        side_effect=[
            mock_execute_response_initial,
            mock_execute_response_poll_done,
        ],
    ):
      schema, operations = client.get_entity_schema_and_operations("entity1")
      assert schema == {"type": "object"}
      assert operations == ["LIST", "GET"]
      assert (
          mock.call(
              f"https://connectors.googleapis.com/v1/projects/{project}/locations/{location}/connections/{connection_name}/connectionSchemaMetadata:getEntityType?entityId=entity1"
          )
          in client._execute_api_call.mock_calls
      )
      assert (
          mock.call(f"https://connectors.googleapis.com/v1/operations/test_op")
          in client._execute_api_call.mock_calls
      )

  def test_get_entity_schema_and_operations_no_operation_id(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_execute_response = mock.MagicMock()
    mock_execute_response.status_code = 200
    mock_execute_response.json.return_value = {}

    with mock.patch.object(
        client, "_execute_api_call", return_value=mock_execute_response
    ):
      with pytest.raises(
          ValueError,
          match=(
              "Failed to get entity schema and operations for entity: entity1"
          ),
      ):
        client.get_entity_schema_and_operations("entity1")

  def test_get_entity_schema_and_operations_execute_api_call_error(
      self, project, location, connection_name
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    with mock.patch.object(
        client, "_execute_api_call", side_effect=ValueError("Request error")
    ):
      with pytest.raises(ValueError, match="Request error"):
        client.get_entity_schema_and_operations("entity1")

  def test_get_action_schema_success(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_execute_response_initial = mock.MagicMock()
    mock_execute_response_initial.status_code = 200
    mock_execute_response_initial.json.return_value = {
        "name": "operations/test_op"
    }

    mock_execute_response_poll_done = mock.MagicMock()
    mock_execute_response_poll_done.status_code = 200
    mock_execute_response_poll_done.json.return_value = {
        "done": True,
        "response": {
            "inputJsonSchema": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
            },
            "outputJsonSchema": {
                "type": "object",
                "properties": {"output": {"type": "string"}},
            },
            "description": "Test Action Description",
            "displayName": "TestAction",
        },
    }

    with mock.patch.object(
        client,
        "_execute_api_call",
        side_effect=[
            mock_execute_response_initial,
            mock_execute_response_poll_done,
        ],
    ):
      schema = client.get_action_schema("action1")
      assert schema == {
          "inputSchema": {
              "type": "object",
              "properties": {"input": {"type": "string"}},
          },
          "outputSchema": {
              "type": "object",
              "properties": {"output": {"type": "string"}},
          },
          "description": "Test Action Description",
          "displayName": "TestAction",
      }
      assert (
          mock.call(
              f"https://connectors.googleapis.com/v1/projects/{project}/locations/{location}/connections/{connection_name}/connectionSchemaMetadata:getAction?actionId=action1"
          )
          in client._execute_api_call.mock_calls
      )
      assert (
          mock.call(f"https://connectors.googleapis.com/v1/operations/test_op")
          in client._execute_api_call.mock_calls
      )

  def test_get_action_schema_no_operation_id(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    mock_execute_response = mock.MagicMock()
    mock_execute_response.status_code = 200
    mock_execute_response.json.return_value = {}

    with mock.patch.object(
        client, "_execute_api_call", return_value=mock_execute_response
    ):
      with pytest.raises(
          ValueError, match="Failed to get action schema for action: action1"
      ):
        client.get_action_schema("action1")

  def test_get_action_schema_execute_api_call_error(
      self, project, location, connection_name
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    with mock.patch.object(
        client, "_execute_api_call", side_effect=ValueError("Request error")
    ):
      with pytest.raises(ValueError, match="Request error"):
        client.get_action_schema("action1")

  def test_get_connector_base_spec(self):
    spec = ConnectionsClient.get_connector_base_spec()
    assert "openapi" in spec
    assert spec["info"]["title"] == "ExecuteConnection"
    assert "components" in spec
    assert "schemas" in spec["components"]
    assert "operation" in spec["components"]["schemas"]

  def test_get_action_operation(self):
    operation = ConnectionsClient.get_action_operation(
        "TestAction", "EXECUTE_ACTION", "TestActionDisplayName", "test_tool"
    )
    assert "post" in operation
    assert operation["post"]["summary"] == "TestActionDisplayName"
    assert "operationId" in operation["post"]
    assert operation["post"]["operationId"] == "test_tool_TestActionDisplayName"

  def test_list_operation(self):
    operation = ConnectionsClient.list_operation(
        "Entity1", '{"type": "object"}', "test_tool"
    )
    assert "post" in operation
    assert operation["post"]["summary"] == "List Entity1"
    assert "operationId" in operation["post"]
    assert operation["post"]["operationId"] == "test_tool_list_Entity1"

  def test_get_operation_static(self):
    operation = ConnectionsClient.get_operation(
        "Entity1", '{"type": "object"}', "test_tool"
    )
    assert "post" in operation
    assert operation["post"]["summary"] == "Get Entity1"
    assert "operationId" in operation["post"]
    assert operation["post"]["operationId"] == "test_tool_get_Entity1"

  def test_create_operation(self):
    operation = ConnectionsClient.create_operation("Entity1", "test_tool")
    assert "post" in operation
    assert operation["post"]["summary"] == "Creates a new Entity1"
    assert "operationId" in operation["post"]
    assert operation["post"]["operationId"] == "test_tool_create_Entity1"

  def test_update_operation(self):
    operation = ConnectionsClient.update_operation("Entity1", "test_tool")
    assert "post" in operation
    assert operation["post"]["summary"] == "Updates the Entity1"
    assert "operationId" in operation["post"]
    assert operation["post"]["operationId"] == "test_tool_update_Entity1"

  def test_delete_operation(self):
    operation = ConnectionsClient.delete_operation("Entity1", "test_tool")
    assert "post" in operation
    assert operation["post"]["summary"] == "Delete the Entity1"
    assert operation["post"]["operationId"] == "test_tool_delete_Entity1"

  def test_create_operation_request(self):
    schema = ConnectionsClient.create_operation_request("Entity1")
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "connectorInputPayload" in schema["properties"]

  def test_update_operation_request(self):
    schema = ConnectionsClient.update_operation_request("Entity1")
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "entityId" in schema["properties"]
    assert "filterClause" in schema["properties"]

  def test_get_operation_request_static(self):
    schema = ConnectionsClient.get_operation_request()
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "entityId" in schema["properties"]

  def test_delete_operation_request(self):
    schema = ConnectionsClient.delete_operation_request()
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "entityId" in schema["properties"]
    assert "filterClause" in schema["properties"]

  def test_list_operation_request(self):
    schema = ConnectionsClient.list_operation_request()
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "filterClause" in schema["properties"]
    assert "sortByColumns" in schema["properties"]

  def test_action_request(self):
    schema = ConnectionsClient.action_request("TestAction")
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "connectorInputPayload" in schema["properties"]

  def test_action_response(self):
    schema = ConnectionsClient.action_response("TestAction")
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "connectorOutputPayload" in schema["properties"]

  def test_execute_custom_query_request(self):
    schema = ConnectionsClient.execute_custom_query_request()
    assert "type" in schema
    assert schema["type"] == "object"
    assert "properties" in schema
    assert "query" in schema["properties"]

  def test_connector_payload(self):
    client = ConnectionsClient("test-project", "us-central1", "test-connection")
    schema = client.connector_payload(
        json_schema={
            "type": "object",
            "properties": {
                "input": {
                    "type": ["null", "string"],
                    "description": "description",
                }
            },
        }
    )
    assert schema == {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "nullable": True,
                "description": "description",
            }
        },
    }

  def test_get_access_token_uses_cached_token(
      self, project, location, connection_name, mock_credentials
  ):
    credentials = {"email": "test@example.com"}
    client = ConnectionsClient(project, location, connection_name, credentials)
    client.credential_cache = mock_credentials
    token = client._get_access_token()
    assert token == "test_token"

  def test_get_access_token_with_service_account_credentials(
      self, project, location, connection_name
  ):
    service_account_json = json.dumps({
        "client_email": "test@example.com",
        "private_key": "test_key",
    })
    client = ConnectionsClient(
        project, location, connection_name, service_account_json
    )
    mock_creds = mock.create_autospec(google.oauth2.service_account.Credentials)
    mock_creds.token = "sa_token"
    mock_creds.expired = False

    with (
        mock.patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=mock_creds,
        ),
        mock.patch.object(mock_creds, "refresh", return_value=None),
    ):
      token = client._get_access_token()
      assert token == "sa_token"
      google.oauth2.service_account.Credentials.from_service_account_info.assert_called_once_with(
          json.loads(service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
      mock_creds.refresh.assert_called_once()

  def test_get_access_token_with_default_credentials(
      self, project, location, connection_name, mock_credentials
  ):
    client = ConnectionsClient(project, location, connection_name, None)
    with (
        mock.patch(
            "google.adk.tools.application_integration_tool.clients.connections_client.default_service_credential",
            return_value=(mock_credentials, "test_project_id"),
        ) as mock_default_service_credential,
        mock.patch.object(mock_credentials, "refresh", return_value=None),
    ):
      token = client._get_access_token()
      assert token == "test_token"
      # Verify default_service_credential is called with the correct scopes parameter
      mock_default_service_credential.assert_called_once_with(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )

  def test_get_access_token_no_valid_credentials(
      self, project, location, connection_name
  ):
    client = ConnectionsClient(project, location, connection_name, None)
    with mock.patch(
        "google.adk.tools.application_integration_tool.clients.connections_client.default_service_credential",
        return_value=(None, None),
    ):
      with pytest.raises(
          ValueError,
          match=(
              "Please provide a service account that has the required"
              " permissions"
          ),
      ):
        client._get_access_token()

  def test_get_access_token_refreshes_expired_token(
      self, project, location, connection_name, mock_credentials
  ):
    client = ConnectionsClient(project, location, connection_name, None)
    mock_credentials.expired = True
    mock_credentials.token = "old_token"
    mock_credentials.refresh.return_value = None

    client.credential_cache = mock_credentials
    with mock.patch(
        "google.adk.tools.application_integration_tool.clients.connections_client.default_service_credential",
        return_value=(mock_credentials, "test_project_id"),
    ):
      # Mock the refresh method directly on the instance within the context
      with mock.patch.object(mock_credentials, "refresh") as mock_refresh:
        mock_credentials.token = "new_token"  # Set the expected new token
        token = client._get_access_token()
        assert token == "new_token"
        mock_refresh.assert_called_once()
