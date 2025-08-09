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
import re
from unittest import mock

from google.adk.tools.application_integration_tool.clients.connections_client import ConnectionsClient
from google.adk.tools.application_integration_tool.clients.integration_client import IntegrationClient
import google.auth
import google.auth.transport.requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
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
def integration_name():
  return "test-integration"


@pytest.fixture
def triggers():
  return ["test-trigger", "test-trigger2"]


@pytest.fixture
def connection_name():
  return "test-connection"


@pytest.fixture
def mock_credentials():
  creds = mock.create_autospec(google.auth.credentials.Credentials)
  creds.token = "test_token"
  return creds


@pytest.fixture
def mock_auth_request():
  return mock.create_autospec(Request)


@pytest.fixture
def mock_connections_client():
  with mock.patch(
      "google.adk.tools.application_integration_tool.clients.integration_client.ConnectionsClient"
  ) as mock_client:
    mock_instance = mock.create_autospec(ConnectionsClient)
    mock_client.return_value = mock_instance
    yield mock_client


class TestIntegrationClient:

  def test_initialization(
      self, project, location, integration_name, triggers, connection_name
  ):
    client = IntegrationClient(
        project=project,
        location=location,
        integration=integration_name,
        triggers=triggers,
        connection=connection_name,
        entity_operations={"entity": ["LIST"]},
        actions=["action1"],
        service_account_json=json.dumps({"email": "test@example.com"}),
    )
    assert client.project == project
    assert client.location == location
    assert client.integration == integration_name
    assert client.triggers == triggers
    assert client.connection == connection_name
    assert client.entity_operations == {"entity": ["LIST"]}
    assert client.actions == ["action1"]
    assert client.service_account_json == json.dumps(
        {"email": "test@example.com"}
    )
    assert client.credential_cache is None

  def test_get_openapi_spec_for_integration_success(
      self,
      project,
      location,
      integration_name,
      triggers,
      mock_credentials,
      mock_connections_client,
  ):
    expected_spec = {"openapi": "3.0.0", "info": {"title": "Test Integration"}}
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"openApiSpec": json.dumps(expected_spec)}

    with (
        mock.patch.object(
            IntegrationClient,
            "_get_access_token",
            return_value=mock_credentials.token,
        ),
        mock.patch("requests.post", return_value=mock_response),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=None,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      spec = client.get_openapi_spec_for_integration()
      assert spec == expected_spec
      requests.post.assert_called_once_with(
          f"https://{location}-integrations.googleapis.com/v1/projects/{project}/locations/{location}:generateOpenApiSpec",
          headers={
              "Content-Type": "application/json",
              "Authorization": f"Bearer {mock_credentials.token}",
          },
          json={
              "apiTriggerResources": [{
                  "integrationResource": integration_name,
                  "triggerId": triggers,
              }],
              "fileFormat": "JSON",
          },
      )

  def test_get_openapi_spec_for_integration_credential_error(
      self,
      project,
      location,
      integration_name,
      triggers,
      mock_connections_client,
  ):
    with mock.patch.object(
        IntegrationClient,
        "_get_access_token",
        side_effect=ValueError(
            "Please provide a service account that has the required permissions"
            " to access the connection."
        ),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=None,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      with pytest.raises(
          Exception,
          match=(
              "An unexpected error occurred: Please provide a service account"
              " that has the required permissions to access the connection."
          ),
      ):
        client.get_openapi_spec_for_integration()

  @pytest.mark.parametrize(
      "status_code, response_text",
      [(404, "Not Found"), (400, "Bad Request"), (404, ""), (400, "")],
  )
  def test_get_openapi_spec_for_integration_request_error_not_found_or_bad_request(
      self,
      project,
      location,
      integration_name,
      triggers,
      mock_credentials,
      status_code,
      response_text,
      mock_connections_client,
  ):
    mock_response = mock.MagicMock()
    mock_response.status_code = status_code
    mock_response.raise_for_status.side_effect = exceptions.HTTPError(
        f"HTTP error {status_code}: {response_text}"
    )

    with (
        mock.patch.object(
            IntegrationClient,
            "_get_access_token",
            return_value=mock_credentials.token,
        ),
        mock.patch("requests.post", return_value=mock_response),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=None,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      with pytest.raises(
          ValueError,
          match=(
              r"Invalid request\. Please check the provided values of"
              rf" project\({project}\), location\({location}\),"
              rf" integration\({integration_name}\)."
          ),
      ):
        client.get_openapi_spec_for_integration()

  def test_get_openapi_spec_for_integration_other_request_error(
      self,
      project,
      location,
      integration_name,
      triggers,
      mock_credentials,
      mock_connections_client,
  ):
    mock_response = mock.MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = exceptions.HTTPError(
        "Internal Server Error"
    )

    with (
        mock.patch.object(
            IntegrationClient,
            "_get_access_token",
            return_value=mock_credentials.token,
        ),
        mock.patch("requests.post", return_value=mock_response),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=None,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      with pytest.raises(ValueError, match="Request error: "):
        client.get_openapi_spec_for_integration()

  def test_get_openapi_spec_for_integration_unexpected_error(
      self,
      project,
      location,
      integration_name,
      triggers,
      mock_credentials,
      mock_connections_client,
  ):
    with (
        mock.patch.object(
            IntegrationClient,
            "_get_access_token",
            return_value=mock_credentials.token,
        ),
        mock.patch(
            "requests.post", side_effect=Exception("Something went wrong")
        ),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=None,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      with pytest.raises(
          Exception, match="An unexpected error occurred: Something went wrong"
      ):
        client.get_openapi_spec_for_integration()

  def test_get_openapi_spec_for_connection_no_entity_operations_or_actions(
      self, project, location, connection_name, mock_connections_client
  ):
    client = IntegrationClient(
        project=project,
        location=location,
        integration=None,
        triggers=None,
        connection=connection_name,
        entity_operations=None,
        actions=None,
        service_account_json=None,
    )
    with pytest.raises(
        ValueError,
        match=(
            "No entity operations or actions provided. Please provide at least"
            " one of them."
        ),
    ):
      client.get_openapi_spec_for_connection()

  def test_get_openapi_spec_for_connection_with_entity_operations(
      self, project, location, connection_name, mock_connections_client
  ):
    entity_operations = {"entity1": ["LIST", "GET"]}

    mock_connections_client_instance = mock_connections_client.return_value
    mock_connections_client_instance.get_connector_base_spec.return_value = {
        "components": {"schemas": {}},
        "paths": {},
    }
    mock_connections_client_instance.get_entity_schema_and_operations.return_value = (
        {"type": "object", "properties": {"id": {"type": "string"}}},
        ["LIST", "GET"],
    )
    mock_connections_client_instance.connector_payload.return_value = {
        "type": "object"
    }
    mock_connections_client_instance.list_operation.return_value = {"get": {}}
    mock_connections_client_instance.list_operation_request.return_value = {
        "type": "object"
    }
    mock_connections_client_instance.get_operation.return_value = {"get": {}}
    mock_connections_client_instance.get_operation_request.return_value = {
        "type": "object"
    }

    client = IntegrationClient(
        project=project,
        location=location,
        integration=None,
        triggers=None,
        connection=connection_name,
        entity_operations=entity_operations,
        actions=None,
        service_account_json=None,
    )
    spec = client.get_openapi_spec_for_connection()
    assert "paths" in spec
    assert (
        f"/v2/projects/{project}/locations/{location}/integrations/ExecuteConnection:execute?triggerId=api_trigger/ExecuteConnection#list_entity1"
        in spec["paths"]
    )
    assert (
        f"/v2/projects/{project}/locations/{location}/integrations/ExecuteConnection:execute?triggerId=api_trigger/ExecuteConnection#get_entity1"
        in spec["paths"]
    )
    mock_connections_client.assert_called_once_with(
        project, location, connection_name, None
    )
    mock_connections_client_instance.get_connector_base_spec.assert_called_once()
    mock_connections_client_instance.get_entity_schema_and_operations.assert_any_call(
        "entity1"
    )
    mock_connections_client_instance.connector_payload.assert_any_call(
        {"type": "object", "properties": {"id": {"type": "string"}}}
    )
    mock_connections_client_instance.list_operation.assert_called_once()
    mock_connections_client_instance.get_operation.assert_called_once()

  def test_get_openapi_spec_for_connection_with_actions(
      self, project, location, connection_name, mock_connections_client
  ):
    actions = ["TestAction"]
    mock_connections_client_instance = (
        mock_connections_client.return_value
    )  # Corrected line
    mock_connections_client_instance.get_connector_base_spec.return_value = {
        "components": {"schemas": {}},
        "paths": {},
    }
    mock_connections_client_instance.get_action_schema.return_value = {
        "inputSchema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
        },
        "outputSchema": {
            "type": "object",
            "properties": {"output": {"type": "string"}},
        },
        "displayName": "TestAction",
    }
    mock_connections_client_instance.connector_payload.side_effect = [
        {"type": "object"},
        {"type": "object"},
    ]
    mock_connections_client_instance.action_request.return_value = {
        "type": "object"
    }
    mock_connections_client_instance.action_response.return_value = {
        "type": "object"
    }
    mock_connections_client_instance.get_action_operation.return_value = {
        "post": {}
    }

    client = IntegrationClient(
        project=project,
        location=location,
        integration=None,
        triggers=None,
        connection=connection_name,
        entity_operations=None,
        actions=actions,
        service_account_json=None,
    )
    spec = client.get_openapi_spec_for_connection()
    assert "paths" in spec
    assert (
        f"/v2/projects/{project}/locations/{location}/integrations/ExecuteConnection:execute?triggerId=api_trigger/ExecuteConnection#TestAction"
        in spec["paths"]
    )
    mock_connections_client.assert_called_once_with(
        project, location, connection_name, None
    )
    mock_connections_client_instance.get_connector_base_spec.assert_called_once()
    mock_connections_client_instance.get_action_schema.assert_called_once_with(
        "TestAction"
    )
    mock_connections_client_instance.connector_payload.assert_any_call(
        {"type": "object", "properties": {"input": {"type": "string"}}}
    )
    mock_connections_client_instance.connector_payload.assert_any_call(
        {"type": "object", "properties": {"output": {"type": "string"}}}
    )
    mock_connections_client_instance.action_request.assert_called_once_with(
        "TestAction"
    )
    mock_connections_client_instance.action_response.assert_called_once_with(
        "TestAction"
    )
    mock_connections_client_instance.get_action_operation.assert_called_once()

  def test_get_openapi_spec_for_connection_invalid_operation(
      self, project, location, connection_name, mock_connections_client
  ):
    entity_operations = {"entity1": ["INVALID"]}
    mock_connections_client_instance = mock_connections_client.return_value
    mock_connections_client_instance.get_connector_base_spec.return_value = {
        "components": {"schemas": {}},
        "paths": {},
    }
    mock_connections_client_instance.get_entity_schema_and_operations.return_value = (
        {"type": "object", "properties": {"id": {"type": "string"}}},
        ["LIST", "GET"],
    )

    client = IntegrationClient(
        project=project,
        location=location,
        integration=None,
        triggers=None,
        connection=connection_name,
        entity_operations=entity_operations,
        actions=None,
        service_account_json=None,
    )
    with pytest.raises(
        ValueError, match="Invalid operation: INVALID for entity: entity1"
    ):
      client.get_openapi_spec_for_connection()

  def test_get_access_token_with_service_account_json(
      self, project, location, integration_name, triggers, connection_name
  ):
    service_account_json = json.dumps({
        "client_email": "test@example.com",
        "private_key": "test_key",
    })
    mock_creds = mock.create_autospec(service_account.Credentials)
    mock_creds.token = "sa_token"
    mock_creds.expired = False

    with (
        mock.patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=mock_creds,
        ),
        mock.patch.object(mock_creds, "refresh", return_value=None),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=connection_name,
          entity_operations=None,
          actions=None,
          service_account_json=service_account_json,
      )
      token = client._get_access_token()
      assert token == "sa_token"
      service_account.Credentials.from_service_account_info.assert_called_once_with(
          json.loads(service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
      mock_creds.refresh.assert_called_once()

  def test_get_access_token_with_default_credentials(
      self,
      project,
      location,
      integration_name,
      triggers,
      connection_name,
      mock_credentials,
  ):
    mock_credentials.expired = False
    with (
        mock.patch(
            "google.adk.tools.application_integration_tool.clients.integration_client.default_service_credential",
            return_value=(mock_credentials, "test_project_id"),
        ) as mock_default_service_credential,
        mock.patch.object(mock_credentials, "refresh", return_value=None),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=connection_name,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      token = client._get_access_token()
      assert token == "test_token"
      # Verify default_service_credential is called with the correct scopes parameter
      mock_default_service_credential.assert_called_once_with(
          scopes=["https://www.googleapis.com/auth/cloud-platform"]
      )

  def test_get_access_token_no_valid_credentials(
      self, project, location, integration_name, triggers, connection_name
  ):
    with (
        mock.patch(
            "google.adk.tools.application_integration_tool.clients.integration_client.default_service_credential",
            return_value=(None, None),
        ),
        mock.patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=None,
        ),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=connection_name,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      try:
        client._get_access_token()
        assert False, "ValueError was not raised"  # Explicitly fail if no error
      except ValueError as e:
        assert (
            "Please provide a service account that has the required permissions"
            " to access the connection."
            in str(e)
        )

  def test_get_access_token_uses_cached_token(
      self,
      project,
      location,
      integration_name,
      triggers,
      connection_name,
      mock_credentials,
  ):
    mock_credentials.token = "cached_token"
    mock_credentials.expired = False
    client = IntegrationClient(
        project=project,
        location=location,
        integration=integration_name,
        triggers=triggers,
        connection=connection_name,
        entity_operations=None,
        actions=None,
        service_account_json=None,
    )
    client.credential_cache = mock_credentials  # Simulate a cached credential
    with (
        mock.patch("google.auth.default") as mock_default,
        mock.patch(
            "google.oauth2.service_account.Credentials.from_service_account_info"
        ) as mock_sa,
    ):
      token = client._get_access_token()
      assert token == "cached_token"
      mock_default.assert_not_called()
      mock_sa.assert_not_called()

  def test_get_access_token_refreshes_expired_token(
      self,
      project,
      location,
      integration_name,
      triggers,
      connection_name,
      mock_credentials,
  ):
    mock_credentials = mock.create_autospec(google.auth.credentials.Credentials)
    mock_credentials.token = "old_token"
    mock_credentials.expired = True
    mock_credentials.refresh.return_value = None
    mock_credentials.token = "new_token"  # Simulate token refresh

    with mock.patch(
        "google.adk.tools.application_integration_tool.clients.integration_client.default_service_credential",
        return_value=(mock_credentials, "test_project_id"),
    ):
      client = IntegrationClient(
          project=project,
          location=location,
          integration=integration_name,
          triggers=triggers,
          connection=connection_name,
          entity_operations=None,
          actions=None,
          service_account_json=None,
      )
      client.credential_cache = mock_credentials
      token = client._get_access_token()
      assert token == "new_token"
      mock_credentials.refresh.assert_called_once()
