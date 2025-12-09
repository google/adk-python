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

from __future__ import annotations

import json
from typing import List
from typing import Optional

from google.adk.tools.application_integration_tool.clients.connections_client import ConnectionsClient
import google.auth
from google.auth import default as default_service_credential
import google.auth.transport.requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests


class IntegrationClient:
  """A client for interacting with Google Cloud Application Integration.

  This class provides methods for retrieving OpenAPI spec for an integration or
  a connection.
  """

  def __init__(
      self,
      project: str,
      location: str,
      connection_template_override: Optional[str] = None,
      integration: Optional[str] = None,
      triggers: Optional[List[str]] = None,
      connection: Optional[str] = None,
      entity_operations: Optional[dict[str, list[str]]] = None,
      actions: Optional[list[str]] = None,
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ApplicationIntegrationClient.

    Args:
        project: The Google Cloud project ID.
        location: The Google Cloud location (e.g., us-central1).
        connection_template_override: Overrides `ExecuteConnection` default
          integration name.
        integration: The integration name.
        triggers: The list of trigger IDs for the integration.
        connection: The connection name.
        entity_operations: A dictionary mapping entity names to a list of
          operations (e.g., LIST, CREATE, UPDATE, DELETE, GET).
        actions: List of actions.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential. Used for fetching
          connection details.
    """
    self.project = project
    self.location = location
    self.connection_template_override = connection_template_override
    self.integration = integration
    self.triggers = triggers
    self.connection = connection
    self.entity_operations = (
        entity_operations if entity_operations is not None else {}
    )
    self.actions = actions if actions is not None else []
    self.service_account_json = service_account_json
    self.credential_cache = None

  def get_openapi_spec_for_integration(self):
    """Gets the OpenAPI spec for the integration.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error or processing error.
        Exception: For any other unexpected errors.
    """
    try:
      url = f"https://{self.location}-integrations.googleapis.com/v1/projects/{self.project}/locations/{self.location}:generateOpenApiSpec"
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._get_access_token()}",
      }
      data = {
          "apiTriggerResources": [
              {
                  "integrationResource": self.integration,
                  "triggerId": self.triggers,
              },
          ],
          "fileFormat": "JSON",
      }
      response = requests.post(url, headers=headers, json=data)
      response.raise_for_status()
      spec = response.json().get("openApiSpec", {})
      return json.loads(spec)
    except google.auth.exceptions.DefaultCredentialsError as e:
      raise PermissionError(f"Credentials error: {e}") from e
    except requests.exceptions.RequestException as e:
      if (
          "404" in str(e)
          or "Not found" in str(e)
          or "400" in str(e)
          or "Bad request" in str(e)
      ):
        raise ValueError(
            "Invalid request. Please check the provided values of"
            f" project({self.project}), location({self.location}),"
            f" integration({self.integration})."
        ) from e
      raise ValueError(f"Request error: {e}") from e
    except Exception as e:
      raise Exception(f"An unexpected error occurred: {e}") from e

  def get_openapi_spec_for_connection(self, tool_name="", tool_instructions=""):
    """Gets the OpenAPI spec for the connection.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        ValueError: If there's an error retrieving the OpenAPI spec.
        PermissionError: If there are credential issues.
        Exception: For any other unexpected errors.
    """
    # Application Integration needs to be provisioned in the same region as connection and an integration with name "ExecuteConnection" and trigger "api_trigger/ExecuteConnection" should be created as per the documentation.
    integration_name = self.connection_template_override or "ExecuteConnection"
    connections_client = ConnectionsClient(
        self.project,
        self.location,
        self.connection,
        self.service_account_json,
    )
    if not self.entity_operations and not self.actions:
      raise ValueError(
          "No entity operations or actions provided. Please provide at least"
          " one of them."
      )
    connector_spec = connections_client.get_connector_base_spec()
    for entity, operations in self.entity_operations.items():
      schema, supported_operations = (
          connections_client.get_entity_schema_and_operations(entity)
      )
      if not operations:
        operations = supported_operations
      json_schema_as_string = json.dumps(schema)
      entity_lower = entity
      connector_spec["components"]["schemas"][
          f"connectorInputPayload_{entity_lower}"
      ] = connections_client.connector_payload(schema)
      for operation in operations:
        operation_lower = operation.lower()
        path = f"/v2/projects/{self.project}/locations/{self.location}/integrations/{integration_name}:execute?triggerId=api_trigger/{integration_name}#{operation_lower}_{entity_lower}"
        if operation_lower == "create":
          connector_spec["paths"][path] = connections_client.create_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"create_{entity_lower}_Request"
          ] = connections_client.create_operation_request(entity_lower)
        elif operation_lower == "update":
          connector_spec["paths"][path] = connections_client.update_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"update_{entity_lower}_Request"
          ] = connections_client.update_operation_request(entity_lower)
        elif operation_lower == "delete":
          connector_spec["paths"][path] = connections_client.delete_operation(
              entity_lower, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"delete_{entity_lower}_Request"
          ] = connections_client.delete_operation_request()
        elif operation_lower == "list":
          connector_spec["paths"][path] = connections_client.list_operation(
              entity_lower, json_schema_as_string, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"list_{entity_lower}_Request"
          ] = connections_client.list_operation_request()
        elif operation_lower == "get":
          connector_spec["paths"][path] = connections_client.get_operation(
              entity_lower, json_schema_as_string, tool_name, tool_instructions
          )
          connector_spec["components"]["schemas"][
              f"get_{entity_lower}_Request"
          ] = connections_client.get_operation_request()
        else:
          raise ValueError(
              f"Invalid operation: {operation} for entity: {entity}"
          )
    for action in self.actions:
      action_details = connections_client.get_action_schema(action)
      input_schema = action_details["inputSchema"]
      output_schema = action_details["outputSchema"]
      # Remove spaces from the display name to generate valid spec
      action_display_name = action_details["displayName"].replace(" ", "")
      operation = "EXECUTE_ACTION"
      if action == "ExecuteCustomQuery":
        connector_spec["components"]["schemas"][
            f"{action_display_name}_Request"
        ] = connections_client.execute_custom_query_request()
        operation = "EXECUTE_QUERY"
      else:
        connector_spec["components"]["schemas"][
            f"{action_display_name}_Request"
        ] = connections_client.action_request(action_display_name)
        connector_spec["components"]["schemas"][
            f"connectorInputPayload_{action_display_name}"
        ] = connections_client.connector_payload(input_schema)
      connector_spec["components"]["schemas"][
          f"connectorOutputPayload_{action_display_name}"
      ] = connections_client.connector_payload(output_schema)
      connector_spec["components"]["schemas"][
          f"{action_display_name}_Response"
      ] = connections_client.action_response(action_display_name)
      path = f"/v2/projects/{self.project}/locations/{self.location}/integrations/{integration_name}:execute?triggerId=api_trigger/{integration_name}#{action}"
      connector_spec["paths"][path] = connections_client.get_action_operation(
          action, operation, action_display_name, tool_name, tool_instructions
      )
    return connector_spec

  def _get_access_token(self) -> str:
    """Gets the access token for the service account or using default credentials.

    Returns:
        The access token.
    """
    if self.credential_cache and not self.credential_cache.expired:
      return self.credential_cache.token

    if self.service_account_json:
      credentials = service_account.Credentials.from_service_account_info(
          json.loads(self.service_account_json),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
    else:
      try:
        credentials, _ = default_service_credential(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
      except:
        credentials = None

    if not credentials:
      raise ValueError(
          "Please provide a service account that has the required permissions"
          " to access the connection."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    return credentials.token
