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

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import json
from typing import Any
from typing import cast
from typing import Protocol

from google.adk.tools.application_integration_tool.clients.connections_client import ConnectionsClient
import google.auth
from google.auth import default as default_service_credential
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import requests

from ....utils import _mtls_utils

_DEFAULT_INTEGRATIONS_REGIONAL_ENDPOINT_TEMPLATE = (
    "{location}-integrations.googleapis.com"
)
_DEFAULT_MTLS_INTEGRATIONS_REGIONAL_ENDPOINT_TEMPLATE = (
    "{location}-integrations.mtls.googleapis.com"
)
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30


class _ServiceAccountCredentialsFactory(Protocol):

  def __call__(
      self,
      info: Mapping[str, object],
      *,
      scopes: Sequence[str],
  ) -> Credentials:
    ...


class IntegrationClient:
  """A client for interacting with Google Cloud Application Integration.

  This class provides methods for retrieving OpenAPI spec for an integration or
  a connection.
  """

  def __init__(
      self,
      project: str,
      location: str,
      connection_template_override: str | None = None,
      integration: str | None = None,
      triggers: list[str] | None = None,
      connection: str | None = None,
      entity_operations: dict[str, list[str]] | None = None,
      actions: list[str] | None = None,
      service_account_json: str | None = None,
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
    self.credential_cache: Credentials | None = None
    self._quota_project_id: str | None = None

  def get_openapi_spec_for_integration(self) -> dict[str, Any]:
    """Gets the OpenAPI spec for the integration.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        PermissionError: If there are credential issues.
        ValueError: If there's a request error or processing error.
        Exception: For any other unexpected errors.
    """
    try:
      if not self.integration or self.triggers is None:
        raise ValueError(
            "Integration name and triggers are required to generate an"
            " integration OpenAPI spec."
        )
      endpoint = _mtls_utils.get_api_endpoint(
          self.location,
          _DEFAULT_INTEGRATIONS_REGIONAL_ENDPOINT_TEMPLATE,
          _DEFAULT_MTLS_INTEGRATIONS_REGIONAL_ENDPOINT_TEMPLATE,
      )
      url = f"https://{endpoint}/v1/projects/{self.project}/locations/{self.location}:generateOpenApiSpec"
      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {self._get_access_token()}",
      }
      if not self.service_account_json:
        headers["x-goog-user-project"] = self._quota_project_id or self.project
      data: dict[str, Any] = {
          "apiTriggerResources": [
              {
                  "integrationResource": self.integration,
                  "triggerId": self.triggers,
              },
          ],
          "fileFormat": "JSON",
      }
      response = requests.post(
          url,
          headers=headers,
          json=data,
          timeout=_DEFAULT_REQUEST_TIMEOUT_SECONDS,
      )
      response.raise_for_status()
      response_payload: object = response.json()
      if not isinstance(response_payload, Mapping):
        raise ValueError("Integration API response must be a JSON object.")
      spec = response_payload.get("openApiSpec")
      if not isinstance(spec, str):
        raise ValueError(
            "Integration API response did not include an OpenAPI spec."
        )
      parsed_spec: object = json.loads(spec)
      if not isinstance(parsed_spec, dict) or not all(
          isinstance(key, str) for key in parsed_spec
      ):
        raise ValueError("Generated OpenAPI spec must be a JSON object.")
      return cast(dict[str, Any], parsed_spec)
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

  def get_openapi_spec_for_connection(
      self, tool_name: str = "", tool_instructions: str = ""
  ) -> dict[str, Any]:
    """Gets the OpenAPI spec for the connection.

    Returns:
        dict: The OpenAPI spec as a dictionary.
    Raises:
        ValueError: If there's an error retrieving the OpenAPI spec.
        PermissionError: If there are credential issues.
        Exception: For any other unexpected errors.
    """
    # Application Integration needs to be provisioned in the same region as connection and an integration with name "ExecuteConnection" and trigger "api_trigger/ExecuteConnection" should be created as per the documentation.
    if not self.connection:
      raise ValueError(
          "Connection name is required to generate a connection OpenAPI spec."
      )
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
      cached_token: object = self.credential_cache.token
      if isinstance(cached_token, str) and cached_token:
        return cached_token

    if self.service_account_json:
      service_account_info: object = json.loads(self.service_account_json)
      if not isinstance(service_account_info, Mapping):
        raise ValueError("Service account JSON must contain an object.")
      credential_factory = cast(
          _ServiceAccountCredentialsFactory,
          service_account.Credentials.from_service_account_info,
      )
      credentials: Credentials | None = credential_factory(
          cast(Mapping[str, object], service_account_info),
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
    else:
      try:
        credentials, project_id = default_service_credential(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
      except google.auth.exceptions.DefaultCredentialsError:
        credentials = None
      if credentials:
        quota_project_id: object = getattr(
            credentials, "quota_project_id", None
        )
        self._quota_project_id = (
            quota_project_id
            if isinstance(quota_project_id, str) and quota_project_id
            else project_id
        )

    if not credentials:
      raise ValueError(
          "Please provide a service account that has the required permissions"
          " to access the connection."
      )

    credentials.refresh(Request())
    self.credential_cache = credentials
    refreshed_token: object = credentials.token
    if not isinstance(refreshed_token, str) or not refreshed_token:
      raise ValueError("Credential refresh did not return an access token.")
    return refreshed_token
