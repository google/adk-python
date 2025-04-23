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

from typing import Dict, List, Optional

from fastapi.openapi.models import HTTPBearer

from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes
from ...auth.auth_credential import ServiceAccount
from ...auth.auth_credential import ServiceAccountCredential
from ..openapi_tool.auth.auth_helpers import service_account_scheme_credential
from ..openapi_tool.openapi_spec_parser.openapi_spec_parser import OpenApiSpecParser
from ..openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from ..openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from .clients.connections_client import ConnectionsClient
from .clients.integration_client import IntegrationClient
from .integration_connector_tool import IntegrationConnectorTool


# TODO(cheliu): Apply a common toolset interface
class ApplicationIntegrationToolset:
  """ApplicationIntegrationToolset generates tools from a given Application

  Integration or Integration Connector resource.
  Example Usage:
  ```
  # Get all available tools for an integration with api trigger
  application_integration_toolset = ApplicationIntegrationToolset(

      project="test-project",
      location="us-central1"
      integration="test-integration",
      trigger="api_trigger/test_trigger",
      service_account_credentials={...},
  )

  # Get all available tools for a connection using entity operations and
  # actions
  # Note: Find the list of supported entity operations and actions for a
  connection
  # using integration connector apis:
  #
  https://cloud.google.com/integration-connectors/docs/reference/rest/v1/projects.locations.connections.connectionSchemaMetadata
  application_integration_toolset = ApplicationIntegrationToolset(
      project="test-project",
      location="us-central1"
      connection="test-connection",
      entity_operations=["EntityId1": ["LIST","CREATE"], "EntityId2": []],
      #empty list for actions means all operations on the entity are supported
      actions=["action1"],
      service_account_credentials={...},
  )

  # Get all available tools
  agent = LlmAgent(tools=[
      ...
      *application_integration_toolset.get_tools(),
  ])
  ```
  """

  def __init__(
      self,
      project: str,
      location: str,
      integration: Optional[str] = None,
      trigger: Optional[str] = None,
      connection: Optional[str] = None,
      entity_operations: Optional[str] = None,
      actions: Optional[str] = None,
      # Optional parameter for the toolset. This is prepended to the generated
      # tool/python function name.
      tool_name: Optional[str] = "",
      # Optional parameter for the toolset. This is appended to the generated
      # tool/python function description.
      tool_instructions: Optional[str] = "",
      service_account_json: Optional[str] = None,
  ):
    """Initializes the ApplicationIntegrationToolset.

    Example Usage:
    ```
    # Get all available tools for an integration with api trigger
    application_integration_toolset = ApplicationIntegrationToolset(

        project="test-project",
        location="us-central1"
        integration="test-integration",
        trigger="api_trigger/test_trigger",
        service_account_credentials={...},
    )

    # Get all available tools for a connection using entity operations and
    # actions
    # Note: Find the list of supported entity operations and actions for a
    connection
    # using integration connector apis:
    #
    https://cloud.google.com/integration-connectors/docs/reference/rest/v1/projects.locations.connections.connectionSchemaMetadata
    application_integration_toolset = ApplicationIntegrationToolset(
        project="test-project",
        location="us-central1"
        connection="test-connection",
        entity_operations=["EntityId1": ["LIST","CREATE"], "EntityId2": []],
        #empty list for actions means all operations on the entity are supported
        actions=["action1"],
        service_account_credentials={...},
    )

    # Get all available tools
    agent = LlmAgent(tools=[
        ...
        *application_integration_toolset.get_tools(),
    ])
    ```

    Args:
        project: The GCP project ID.
        location: The GCP location.
        integration: The integration name.
        trigger: The trigger name.
        connection: The connection name.
        entity_operations: The entity operations supported by the connection.
        actions: The actions supported by the connection.
        tool_name: The name of the tool.
        tool_instructions: The instructions for the tool.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential. Used for fetching
          the Application Integration or Integration Connector resource.

    Raises:
        ValueError: If neither integration and trigger nor connection and
            (entity_operations or actions) is provided.
        Exception: If there is an error during the initialization of the
            integration or connection client.
    """
    self.project = project
    self.location = location
    self.integration = integration
    self.trigger = trigger
    self.connection = connection
    self.entity_operations = entity_operations
    self.actions = actions
    self.tool_name = tool_name
    self.tool_instructions = tool_instructions
    self.service_account_json = service_account_json
    self.generated_tools: Dict[str, RestApiTool] = {}

    integration_client = IntegrationClient(
        project,
        location,
        integration,
        trigger,
        connection,
        entity_operations,
        actions,
        service_account_json,
    )
    connection_details = {}
    if integration and trigger:
      spec = integration_client.get_openapi_spec_for_integration()
    elif connection and (entity_operations or actions):
      connections_client = ConnectionsClient(
          project, location, connection, service_account_json
      )
      connection_details = connections_client.get_connection_details()
      spec = integration_client.get_openapi_spec_for_connection(
          tool_name,
          tool_instructions,
      )
    else:
      raise ValueError(
          "Either (integration and trigger) or (connection and"
          " (entity_operations or actions)) should be provided."
      )
    self._parse_spec_to_tools(spec, connection_details)

  def _parse_spec_to_tools(self, spec_dict, connection_details):
    """Parses the spec dict to a list of RestApiTool."""
    if self.service_account_json:
      sa_credential = ServiceAccountCredential.model_validate_json(
          self.service_account_json
      )
      service_account = ServiceAccount(
          service_account_credential=sa_credential,
          scopes=["https://www.googleapis.com/auth/cloud-platform"],
      )
      auth_scheme, auth_credential = service_account_scheme_credential(
          config=service_account
      )
    else:
      auth_credential = AuthCredential(
          auth_type=AuthCredentialTypes.SERVICE_ACCOUNT,
          service_account=ServiceAccount(
              use_default_credential=True,
              scopes=["https://www.googleapis.com/auth/cloud-platform"],
          ),
      )
      auth_scheme = HTTPBearer(bearerFormat="JWT")

    if self.integration and self.trigger:
      tools = OpenAPIToolset(
          spec_dict=spec_dict,
          auth_credential=auth_credential,
          auth_scheme=auth_scheme,
      ).get_tools()
      for tool in tools:
        self.generated_tools[tool.name] = tool
      return

    operations = OpenApiSpecParser().parse(spec_dict)

    for open_api_operation in operations:
      operation = getattr(open_api_operation.operation, "x-operation")
      entity = None
      action = None
      if hasattr(open_api_operation.operation, "x-entity"):
        entity = getattr(open_api_operation.operation, "x-entity")
      elif hasattr(open_api_operation.operation, "x-action"):
        action = getattr(open_api_operation.operation, "x-action")
      rest_api_tool = RestApiTool.from_parsed_operation(open_api_operation)
      if auth_scheme:
        rest_api_tool.configure_auth_scheme(auth_scheme)
      if auth_credential:
        rest_api_tool.configure_auth_credential(auth_credential)
      tool = IntegrationConnectorTool(
          name=rest_api_tool.name,
          description=rest_api_tool.description,
          connection_name=connection_details["name"],
          connection_host=connection_details["host"],
          connection_service_name=connection_details["serviceName"],
          entity=entity,
          action=action,
          operation=operation,
          rest_api_tool=rest_api_tool,
      )
      self.generated_tools[tool.name] = tool

  def get_tools(self) -> List[RestApiTool]:
    return list(self.generated_tools.values())
