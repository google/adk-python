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

import logging
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import HTTPBearer
from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes
from ...auth.auth_credential import ServiceAccount
from ...auth.auth_credential import ServiceAccountCredential
from ...auth.auth_schemes import AuthScheme
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..openapi_tool.auth.auth_helpers import service_account_scheme_credential
from ..openapi_tool.openapi_spec_parser.openapi_spec_parser import OpenApiSpecParser
from ..openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from ..openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from .clients.connections_client import ConnectionsClient
from .clients.integration_client import IntegrationClient
from .integration_connector_tool import IntegrationConnectorTool

logger = logging.getLogger("google_adk." + __name__)


# TODO(cheliu): Apply a common toolset interface
class ApplicationIntegrationToolset(BaseToolset):
  """ApplicationIntegrationToolset generates tools from a given Application
  Integration or Integration Connector resource.

  Example Usage::

    # Get all available tools for an integration with api trigger
    application_integration_toolset = ApplicationIntegrationToolset(
        project="test-project",
        location="us-central1"
        integration="test-integration",
        triggers=["api_trigger/test_trigger"],
        service_account_credentials={...},
    )

    # Get all available tools for a connection using entity operations and
    # actions
    # Note: Find the list of supported entity operations and actions for a
    # connection using integration connector apis:
    # https://cloud.google.com/integration-connectors/docs/reference/rest/v1/projects.locations.connections.connectionSchemaMetadata
    application_integration_toolset = ApplicationIntegrationToolset(
        project="test-project",
        location="us-central1"
        connection="test-connection",
        entity_operations=["EntityId1": ["LIST","CREATE"], "EntityId2": []],
        #empty list for actions means all operations on the entity are supported
        actions=["action1"],
        service_account_credentials={...},
    )

    # Feed the toolset to agent
    agent = LlmAgent(tools=[
        ...,
        application_integration_toolset,
    ])
  """

  def __init__(
      self,
      project: str,
      location: str,
      connection_template_override: Optional[str] = None,
      integration: Optional[str] = None,
      triggers: Optional[List[str]] = None,
      connection: Optional[str] = None,
      entity_operations: Optional[str] = None,
      actions: Optional[list[str]] = None,
      # Optional parameter for the toolset. This is prepended to the generated
      # tool/python function name.
      tool_name_prefix: Optional[str] = "",
      # Optional parameter for the toolset. This is appended to the generated
      # tool/python function description.
      tool_instructions: Optional[str] = "",
      service_account_json: Optional[str] = None,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    """Args:

    Args:
        project: The GCP project ID.
        location: The GCP location.
        connection_template_override: Overrides `ExecuteConnection` default
          integration name.
        integration: The integration name.
        triggers: The list of trigger names in the integration.
        connection: The connection name.
        entity_operations: The entity operations supported by the connection.
        actions: The actions supported by the connection.
        tool_name_prefix: The name prefix of the generated tools.
        tool_instructions: The instructions for the tool.
        service_account_json: The service account configuration as a dictionary.
          Required if not using default service credential. Used for fetching
          the Application Integration or Integration Connector resource.
        tool_filter: The filter used to filter the tools in the toolset. It can
          be either a tool predicate or a list of tool names of the tools to
          expose.

    Raises:
        ValueError: If none of the following conditions are met:
          - ``integration`` is provided.
          - ``connection`` is provided and at least one of ``entity_operations``
            or ``actions`` is provided.
        Exception: If there is an error during the initialization of the
          integration or connection client.
    """
    super().__init__(tool_filter=tool_filter)
    self.project = project
    self.location = location
    self._connection_template_override = connection_template_override
    self._integration = integration
    self._triggers = triggers
    self._connection = connection
    self._entity_operations = entity_operations
    self._actions = actions
    self._tool_instructions = tool_instructions
    self._service_account_json = service_account_json
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

    integration_client = IntegrationClient(
        project,
        location,
        connection_template_override,
        integration,
        triggers,
        connection,
        entity_operations,
        actions,
        service_account_json,
    )
    connection_details = {}
    if integration:
      spec = integration_client.get_openapi_spec_for_integration()
    elif connection and (entity_operations or actions):
      connections_client = ConnectionsClient(
          project, location, connection, service_account_json
      )
      connection_details = connections_client.get_connection_details()
      spec = integration_client.get_openapi_spec_for_connection(
          tool_name_prefix,
          tool_instructions,
      )
    else:
      raise ValueError(
          "Invalid request, Either integration or (connection and"
          " (entity_operations or actions)) should be provided."
      )
    self._openapi_toolset = None
    self._tools = []
    self._parse_spec_to_toolset(spec, connection_details)

  def _parse_spec_to_toolset(self, spec_dict, connection_details):
    """Parses the spec dict to OpenAPI toolset."""
    if self._service_account_json:
      sa_credential = ServiceAccountCredential.model_validate_json(
          self._service_account_json
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

    if self._integration:
      self._openapi_toolset = OpenAPIToolset(
          spec_dict=spec_dict,
          auth_credential=auth_credential,
          auth_scheme=auth_scheme,
          tool_filter=self.tool_filter,
      )
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

      auth_override_enabled = connection_details.get(
          "authOverrideEnabled", False
      )

      if (
          self._auth_scheme
          and self._auth_credential
          and not auth_override_enabled
      ):
        # Case: Auth provided, but override is OFF. Don't use provided auth.
        logger.warning(
            "Authentication schema and credentials are not used because"
            " authOverrideEnabled is not enabled in the connection."
        )
        connector_auth_scheme = None
        connector_auth_credential = None
      else:
        connector_auth_scheme = self._auth_scheme
        connector_auth_credential = self._auth_credential

      self._tools.append(
          IntegrationConnectorTool(
              name=rest_api_tool.name,
              description=rest_api_tool.description,
              connection_name=connection_details["name"],
              connection_host=connection_details["host"],
              connection_service_name=connection_details["serviceName"],
              entity=entity,
              action=action,
              operation=operation,
              rest_api_tool=rest_api_tool,
              auth_scheme=connector_auth_scheme,
              auth_credential=connector_auth_credential,
          )
      )

  @override
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> List[RestApiTool]:
    return (
        [
            tool
            for tool in self._tools
            if self._is_tool_selected(tool, readonly_context)
        ]
        if self._openapi_toolset is None
        else await self._openapi_toolset.get_tools(readonly_context)
    )

  @override
  async def close(self) -> None:
    if self._openapi_toolset:
      await self._openapi_toolset.close()
