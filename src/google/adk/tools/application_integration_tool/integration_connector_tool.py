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


import logging
from typing import Any
from typing import Dict
from typing import Optional

from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import to_gemini_schema
from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .. import BaseTool
from ..tool_context import ToolContext

logger = logging.getLogger(__name__)


class IntegrationConnectorTool(BaseTool):
  """A tool that wraps a RestApiTool to interact with a specific Application Integration endpoint.

  This tool adds Application Integration specific context like connection
  details, entity, operation, and action to the underlying REST API call
  handled by RestApiTool. It prepares the arguments and then delegates the
  actual API call execution to the contained RestApiTool instance.

  * Generates request params and body
  * Attaches auth credentials to API call.

  Example:
  ```
    # Each API operation in the spec will be turned into its own tool
    # Name of the tool is the operationId of that operation, in snake case
    operations = OperationGenerator().parse(openapi_spec_dict)
    tool = [RestApiTool.from_parsed_operation(o) for o in operations]
  ```
  """

  EXCLUDE_FIELDS = [
      'connection_name',
      'service_name',
      'host',
      'entity',
      'operation',
      'action',
  ]

  OPTIONAL_FIELDS = [
      'page_size',
      'page_token',
      'filter',
  ]

  def __init__(
      self,
      name: str,
      description: str,
      connection_name: str,
      connection_host: str,
      connection_service_name: str,
      entity: str,
      operation: str,
      action: str,
      rest_api_tool: RestApiTool,
  ):
    """Initializes the ApplicationIntegrationTool.

    Args:
        name: The name of the tool, typically derived from the API operation.
          Should be unique and adhere to Gemini function naming conventions
          (e.g., less than 64 characters).
        description: A description of what the tool does, usually based on the
          API operation's summary or description.
        connection_name: The name of the Integration Connector connection.
        connection_host: The hostname or IP address for the connection.
        connection_service_name: The specific service name within the host.
        entity: The Integration Connector entity being targeted.
        operation: The specific operation being performed on the entity.
        action: The action associated with the operation (e.g., 'execute').
        rest_api_tool: An initialized RestApiTool instance that handles the
          underlying REST API communication based on an OpenAPI specification
          operation. This tool will be called by ApplicationIntegrationTool with
          added connection and context arguments. tool =
          [RestApiTool.from_parsed_operation(o) for o in operations]
    """
    # Gemini restrict the length of function name to be less than 64 characters
    super().__init__(
        name=name,
        description=description,
    )
    self.connection_name = connection_name
    self.connection_host = connection_host
    self.connection_service_name = connection_service_name
    self.entity = entity
    self.operation = operation
    self.action = action
    self.rest_api_tool = rest_api_tool

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    """Returns the function declaration in the Gemini Schema format."""
    schema_dict = self.rest_api_tool._operation_parser.get_json_schema()
    for field in self.EXCLUDE_FIELDS:
      if field in schema_dict['properties']:
        del schema_dict['properties'][field]
    for field in self.OPTIONAL_FIELDS + self.EXCLUDE_FIELDS:
      if field in schema_dict['required']:
        schema_dict['required'].remove(field)

    parameters = to_gemini_schema(schema_dict)
    function_decl = FunctionDeclaration(
        name=self.name, description=self.description, parameters=parameters
    )
    return function_decl

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    args['connection_name'] = self.connection_name
    args['service_name'] = self.connection_service_name
    args['host'] = self.connection_host
    args['entity'] = self.entity
    args['operation'] = self.operation
    args['action'] = self.action
    logger.info('Running tool: %s with args: %s', self.name, args)
    return self.rest_api_tool.call(args=args, tool_context=tool_context)

  def __str__(self):
    return (
        f'ApplicationIntegrationTool(name="{self.name}",'
        f' description="{self.description}",'
        f' connection_name="{self.connection_name}", entity="{self.entity}",'
        f' operation="{self.operation}", action="{self.action}")'
    )

  def __repr__(self):
    return (
        f'ApplicationIntegrationTool(name="{self.name}",'
        f' description="{self.description}",'
        f' connection_name="{self.connection_name}",'
        f' connection_host="{self.connection_host}",'
        f' connection_service_name="{self.connection_service_name}",'
        f' entity="{self.entity}", operation="{self.operation}",'
        f' action="{self.action}", rest_api_tool={repr(self.rest_api_tool)})'
    )
