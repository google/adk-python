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
import logging
from typing import Any
from typing import Dict
from typing import Final
from typing import List
from typing import Literal
from typing import Optional

import yaml

from ....auth.auth_credential import AuthCredential
from ....auth.auth_schemes import AuthScheme
from .openapi_spec_parser import OpenApiSpecParser
from .rest_api_tool import RestApiTool

logger = logging.getLogger(__name__)


class OpenAPIToolset:
  """Class for parsing OpenAPI spec into a list of RestApiTool.

  Usage:
  ```
    # Initialize OpenAPI toolset from a spec string.
    openapi_toolset = OpenAPIToolset(spec_str=openapi_spec_str,
      spec_str_type="json")
    # Or, initialize OpenAPI toolset from a spec dictionary.
    openapi_toolset = OpenAPIToolset(spec_dict=openapi_spec_dict)

    # Add all tools to an agent.
    agent = Agent(
      tools=[*openapi_toolset.get_tools()]
    )
    # Or, add a single tool to an agent.
    agent = Agent(
      tools=[openapi_toolset.get_tool('tool_name')]
    )
  ```
  """

  def __init__(
      self,
      *,
      spec_dict: Optional[Dict[str, Any]] = None,
      spec_str: Optional[str] = None,
      spec_str_type: Literal["json", "yaml"] = "json",
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
  ):
    """Initializes the OpenAPIToolset.

    Usage:
    ```
      # Initialize OpenAPI toolset from a spec string.
      openapi_toolset = OpenAPIToolset(spec_str=openapi_spec_str,
        spec_str_type="json")
      # Or, initialize OpenAPI toolset from a spec dictionary.
      openapi_toolset = OpenAPIToolset(spec_dict=openapi_spec_dict)

      # Add all tools to an agent.
      agent = Agent(
        tools=[*openapi_toolset.get_tools()]
      )
      # Or, add a single tool to an agent.
      agent = Agent(
        tools=[openapi_toolset.get_tool('tool_name')]
      )
    ```

    Args:
      spec_dict: The OpenAPI spec dictionary. If provided, it will be used
        instead of loading the spec from a string.
      spec_str: The OpenAPI spec string in JSON or YAML format. It will be used
        when spec_dict is not provided.
      spec_str_type: The type of the OpenAPI spec string. Can be "json" or
        "yaml".
      auth_scheme: The auth scheme to use for all tools. Use AuthScheme or use
        helpers in `google.adk.tools.openapi_tool.auth.auth_helpers`
      auth_credential: The auth credential to use for all tools. Use
        AuthCredential or use helpers in
        `google.adk.tools.openapi_tool.auth.auth_helpers`
    """
    if not spec_dict:
      spec_dict = self._load_spec(spec_str, spec_str_type)
    self.tools: Final[List[RestApiTool]] = list(self._parse(spec_dict))
    if auth_scheme or auth_credential:
      self._configure_auth_all(auth_scheme, auth_credential)

  def _configure_auth_all(
      self, auth_scheme: AuthScheme, auth_credential: AuthCredential
  ):
    """Configure auth scheme and credential for all tools."""

    for tool in self.tools:
      if auth_scheme:
        tool.configure_auth_scheme(auth_scheme)
      if auth_credential:
        tool.configure_auth_credential(auth_credential)

  def get_tools(self) -> List[RestApiTool]:
    """Get all tools in the toolset."""
    return self.tools

  def get_tool(self, tool_name: str) -> Optional[RestApiTool]:
    """Get a tool by name."""
    matching_tool = filter(lambda t: t.name == tool_name, self.tools)
    return next(matching_tool, None)

  def _load_spec(
      self, spec_str: str, spec_type: Literal["json", "yaml"]
  ) -> Dict[str, Any]:
    """Loads the OpenAPI spec string into a dictionary."""
    if spec_type == "json":
      return json.loads(spec_str)
    elif spec_type == "yaml":
      return yaml.safe_load(spec_str)
    else:
      raise ValueError(f"Unsupported spec type: {spec_type}")

  def _parse(self, openapi_spec_dict: Dict[str, Any]) -> List[RestApiTool]:
    """Parse OpenAPI spec into a list of RestApiTool."""
    operations = OpenApiSpecParser().parse(openapi_spec_dict)

    tools = []
    for o in operations:
      tool = RestApiTool.from_parsed_operation(o)
      logger.info("Parsed tool: %s", tool.name)
      tools.append(tool)
    return tools
