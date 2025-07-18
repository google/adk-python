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

from typing import List
from typing import Optional
from typing import Union

from typing_extensions import override
import yaml

from ...agents.readonly_context import ReadonlyContext
from ...auth.auth_credential import AuthCredential
from ...auth.auth_schemes import AuthScheme
from .._gemini_schema_util import _to_snake_case
from ..base_toolset import BaseToolset
from ..base_toolset import ToolPredicate
from ..openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from ..openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from .clients.apihub_client import APIHubClient


class APIHubToolset(BaseToolset):
  """APIHubTool generates tools from a given API Hub resource.

  Examples::

    apihub_toolset = APIHubToolset(
        apihub_resource_name="projects/test-project/locations/us-central1/apis/test-api",
        service_account_json="...",
        tool_filter=lambda tool, ctx=None: tool.name in ('my_tool',
        'my_other_tool')
    )

    # Get all available tools
    agent = LlmAgent(tools=apihub_toolset)

  **apihub_resource_name** is the resource name from API Hub. It must include
  API name, and can optionally include API version and spec name.

  - If apihub_resource_name includes a spec resource name, the content of that
    spec will be used for generating the tools.
  - If apihub_resource_name includes only an api or a version name, the
    first spec of the first version of that API will be used.
  """

  def __init__(
      self,
      *,
      # Parameters for fetching API Hub resource
      apihub_resource_name: str,
      access_token: Optional[str] = None,
      service_account_json: Optional[str] = None,
      # Parameters for the toolset itself
      name: str = '',
      description: str = '',
      # Parameters for generating tools
      lazy_load_spec=False,
      auth_scheme: Optional[AuthScheme] = None,
      auth_credential: Optional[AuthCredential] = None,
      # Optionally, you can provide a custom API Hub client
      apihub_client: Optional[APIHubClient] = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    """Initializes the APIHubTool with the given parameters.

    Examples::

      apihub_toolset = APIHubToolset(
          apihub_resource_name="projects/test-project/locations/us-central1/apis/test-api",
          service_account_json="...",
      )

      # Get all available tools
      agent = LlmAgent(tools=[apihub_toolset])

      apihub_toolset = APIHubToolset(
          apihub_resource_name="projects/test-project/locations/us-central1/apis/test-api",
          service_account_json="...",
          tool_filter = ['my_tool']
      )
      # Get a specific tool
      agent = LlmAgent(tools=[
          ...,
          apihub_toolset,
      ])

    **apihub_resource_name** is the resource name from API Hub. It must include
    API name, and can optionally include API version and spec name.

    - If apihub_resource_name includes a spec resource name, the content of that
      spec will be used for generating the tools.
    - If apihub_resource_name includes only an api or a version name, the
      first spec of the first version of that API will be used.

    Example:

    * projects/xxx/locations/us-central1/apis/apiname/...
    * https://console.cloud.google.com/apigee/api-hub/apis/apiname?project=xxx

    Args:
        apihub_resource_name: The resource name of the API in API Hub.
          Example: ``projects/test-project/locations/us-central1/apis/test-api``.
        access_token: Google Access token. Generate with gcloud cli
          ``gcloud auth auth print-access-token``. Used for fetching API Specs from API Hub.
        service_account_json: The service account config as a json string.
          Required if not using default service credential. It is used for
          creating the API Hub client and fetching the API Specs from API Hub.
        apihub_client: Optional custom API Hub client.
        name: Name of the toolset. Optional.
        description: Description of the toolset. Optional.
        auth_scheme: Auth scheme that applies to all the tool in the toolset.
        auth_credential: Auth credential that applies to all the tool in the
          toolset.
        lazy_load_spec: If True, the spec will be loaded lazily when needed.
          Otherwise, the spec will be loaded immediately and the tools will be
          generated during initialization.
        tool_filter: The filter used to filter the tools in the toolset. It can
          be either a tool predicate or a list of tool names of the tools to
          expose.
    """
    super().__init__(tool_filter=tool_filter)
    self.name = name
    self.description = description
    self._apihub_resource_name = apihub_resource_name
    self._lazy_load_spec = lazy_load_spec
    self._apihub_client = apihub_client or APIHubClient(
        access_token=access_token,
        service_account_json=service_account_json,
    )

    self._openapi_toolset = None
    self._auth_scheme = auth_scheme
    self._auth_credential = auth_credential

    if not self._lazy_load_spec:
      self._prepare_toolset()

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[RestApiTool]:
    """Retrieves all available tools.

    Returns:
        A list of all available RestApiTool objects.
    """
    if not self._openapi_toolset:
      self._prepare_toolset()
    if not self._openapi_toolset:
      return []
    return await self._openapi_toolset.get_tools(readonly_context)

  def _prepare_toolset(self) -> None:
    """Fetches the spec from API Hub and generates the toolset."""
    # For each API, get the first version and the first spec of that version.
    spec_str = self._apihub_client.get_spec_content(self._apihub_resource_name)
    spec_dict = yaml.safe_load(spec_str)
    if not spec_dict:
      return

    self.name = self.name or _to_snake_case(
        spec_dict.get('info', {}).get('title', 'unnamed')
    )
    self.description = self.description or spec_dict.get('info', {}).get(
        'description', ''
    )
    self._openapi_toolset = OpenAPIToolset(
        spec_dict=spec_dict,
        auth_credential=self._auth_credential,
        auth_scheme=self._auth_scheme,
        tool_filter=self.tool_filter,
    )

  @override
  async def close(self):
    if self._openapi_toolset:
      await self._openapi_toolset.close()
