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

import inspect
import os
from typing import Any
from typing import List
from typing import Optional
from typing import Type
from typing import Union

from typing_extensions import override

from ...agents.readonly_context import ReadonlyContext
from ...auth import OpenIdConnectWithConfig
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ..openapi_tool import OpenAPIToolset
from .google_api_tool import GoogleApiTool
from .googleapi_to_openapi_converter import GoogleApiToOpenApiConverter


class GoogleApiToolset(BaseToolset):
  """Google API Toolset contains tools for interacting with Google APIs.

  Usually one toolsets will contains tools only replated to one Google API, e.g.
  Google Bigquery API toolset will contains tools only related to Google
  Bigquery API, like list dataset tool, list table tool etc.
  """

  def __init__(
      self,
      openapi_toolset: OpenAPIToolset,
      client_id: Optional[str] = None,
      client_secret: Optional[str] = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    self.openapi_toolset = openapi_toolset
    self.tool_filter = tool_filter
    self.client_id = client_id
    self.client_secret = client_secret

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[GoogleApiTool]:
    """Get all tools in the toolset."""
    tools = []

    for tool in await self.openapi_toolset.get_tools(readonly_context):
      if self.tool_filter and (
          isinstance(self.tool_filter, ToolPredicate)
          and not self.tool_filter(tool, readonly_context)
          or isinstance(self.tool_filter, list)
          and tool.name not in self.tool_filter
      ):
        continue
      google_api_tool = GoogleApiTool(tool)
      google_api_tool.configure_auth(self.client_id, self.client_secret)
      tools.append(google_api_tool)

    return tools

  def set_tool_filter(self, tool_filter: Union[ToolPredicate, List[str]]):
    self.tool_filter = tool_filter

  @staticmethod
  def _load_toolset_with_oidc_auth(
      spec_file: Optional[str] = None,
      spec_dict: Optional[dict[str, Any]] = None,
      scopes: Optional[list[str]] = None,
  ) -> OpenAPIToolset:
    spec_str = None
    if spec_file:
      # Get the frame of the caller
      caller_frame = inspect.stack()[1]
      # Get the filename of the caller
      caller_filename = caller_frame.filename
      # Get the directory of the caller
      caller_dir = os.path.dirname(os.path.abspath(caller_filename))
      # Join the directory path with the filename
      yaml_path = os.path.join(caller_dir, spec_file)
      with open(yaml_path, 'r', encoding='utf-8') as file:
        spec_str = file.read()
    toolset = OpenAPIToolset(
        spec_dict=spec_dict,
        spec_str=spec_str,
        spec_str_type='yaml',
        auth_scheme=OpenIdConnectWithConfig(
            authorization_endpoint=(
                'https://accounts.google.com/o/oauth2/v2/auth'
            ),
            token_endpoint='https://oauth2.googleapis.com/token',
            userinfo_endpoint=(
                'https://openidconnect.googleapis.com/v1/userinfo'
            ),
            revocation_endpoint='https://oauth2.googleapis.com/revoke',
            token_endpoint_auth_methods_supported=[
                'client_secret_post',
                'client_secret_basic',
            ],
            grant_types_supported=['authorization_code'],
            scopes=scopes,
        ),
    )
    return toolset

  def configure_auth(self, client_id: str, client_secret: str):
    self.client_id = client_id
    self.client_secret = client_secret

  @classmethod
  def load_toolset(
      cls: Type[GoogleApiToolset],
      api_name: str,
      api_version: str,
  ) -> GoogleApiToolset:
    spec_dict = GoogleApiToOpenApiConverter(api_name, api_version).convert()
    scope = list(
        spec_dict['components']['securitySchemes']['oauth2']['flows'][
            'authorizationCode'
        ]['scopes'].keys()
    )[0]
    return cls(
        cls._load_toolset_with_oidc_auth(spec_dict=spec_dict, scopes=[scope])
    )

  @override
  async def close(self):
    if self.openapi_toolset:
      await self.openapi_toolset.close()
