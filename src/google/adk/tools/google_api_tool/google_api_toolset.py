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

  Usually one toolsets will contains tools only related to one Google API, e.g.
  Google Bigquery API toolset will contains tools only related to Google
  Bigquery API, like list dataset tool, list table tool etc.
  """

  def __init__(
      self,
      api_name: str,
      api_version: str,
      client_id: Optional[str] = None,
      client_secret: Optional[str] = None,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
  ):
    self.api_name = api_name
    self.api_version = api_version
    self._client_id = client_id
    self._client_secret = client_secret
    self._openapi_toolset = self._load_toolset_with_oidc_auth()
    self.tool_filter = tool_filter

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> List[GoogleApiTool]:
    """Get all tools in the toolset."""
    tools = []

    return [
        GoogleApiTool(tool, self._client_id, self._client_secret)
        for tool in await self._openapi_toolset.get_tools(readonly_context)
        if self._is_tool_selected(tool, readonly_context)
    ]

  def set_tool_filter(self, tool_filter: Union[ToolPredicate, List[str]]):
    self.tool_filter = tool_filter

  def _load_toolset_with_oidc_auth(self) -> OpenAPIToolset:
    spec_dict = GoogleApiToOpenApiConverter(
        self.api_name, self.api_version
    ).convert()
    scope = list(
        spec_dict['components']['securitySchemes']['oauth2']['flows'][
            'authorizationCode'
        ]['scopes'].keys()
    )[0]
    return OpenAPIToolset(
        spec_dict=spec_dict,
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
            scopes=[scope],
        ),
    )

  def configure_auth(self, client_id: str, client_secret: str):
    self._client_id = client_id
    self._client_secret = client_secret

  @override
  async def close(self):
    if self._openapi_toolset:
      await self._openapi_toolset.close()
