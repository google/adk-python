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
from typing import Final
from typing import List
from typing import Optional
from typing import Type

from ...auth import OpenIdConnectWithConfig
from ..openapi_tool import OpenAPIToolset
from ..openapi_tool import RestApiTool
from .google_api_tool import GoogleApiTool
from .googleapi_to_openapi_converter import GoogleApiToOpenApiConverter


class GoogleApiToolSet:
  """Google API Tool Set."""

  def __init__(self, tools: List[RestApiTool]):
    self.tools: Final[List[GoogleApiTool]] = [
        GoogleApiTool(tool) for tool in tools
    ]

  def get_tools(self) -> List[GoogleApiTool]:
    """Get all tools in the toolset."""
    return self.tools

  def get_tool(self, tool_name: str) -> Optional[GoogleApiTool]:
    """Get a tool by name."""
    matching_tool = filter(lambda t: t.name == tool_name, self.tools)
    return next(matching_tool, None)

  @staticmethod
  def _load_tool_set_with_oidc_auth(
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
    tool_set = OpenAPIToolset(
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
    return tool_set

  def configure_auth(self, client_id: str, client_secret: str):
    for tool in self.tools:
      tool.configure_auth(client_id, client_secret)

  @classmethod
  def load_tool_set(
      cls: Type[GoogleApiToolSet],
      api_name: str,
      api_version: str,
  ) -> GoogleApiToolSet:
    spec_dict = GoogleApiToOpenApiConverter(api_name, api_version).convert()
    scope = list(
        spec_dict['components']['securitySchemes']['oauth2']['flows'][
            'authorizationCode'
        ]['scopes'].keys()
    )[0]
    return cls(
        cls._load_tool_set_with_oidc_auth(
            spec_dict=spec_dict, scopes=[scope]
        ).get_tools()
    )
