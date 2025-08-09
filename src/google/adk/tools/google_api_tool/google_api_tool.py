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

from typing import Any
from typing import Dict
from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from ...auth.auth_credential import AuthCredential
from ...auth.auth_credential import AuthCredentialTypes
from ...auth.auth_credential import OAuth2Auth
from ...auth.auth_credential import ServiceAccount
from ..base_tool import BaseTool
from ..openapi_tool import RestApiTool
from ..openapi_tool.auth.auth_helpers import service_account_scheme_credential
from ..tool_context import ToolContext


class GoogleApiTool(BaseTool):

  def __init__(
      self,
      rest_api_tool: RestApiTool,
      client_id: Optional[str] = None,
      client_secret: Optional[str] = None,
      service_account: Optional[ServiceAccount] = None,
  ):
    super().__init__(
        name=rest_api_tool.name,
        description=rest_api_tool.description,
        is_long_running=rest_api_tool.is_long_running,
    )
    self._rest_api_tool = rest_api_tool
    if service_account is not None:
      self.configure_sa_auth(service_account)
    else:
      self.configure_auth(client_id, client_secret)

  @override
  def _get_declaration(self) -> FunctionDeclaration:
    return self._rest_api_tool._get_declaration()

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: Optional[ToolContext]
  ) -> Dict[str, Any]:
    return await self._rest_api_tool.run_async(
        args=args, tool_context=tool_context
    )

  def configure_auth(self, client_id: str, client_secret: str):
    self._rest_api_tool.auth_credential = AuthCredential(
        auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
        oauth2=OAuth2Auth(
            client_id=client_id,
            client_secret=client_secret,
        ),
    )

  def configure_sa_auth(self, service_account: ServiceAccount):
    auth_scheme, auth_credential = service_account_scheme_credential(
        service_account
    )
    self._rest_api_tool.auth_scheme = auth_scheme
    self._rest_api_tool.auth_credential = auth_credential
