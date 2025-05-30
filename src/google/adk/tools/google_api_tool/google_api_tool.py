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

from typing import Any
from typing import Dict
from typing import Optional

from google.genai.types import FunctionDeclaration
from typing_extensions import override

from .. import BaseTool
from ...auth import AuthCredential
from ...auth import AuthCredentialTypes
from ...auth import OAuth2Auth
from ..openapi_tool import RestApiTool
from ..tool_context import ToolContext


class GoogleApiTool(BaseTool):

  def __init__(
      self,
      rest_api_tool: RestApiTool,
      client_id: Optional[str] = None,
      client_secret: Optional[str] = None,
  ):
    super().__init__(
        name=rest_api_tool.name,
        description=rest_api_tool.description,
        is_long_running=rest_api_tool.is_long_running,
    )
    self._rest_api_tool = rest_api_tool
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
