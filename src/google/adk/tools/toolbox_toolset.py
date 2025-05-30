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
from typing import Callable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import toolbox_core as toolbox
from typing_extensions import override

from ..agents.readonly_context import ReadonlyContext
from .base_tool import BaseTool
from .base_toolset import BaseToolset
from .function_tool import FunctionTool


class ToolboxToolset(BaseToolset):
  """A class that provides access to toolbox toolsets.

  Example:
  ```python
  toolbox_toolset = ToolboxToolset("http://127.0.0.1:5000",
  toolset_name="my-toolset")
  )
  ```
  """

  def __init__(
      self,
      server_url: str,
      toolset_name: Optional[str] = None,
      tool_names: Optional[List[str]] = None,
      auth_token_getters: Optional[dict[str, Callable[[], str]]] = None,
      bound_params: Optional[
          Mapping[str, Union[Callable[[], Any], Any]]
      ] = None,
  ):
    """Args:

      server_url: The URL of the toolbox server.
      toolset_name: The name of the toolbox toolset to load.
      tool_names: The names of the tools to load.
      auth_token_getters: A mapping of authentication service names to
        callables that return the corresponding authentication token. see:
        https://github.com/googleapis/mcp-toolbox-sdk-python/tree/main/packages/toolbox-core#authenticating-tools
        for details.
      bound_params: A mapping of parameter names to bind to specific values or
        callables that are called to produce values as needed. see:
        https://github.com/googleapis/mcp-toolbox-sdk-python/tree/main/packages/toolbox-core#binding-parameter-values
        for details.
    The resulting ToolboxToolset will contain both tools loaded by tool_names
    and toolset_name.
    """
    if not tool_names and not toolset_name:
      raise ValueError("tool_names and toolset_name cannot both be None")
    super().__init__()
    self._server_url = server_url
    self._toolbox_client = toolbox.ToolboxClient(server_url)
    self._toolset_name = toolset_name
    self._tool_names = tool_names
    self._auth_token_getters = auth_token_getters or {}
    self._bound_params = bound_params or {}

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    tools = []
    if self._toolset_name:
      tools.extend([
          FunctionTool(tool)
          for tool in await self._toolbox_client.load_toolset(
              self._toolset_name,
              auth_token_getters=self._auth_token_getters,
              bound_params=self._bound_params,
          )
      ])
    if self._tool_names:
      tools.extend([
          FunctionTool(
              await self._toolbox_client.load_tool(
                  tool_name,
                  auth_token_getters=self._auth_token_getters,
                  bound_params=self._bound_params,
              )
          )
          for tool_name in self._tool_names
      ])
    return tools

  @override
  async def close(self):
    self._toolbox_client.close()
