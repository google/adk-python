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
from typing import Callable
from typing import List
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from typing_extensions import override

from ..agents.readonly_context import ReadonlyContext
from .base_tool import BaseTool
from .base_toolset import BaseToolset

if TYPE_CHECKING:
  from toolbox_adk import CredentialConfig


class ToolboxToolset(BaseToolset):
  """A class that provides access to toolbox toolsets.

  This class acts as a bridge to the `toolbox-adk` package.
  You must install `toolbox-adk` to use this class.

  Example:
  ```python
  from toolbox_adk import CredentialStrategy

  toolbox_toolset = ToolboxToolset(
      server_url="http://127.0.0.1:5000",
      # toolset_name and tool_names are optional. If omitted, all tools are
      loaded.
      credentials=CredentialStrategy.toolbox_identity()
  )
  ```
  """

  def __init__(
      self,
      server_url: str,
      toolset_name: Optional[str] = None,
      tool_names: Optional[List[str]] = None,
      auth_token_getters: Optional[Mapping[str, Callable[[], str]]] = None,
      bound_params: Optional[
          Mapping[str, Union[Callable[[], Any], Any]]
      ] = None,
      credentials: Optional[CredentialConfig] = None,
      additional_headers: Optional[Mapping[str, str]] = None,
      **kwargs,
  ):
    """Args:

    server_url: The URL of the toolbox server.
    toolset_name: The name of the toolbox toolset to load.
    tool_names: The names of the tools to load.
    auth_token_getters: (Deprecated) Map of auth token getters.
    bound_params: Parameters to bind to the tools.
    credentials: (Optional) toolbox_adk.CredentialConfig object.
    additional_headers: (Optional) Static headers dictionary.
    **kwargs: Additional arguments passed to the underlying
    toolbox_adk.ToolboxToolset.
    """
    if not toolset_name and not tool_names:
      raise ValueError(
          "Either 'toolset_name' or 'tool_names' must be provided."
      )

    try:
      from toolbox_adk import ToolboxToolset as RealToolboxToolset  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
      raise ImportError(
          "ToolboxToolset requires the 'toolbox-adk' package. "
          "Please install it using `pip install toolbox-adk`."
      ) from exc

    super().__init__()

    self._delegate = RealToolboxToolset(
        server_url=server_url,
        toolset_name=toolset_name,
        tool_names=tool_names,
        credentials=credentials,
        additional_headers=additional_headers,
        bound_params=bound_params,
        auth_token_getters=auth_token_getters,
        **kwargs,
    )

  @override
  async def get_tools(
      self, readonly_context: Optional[ReadonlyContext] = None
  ) -> list[BaseTool]:
    return await self._delegate.get_tools(readonly_context)

  @override
  async def close(self):
    await self._delegate.close()
