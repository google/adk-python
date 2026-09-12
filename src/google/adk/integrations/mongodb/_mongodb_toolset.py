# Copyright 2026 Google LLC
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

"""MongoDB toolset for vector search and hybrid search operations."""

import asyncio
import inspect
from typing import Any
from typing import Callable
from typing import TYPE_CHECKING

from typing_extensions import override

from . import _client
from . import _search_tool
from ...agents.readonly_context import ReadonlyContext
from ...features import experimental
from ...features import FeatureName
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.function_tool import FunctionTool
from ...tools.tool_context import ToolContext
from ._settings import MongoDbToolSettings

if TYPE_CHECKING:
  from pymongo import MongoClient


class _MongoDbTool(FunctionTool):
  """FunctionTool that injects the bound MongoDB client, database and settings.

  The `client`, `database_name` and `settings` parameters are configured on
  the toolset and hidden from the LLM, so the model only sees the search
  parameters of each tool.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      client: MongoClient,
      database_name: str,
      settings: MongoDbToolSettings,
  ):
    super().__init__(func=func)
    self._ignore_params.append("client")
    self._ignore_params.append("database_name")
    self._ignore_params.append("settings")
    self._client = client
    self._database_name = database_name
    self._settings = settings

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    args_to_call = args.copy()
    signature = inspect.signature(self.func)
    if "client" in signature.parameters:
      args_to_call["client"] = self._client
    if "database_name" in signature.parameters:
      args_to_call["database_name"] = self._database_name
    if "settings" in signature.parameters:
      args_to_call["settings"] = self._settings
    return await super().run_async(args=args_to_call, tool_context=tool_context)


DEFAULT_MONGODB_TOOL_NAME_PREFIX = "mongodb"


@experimental(FeatureName.MONGODB_TOOLSET)
class MongoDbToolset(BaseToolset):
  """MongoDB Toolset contains tools for vector search and hybrid search.

  The tool names are:
    - mongodb_vector_search
    - mongodb_hybrid_search

  Example:
      ```python
      toolset = MongoDbToolset(
          connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
          database_name="products_db",
      )
      agent = Agent(model="gemini-2.5-flash", tools=[toolset])
      ```
  """

  def __init__(
      self,
      *,
      database_name: str,
      connection_string: str | None = None,
      mongo_client: MongoClient | None = None,
      tool_filter: ToolPredicate | list[str] | None = None,
      settings: MongoDbToolSettings | None = None,
  ):
    """Initializes the MongoDbToolset.

    Args:
      database_name: The MongoDB database the search tools operate on.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this toolset. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`).
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client.
      tool_filter: Filter to apply to tools.
      settings: The settings for the MongoDB tools.
    """
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_MONGODB_TOOL_NAME_PREFIX,
    )
    if mongo_client is not None and connection_string is not None:
      raise ValueError(
          "Only one of `connection_string` and `mongo_client` may be provided."
      )
    if mongo_client is not None:
      self._client = mongo_client
      self._owns_client = False
    elif connection_string is not None:
      self._client = _client.get_mongo_client(connection_string)
      self._owns_client = True
    else:
      raise ValueError(
          "Either `connection_string` or `mongo_client` must be provided."
      )
    self._database_name = database_name
    self._settings = settings if settings else MongoDbToolSettings()

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Get tools from the toolset."""
    all_tools = [
        _MongoDbTool(
            func=func,
            client=self._client,
            database_name=self._database_name,
            settings=self._settings,
        )
        for func in [
            _search_tool.vector_search,
            _search_tool.hybrid_search,
        ]
    ]
    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]

  @override
  async def close(self) -> None:
    """Closes the MongoDB client if it was created by this toolset."""
    if self._owns_client:
      await asyncio.to_thread(self._client.close)
