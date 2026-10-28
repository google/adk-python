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

"""MongoDB toolset for vector search and hybrid search operations."""

from __future__ import annotations

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
  from google.genai import Client
  from pymongo import MongoClient


class _MongoDbTool(FunctionTool):
  """FunctionTool that injects the bound MongoDB client, database and settings.

  The `client`, `database_name`, `settings` and `genai_client` parameters are
  configured on the toolset and hidden from the LLM, so the model only sees
  the search parameters of each tool.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      client: MongoClient,
      database_name: str,
      settings: MongoDbToolSettings,
      genai_client: Client | None = None,
  ):
    super().__init__(func=func)
    self._ignore_params.append("client")
    self._ignore_params.append("database_name")
    self._ignore_params.append("settings")
    self._ignore_params.append("genai_client")
    self._client = client
    self._database_name = database_name
    self._settings = settings
    self._genai_client = genai_client

  @override
  def _detect_error_in_response(self, response: Any) -> str | None:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("status") == "ERROR":
      return "TOOL_ERROR"
    return super()._detect_error_in_response(response)

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
    if self._genai_client and "genai_client" in signature.parameters:
      args_to_call["genai_client"] = self._genai_client
    return await super().run_async(args=args_to_call, tool_context=tool_context)


DEFAULT_MONGODB_TOOL_NAME_PREFIX = "mongodb"


@experimental(FeatureName.MONGODB_TOOLSET)
class MongoDbToolset(BaseToolset):
  """MongoDB Toolset contains tools for vector search and hybrid search.

  The tool names are:
    - mongodb_vector_search
    - mongodb_hybrid_search

  The toolset binds one database. Within it the model picks the collection,
  the search index and the fields to return on each call, so every collection
  in `database_name` that carries a search index is reachable: point it at
  data the agent is allowed to read. `tool_filter` does not narrow this,
  because it selects whole tools and never sees call arguments. A
  per-collection policy belongs in a `before_tool_callback`, which does see
  them.

  Example:
      ```python
      toolset = MongoDbToolset(
          connection_string="mongodb+srv://user:pass@cluster.mongodb.net/",
          database_name="products_db",
      )
      agent = Agent(model="gemini-3.5-flash", tools=[toolset])
      ```

  When constructed with `connection_string` or `connection_string_env_var`
  and no `genai_client`, the toolset is picklable, so the agent can ship to
  Agent Engine (which packages apps with cloudpickle): the live `MongoClient`
  is dropped at pickle time and rebuilt from the connection string on the
  runtime. A toolset built with a caller-owned `mongo_client` or
  `genai_client` cannot be pickled.

  Prefer `connection_string_env_var` for Agent Engine deployments: a plain
  `connection_string` is stored on the toolset and would be pickled into the
  deployment package in plaintext, password included. The env-var form keeps
  only the variable's name on the toolset and resolves the value on the
  runtime, so the package holds no credential.
  """

  def __init__(
      self,
      *,
      database_name: str,
      connection_string: str | None = None,
      connection_string_env_var: str | None = None,
      mongo_client: MongoClient | None = None,
      tool_filter: ToolPredicate | list[str] | None = None,
      settings: MongoDbToolSettings | None = None,
      genai_client: Client | None = None,
  ):
    """Initializes the MongoDbToolset.

    Args:
      database_name: The MongoDB database the search tools operate on. The
        model chooses the collection within it, so it should hold only data
        the agent may read.
      connection_string: The MongoDB connection string (URI) used to create a
        client owned by this toolset. Requires the `pymongo` package
        (`pip install google-adk[mongodb]`).
      connection_string_env_var: Name of an environment variable holding the
        connection string, resolved where the client is built. Prefer this
        over `connection_string` for Agent Engine deployments so the
        credential is not pickled into the deployment package.
      mongo_client: An existing PyMongo client to use instead of creating one
        from `connection_string`. The caller keeps ownership of the client,
        and `settings.timeout_ms` is applied per operation rather than to the
        client.
      tool_filter: Filter to apply to tools.
      settings: The settings for the MongoDB tools.
      genai_client: The `google.genai` client used to embed the query text.
        Defaults to a client built from the ambient environment.

    Raises:
      ValueError: If more than one of `connection_string`,
        `connection_string_env_var` and `mongo_client` are given, or none.
    """
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_MONGODB_TOOL_NAME_PREFIX,
    )
    self._settings = settings if settings else MongoDbToolSettings()
    sources = [
        source is not None
        for source in (
            connection_string,
            connection_string_env_var,
            mongo_client,
        )
    ]
    if sum(sources) != 1:
      raise ValueError(
          "Exactly one of `connection_string`, `connection_string_env_var` or"
          " `mongo_client` must be provided."
      )
    if mongo_client is not None:
      self._client = mongo_client
      self._owns_client = False
      self._connection_string = None
    else:
      if connection_string is not None:
        resolved = connection_string
      else:
        assert connection_string_env_var is not None
        resolved = _client.connection_string_from_env(connection_string_env_var)
      self._client = _client.get_mongo_client(
          resolved, timeout_ms=self._settings.timeout_ms
      )
      self._owns_client = True
      # None in env-var mode, so the credential is never pickled.
      self._connection_string = connection_string
    self._connection_string_env_var = connection_string_env_var
    self._database_name = database_name
    self._genai_client = genai_client

  def __getstate__(self) -> dict[str, Any]:
    """Drops the unpicklable client so the toolset can ship to Agent Engine.

    Agent Engine packages the app with cloudpickle; neither a MongoClient
    (sockets, locks, background threads) nor a genai Client (httpx pools)
    crosses that boundary. The MongoClient is rebuilt on restore from the
    connection string, or, with `connection_string_env_var`, re-resolved from
    the named environment variable on the destination. A caller-owned
    `mongo_client` or `genai_client` raises `TypeError`.
    """
    state = _client.drop_client_for_pickle(
        self.__dict__, owns_client=self._owns_client, owner="MongoDbToolset"
    )
    if self._genai_client is not None:
      raise TypeError(
          "MongoDbToolset cannot be pickled when constructed with"
          " `genai_client`: a caller-owned client cannot be rebuilt on the"
          " destination."
      )
    # Caches hold tool instances that reference the dropped client.
    state["_cached_invocation_id"] = None
    state["_cached_prefixed_tools"] = None
    return state

  def __setstate__(self, state: dict[str, Any]) -> None:
    self.__dict__.update(state)
    connection_string = self._connection_string
    if connection_string is None:
      # Env-var form: the credential was never pickled; resolve it on the
      # destination (e.g. injected by Secret Manager on Agent Engine).
      assert self._connection_string_env_var is not None
      connection_string = _client.connection_string_from_env(
          self._connection_string_env_var
      )
    self._client = _client.get_mongo_client(
        connection_string, timeout_ms=self._settings.timeout_ms
    )

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> list[BaseTool]:
    """Get tools from the toolset."""
    funcs: list[Callable[..., Any]] = [
        _search_tool.vector_search,
        _search_tool.hybrid_search,
    ]
    all_tools = [
        _MongoDbTool(
            func=func,
            client=self._client,
            database_name=self._database_name,
            settings=self._settings,
            genai_client=self._genai_client,
        )
        for func in funcs
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
