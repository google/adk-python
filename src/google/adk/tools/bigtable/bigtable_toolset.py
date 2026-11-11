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

from collections.abc import Awaitable
from collections.abc import Mapping
from contextvars import ContextVar
import inspect
import logging
from typing import Any
from typing import Callable
from typing import List

from google.adk.agents.readonly_context import ReadonlyContext
from google.auth.credentials import Credentials
from pydantic import BaseModel
from typing_extensions import override

from . import metadata_tool
from . import query_tool
from ...tools.base_tool import BaseTool
from ...tools.base_toolset import BaseToolset
from ...tools.base_toolset import ToolPredicate
from ...tools.google_tool import GoogleTool
from ..tool_context import ToolContext
from .bigtable_credentials import BigtableCredentialsConfig
from .settings import BigtableToolSettings

logger = logging.getLogger("google_adk." + __name__)

_view_param_error: ContextVar[list[Exception] | None] = ContextVar(
    "_view_param_error", default=None
)


def _validate_view_parameters(view_parameters: object) -> None:
  if (
      view_parameters is not None
      and not callable(view_parameters)
      and not isinstance(view_parameters, Mapping)
  ):
    raise TypeError(
        "view_parameters must be a Mapping, callable, or None, got"
        f" {type(view_parameters).__name__}"
    )


class _BigtableQueryTool(GoogleTool):
  """Plain execute_sql tool that keeps _view_parameters away from the model."""

  def __init__(
      self,
      *,
      credentials_config: BigtableCredentialsConfig | None = None,
      tool_settings: BigtableToolSettings | None = None,
  ):
    super().__init__(
        func=query_tool.execute_sql,
        credentials_config=credentials_config,
        tool_settings=tool_settings,
    )
    self._ignore_params.append("_view_parameters")

  @override
  async def _run_async_with_credential(
      self,
      credentials: Credentials | None,
      tool_settings: BaseModel | None,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    args = {k: v for k, v in args.items() if k != "_view_parameters"}
    return await super()._run_async_with_credential(
        credentials, tool_settings, args, tool_context
    )


class BigtableParameterizedViewTool(GoogleTool):
  """Wrapper FunctionTool for Bigtable execute_sql query tool that passes view parameters.

  This tool wraps the Bigtable query tool to automatically resolve and inject
  parameters into the query's `view_parameters`. Top-level `ToolContext`
  attribute names (e.g. `user_id`) are configured via `view_parameter_names`,
  and static or per-invocation resolved parameter values (e.g. `tenant_id`) are
  configured via `view_parameters`.

  Example:
      If a parameterized view `purchase_history_pv` was created with the query:
      `SELECT * FROM purchases WHERE user_id = VIEW_PARAMETERS('user_id')`

      By configuring `view_parameter_names=["user_id"]`, the wrapper will
      resolve the `user_id` value from the `tool_context.user_id` at runtime and
      pass it as `view_parameters={"user_id": user_id}`.
      This securely restricts query execution to the logged-in user's data
      without exposing the `user_id` parameter to the LLM.

      Only names that resolve to an attribute of `tool_context` itself (such as
      `user_id`) are honored by `view_parameter_names`. `tool_context.state` is
      not consulted: state is writable by the caller, so a name that fell back
      to state could be overridden to run the query in a different user's scope,
      defeating the view's own row-level restriction. Additional parameters such
      as `tenant_id` should be supplied via `view_parameters` from a trusted
      source that the caller and model cannot write (for example, a lookup keyed
      on `tool_context.user_id`), and never from `tool_context.state` or
      `tool_context.custom_metadata`:

      - Mapping: `view_parameters={"tenant_id": "tenant-abc"}`
      - Mapping of callables:
        `view_parameters={"tenant_id": lambda ctx: get_tenant(ctx.user_id)}`
      - Callable receiving `ToolContext`:
        `view_parameters=lambda ctx: {"tenant_id": get_tenant(ctx.user_id)}`

  Attributes:
    view_parameter_names: Parameter names to resolve from `tool_context`'s own
      attributes (not `tool_context.state`) and pass into `view_parameters`.
    view_parameters: Mapping of parameter values or per-value callables, or a
      callable receiving `ToolContext` that returns a mapping of trusted
      parameter values (or `None`) resolved per invocation.
  """

  def __init__(
      self,
      func: Callable[..., Any],
      *,
      credentials_config: BigtableCredentialsConfig | None = None,
      tool_settings: BigtableToolSettings | None = None,
      view_parameter_names: list[str] | None = None,
      view_parameters: (
          Mapping[str, Any]
          | Callable[
              [ToolContext],
              Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None],
          ]
          | None
      ) = None,
  ):
    """Initializes the BigtableParameterizedViewTool.

    Args:
        func: The Bigtable query function to wrap.
        credentials_config: The credentials configuration.
        tool_settings: The tool settings.
        view_parameter_names: A list of parameter names to resolve from
          `tool_context`'s own attributes (such as `user_id`, not
          `tool_context.state`) and pass into `view_parameters`. This is
          configured on `BigtableToolset` and forwarded here.
        view_parameters: A mapping of parameter values, a mapping of per-value
          callables (sync or async), or a callable (sync or async) receiving
          `ToolContext` that returns a mapping of parameter values (such as
          `tenant_id`). Values must come from a source that the caller and model
          cannot write (e.g. a lookup keyed on `ctx.user_id`), and must never be
          read from `ctx.state` or `ctx.custom_metadata`. Keys from
          `view_parameters` take precedence over and overwrite keys resolved
          from `view_parameter_names` (including `user_id`). If a top-level
          callable returns `None`, no parameters from `view_parameters` are
          merged; any key whose value or per-value callable resolves to `None`
          is omitted (and removes any value resolved from
          `view_parameter_names`); if a top-level callable returns a
          non-mapping value, a `TypeError` is raised. Any exception raised by a
          resolver callable propagates to the caller and fails the tool
          invocation.
    """
    _validate_view_parameters(view_parameters)
    super().__init__(
        func=func,
        credentials_config=credentials_config,
        tool_settings=tool_settings,
    )
    self.name = "execute_sql_parameterized"
    self.description = (
        "Execute a GoogleSQL query from a Bigtable table using parameterized"
        " views to securely check permissions."
    )
    self.view_parameter_names = view_parameter_names
    self.view_parameters = view_parameters
    # Exclude from being parsed and exposed to the LLM when generating tool schemas
    self._ignore_params.append("_view_parameters")

  async def _resolve_view_parameters(
      self, tool_context: ToolContext
  ) -> dict[str, Any]:
    view_params: dict[str, Any] = {}
    for param_name in self.view_parameter_names or ():
      if (val := getattr(tool_context, param_name, None)) is not None:
        view_params[param_name] = val

    if self.view_parameters is not None:
      if callable(self.view_parameters):
        resolved = self.view_parameters(tool_context)
        if inspect.isawaitable(resolved):
          resolved = await resolved
        if resolved is not None:
          if not isinstance(resolved, Mapping):
            raise TypeError(
                "view_parameters callable must return a Mapping or None,"
                f" got {type(resolved).__name__}"
            )
          for k, val in resolved.items():
            if val is not None:
              view_params[k] = val
            else:
              view_params.pop(k, None)
      elif isinstance(self.view_parameters, Mapping):
        for k, v in self.view_parameters.items():
          if callable(v):
            val = v(tool_context)
            if inspect.isawaitable(val):
              val = await val
          else:
            val = v
          if val is not None:
            view_params[k] = val
          else:
            view_params.pop(k, None)
      else:
        raise TypeError(
            "view_parameters must be a Mapping, callable, or None, got"
            f" {type(self.view_parameters).__name__}"
        )

    for param_name in self.view_parameter_names or ():
      if (
          param_name not in view_params
          and tool_context.state
          and param_name in tool_context.state
      ):
        logger.warning(
            "View parameter %r is no longer read from session state; pass"
            " it via view_parameters instead.",
            param_name,
        )

    return view_params

  @override
  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    error_holder: list[Exception] = []
    token = _view_param_error.set(error_holder)
    try:
      result = await super().run_async(args=args, tool_context=tool_context)
    finally:
      _view_param_error.reset(token)
    if error_holder:
      raise error_holder[0]
    return result

  @override
  async def _run_async_with_credential(
      self,
      credentials: Credentials | None,
      tool_settings: BaseModel | None,
      args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Any:
    args_to_call = args.copy()
    args_to_call.pop("_view_parameters", None)
    signature = inspect.signature(self.func)
    if "_view_parameters" in signature.parameters and (
        self.view_parameter_names or self.view_parameters is not None
    ):
      try:
        args_to_call["_view_parameters"] = await self._resolve_view_parameters(
            tool_context
        )
      except Exception as ex:
        if (error_holder := _view_param_error.get()) is not None:
          error_holder.append(ex)
        raise
    return await super()._run_async_with_credential(
        credentials, tool_settings, args_to_call, tool_context
    )


DEFAULT_BIGTABLE_TOOL_NAME_PREFIX = "bigtable"


class BigtableToolset(BaseToolset):
  """Bigtable Toolset contains tools for interacting with Bigtable data and metadata.

  The tool names are:
    - bigtable_list_instances
    - bigtable_get_instance_info
    - bigtable_list_tables
    - bigtable_get_table_info
    - bigtable_list_clusters
    - bigtable_get_cluster_info
    - bigtable_execute_sql

  When view_parameter_names or view_parameters is configured,
  bigtable_execute_sql is replaced by bigtable_execute_sql_parameterized, which
  resolves view parameters from `tool_context`'s own attributes (such as
  `user_id`, never session state `tool_context.state`) and from
  `view_parameters`.

  Attributes:
    view_parameter_names: Optional list of parameter names to resolve from
      `tool_context`'s own attributes (such as `user_id`, not
      `tool_context.state`) and pass into `view_parameters` for parameterized
      views.
    view_parameters: Optional mapping of parameter values, mapping of per-value
      callables (sync or async), or a callable (sync or async) receiving
      `ToolContext` that returns a mapping of trusted parameter values (such as
      `tenant_id`, or `None`) resolved per invocation for parameterized views:
      - Mapping: `view_parameters={"tenant_id": "tenant-abc"}`
      - Mapping of callables:
        `view_parameters={"tenant_id": lambda ctx: get_tenant(ctx.user_id)}`
      - Callable receiving `ToolContext`:
        `view_parameters=lambda ctx: {"tenant_id": get_tenant(ctx.user_id)}`
      Values must come from a source that the caller and model cannot write, and
      never from `tool_context.state` or `tool_context.custom_metadata`. Keys
      from `view_parameters` take precedence over keys resolved from
      `view_parameter_names`; `None` returns or `None` values are omitted (and
      remove any value resolved from `view_parameter_names`); a non-mapping
      callable return raises `TypeError`; and any exception raised by a resolver
      callable propagates to the caller and fails the tool invocation.
  """

  def __init__(
      self,
      *,
      tool_filter: ToolPredicate | list[str] | None = None,
      credentials_config: BigtableCredentialsConfig | None = None,
      bigtable_tool_settings: BigtableToolSettings | None = None,
      view_parameter_names: list[str] | None = None,
      view_parameters: (
          Mapping[str, Any]
          | Callable[
              [ToolContext],
              Mapping[str, Any] | None | Awaitable[Mapping[str, Any] | None],
          ]
          | None
      ) = None,
  ):
    _validate_view_parameters(view_parameters)
    super().__init__(
        tool_filter=tool_filter,
        tool_name_prefix=DEFAULT_BIGTABLE_TOOL_NAME_PREFIX,
    )
    self._credentials_config = credentials_config
    self._tool_settings = (
        bigtable_tool_settings
        if bigtable_tool_settings
        else BigtableToolSettings()
    )
    self.view_parameter_names = view_parameter_names
    self.view_parameters = view_parameters

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: ReadonlyContext | None
  ) -> bool:
    if self.tool_filter is None:
      return True

    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)

    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter

    return False

  def _warn_on_filtered_out_query_tool(self) -> None:
    if not (
        self.view_parameter_names or self.view_parameters
    ) or not isinstance(self.tool_filter, list):
      return
    if (
        "execute_sql" in self.tool_filter
        and "execute_sql_parameterized" not in self.tool_filter
    ):
      logger.warning(
          "tool_filter names execute_sql, which view_parameter_names or"
          " view_parameters replaces with execute_sql_parameterized, so no"
          " query tool is returned. Add execute_sql_parameterized to"
          " tool_filter."
      )

  @override
  async def get_tools(
      self, readonly_context: ReadonlyContext | None = None
  ) -> List[BaseTool]:
    """Get tools from the toolset."""
    tool_functions: tuple[Callable[..., Any], ...] = (
        metadata_tool.list_instances,
        metadata_tool.get_instance_info,
        metadata_tool.list_tables,
        metadata_tool.get_table_info,
        metadata_tool.list_clusters,
        metadata_tool.get_cluster_info,
    )
    all_tools = [
        GoogleTool(
            func=func,
            credentials_config=self._credentials_config,
            tool_settings=self._tool_settings,
        )
        for func in tool_functions
    ]
    # Registering both would let the model pick its own view parameters.
    if self.view_parameter_names or self.view_parameters:
      all_tools.append(
          BigtableParameterizedViewTool(
              func=query_tool.execute_sql,
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
              view_parameter_names=self.view_parameter_names,
              view_parameters=self.view_parameters,
          )
      )
    else:
      all_tools.append(
          _BigtableQueryTool(
              credentials_config=self._credentials_config,
              tool_settings=self._tool_settings,
          )
      )
    self._warn_on_filtered_out_query_tool()
    return [
        tool
        for tool in all_tools
        if self._is_tool_selected(tool, readonly_context)
    ]
