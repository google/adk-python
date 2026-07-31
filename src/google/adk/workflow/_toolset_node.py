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

"""A node that runs a named tool from a toolset."""

from collections.abc import AsyncGenerator
import re
from typing import Any

from pydantic import ConfigDict
from pydantic import Field
from typing_extensions import override

from ..agents.context import Context
from ..agents.readonly_context import ReadonlyContext
from ..tools.base_tool import BaseTool
from ..tools.base_toolset import BaseToolset
from ._base_node import BaseNode
from ._retry_config import RetryConfig
from ._tool_node import _run_tool


def _to_node_name(tool_name: str) -> str:
  """Converts a tool name into a valid Python identifier.

  Tool names from external servers are not constrained to Python identifiers
  (MCP servers commonly use dashes), but node names are.
  """
  name = re.sub(r'\W', '_', tool_name)
  if not name or name[0].isdigit():
    name = f'_{name}'
  return name


class ToolsetNode(BaseNode):
  """A node that runs a named tool from a toolset.

  Unlike passing a ``BaseTool`` directly into a workflow's edges, the tool is
  resolved lazily when the node runs, rather than when the graph is built. This
  is what makes toolsets whose tools are only discoverable asynchronously --
  such as ``McpToolset``, which lists them over a live connection -- usable as
  workflow nodes::

      toolset = McpToolset(connection_params=StdioConnectionParams(...))

      workflow = Workflow(
          name='research',
          edges=[
              (START, build_query),
              (build_query, ToolsetNode(toolset=toolset, tool_name='search')),
              (ToolsetNode(...), summarize),
          ],
      )

  Resolution goes through ``BaseToolset.get_tools_with_prefix()``, which caches
  per invocation, so several ``ToolsetNode``s sharing one toolset only list its
  tools once per run.

  The node input must be a dict of tool arguments, a JSON object string, or
  ``None`` for no arguments. The tool's response becomes the node's output.

  Closing the toolset remains the caller's responsibility, except when the
  workflow is run by a ``Runner``, which closes the toolsets it finds on the
  agent it runs.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  toolset: BaseToolset = Field(...)
  """The toolset to resolve the tool from."""

  tool_name: str = Field(...)
  """The name of the tool to run.

  This is matched against the tool names the toolset reports, so it includes
  the toolset's ``tool_name_prefix`` if one is set.
  """

  def __init__(
      self,
      *,
      toolset: BaseToolset,
      tool_name: str,
      name: str | None = None,
      description: str = '',
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    """Initializes the ToolsetNode.

    Args:
      toolset: The toolset to resolve the tool from.
      tool_name: The name of the tool to run.
      name: The node's name. Defaults to ``tool_name`` with any character that
        is not valid in a Python identifier replaced by an underscore.
      description: A human-readable description of what this node does.
      retry_config: Configuration for retrying the node on failure.
      timeout: Maximum time in seconds for this node to complete.
    """
    super().__init__(
        toolset=toolset,
        tool_name=tool_name,
        name=name or _to_node_name(tool_name),
        description=description,
        rerun_on_resume=False,
        retry_config=retry_config,
        timeout=timeout,
    )

  async def _resolve_tool(self, ctx: Context) -> BaseTool:
    """Finds the named tool in the toolset.

    Raises:
      ValueError: If the toolset does not offer a tool by that name.
    """
    readonly_context = ReadonlyContext(ctx.get_invocation_context())
    tools = await self.toolset.get_tools_with_prefix(readonly_context)
    for tool in tools:
      if tool.name == self.tool_name:
        return tool
    available = ', '.join(sorted(tool.name for tool in tools)) or '<none>'
    raise ValueError(
        f"Tool '{self.tool_name}' was not found in"
        f' {type(self.toolset).__name__}. Available tools: {available}.'
    )

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    tool = await self._resolve_tool(ctx)
    async for event in _run_tool(tool, ctx=ctx, node_input=node_input):
      yield event
