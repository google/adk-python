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

from abc import ABC
from abc import abstractmethod
import copy
from typing import final
from typing import List
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import TYPE_CHECKING
from typing import Union

from ..agents.readonly_context import ReadonlyContext
from .base_tool import BaseTool

if TYPE_CHECKING:
  from ..models.llm_request import LlmRequest
  from .tool_context import ToolContext


@runtime_checkable
class ToolPredicate(Protocol):
  """Base class for a predicate that defines the interface to decide whether a

  tool should be exposed to LLM. Toolset implementer could consider whether to
  accept such instance in the toolset's constructor and apply the predicate in
  get_tools method.
  """

  def __call__(
      self, tool: BaseTool, readonly_context: Optional[ReadonlyContext] = None
  ) -> bool:
    """Decide whether the passed-in tool should be exposed to LLM based on the

    current context. True if the tool is usable by the LLM.

    It's used to filter tools in the toolset.
    """


class BaseToolset(ABC):
  """Base class for toolset.

  A toolset is a collection of tools that can be used by an agent.
  """

  def __init__(
      self,
      *,
      tool_filter: Optional[Union[ToolPredicate, List[str]]] = None,
      tool_name_prefix: Optional[str] = None,
  ):
    """Initialize the toolset.

    Args:
      tool_filter: Filter to apply to tools.
      tool_name_prefix: The prefix to prepend to the names of the tools returned by the toolset.
    """
    self.tool_filter = tool_filter
    self.tool_name_prefix = tool_name_prefix

  @abstractmethod
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
      readonly_context (ReadonlyContext, optional): Context used to filter tools
        available to the agent. If None, all tools in the toolset are returned.

    Returns:
      list[BaseTool]: A list of tools available under the specified context.
    """

  @final
  async def get_tools_with_prefix(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Return all tools with optional prefix applied to tool names.

    This method calls get_tools() and applies prefixing if tool_name_prefix is provided.

    Args:
      readonly_context (ReadonlyContext, optional): Context used to filter tools
        available to the agent. If None, all tools in the toolset are returned.

    Returns:
      list[BaseTool]: A list of tools with prefixed names if tool_name_prefix is provided.
    """
    tools = await self.get_tools(readonly_context)

    if not self.tool_name_prefix:
      return tools

    prefix = self.tool_name_prefix

    # Create copies of tools to avoid modifying original instances
    prefixed_tools = []
    for tool in tools:
      # Create a shallow copy of the tool
      tool_copy = copy.copy(tool)

      # Apply prefix to the copied tool
      prefixed_name = f"{prefix}_{tool.name}"
      tool_copy.name = prefixed_name

      # Also update the function declaration name if the tool has one
      # Use default parameters to capture the current values in the closure
      def _create_prefixed_declaration(
          original_get_declaration=tool._get_declaration,
          prefixed_name=prefixed_name,
      ):
        def _get_prefixed_declaration():
          declaration = original_get_declaration()
          if declaration is not None:
            declaration.name = prefixed_name
            return declaration
          return None

        return _get_prefixed_declaration

      tool_copy._get_declaration = _create_prefixed_declaration()
      prefixed_tools.append(tool_copy)

    return prefixed_tools

  async def close(self) -> None:
    """Performs cleanup and releases resources held by the toolset.

    NOTE:
      This method is invoked, for example, at the end of an agent server's
      lifecycle or when the toolset is no longer needed. Implementations
      should ensure that any open connections, files, or other managed
      resources are properly released to prevent leaks.
    """

  def _is_tool_selected(
      self, tool: BaseTool, readonly_context: ReadonlyContext
  ) -> bool:
    if not self.tool_filter:
      return True

    if isinstance(self.tool_filter, ToolPredicate):
      return self.tool_filter(tool, readonly_context)

    if isinstance(self.tool_filter, list):
      return tool.name in self.tool_filter

    return False

  async def process_llm_request(
      self, *, tool_context: ToolContext, llm_request: LlmRequest
  ) -> None:
    """Processes the outgoing LLM request for this toolset. This method will be
    called before each tool processes the llm request.

    Use cases:
    - Instead of let each tool process the llm request, we can let the toolset
      process the llm request. e.g. ComputerUseToolset can add computer use
      tool to the llm request.

    Args:
      tool_context: The context of the tool.
      llm_request: The outgoing LLM request, mutable this method.
    """
    pass
