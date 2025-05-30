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


from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import Union

from ..agents.readonly_context import ReadonlyContext
from .base_tool import BaseTool


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
      self, *, tool_filter: Optional[Union[ToolPredicate, List[str]]] = None
  ):
    self.tool_filter = tool_filter

  @abstractmethod
  async def get_tools(
      self,
      readonly_context: Optional[ReadonlyContext] = None,
  ) -> list[BaseTool]:
    """Return all tools in the toolset based on the provided context.

    Args:
      readony_context (ReadonlyContext, optional): Context used to filter tools
        available to the agent. If None, all tools in the toolset are returned.

    Returns:
      list[BaseTool]: A list of tools available under the specified context.
    """

  @abstractmethod
  async def close(self) -> None:
    """Performs cleanup and releases resources held by the toolset.

    NOTE: This method is invoked, for example, at the end of an agent server's
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
