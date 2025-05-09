from abc import ABC
from abc import abstractmethod
from typing import Protocol

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.base_tool import BaseTool


class ToolPredicate(Protocol):
  """Base class for a predicate that defines the interface to decide whether a

  tool should be exposed to LLM. Toolset implementer could consider whether to
  accept such instance in the toolset's constructor and apply the predicate in
  get_tools method.
  """

  def __call__(
      self, tool: BaseTool, readonly_context: ReadonlyContext = None
  ) -> bool:
    """Decide whether the passed-in tool should be exposed to LLM based on the

    current context. True if the tool is usable by the LLM.

    It's used to filter tools in the toolset.
    """


class BaseToolset(ABC):
  """Base class for toolset.

  A toolset is a collection of tools that can be used by an agent.
  """

  @abstractmethod
  async def get_tools(
      self, readony_context: ReadonlyContext = None
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
