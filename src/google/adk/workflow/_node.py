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

"""A wrapper class and @node decorator for creating workflow nodes."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any
from typing import overload
from typing import TYPE_CHECKING
from typing import TypeVar

from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from . import _base_node as base_node
from . import _function_node as function_node
from . import _graph as definitions
from . import _parallel_worker as parallel_worker_lib
from ._retry_config import RetryConfig
from .utils import _workflow_graph_utils as workflow_graph_utils

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..auth.auth_tool import AuthConfig

T = TypeVar("T", bound=Callable[..., Any])


@overload
def node(
    *,
    name: str | None = None,
    rerun_on_resume: bool | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    parallel_worker: bool = False,
    auth_config: AuthConfig | None = None,
) -> Callable[
    [T], function_node.FunctionNode | parallel_worker_lib._ParallelWorker
]:
  ...


@overload
def node(
    node_like: definitions.NodeLike,
    *,
    name: str | None = None,
    rerun_on_resume: bool | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    parallel_worker: bool = False,
    auth_config: AuthConfig | None = None,
) -> base_node.BaseNode:
  ...


def node(
    node_like: definitions.NodeLike | None = None,
    *,
    name: str | None = None,
    rerun_on_resume: bool | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    parallel_worker: bool = False,
    auth_config: AuthConfig | None = None,
) -> Any:
  """Decorator or function to wrap a NodeLike in a node or override its properties.

  This can be used as a decorator on a function:
  @node
  async def my_func(): ...

  @node()
  async def my_func2(): ...

  @node(name='my_node', rerun_on_resume=True)
  async def my_func3(): ...

  Or as a function on a NodeLike:
  my_node = node(my_func, name='other_name')

  Args:
    node_like: The item to be wrapped as a node. Can be a BaseNode, BaseAgent,
      BaseTool, or callable.
    name: If provided, overrides the name of the wrapped node.
    rerun_on_resume: If provided, overrides the rerun_on_resume property of the
      wrapped node.
    retry_config: If provided, overrides the retry_config property of the
      wrapped node.
    timeout: If provided, overrides the timeout property of the wrapped node.
    parallel_worker: If True, wraps the node in a _ParallelWorker.
    auth_config: If provided, the framework requests user authentication
      before running the node. Requires rerun_on_resume=True.

  Returns:
    If used as a decorator factory (@node() or @node(...)), returns a decorator.
    If used as a decorator (@node) or function (node(node_like, ...)), returns
    a BaseNode instance.
  """

  def wrapper(
      func: T,
  ) -> function_node.FunctionNode | parallel_worker_lib._ParallelWorker:
    built_node = function_node.FunctionNode(
        func=func,
        name=name,
        rerun_on_resume=rerun_on_resume
        if rerun_on_resume is not None
        else False,
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
    )
    if parallel_worker:
      return parallel_worker_lib._ParallelWorker(node=built_node)
    return built_node

  if node_like is None:
    # If no node_like is provided, return a decorator factory.
    return wrapper  # type: ignore
  else:
    built_node = workflow_graph_utils.build_node(
        node_like,
        name=name,
        rerun_on_resume=rerun_on_resume,
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
    )
    if parallel_worker:
      return parallel_worker_lib._ParallelWorker(node=built_node)
    return built_node


class Node(base_node.BaseNode):
  """A node class designed for subclassing.

  Subclasses can directly benefit from advanced flags like parallel_worker
  by implementing the run_node_impl() method.
  """

  parallel_worker: bool = Field(default=False, frozen=True)
  _inner_node: base_node.BaseNode | None = PrivateAttr(default=None)

  def model_post_init(self, __context: Any) -> None:
    super().model_post_init(__context)
    if self.parallel_worker:
      # If parallel_worker is True, we wrap a clone of the current node
      # in a _ParallelWorker. We disable parallel_worker on the clone
      # to avoid infinite recursion when its run() method is called.
      # The cloned node preserves the class identity and behavior of the
      # original (essential for LlmAgent and Workflow subclasses).
      worker_node = self.model_copy(update={"parallel_worker": False})

      inner = parallel_worker_lib._ParallelWorker(node=worker_node)
      self._inner_node = inner
      # Synchronize rerun_on_resume with the inner node.
      self.rerun_on_resume = inner.rerun_on_resume

  @override
  def model_copy(
      self, *, update: dict[str, Any] | None = None, deep: bool = False
  ) -> Any:
    """Clones the node with updated fields."""
    copied = super().model_copy(update=update, deep=deep)

    if copied.parallel_worker:
      worker_node = copied.model_copy(update={"parallel_worker": False})
      copied._inner_node = parallel_worker_lib._ParallelWorker(node=worker_node)
      copied.rerun_on_resume = copied._inner_node.rerun_on_resume

    return copied

  async def run_node_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    """Implement this method when designing a child class that inherits from Node.

    Subclasses can directly benefit from advanced flags like parallel_worker
    by providing their custom execution logic here.
    """
    raise NotImplementedError("run_node_impl must be implemented.")
    yield

  @override
  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    """Dispatches to run_node_impl() or parallel_worker inner node."""
    if self.parallel_worker:
      if self._inner_node is None:
        raise ValueError("inner_node is not initialized for parallel worker.")
      async for output in self._inner_node.run(ctx=ctx, node_input=node_input):
        yield output
    else:
      async for output in self.run_node_impl(ctx=ctx, node_input=node_input):
        yield output
