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

"""Utility functions for building workflow graphs."""

from __future__ import annotations

from typing import Any
from typing import cast

from ...tools.base_tool import BaseTool
from .._base_node import BaseNode
from .._base_node import START
from .._function_node import FunctionNode
from .._graph import NodeLike
from .._retry_config import RetryConfig
from .._tool_node import _ToolNode


def is_node_like(item: Any) -> bool:
  """Checks if an object is NodeLike."""
  return (
      isinstance(item, (BaseNode, BaseTool))
      or callable(item)
      or item == 'START'
  )


def build_node(
    node_like: NodeLike,
    *,
    name: str | None = None,
    rerun_on_resume: bool | None = None,
    retry_config: RetryConfig | None = None,
    timeout: float | None = None,
    auth_config: Any = None,
) -> BaseNode:
  """Converts a NodeLike to a BaseNode, wrapping async funcs in FunctionNode.

  Args:
    node_like: The item to convert to a BaseNode.
    name: If provided, overrides the name of the wrapped node.
    rerun_on_resume: If provided, overrides the rerun_on_resume property of the
      wrapped node.
    retry_config: If provided, overrides the retry_config property of the
      wrapped node.
    timeout: If provided, overrides the timeout property of the wrapped node.
    auth_config: If provided, passed to FunctionNode for authentication.

  Returns:
    A BaseNode instance.

  Raises:
    ValueError: If node_like is not a valid type (BaseNode, BaseAgent,
      BaseTool, callable, or 'START').
  """

  if node_like == 'START':
    return START

  # Lazy import to avoid circular dependency:
  # workflow_graph_utils -> agents.llm_agent -> ... -> workflow_graph_utils
  from ...agents.llm_agent import LlmAgent

  if isinstance(node_like, BaseNode):
    kwargs: dict[str, Any] = {}
    if name is not None:
      kwargs['name'] = name
    if rerun_on_resume is not None:
      kwargs['rerun_on_resume'] = rerun_on_resume
    if retry_config is not None:
      kwargs['retry_config'] = retry_config
    if timeout is not None:
      kwargs['timeout'] = timeout

    if isinstance(node_like, LlmAgent):
      if rerun_on_resume is None:
        kwargs['rerun_on_resume'] = True
      agent = node_like.clone(update=kwargs)
      # Preserve parent agent reference that was lost during clone
      agent.parent_agent = node_like.parent_agent

      if agent.mode is None:
        agent.mode = 'single_turn'

      if agent.mode in ('task', 'chat'):
        agent.wait_for_output = True

      if agent.parallel_worker:
        from .._parallel_worker import _ParallelWorker

        agent.parallel_worker = False
        return _ParallelWorker(node=agent)
      return cast(BaseNode, agent)
    else:
      if kwargs:
        return cast(BaseNode, node_like.model_copy(update=kwargs))
      return node_like
  elif isinstance(node_like, BaseTool):
    return _ToolNode(
        tool=node_like,
        name=name,
        retry_config=retry_config,
        timeout=timeout,
    )
  elif callable(node_like):
    return FunctionNode(
        func=node_like,
        name=name,
        rerun_on_resume=rerun_on_resume or False,
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
    )
  else:
    raise ValueError(
        f'Invalid node type: {type(node_like)}. Node must be a BaseNode, a'
        ' BaseAgent, a BaseTool, or a callable.'
    )
