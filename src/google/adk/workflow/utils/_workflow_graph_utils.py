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

from collections.abc import Callable
from typing import Any
from typing import cast
from typing import Literal

from ...tools.base_tool import BaseTool
from ...utils._agent_mode import AgentMode
from ...utils._agent_mode import DefaultLlmNodeMode
from .._base_node import BaseNode
from .._base_node import START
from .._errors import WorkflowConfigurationError
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
    require_confirmation: bool | Callable[..., bool] | None = None,
    parameter_binding: Literal['state', 'node_input'] = 'state',
    default_llm_mode: DefaultLlmNodeMode = AgentMode.SINGLE_TURN,
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
    require_confirmation: If provided, passed to FunctionNode for user
      confirmation before running.
    parameter_binding: How function parameters are bound. ``'state'`` (default)
      binds parameters from ``ctx.state``. ``'node_input'`` binds parameters
      from ``node_input`` dict and infers ``input_schema`` / ``output_schema``
      from the function signature (used when the node acts as an agent's tool).
    default_llm_mode: Default mode applied to the cloned LlmAgent when its
      ``mode`` is ``None`` and ``parent_agent`` is ``None``.

  Returns:
    A BaseNode instance.

  Raises:
    WorkflowConfigurationError: If node_like is not a valid type (BaseNode,
      BaseAgent, BaseTool, callable, or 'START'), or if it is a task-mode
      RemoteA2aAgent with no parent agent.
  """

  if node_like == 'START':
    return START

  # Lazy import to avoid circular dependency:
  # workflow_graph_utils -> agents.llm_agent -> ... -> workflow_graph_utils
  from ...agents.llm_agent import LlmAgent

  # Optional dependency: RemoteA2aAgent is only available if a2a is installed.
  _remote_a2a_agent_type: Any = None
  try:
    from ...agents.remote_a2a_agent import RemoteA2aAgent  # pylint: disable=g-import-not-at-top

    _remote_a2a_agent_type = RemoteA2aAgent
  except ImportError:
    pass

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
    if isinstance(node_like, FunctionNode):
      effective_confirmation = (
          require_confirmation
          if require_confirmation is not None
          else node_like.require_confirmation
      )
      if effective_confirmation and rerun_on_resume is False:
        raise WorkflowConfigurationError(
            'FunctionNode with require_confirmation requires'
            ' rerun_on_resume=True. The node must rerun after confirmation is'
            ' provided.'
        )
      if require_confirmation is not None:
        kwargs['require_confirmation'] = require_confirmation
        if require_confirmation and rerun_on_resume is None:
          kwargs['rerun_on_resume'] = True
    elif require_confirmation:
      raise WorkflowConfigurationError(
          'require_confirmation is only supported on FunctionNode, got'
          f' {type(node_like).__name__}.'
      )

    is_remote_a2a_task = False
    if _remote_a2a_agent_type is not None:
      is_remote_a2a_task = (
          isinstance(node_like, _remote_a2a_agent_type)
          and node_like.mode == AgentMode.TASK
      )
    if is_remote_a2a_task and getattr(node_like, 'parent_agent', None) is None:
      raise WorkflowConfigurationError(
          'RemoteA2aAgent in task mode is not supported as a standalone '
          'workflow node. It is only supported in tool-delegation mode.'
      )

    if isinstance(node_like, LlmAgent) or is_remote_a2a_task:
      if rerun_on_resume is None:
        kwargs['rerun_on_resume'] = True
      agent_node = cast(Any, node_like)
      agent = agent_node.clone(update=kwargs)
      # Preserve parent agent reference that was lost during clone
      agent.parent_agent = agent_node.parent_agent

      if isinstance(agent, LlmAgent) and agent.mode is None:
        # Sub-agents dynamically attached to a parent agent default to 'chat'
        # mode to enable agent transfer.
        # Standalone agents in a workflow graph default to `default_llm_mode`.
        if agent.parent_agent is not None:
          agent.mode = AgentMode.CHAT.value
        else:
          agent.mode = str(default_llm_mode)

      if (
          isinstance(agent, LlmAgent)
          and agent.mode == AgentMode.SINGLE_TURN
          and 'include_contents' not in node_like.model_fields_set
      ):
        agent.include_contents = 'none'

      if agent.mode in (AgentMode.TASK, AgentMode.CHAT):
        agent.wait_for_output = True

      if isinstance(agent, LlmAgent) and agent.parallel_worker:
        from .._parallel_worker import _ParallelWorker

        agent.parallel_worker = False
        return _ParallelWorker(node=agent)
      return cast(BaseNode, agent)
    else:
      if kwargs:
        return node_like.model_copy(update=kwargs)
      return node_like
  elif isinstance(node_like, BaseTool):
    # Lazy imports to avoid circular dependency
    from ...tools._node_tool import NodeTool

    # 1. Unpack NodeTool: directly return the underlying native node, applying any overrides
    if isinstance(node_like, NodeTool):
      target_name = name or node_like.name
      return build_node(
          node_like.node,
          name=(
              target_name
              if target_name != getattr(node_like.node, 'name', None)
              else None
          ),
          rerun_on_resume=rerun_on_resume,
          retry_config=retry_config,
          timeout=timeout,
          require_confirmation=require_confirmation,
      )

    # 2. Other generic BaseTools (including FunctionTool) are wrapped in _ToolNode
    if require_confirmation:
      raise WorkflowConfigurationError(
          'require_confirmation is only supported for FunctionNode or callable'
          f' functions, got {type(node_like).__name__}.'
      )
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
        rerun_on_resume=(
            True
            if (require_confirmation and rerun_on_resume is None)
            else (rerun_on_resume or False)
        ),
        retry_config=retry_config,
        timeout=timeout,
        auth_config=auth_config,
        require_confirmation=(
            require_confirmation if require_confirmation is not None else False
        ),
        parameter_binding=parameter_binding,
    )
  else:
    raise WorkflowConfigurationError(
        f'Invalid node type: {type(node_like)}. Node must be a BaseNode, a'
        ' BaseAgent, a BaseTool, or a callable.'
    )
