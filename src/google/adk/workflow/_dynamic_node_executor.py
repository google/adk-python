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

"""Dynamic node execution algorithms for ADK workflows."""

from __future__ import annotations

from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from ..agents.base_agent import BaseAgent
from ._base_node import BaseNode
from ._errors import DynamicNodeFailError
from ._errors import NodeInterruptedError
from ._graph import NodeLike
from ._node_runner import NodeRunner
from ._workflow import Workflow
from .utils._transfer_utils import resolve_and_derive_transfer_context
from .utils._workflow_graph_utils import build_node

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._schedule_dynamic_node import ScheduleDynamicNode


async def run_node_internal(
    ctx: Context,
    node: NodeLike,
    node_input: Any = None,
    *,
    use_as_output: bool = False,
    run_id: str | None = None,
    use_sub_branch: bool = False,
    override_branch: str | None = None,
    override_isolation_scope: str | None = None,
    raise_on_wait: bool = False,
    return_ctx: bool = False,
    resume_inputs: dict[str, Any] | None = None,
    skip_run_id_validation: bool = False,
) -> Any:
  """Executes a node dynamically (Internal Orchestration API).

  See public ``run_node`` for public argument details.
  Additional internal args:
    return_ctx: If True, returns the child's Context instead of its output.
  """
  if not ctx._node_rerun_on_resume:
    raise ValueError(
        'A node must have rerun_on_resume=True. Reason is that dynamically'
        ' scheduled nodes might be interrupted, and the workflow'
        ' wakes-up/re-runs the parent node, so it can get the child node'
        ' response.'
    )

  built_node = build_node(node)

  if isinstance(node, BaseAgent) and isinstance(built_node, BaseAgent):
    built_node.parent_agent = node.parent_agent

  # Output delegation: once set, the calling node's own output
  # events are suppressed — the child's output (annotated with
  # output_for) becomes the calling node's output.
  # We validate and set this upfront before entering the loop.
  if use_as_output:
    if not isinstance(ctx.node, Workflow):
      if ctx._output_delegated:
        raise ValueError(
            f'Node {ctx.node_path} already has a use_as_output delegate.'
        )
      ctx._output_delegated = True

  # Pointers to track the active execution state in the transfer loop.
  # These will be updated dynamically if an agent transfers execution.
  curr_parent_ctx = ctx
  curr_node = built_node
  curr_run_id = run_id
  curr_input = node_input

  # Active Execution Loop: Handles both standard execution and sequential Agent Transfers
  # (e.g. Agent A transferring to Agent B). Instead of recursive execution, we use this
  # loop to execute the target agent in-place, updating pointers and 'continuing' the loop.
  while True:
    curr_use_as_output = use_as_output if (curr_parent_ctx is ctx) else False
    if ctx._workflow_scheduler:
      # --- Mode 1: Workflow Execution ---
      # The node is running as part of a Workflow graph. We must delegate execution
      # to the workflow scheduler to handle graph dependencies and state.

      # Validate the caller-supplied run_id. A None run_id is passed through
      # unchanged: the scheduler owns sequential run_id allocation.
      if curr_run_id and curr_run_id.isdigit() and not skip_run_id_validation:
        raise ValueError(
            f'Explicit run_id "{curr_run_id}" for node "{curr_node.name}"'
            ' must contain non-numeric characters to prevent collision'
            ' with auto-generated IDs.'
        )

      scheduler = cast(
          'ScheduleDynamicNode', curr_parent_ctx._workflow_scheduler
      )
      child_ctx = await scheduler(
          curr_parent_ctx,
          curr_node,
          curr_input,
          node_name=curr_node.name,
          use_as_output=curr_use_as_output,
          run_id=curr_run_id,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
      )
    else:
      # --- Mode 2: Standalone Execution ---
      # The node is running independently (outside of a workflow).
      # We run it directly using NodeRunner.
      child_ctx = await run_node_standalone(
          curr_parent_ctx,
          curr_node,
          curr_input,
          use_as_output=curr_use_as_output,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
          run_id=curr_run_id,
          resume_inputs=resume_inputs,
      )

    # Extract the transfer target if the node requested an agent transfer.
    transfer_to_agent = (
        child_ctx.actions.transfer_to_agent if child_ctx else None
    )

    # Post-Execution Validation: If the caller expects the raw output (not the Context),
    # we check for errors or interrupts and raise them immediately.
    if not return_ctx:
      if child_ctx.error:
        raise DynamicNodeFailError(
            message=f'Dynamic node {curr_node.name} failed',
            error=child_ctx.error,
            error_node_path=child_ctx.error_node_path,
        )
      if child_ctx.interrupt_ids:
        # Propagate child's interrupt_ids to this node's ctx
        # so NodeRunner sees them after catching the error.
        curr_parent_ctx._interrupt_ids.update(child_ctx.interrupt_ids)
        raise NodeInterruptedError()
      # When the caller passes raise_on_wait=True, surface a child
      # execution that's WAITING (wait_for_output, no output, not transferring)
      # as NodeInterruptedError so the parent's NodeRunner records
      # the parent as WAITING instead of falsely COMPLETED.
      if raise_on_wait and child_ctx.output is None and not transfer_to_agent:
        if isinstance(curr_node, Workflow) or getattr(
            curr_node, 'wait_for_output', False
        ):
          raise NodeInterruptedError()

    # Handle Agent Transfer: If a transfer was requested, we resolve the target agent
    # and its parent context, update loop pointers, and continue to the next iteration.
    if isinstance(transfer_to_agent, str):
      if not isinstance(curr_node, BaseAgent):
        raise ValueError('Only agents can request an agent transfer.')
      target_name = transfer_to_agent
      root_agent = getattr(curr_node, 'root_agent', None)
      if not root_agent:
        raise ValueError(f'Cannot find root_agent on node {curr_node.name}')

      target_agent, next_parent_ctx = resolve_and_derive_transfer_context(
          target_name=target_name,
          current_agent=curr_node,
          root_agent=root_agent,
          curr_ctx=child_ctx,
          curr_parent_ctx=curr_parent_ctx,
      )
      if not target_agent:
        raise ValueError(f"Transfer target agent '{target_name}' not found.")
      if not next_parent_ctx:
        available = []
        if hasattr(curr_node, '_get_available_agent_names'):
          available = curr_node._get_available_agent_names()
        available_str = (
            f"\nAvailable agents: {', '.join(available)}" if available else ''
        )
        raise ValueError(
            f"Cannot transfer from '{curr_node.name}' to unrelated agent"
            f" '{target_name}'.{available_str}"
        )
      curr_parent_ctx = next_parent_ctx

      # Set up parameters for next iteration (the transfer target).
      curr_node = target_agent
      curr_run_id = None
      curr_input = None  # Input for transfer target is usually empty.
      resume_inputs = None

      if not curr_parent_ctx:
        raise AssertionError(
            'curr_parent_ctx cannot be None during active workflow execution'
        )

      continue

    # If no transfer occurred, execution of the branch is complete.
    if return_ctx:
      return child_ctx
    return child_ctx.output


async def run_node_standalone(
    ctx: Context,
    node: BaseNode,
    node_input: Any = None,
    *,
    use_as_output: bool = False,
    run_id: str | None = None,
    use_sub_branch: bool = False,
    override_branch: str | None = None,
    override_isolation_scope: str | None = None,
    resume_inputs: dict[str, Any] | None = None,
) -> Context:
  """Run a node directly via NodeRunner without an orchestrator."""
  runner = NodeRunner(
      node=node,
      parent_ctx=ctx,
      run_id=run_id,
      use_as_output=use_as_output,
      use_sub_branch=use_sub_branch,
      override_branch=override_branch,
      override_isolation_scope=override_isolation_scope,
  )
  return await runner.run(node_input=node_input, resume_inputs=resume_inputs)
