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
from typing import TYPE_CHECKING

from ..agents.base_agent import BaseAgent
from ._base_node import BaseNode
from ._errors import DynamicNodeFailError
from ._errors import NodeInterruptedError
from ._graph import NodeLike
from ._node_runner import NodeRunner
from ._workflow import Workflow
from .utils._workflow_graph_utils import build_node

if TYPE_CHECKING:
  from ..agents.context import Context


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

  # Validate the caller-supplied run_id when running inside a workflow.
  # A None run_id is passed through unchanged: the scheduler owns sequential run_id allocation.
  # Standalone runs (outside a workflow) allow explicit numeric IDs since there is no auto-allocation collision risk.
  if ctx._workflow_scheduler is not None:
    if run_id and run_id.isdigit() and not skip_run_id_validation:
      raise ValueError(
          f'Explicit run_id "{run_id}" for node "{built_node.name}"'
          ' must contain non-numeric characters to prevent collision'
          ' with auto-generated IDs.'
      )

  scheduler = ctx._workflow_scheduler
  if scheduler is None:
    from ._dynamic_node_scheduler import DynamicNodeScheduler
    from ._dynamic_node_scheduler import DynamicNodeState

    # No orchestrator installed one, so this call is not part of a replayable
    # workflow. Use a transfer-only scheduler: it drives the sequential
    # agent transfer loop but skips session event rehydration, keeping this
    # path a direct pass-through to NodeRunner as it was before, without
    # attaching a scheduler to ctx._workflow_scheduler.
    #
    # IMPORTANT: ctx._workflow_scheduler MUST remain None for standalone runs.
    # Across ADK, `ctx._workflow_scheduler is not None` is the canonical check
    # for whether execution is inside a workflow graph (e.g. for numeric run_id
    # validation and replay semantics).
    scheduler = DynamicNodeScheduler(
        state=DynamicNodeState(), enable_replay=False
    )

  child_ctx = await scheduler(
      ctx,
      built_node,
      node_input,
      node_name=built_node.name,
      use_as_output=use_as_output,
      run_id=run_id,
      use_sub_branch=use_sub_branch,
      override_branch=override_branch,
      override_isolation_scope=override_isolation_scope,
      resume_inputs=resume_inputs,
  )

  transfer_to_agent = child_ctx.actions.transfer_to_agent if child_ctx else None

  # Post-Execution Validation: If the caller expects the raw output (not the Context),
  # we check for errors or interrupts and raise them immediately.
  if not return_ctx:
    if child_ctx.error:
      executed_name = child_ctx.node.name if child_ctx.node else built_node.name
      raise DynamicNodeFailError(
          message=f'Dynamic node {executed_name} failed',
          error=child_ctx.error,
          error_node_path=child_ctx.error_node_path,
      )
    if child_ctx.interrupt_ids:
      # Propagate child's interrupt_ids to this node's ctx
      # so NodeRunner sees them after catching the error.
      ctx._interrupt_ids.update(child_ctx.interrupt_ids)
      raise NodeInterruptedError()
    # When the caller passes raise_on_wait=True, surface a child
    # execution that's WAITING (wait_for_output, no output, not transferring)
    # as NodeInterruptedError so the parent's NodeRunner records
    # the parent as WAITING instead of falsely COMPLETED.
    if raise_on_wait and child_ctx.output is None and not transfer_to_agent:
      # After a transfer chain, child_ctx belongs to the last agent that ran,
      # not to built_node, so the wait decision must follow child_ctx.node.
      executed_node = child_ctx.node
      if isinstance(executed_node, Workflow) or getattr(
          executed_node, 'wait_for_output', False
      ):
        raise NodeInterruptedError()

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
