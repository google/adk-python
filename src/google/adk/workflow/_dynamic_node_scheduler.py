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

"""Dynamic node scheduler for Workflow.

Handles ctx.run_node() calls by tracking dynamic nodes in the
Workflow's _LoopState or a local DynamicNodeState. Supports dedup
(cached output), resume (lazy event scan + re-run), and fresh execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Any
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..events._node_path_builder import _NodePathBuilder
from ._node_state import NodeState
from ._node_status import NodeStatus
from ._schedule_dynamic_node import ScheduleDynamicNode
from .utils._rehydration_utils import _ChildScanState
from .utils._rehydration_utils import _reconstruct_node_states
from .utils._replay_interceptor import check_interception
from .utils._replay_interceptor import create_mock_context
from .utils._replay_manager import ReplayManager

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._base_node import BaseNode


logger = logging.getLogger('google_adk.' + __name__)


@dataclass(kw_only=True)
class DynamicNodeRun:
  """Combines state, output, and running task for a single node execution."""

  state: NodeState
  """The tracking state (status, interrupts, run_id)."""

  output: Any = None
  """The final output of the node once it completes."""

  task: asyncio.Task[Context] | None = None
  """The running asyncio Task for this node execution."""

  transfer_to_agent: str | None = None
  """The target agent name if this node execution transferred."""

  recovered_state: _ChildScanState | None = None
  """The raw scan state from events, used for replay interception."""


@dataclass(kw_only=True)
class DynamicNodeState:
  """State for tracking dynamic nodes scheduled via ctx.run_node().

  Base class for both Workflow's ``_LoopState`` and standalone
  ``DefaultNodeScheduler``. DynamicNodeScheduler reads/writes
  these fields for dedup, resume, and interrupt propagation.
  """

  runs: dict[str, DynamicNodeRun] = field(default_factory=dict)
  """Dynamic node runs keyed by unique node_path (e.g. /wf@1/node_a@1)."""

  # --- Shared (static + dynamic) ---

  interrupt_ids: set[str] = field(default_factory=set)
  """Union of all unresolved interrupt IDs across static and
  dynamic child nodes.

  Populated by:
  - _restore_static_nodes_from_events: from WAITING static nodes
  - _handle_completion: when a static node interrupts at runtime
  - schedule callback: when a dynamic node interrupts

  Read by _finalize to propagate to the Workflow's own ctx,
  which the parent orchestrator checks after this Workflow
  completes.
  """

  def get_dynamic_tasks(self) -> list[asyncio.Task[Context]]:
    """Get all active dynamic node tasks."""
    return [
        run.task
        for run in self.runs.values()
        if run.task and not run.task.done()
    ]


class DynamicNodeScheduler(ScheduleDynamicNode):
  """Handles ctx.run_node() calls for a Workflow.

  Implements ScheduleDynamicNode protocol via __call__. Tracks
  dynamic nodes in loop_state, handles dedup via lazy event
  scanning, and manages resume/interrupt propagation.

  Three cases:
  1. Fresh: no prior events → execute normally.
  2. Completed: prior events show output → return cached.
  3. Waiting: prior events show interrupt → resolve or propagate.
  """

  def __init__(self, *, state: DynamicNodeState) -> None:
    self._state = state
    self._replay_manager = ReplayManager()

  async def __call__(
      self,
      ctx: Context,
      node: BaseNode,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
      run_id: str,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
  ) -> Context:
    """Schedule a dynamic node: dedup, resume, or fresh run.

    Args:
      ctx: The calling node's Context.
      node: The BaseNode to execute (original, before renaming).
      node_input: Input data for the node.
      node_name: Deterministic tracking name from ctx.run_node().
        Always provided (user-specified or auto-generated).
      use_as_output: If True, the child's output replaces the
        calling node's output.
      run_id: Custom run ID for the child node execution.
      use_sub_branch: Whether the node should use a sub-branch.
      override_branch: Optional branch to use instead of parent's branch.

    Returns:
      Child Context with output, route, and interrupt_ids set.
    """
    curr_parent_path = ctx.node_path if ctx else None
    base_path_builder = (
        _NodePathBuilder.from_string(curr_parent_path)
        if curr_parent_path
        else _NodePathBuilder([])
    )
    node_path = str(base_path_builder.append(node_name or node.name, run_id))

    # Rehydration chronological sequence barrier setup for the parent path
    parent_path = ctx.node_path if ctx else ''
    if parent_path:
      self._replay_manager.prepare_parent_sequence_barrier(ctx, parent_path)

    # Runtime schema validation.
    if node_input is not None:
      try:
        node_input = node._validate_input_data(node_input)
      except ValidationError as e:
        raise ValueError(
            'Runtime schema validation failed for dynamic node'
            f" '{node_name or node.name}'. Input does not match"
            f' input_schema: {e}'
        ) from e

    logger.debug('node %s schedule start.', node_path)

    # Phase 1: Lazy rehydration from session events.
    if node_path not in self._state.runs:
      self._rehydrate_from_events(ctx, node_path)

    # Check existing run and determine if fresh execution is needed.
    child_ctx, run_completed = await self._check_existing_run(
        ctx,
        node,
        node_name or node.name,
        node_path,
        run_id,
        node_input,
        use_as_output,
        use_sub_branch,
        override_branch,
        override_isolation_scope=override_isolation_scope,
    )

    if not run_completed:
      # Phase 3: Fresh execution.
      logger.debug('node %s schedule: Fresh execution.', node_path)
      child_ctx = await self._run_node_internal(
          ctx,
          node,
          node_name or node.name,
          node_path,
          run_id,
          node_input,
          use_as_output,
          is_fresh=True,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
      )

    logger.debug('node %s schedule end.', node_path)

    # Advance chronological sequence for this parent path and key
    parent_path = ctx.node_path if ctx else ''
    key = f'{node_name or node.name}@{run_id}'
    await self._replay_manager.advance_sequence(parent_path, key)

    return child_ctx

  async def _check_existing_run(
      self,
      curr_parent_ctx: Context | None,
      curr_node: BaseNode,
      curr_name: str,
      node_path: str,
      curr_run_id: str,
      curr_input: Any,
      use_as_output: bool,
      use_sub_branch: bool,
      override_branch: str | None,
      override_isolation_scope: str | None = None,
  ) -> tuple[Context | None, bool]:
    """Scan and process cached status for waiting or completed runs.

    Returns a tuple of (child_ctx, run_completed_flag).
    """
    if node_path not in self._state.runs:
      return None, False

    run = self._state.runs[node_path]

    # Deduplication of concurrent calls!
    if run.task and not run.task.done():
      logger.debug('node %s schedule: Awaiting existing task.', node_path)
      return await run.task, True

    if run.recovered_state:
      recovered = run.recovered_state
      unresolved = recovered.interrupt_ids - recovered.resolved_ids
      if recovered.interrupt_ids and not unresolved:
        if curr_node.wait_for_output and not curr_node.rerun_on_resume:
          raise ValueError(
              f'Node {node_path} is waiting for output but was called again'
              ' with rerun_on_resume=False. This would cause it to'
              ' auto-complete with empty output, which is likely a'
              ' configuration error. Consider setting rerun_on_resume=True.'
          )

    # Delegate replay and same-turn interception check to ReplayInterceptor.
    result = check_interception(
        node=curr_node,
        recovered=run.recovered_state,
        current_run=run,
    )

    if not result.should_run:
      if result.interrupts:
        self._state.interrupt_ids.update(result.interrupts)
        logger.debug(
            'node %s schedule: Unresolved interrupts remain.', node_path
        )
      else:
        logger.debug(
            'node %s schedule: Fast-forwarding completed execution.', node_path
        )
        # Sync output and transfer decisions with the current run state.
        run.output = result.output
        run.transfer_to_agent = result.transfer_to_agent

      # Create a high-fidelity mock context with cached results.
      mock_ctx = create_mock_context(
          parent_ctx=curr_parent_ctx,
          node=curr_node,
          run_id=curr_run_id,
          result=result,
          ancestors=[],
          node_path=node_path,
          branch=(run.recovered_state.branch if run.recovered_state else None),
      )

      # Chronological sequence barrier wait for replayed dynamic nodes
      parent_path = curr_parent_ctx.node_path if curr_parent_ctx else ''
      key = f'{curr_name}@{curr_run_id}'
      await self._replay_manager.wait_sequence(parent_path, key)

      return mock_ctx, True

    else:
      # Rerun!
      run.state.resume_inputs = result.resume_inputs
      logger.debug('node %s schedule: Rerunning execution.', node_path)
      return (
          await self._run_node_internal(
              curr_parent_ctx,
              curr_node,
              curr_name,
              node_path,
              curr_run_id,
              curr_input,
              use_as_output,
              is_fresh=False,
              use_sub_branch=use_sub_branch,
              override_branch=override_branch,
              override_isolation_scope=override_isolation_scope,
          ),
          True,
      )

  # --- Lazy scan ---

  def _rehydrate_from_events(self, ctx: Context, node_path: str) -> None:
    """Scan session events for a dynamic node's prior state."""
    logger.debug('node %s rehydrate start.', node_path)
    ic = ctx._invocation_context  # pylint: disable=protected-access

    results = _reconstruct_node_states(
        events=ic.session.events,
        base_path=node_path,
        group_by_direct_child=False,
        invocation_id=ic.invocation_id,
    )

    target_state = results.get(node_path)

    if target_state:
      self._state.runs[node_path] = DynamicNodeRun(
          state=NodeState(run_id=target_state.run_id),
          recovered_state=target_state,
      )

    logger.debug('node %s rehydrate end.', node_path)

  # --- Execution ---

  async def _run_node_internal(
      self,
      ctx: Context,
      node: BaseNode,
      name: str,
      node_path: str,
      run_id: str,
      node_input: Any,
      use_as_output: bool,
      is_fresh: bool,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
  ) -> Context:
    """Unified runner for both fresh and resume executions."""
    if is_fresh:
      state = NodeState(
          status=NodeStatus.RUNNING,
          input=node_input,
          run_id=run_id,
          parent_run_id=ctx.run_id,
      )
      run = DynamicNodeRun(state=state)
      self._state.runs[node_path] = run
      resume_inputs = None
    else:
      run = self._state.runs[node_path]
      run.state.status = NodeStatus.RUNNING
      resume_inputs = (
          dict(run.state.resume_inputs) if run.state.resume_inputs else None
      )

    target_node = node.model_copy(update={'name': name})
    run.task = asyncio.create_task(
        ctx._run_node_standalone(
            target_node,
            node_input=node_input,
            use_as_output=use_as_output,
            run_id=run_id,
            use_sub_branch=use_sub_branch,
            override_branch=override_branch,
            override_isolation_scope=override_isolation_scope,
            resume_inputs=resume_inputs,
        )
    )
    try:
      child_ctx = await run.task
    except asyncio.CancelledError:
      if node_path in self._state.runs:
        del self._state.runs[node_path]
      raise
    self._record_result(run, child_ctx, node)
    return child_ctx

  def _record_result(
      self,
      run: DynamicNodeRun,
      child_ctx: Context,
      node: BaseNode,
  ) -> None:
    """Update dynamic node state after execution."""
    state = run.state
    if child_ctx.error:
      state.status = NodeStatus.FAILED
    elif child_ctx.interrupt_ids:
      state.status = NodeStatus.WAITING
      state.interrupts = list(child_ctx.interrupt_ids)
      self._state.interrupt_ids.update(child_ctx.interrupt_ids)
    elif child_ctx.actions.transfer_to_agent:
      state.status = NodeStatus.COMPLETED
      run.transfer_to_agent = child_ctx.actions.transfer_to_agent
    elif (
        node.wait_for_output
        and child_ctx.output is None
        and child_ctx.route is None
    ):
      state.status = NodeStatus.WAITING
    else:
      state.status = NodeStatus.COMPLETED
      run.output = child_ctx.output
