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
from .utils._rehydration_utils import _unwrap_response
from .utils._rehydration_utils import is_terminal_event
from .utils._replay_interceptor import check_interception
from .utils._replay_interceptor import create_mock_context
from .utils._replay_sequence_barrier import ReplaySequenceBarrier

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
    return [run.task for run in self.runs.values() if run.task]


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
    self._parent_sequence_barriers: dict[str, ReplaySequenceBarrier] = {}

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
    curr_node = node
    curr_name = node_name or node.name
    curr_run_id = run_id
    curr_input = node_input
    curr_parent_ctx: Context | None = ctx

    while True:
      curr_parent_path = curr_parent_ctx.node_path if curr_parent_ctx else None
      base_path_builder = (
          _NodePathBuilder.from_string(curr_parent_path)
          if curr_parent_path
          else _NodePathBuilder([])
      )
      node_path = str(base_path_builder.append(curr_name, curr_run_id))

      # Rehydration chronological sequence barrier setup for the parent path
      parent_path = curr_parent_ctx.node_path if curr_parent_ctx else ''
      if parent_path and parent_path not in self._parent_sequence_barriers:
        seq = self._scan_parent_child_sequence(curr_parent_ctx, parent_path)
        self._parent_sequence_barriers[parent_path] = ReplaySequenceBarrier(seq)

      # Runtime schema validation.
      if curr_input is not None:
        try:
          curr_input = curr_node._validate_input_data(curr_input)
        except ValidationError as e:
          raise ValueError(
              'Runtime schema validation failed for dynamic node'
              f" '{curr_name}'. Input does not match input_schema: {e}"
          ) from e

      logger.debug('node %s schedule start.', node_path)

      # Phase 1: Lazy rehydration from session events.
      if node_path not in self._state.runs:
        self._rehydrate_from_events(curr_parent_ctx, node_path)

      # Check existing run and determine if fresh execution is needed.
      child_ctx, run_completed = await self._check_existing_run(
          curr_parent_ctx,
          curr_node,
          curr_name,
          node_path,
          curr_run_id,
          curr_input,
          use_as_output,
          use_sub_branch,
          override_branch,
          override_isolation_scope=override_isolation_scope,
      )

      if not run_completed:
        # Phase 3: Fresh execution.
        logger.debug('node %s schedule: Fresh execution.', node_path)
        child_ctx = await self._run_node_internal(
            curr_parent_ctx,
            curr_node,
            curr_name,
            node_path,
            curr_run_id,
            curr_input,
            use_as_output,
            is_fresh=True,
            use_sub_branch=use_sub_branch,
            override_branch=override_branch,
            override_isolation_scope=override_isolation_scope,
        )

      logger.debug('node %s schedule end.', node_path)

      # Advance chronological sequence for this parent path and key
      parent_path = curr_parent_ctx.node_path if curr_parent_ctx else ''
      key = f'{curr_name}@{curr_run_id}'
      if parent_path in self._parent_sequence_barriers:
        self._parent_sequence_barriers[parent_path].check_and_advance(key)

      # Check for transfer_to_agent signal.
      transfer_to_agent = (
          child_ctx.actions.transfer_to_agent if child_ctx else None
      )
      if isinstance(transfer_to_agent, str):
        target_name = transfer_to_agent
        root_agent = getattr(curr_node, 'root_agent', None)
        if not root_agent:
          raise ValueError(f'Cannot find root_agent on node {curr_node.name}')

        # Local import to avoid runtime circular dependencies with Context
        from .utils._transfer_utils import resolve_and_derive_transfer_context

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
              f"Cannot transfer from '{curr_name}' to unrelated agent"
              f" '{target_name}'.{available_str}"
          )
        curr_parent_ctx = next_parent_ctx

        # Set up parameters for next iteration.
        curr_node = target_agent
        curr_name = target_agent.name

        if not curr_parent_ctx:
          raise AssertionError(
              'curr_parent_ctx cannot be None during active workflow execution'
          )

        curr_parent_ctx._child_run_counters[target_agent.name] = (
            curr_parent_ctx._child_run_counters.get(target_agent.name, 0) + 1
        )
        curr_run_id = str(
            curr_parent_ctx._child_run_counters[target_agent.name]
        )
        curr_input = None  # Input for transfer target is usually empty.

        # Loop continues to execute the next agent
        continue

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
        node_path=node_path,
        node=curr_node,
        recovered=run.recovered_state,
        current_run=run,
        curr_parent_ctx=curr_parent_ctx,
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
      if parent_path in self._parent_sequence_barriers:
        await self._parent_sequence_barriers[parent_path].wait(key)

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

  def _scan_parent_child_sequence(
      self, ctx: Context, parent_path: str
  ) -> list[str]:
    """Scan historical events and extract direct dynamic child completion sequence."""
    ic = ctx._invocation_context
    base_path_builder = _NodePathBuilder.from_string(parent_path)
    sequence: list[str] = []

    for event in ic.session.events:
      if event.invocation_id != ic.invocation_id:
        continue
      event_node_path = event.node_info.path or ''
      event_path_builder = _NodePathBuilder.from_string(event_node_path)

      if not event_path_builder.is_descendant_of(base_path_builder):
        continue

      child_path = base_path_builder.get_direct_child(event_path_builder)
      if event_path_builder != child_path:
        continue

      segment = child_path.leaf_segment

      if is_terminal_event(event):
        if segment in sequence:
          sequence.remove(segment)
        sequence.append(segment)

    return sequence

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
    elif node.wait_for_output and child_ctx.output is None:
      state.status = NodeStatus.WAITING
    else:
      state.status = NodeStatus.COMPLETED
      run.output = child_ctx.output
