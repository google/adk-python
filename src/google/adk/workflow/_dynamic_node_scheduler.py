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

from ..agents.base_agent import BaseAgent
from ..events._node_path_builder import _NodePathBuilder
from ._base_node import BaseNode
from ._errors import DynamicNodeFailError
from ._errors import NodeInterruptedError
from ._errors import WorkflowConfigurationError
from ._errors import WorkflowInvariantError
from ._graph import NodeLike
from ._node_runner import NodeRunner
from ._node_state import NodeState
from ._node_status import NodeStatus
from .utils._rehydration_utils import _ChildScanState
from .utils._rehydration_utils import _reconstruct_node_states
from .utils._replay_interceptor import check_interception
from .utils._replay_interceptor import create_mock_context
from .utils._replay_manager import ReplayManager
from .utils._transfer_utils import resolve_and_derive_transfer_context
from .utils._workflow_graph_utils import build_node

if TYPE_CHECKING:
  from ..agents.context import Context


logger = logging.getLogger('google_adk.' + __name__)


@dataclass(kw_only=True)
class DynamicNodeRun:
  """Combines state, output, and running task for a single node execution."""

  state: NodeState
  """The tracking state (status, interrupts, run_id)."""

  output: Any = None
  """The final output of the node once it completes."""

  task: asyncio.Task[Context] | None = None
  """The running asyncio Task, or None for a run replayed from cache."""

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

  run_counters: dict[str, dict[str, int]] = field(default_factory=dict)
  """Sequential execution counters per parent_path and node_name, used to allocate run IDs."""

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

  replay_manager: ReplayManager = field(default_factory=ReplayManager)
  """The replay manager for this loop state, containing event indexes."""

  def next_run_id(self, node_name: str, parent_path: str = '') -> str:
    """Increment and return the next sequential run_id for a node name under parent_path."""
    counters = self.run_counters.setdefault(parent_path, {})
    counters[node_name] = counters.get(node_name, 0) + 1
    return str(counters[node_name])

  def get_dynamic_tasks(self) -> list[asyncio.Task[Context]]:
    """Get all active dynamic node tasks."""
    return [
        run.task
        for run in self.runs.values()
        if run.task and not run.task.done()
    ]


class DynamicNodeScheduler:
  """Handles dynamic node scheduling and sequential agent transfers.

  Serves as the single runtime driver for both workflow-integrated dynamic
  execution (with state tracking, deduplication, and replay) and standalone
  sequential agent transfers.

  The scheduler manages four core execution concerns:
  1. Fresh Execution: Runs a node for the first time via NodeRunner.
  2. Deduplication: Replays cached output from prior turn events without
     re-execution.
  3. Resumption: Rehydrates state from session events after an interrupt,
     re-executing with resolved resume_inputs or propagating pending interrupts.
  4. Agent Transfer: Drives sequential agent handoffs (transfer_to_agent)
     in-place within a loop until a terminal result or interrupt is reached.

  When enable_replay=False (used for standalone executions outside a workflow),
  the scheduler operates in pass-through mode: it drives agent transfers while
  skipping event scanning, defaulting run_id to '1', and bypassing run caching.
  """

  def __init__(
      self, *, state: DynamicNodeState, enable_replay: bool = True
  ) -> None:
    """Initialize the scheduler.

    Args:
      state: The shared dynamic node state.
      enable_replay: Whether to reconstruct prior runs from session events.
        Set to False when the scheduler is installed only to drive sequential
        agent transfers, so scheduling stays a direct pass-through to
        NodeRunner without session event scanning.
    """
    self._state = state
    self._replay_manager = state.replay_manager
    self._enable_replay = enable_replay

  async def __call__(
      self,
      ctx: Context,
      node: BaseNode,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
      run_id: str | None = None,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
      resume_inputs: dict[str, Any] | None = None,
  ) -> Context:
    """Schedule a dynamic node, executing any sequential agent transfers.

    Args:
      ctx: The calling node's Context.
      node: The BaseNode to execute (original, before renaming).
      node_input: Input data for the node.
      node_name: Deterministic tracking name from ctx.run_node(). Always
        provided (user-specified or auto-generated).
      use_as_output: If True, the child's output replaces the calling node's
        output.
      run_id: Custom run ID for the child node execution. If None, the scheduler
        assigns a sequential run ID.
      use_sub_branch: Whether the node should use a sub-branch.
      override_branch: Optional branch to use instead of parent's branch.
      override_isolation_scope: Optional isolation scope override.
      resume_inputs: Optional inputs to pass when resuming an interrupted node.

    Returns:
      Child Context with output, route, and interrupt_ids set.
    """
    curr_parent_ctx = ctx
    curr_node = node
    curr_input = node_input
    curr_name = node_name
    curr_run_id = run_id
    curr_resume_inputs = resume_inputs

    while True:
      curr_use_as_output = use_as_output if (curr_parent_ctx is ctx) else False

      active_scheduler = curr_parent_ctx._workflow_scheduler
      if active_scheduler is not None and active_scheduler is not self:
        # The transfer target's parent context is owned by a different
        # scheduler (and therefore a different DynamicNodeState).
        # We delegate only the single step (_execute_step) to that scheduler so
        # that run IDs, replay barriers, and session rehydration are governed
        # by its state.
        #
        # Crucially, the transfer *loop* remains under the control of `self`
        # (the initiating scheduler). A foreign scheduler cannot own a
        # transfer hop because the loop's use_as_output and run-id ownership
        # would be lost.
        if not isinstance(active_scheduler, DynamicNodeScheduler):
          raise WorkflowInvariantError(
              f'Foreign scheduler of type {type(active_scheduler).__name__}'
              " cannot own a transfer hop because the loop's use_as_output and"
              ' run-id ownership would be lost.'
          )
        step_scheduler = active_scheduler
      else:
        step_scheduler = self

      child_ctx = await step_scheduler._execute_step(
          curr_parent_ctx,
          curr_node,
          curr_input,
          node_name=curr_name,
          use_as_output=curr_use_as_output,
          run_id=curr_run_id,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
          resume_inputs=curr_resume_inputs,
      )

      if child_ctx.error or child_ctx.interrupt_ids:
        if self._enable_replay and child_ctx.interrupt_ids:
          self._state.interrupt_ids.update(child_ctx.interrupt_ids)
        return child_ctx

      transfer_to_agent = (
          child_ctx.actions.transfer_to_agent if child_ctx else None
      )

      if not isinstance(transfer_to_agent, str):
        return child_ctx

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
      curr_node = target_agent
      curr_name = target_agent.name
      curr_run_id = None
      curr_input = None
      curr_resume_inputs = None

  async def _execute_step(
      self,
      ctx: Context,
      node: BaseNode,
      node_input: Any,
      *,
      node_name: str | None = None,
      use_as_output: bool = False,
      run_id: str | None = None,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
      resume_inputs: dict[str, Any] | None = None,
  ) -> Context:
    """Execute a single dynamic node step: dedup, resume, or fresh run.

    In workflow mode (enable_replay=True):
      - Allocates auto-incrementing sequential run IDs (_state.next_run_id).
      - Rehydrates and deduplicates executions from session events.
      - Sets up chronological sequence barriers and registers runs in _state.runs.

    In standalone mode (enable_replay=False):
      - Defaults run_id to '1' to keep node paths stable across repeat runs.
      - Skips event rehydration and bypasses run caching, operating as a direct
        pass-through to NodeRunner.
    """
    curr_parent_path = ctx.node_path if ctx else ''
    target_node_name = node_name or node.name
    if not run_id:
      if self._enable_replay:
        run_id = self._state.next_run_id(
            target_node_name, parent_path=curr_parent_path
        )
      else:
        run_id = '1'

    base_path_builder = (
        _NodePathBuilder.from_string(curr_parent_path)
        if curr_parent_path
        else _NodePathBuilder([])
    )
    node_path = str(base_path_builder.append(target_node_name, run_id))

    # Rehydration chronological sequence barrier setup for the parent path
    if self._enable_replay and curr_parent_path:
      self._replay_manager.prepare_parent_sequence_barrier(
          ctx, curr_parent_path
      )

    # Runtime schema validation.
    if self._enable_replay and node_input is not None:
      try:
        node_input = node._validate_input_data(node_input)
      except ValidationError as e:
        raise ValidationError.from_exception_data(
            title=f"dynamic node '{node_name or node.name}'",
            line_errors=e.errors(),  # type: ignore[arg-type]
        ) from e

    logger.debug('node %s schedule start.', node_path)

    child_ctx: Context | None = None
    run_completed = False
    if self._enable_replay:
      # Phase 1: Lazy rehydration from session events.
      if node_path not in self._state.runs:
        self._rehydrate_from_events(ctx, node_path)

      # Check existing run and determine if fresh execution is needed.
      child_ctx, run_completed = await self._check_existing_run(
          ctx,
          node,
          target_node_name,
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
          target_node_name,
          node_path,
          run_id,
          node_input,
          use_as_output,
          is_fresh=True,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
          resume_inputs=resume_inputs,
      )

    if child_ctx is None:
      raise WorkflowInvariantError(
          f'Dynamic node {node_path} completed without a child context.'
      )

    logger.debug('node %s schedule end.', node_path)

    # Advance chronological sequence for this parent path and key
    if self._enable_replay:
      key = f'{target_node_name}@{run_id}'
      await self._replay_manager.advance_sequence(curr_parent_path, key)

    return child_ctx

  async def _check_existing_run(
      self,
      curr_parent_ctx: Context,
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
          raise WorkflowConfigurationError(
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
      run.state.resume_inputs = result.resume_inputs or {}
      logger.debug('node %s schedule: Rerunning execution.', node_path)
      actual_isolation_scope = (
          run.recovered_state.isolation_scope
          if (run.recovered_state and run.recovered_state.isolation_scope)
          else override_isolation_scope
      )
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
              override_isolation_scope=actual_isolation_scope,
          ),
          True,
      )

  # --- Lazy scan ---

  def _rehydrate_from_events(self, ctx: Context, node_path: str) -> None:
    """Scan session events for a dynamic node's prior state."""
    logger.debug('node %s rehydrate start.', node_path)
    ic = ctx._invocation_context  # pylint: disable=protected-access

    filtered_events = self._replay_manager.get_events_for_rehydration(
        ctx, node_path
    )
    results = _reconstruct_node_states(
        events=filtered_events,
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
      resume_inputs: dict[str, Any] | None = None,
  ) -> Context:
    """Unified runner for both fresh and resume executions."""
    if is_fresh:
      state = NodeState(
          status=NodeStatus.RUNNING,
          input=node_input,
          run_id=run_id,
      )
      run = DynamicNodeRun(state=state)
      if self._enable_replay:
        # With replay off nothing reads this map back, and the throwaway
        # state is unreachable from outside, so registering the run would
        # only pin the task and its child Context for the life of the ctx.
        self._state.runs[node_path] = run
      actual_resume_inputs = resume_inputs
    else:
      run = self._state.runs[node_path]
      run.state.status = NodeStatus.RUNNING
      # The rerun path is only reached from _check_existing_run, which does
      # not forward resume_inputs; the recovered state is the only source.
      actual_resume_inputs = (
          dict(run.state.resume_inputs) if run.state.resume_inputs else None
      )

    if not self._enable_replay:
      # Standalone mode: pass node straight through without cloning, and execute
      # directly without creating a new asyncio task, preserving the caller's
      # contextvars and cancellation scope.
      child_ctx = await ctx._run_node_standalone(
          node,
          node_input=node_input,
          use_as_output=use_as_output,
          run_id=run_id,
          use_sub_branch=use_sub_branch,
          override_branch=override_branch,
          override_isolation_scope=override_isolation_scope,
          resume_inputs=actual_resume_inputs,
      )
      return child_ctx

    if hasattr(node, 'clone'):
      target_node = node.clone(update={'name': name})
      parent_agent = getattr(node, 'parent_agent', None)
      if (
          parent_agent is not None
          and getattr(target_node, 'parent_agent', None) is None
      ):
        target_node.parent_agent = parent_agent
    else:
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
            resume_inputs=actual_resume_inputs,
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
  """Executes a node dynamically (Internal Orchestration API)."""
  from ._workflow import Workflow

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

  if use_as_output:
    if not isinstance(ctx.node, Workflow):
      if ctx._output_delegated:
        raise ValueError(
            f'Node {ctx.node_path} already has a use_as_output delegate.'
        )
      ctx._output_delegated = True

  if ctx._workflow_scheduler is not None:
    if run_id and run_id.isdigit() and not skip_run_id_validation:
      raise ValueError(
          f'Explicit run_id "{run_id}" for node "{built_node.name}"'
          ' must contain non-numeric characters to prevent collision'
          ' with auto-generated IDs.'
      )

  scheduler = ctx._workflow_scheduler
  if scheduler is None:
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

  if not return_ctx:
    if child_ctx.error:
      executed_name = child_ctx.node.name if child_ctx.node else built_node.name
      raise DynamicNodeFailError(
          message=f'Dynamic node {executed_name} failed',
          error=child_ctx.error,
          error_node_path=child_ctx.error_node_path,
      )
    if child_ctx.interrupt_ids:
      ctx._interrupt_ids.update(child_ctx.interrupt_ids)
      raise NodeInterruptedError()
    if raise_on_wait and child_ctx.output is None and not transfer_to_agent:
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
