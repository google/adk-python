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

"""New Workflow implementation — BaseNode with graph orchestration.

Combines user-facing graph definition with the execution engine.
Workflow(BaseNode) with _run_impl() as the orchestration loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from dataclasses import field
import logging
from typing import Any
from typing import TYPE_CHECKING

from pydantic import Field

from ._base_node import BaseNode
from ._base_node import START
from ._dynamic_node_scheduler import DynamicNodeScheduler
from ._dynamic_node_scheduler import DynamicNodeState
from ._graph import EdgeItem
from ._graph import Graph
from ._node_runner import NodeRunner
from ._node_state import NodeState
from ._node_status import NodeStatus
from ._trigger import Trigger
from .utils._rehydration_utils import _ChildScanState
from .utils._rehydration_utils import _reconstruct_node_states
from .utils._rehydration_utils import is_terminal_event
from .utils._replay_interceptor import check_interception
from .utils._replay_interceptor import create_mock_context
from .utils._replay_sequence_barrier import ReplaySequenceBarrier

if TYPE_CHECKING:
  from ..agents.context import Context
  from ._schedule_dynamic_node import ScheduleDynamicNode

logger = logging.getLogger('google_adk.' + __name__)


def get_common_branch_prefix(branches: list[str]) -> str:
  """Find the common prefix of dot-separated branch strings."""
  if not branches:
    return ''
  split_branches = [b.split('.') if b else [] for b in branches]

  common = []
  for segments in zip(*split_branches):
    if len(set(segments)) == 1:
      common.append(segments[0])
    else:
      break
  return '.'.join(common)


# ---------------------------------------------------------------------------
# Loop state (mutable, not persisted)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class _LoopState(DynamicNodeState):
  """Mutable, in-memory state for one Workflow execution.

  Extends ``DynamicNodeState`` (which provides dynamic_nodes,
  dynamic_outputs, dynamic_pending_tasks, interrupt_ids) with
  graph-specific fields for static nodes and triggers.

  Scoped to a single _run_impl invocation. Not persisted —
  static node state is reconstructed from session events on
  resume; dynamic node state is lazily scanned on demand.
  Discarded when _run_impl returns.
  """

  # --- Static graph nodes (keyed by node name) ---

  nodes: dict[str, NodeState] = field(default_factory=dict)
  """Static node states."""

  recovered_executions: dict[str, _ChildScanState] = field(default_factory=dict)
  """Raw node states reconstructed from session events, keyed by node_name@run_id."""

  sequence_barrier: ReplaySequenceBarrier | None = None
  """Chronological sequence barrier to ensure deterministic replay ordering."""

  error_shut_down: bool = False
  """Flag indicating that the workflow is shutting down due to an error."""

  node_outputs: dict[str, Any] = field(default_factory=dict)
  """Cached static node outputs."""

  node_branches: dict[str, str] = field(default_factory=dict)
  """Cached static node branches."""

  pending_tasks: dict[str, asyncio.Task[Context]] = field(default_factory=dict)
  """Running static node tasks."""

  trigger_buffer: dict[str, list[Trigger]] = field(default_factory=dict)
  """Queued triggers waiting to be dispatched, keyed by target node name.

  Producers:
  - _seed_start_triggers: initial triggers for START successors
  - _buffer_downstream_triggers: when a node completes, triggers
    its downstream successors
  - _process_resume: seeds triggers for PENDING nodes on resume

  Consumer:
  - _schedule_ready_nodes: pops triggers, creates NodeRunners,
    moves nodes to RUNNING
  """

  schedule_dynamic_node: ScheduleDynamicNode | None = None
  """Closure that handles ctx.run_node() calls from child nodes.

  Tracks dynamic nodes in this Workflow's loop state
  (dynamic_nodes, dynamic_outputs, dynamic_pending_tasks).
  Handles dedup (cached output), resume (lazy scan + re-run),
  and fresh execution.

  Set on ctx at Workflow setup, propagated down to descendants
  via NodeRunner until a nested orchestration node overrides it.
  """


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class Workflow(BaseNode):
  """A graph-based workflow node.

  _run_impl() IS the graph orchestration loop:
  - SETUP: build graph, seed triggers
  - LOOP: schedule ready nodes via NodeRunner, handle completions
  - FINALIZE: collect terminal outputs
  """

  rerun_on_resume: bool = Field(default=True)

  edges: list[EdgeItem] = Field(
      description='Edges to build the workflow graph.',
      default_factory=list,
  )

  max_concurrency: int | None = None
  """Maximum parallel graph-scheduled nodes. None means unlimited.

  Only applies to nodes triggered by graph edges. Dynamic nodes
  (via ctx.run_node()) are excluded — they are awaited inline by
  their parent and throttling them would cause deadlock.
  """

  graph: Graph | None = Field(
      description='The compiled workflow graph.',
      default=None,
  )

  # --- Construction ---

  def model_post_init(self, context: Any) -> None:
    super().model_post_init(context)
    if self.edges and self.graph is None:
      self.graph = self._build_graph()
    self._validate_state_schema()
    self._validate_no_task_mode_graph_nodes()

  def _build_graph(self) -> Graph:
    """Convert edge definitions to a validated Graph."""
    graph = Graph.from_edge_items(self.edges)
    graph.validate_graph()
    return graph

  def _validate_no_task_mode_graph_nodes(self) -> None:
    """Reject ``mode='task'`` LlmAgents that appear as static graph nodes.

    Task-mode agents are multi-turn — they pause for user replies and
    expect the original ``node_input`` (the task brief) to remain visible
    across re-dispatches.  The workflow scheduler currently overwrites
    ``node_input`` with the latest user message on every re-entry, so the
    task brief is lost and the agent loses context.  Until the scheduler
    preserves the originating ``node_input`` on resume, task agents may
    only be used:

      * as chat sub-agents of an LlmAgent coordinator (FC delegation
        via ``_TaskAgentTool`` / ``_dispatch_task_fc``), or
      * dispatched dynamically via ``ctx.run_node`` from a custom
        function node — never as static workflow graph nodes.
    """
    if not self.graph:
      return
    from ..agents.llm_agent import LlmAgent

    for graph_node in self.graph.nodes:
      if isinstance(graph_node, LlmAgent) and graph_node.mode == 'task':
        raise ValueError(
            f"Agent {graph_node.name!r} has mode='task' and cannot be "
            'used as a workflow graph node. Use a chat coordinator with '
            'task sub-agents, or dispatch dynamically via ctx.run_node '
            'from a function node.'
        )

  def _validate_state_schema(self) -> None:
    """Raises when FunctionNode params don't match state_schema fields."""
    if not self.state_schema or not self.graph:
      return

    from ..sessions.state import StateSchemaError
    from ._function_node import FunctionNode

    schema_fields = set(self.state_schema.model_fields.keys())

    for graph_node in self.graph.nodes:
      if not isinstance(graph_node, FunctionNode):
        continue

      for param_name in graph_node._sig.parameters:
        if param_name in ('ctx', 'node_input', 'self'):
          continue

        if param_name not in schema_fields:
          raise StateSchemaError(
              f'FunctionNode {graph_node.name!r} parameter '
              f'{param_name!r} is not declared in state_schema '
              f'{self.state_schema.__name__!r}. Declared fields: '
              f'{sorted(schema_fields)}'
          )

  # --- _run_impl: the orchestration loop ---

  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    """Orchestration loop: SETUP -> LOOP -> FINALIZE."""
    if self.graph is None:
      return

    # Set event_author so child events are attributed to this workflow.
    ctx.event_author = self.name

    # --- SETUP: resume from events or start fresh ---
    # TODO: resume from checkpoint event.
    loop_state = _LoopState()
    loop_state.recovered_executions, recovered_sequence = (
        self._scan_child_events(ctx)
    )
    loop_state.sequence_barrier = ReplaySequenceBarrier(recovered_sequence)

    if ctx.resume_inputs and not loop_state.recovered_executions:
      logger.warning(
          'Workflow %s: resume_inputs provided but no recovered executions'
          ' found.',
          self.name,
      )

    self._seed_start_triggers(loop_state, node_input)

    # Create closure for dynamic node scheduling
    loop_state.schedule_dynamic_node = self._make_schedule_dynamic_node(
        loop_state
    )
    ctx._workflow_scheduler = loop_state.schedule_dynamic_node

    # --- LOOP ---
    try:
      await self._run_loop(loop_state, ctx)
    finally:
      await self._cleanup_all_tasks(loop_state)

    if loop_state.error_shut_down:
      return

    # Collect remaining interrupts from WAITING nodes
    self._collect_remaining_interrupts(loop_state)

    # --- FINALIZE ---
    # Terminal node output already has output_for including this
    # workflow's path. Mark output as delegated so the workflow's
    # NodeRunner skips creating a duplicate output event.
    if self._has_terminal_output(loop_state):
      ctx._output_delegated = True
    self._finalize(loop_state, ctx)
    return
    yield  # required to keep _run_impl as async generator

  # --- LOOP ---

  async def _run_loop(self, loop_state: _LoopState, ctx: Context) -> None:
    """Schedule and execute nodes until no more work."""
    logger.debug('node %s execute loop start.', ctx.node_path)

    recovered_sequence_indices = {
        node_path: i
        for i, node_path in enumerate(
            loop_state.sequence_barrier.sequence
            if loop_state.sequence_barrier
            else []
        )
    }

    while True:
      self._schedule_ready_nodes(loop_state, ctx)

      if not loop_state.pending_tasks:
        break

      done, _ = await asyncio.wait(
          loop_state.pending_tasks.values(),
          return_when=asyncio.FIRST_COMPLETED,
      )

      # To ensure deterministic processing order even for fresh executions,
      # first order the done tasks by their insertion order in pending_tasks.
      # Since pending_tasks is a dict, its values() preserve the exact order
      # in which the tasks were originally scheduled.
      ordered_done = [
          task for task in loop_state.pending_tasks.values() if task in done
      ]

      task_to_name = {
          task: name for name, task in loop_state.pending_tasks.items()
      }

      # Sort done tasks by their order in recovered_sequence to ensure
      # processing order matches history.
      # This is needed because python asyncio.wait returns a set of completed
      # tasks (done) when multiple tasks finish at the same time, which loses
      # the order in which the original pending tasks were enqueued.
      # Tasks not found in the sequence (e.g., new executions) will be placed
      # at the end, preserving their original insertion order due to Python's
      # stable sort.
      def get_recovered_sequence_index(t):
        name = task_to_name.get(t)
        if not name:
          return float('inf')
        node_state = loop_state.nodes.get(name)
        if not node_state:
          return float('inf')
        node_path = f'{name}@{node_state.run_id}'
        return recovered_sequence_indices.get(node_path, float('inf'))

      sorted_done = sorted(ordered_done, key=get_recovered_sequence_index)

      for task in sorted_done:
        name = self._pop_completed_task(loop_state, task)

        node = self._get_static_node_by_name(name)
        child_ctx: Context = task.result()
        if loop_state.sequence_barrier:
          loop_state.sequence_barrier.check_and_advance(
              f'{name}@{child_ctx.run_id}'
          )

        if child_ctx.error:
          node_state = loop_state.nodes[name]
          node_state.status = NodeStatus.FAILED

          ctx._error = child_ctx.error
          ctx._error_node_path = child_ctx.error_node_path

          loop_state.error_shut_down = True
          logger.debug('node %s execute loop end.', ctx.node_path)
          return

        self._handle_completion(loop_state, name, node, child_ctx)

    # Await fire-and-forget dynamic tasks.
    # TODO: Handle dynamic task failures and interrupts here.
    # Currently, dynamic node completion is handled inline in the
    # _schedule_dynamic_node_callback closure. But failures are not caught.
    dynamic_tasks = loop_state.get_dynamic_tasks()
    if dynamic_tasks:
      await asyncio.wait(dynamic_tasks)
    logger.debug('node %s execute loop end.', ctx.node_path)

  # --- Scheduling ---

  def _seed_start_triggers(
      self,
      loop_state: _LoopState,
      node_input: Any,
  ) -> None:
    """Seed triggers for START's direct successors."""
    assert self.graph is not None

    start_edges = [
        e for e in self.graph.edges if e.from_node.name == START.name
    ]
    use_sub_branch = len(start_edges) > 1
    for edge in start_edges:
      loop_state.trigger_buffer.setdefault(edge.to_node.name, []).append(
          Trigger(
              input=node_input,
              use_sub_branch=use_sub_branch,
          )
      )

  def _schedule_ready_nodes(self, loop_state: _LoopState, ctx: Context) -> None:
    """Pop triggers from buffer and schedule ready nodes."""
    # loop_state.trigger_buffer is a dict, and Python 3.7+ dicts preserve insertion order.
    # Therefore, nodes are processed strictly in the order their triggers arrived,
    # ensuring deterministic scheduling order for parallel branches.
    for node_name in list(loop_state.trigger_buffer.keys()):
      if node_name in loop_state.pending_tasks:
        continue
      node_state = loop_state.nodes.get(node_name)
      if node_state:
        # Serialize executions of the same node to prevent race conditions.
        # If a node is already RUNNING, or WAITING for user input (interrupts),
        # we skip dequeuing new triggers for it until it completes its current turn.
        if node_state.status == NodeStatus.RUNNING:
          continue
        # We only skip WAITING nodes if they have unresolved interrupts (waiting for user).
        # If they are WAITING because of wait_for_output=True but produced no output yet,
        # they should still process new triggers to accumulate state.
        if node_state.status == NodeStatus.WAITING and node_state.interrupts:
          continue

      if self._at_concurrency_limit(loop_state):
        break

      trigger = self._pop_trigger(loop_state, node_name)
      if trigger is None:
        continue

      self._prepare_node_state_for_starting(loop_state, node_name, trigger)
      self._start_node_task(loop_state, ctx, node_name, trigger)

  def _at_concurrency_limit(self, loop_state: _LoopState) -> bool:
    """Check if max_concurrency has been reached."""
    return (
        bool(self.max_concurrency)
        and len(loop_state.pending_tasks) >= self.max_concurrency
    )

  def _pop_trigger(
      self, loop_state: _LoopState, node_name: str
  ) -> Trigger | None:
    """Pop the next trigger for a node, or None if empty."""
    buffer = loop_state.trigger_buffer.get(node_name, [])
    if not buffer:
      return None
    trigger = buffer.pop(0)
    if not buffer:
      del loop_state.trigger_buffer[node_name]
    return trigger

  @staticmethod
  def _next_run_id(node_state: NodeState) -> str:
    """Increment and return the next sequential run_id for a node."""
    node_state.run_counter += 1
    return str(node_state.run_counter)

  @staticmethod
  def _compute_isolation_scope_for_node(
      node: BaseNode,
      trigger: Trigger,
      parent_ctx: Context | None,
      run_id: str,
  ) -> str | None:
    """Decide the isolation_scope for a node about to run.

    Order of precedence:
      1. Explicit ``trigger.isolation_scope`` — set by the resume path
         (``loop_state.recovered_executions[key].isolation_scope``) so a
         resumed run continues in its original scope.
      2. Task-mode LlmAgent node — gets the task agent's full node_path
         (``<parent_path>/<name>@<run_id>``) as its scope so its
         multi-turn conversation is isolated from peer workflow nodes.
         The full path (not just ``<name>@<run_id>``) is required so
         scopes stay unique across nested workflows or re-used node
         names in different graph positions.
      3. Otherwise unscoped — workflow nodes share the workflow's
         conversation view by default.

    Note: FC-driven task delegations (chat coordinator → task agent
    via ``ctx.run_node``) take a different path and set
    ``override_isolation_scope=fc.id`` directly on the NodeRunner.
    """
    if trigger.isolation_scope is not None:
      return trigger.isolation_scope
    if getattr(node, 'mode', None) == 'task':
      parent_path = parent_ctx.node_path if parent_ctx else ''
      segment = f'{node.name}@{run_id}'
      return f'{parent_path}/{segment}' if parent_path else segment
    return None

  @classmethod
  def _create_node_state_for_new_run(cls, old_state: NodeState) -> NodeState:
    """Create a fresh NodeState for a new run, preserving the run counter."""
    return NodeState(run_counter=old_state.run_counter)

  def _prepare_node_state_for_starting(
      self, loop_state: _LoopState, node_name: str, trigger: Trigger
  ) -> None:
    """Prepare NodeState for starting a node.

    This method determines whether to reuse or recreate the node's state:
    *   Creates a brand new `NodeState` if none exists.
    *   Creates a fresh `NodeState` (preserving `run_counter`) if this is a new execution
        (not resuming and not waiting) to avoid state carryover.
    *   Reuses the existing `NodeState` if resuming from interrupt or waiting for inputs.

    Outcome: The node's state is updated with the trigger's input and source,
    and its status is set to `RUNNING`.
    """
    if node_name not in loop_state.nodes:
      node_state = NodeState()
      loop_state.nodes[node_name] = node_state
    else:
      node_state = loop_state.nodes[node_name]
      # Create a new NodeState for a fresh execution to avoid carryover bugs.
      node_state = self._create_node_state_for_new_run(node_state)
      loop_state.nodes[node_name] = node_state

    node_state.input = trigger.input
    node_state.status = NodeStatus.RUNNING

  def _start_node_task(
      self,
      loop_state: _LoopState,
      ctx: Context,
      node_name: str,
      trigger: Trigger,
  ) -> None:
    """Create NodeRunner and start asyncio task for a node."""

    assert self.graph is not None

    node = self._get_static_node_by_name(node_name)
    is_terminal = node_name in self.graph._terminal_node_names

    node_state = loop_state.nodes[node_name]
    # Reuse run_id on resume; assign a new sequential id for fresh runs.
    run_id = node_state.run_id
    if not run_id:
      run_id = self._next_run_id(node_state)
    node_state.run_id = run_id

    # Intercept execution based on historical session events.
    key = f'{node_name}@{run_id}'
    if key in loop_state.recovered_executions:
      recovered = loop_state.recovered_executions[key]

      result = check_interception(
          node_path=f'{ctx.node_path}/{node_name}@{run_id}',
          node=node,
          recovered=recovered,
          curr_parent_ctx=ctx,
      )

      if not result.should_run:
        is_terminal = node_name in self.graph._terminal_node_names
        ancestor_path = ctx.node_path if is_terminal else None

        if ancestor_path:
          ancestors = [ancestor_path] + list(ctx._output_for_ancestors or [])
        else:
          ancestors = list(ctx._output_for_ancestors or [])

        mock_ctx = create_mock_context(
            parent_ctx=ctx,
            node=node,
            run_id=run_id,
            result=result,
            ancestors=ancestors,
            branch=recovered.branch,
        )

        async def return_ctx():
          if loop_state.sequence_barrier:
            await loop_state.sequence_barrier.wait(key)
          return mock_ctx

        loop_state.pending_tasks[node_name] = asyncio.create_task(return_ctx())
        return

      node_state.resume_inputs = result.resume_inputs or {}

    # when re-running a node from replay, prefer the
    # recovered isolation_scope so the resumed run continues in its
    # original scope (rather than computing a fresh wf:<eid>).
    if (
        key in loop_state.recovered_executions
        and loop_state.recovered_executions[key].isolation_scope
        and trigger.isolation_scope is None
    ):
      trigger.isolation_scope = loop_state.recovered_executions[
          key
      ].isolation_scope

    runner = NodeRunner(
        node=node,
        parent_ctx=ctx,
        run_id=run_id,
        use_as_output=is_terminal,
        use_sub_branch=trigger.use_sub_branch,
        override_branch=trigger.branch,
        override_isolation_scope=self._compute_isolation_scope_for_node(
            node, trigger, ctx, run_id
        ),
    )
    resume_inputs = (
        dict(node_state.resume_inputs) if node_state.resume_inputs else None
    )
    loop_state.pending_tasks[node_name] = asyncio.create_task(
        runner.run(node_input=trigger.input, resume_inputs=resume_inputs)
    )

  def _make_schedule_dynamic_node(
      self, loop_state: _LoopState
  ) -> ScheduleDynamicNode:
    """Create a DynamicNodeScheduler for this Workflow's loop state."""
    return DynamicNodeScheduler(state=loop_state)

  # --- Completion handling ---

  def _handle_completion(
      self,
      loop_state: _LoopState,
      node_name: str,
      node: BaseNode,
      child_ctx: Context,
  ) -> None:
    """Update state and trigger downstream after node completes."""
    node_state = loop_state.nodes[node_name]

    if child_ctx.interrupt_ids:
      node_state.status = NodeStatus.WAITING
      node_state.interrupts = list(child_ctx.interrupt_ids)
      loop_state.interrupt_ids.update(child_ctx.interrupt_ids)
      return

    if (
        node.wait_for_output
        and child_ctx.output is None
        and child_ctx.route is None
    ):
      node_state.status = NodeStatus.WAITING
      return

    node_state.status = NodeStatus.COMPLETED
    if node_state.resume_inputs:
      node_state.resume_inputs.clear()
    if child_ctx.output is not None:
      loop_state.node_outputs[node_name] = child_ctx.output
    loop_state.node_branches[node_name] = (
        child_ctx._invocation_context.branch or ''
    )

    # Buffer downstream triggers.
    self._buffer_downstream_triggers(
        loop_state,
        node_name,
        child_ctx.output,
        child_ctx.route,
        child_ctx._invocation_context.branch,
    )

  def _buffer_downstream_triggers(
      self,
      loop_state: _LoopState,
      node_name: str,
      output: Any,
      route: Any,
      branch: str | None = None,
  ) -> None:
    """Find downstream edges and add triggers to the buffer."""
    assert self.graph is not None
    next_nodes = self.graph.get_next_pending_nodes(
        node_name=node_name,
        routes_to_match=route,
    )
    use_sub_branch = len(next_nodes) > 1
    for target_name in next_nodes:
      target_node = self._get_static_node_by_name(target_name)
      target_state = loop_state.nodes.get(target_name)

      if target_node._requires_all_predecessors:
        # Wait for all predecessors
        predecessors = {
            e.from_node.name
            for e in self.graph.edges
            if e.to_node.name == target_name
        }
        if all(
            loop_state.nodes.get(p)
            and loop_state.nodes[p].status == NodeStatus.COMPLETED
            for p in predecessors
        ):
          # All predecessors have completed!
          outputs = {p: loop_state.node_outputs.get(p) for p in predecessors}
          branches = [loop_state.node_branches.get(p, '') for p in predecessors]
          common_branch = get_common_branch_prefix(branches)

          loop_state.trigger_buffer.setdefault(target_name, []).append(
              Trigger(
                  input=outputs,
                  use_sub_branch=False,
                  branch=common_branch,
              )
          )
      else:
        # Normal node logic
        loop_state.trigger_buffer.setdefault(target_name, []).append(
            Trigger(
                input=output,
                use_sub_branch=use_sub_branch,
                branch=branch,
            )
        )

  def _collect_remaining_interrupts(self, loop_state: _LoopState) -> None:
    """Gather interrupt_ids from nodes still WAITING after the loop."""
    for node_state in loop_state.nodes.values():
      if node_state.status == NodeStatus.WAITING and node_state.interrupts:
        loop_state.interrupt_ids.update(node_state.interrupts)

  # --- Resume ---

  def _scan_child_events(
      self, ctx: Context
  ) -> tuple[dict[str, _ChildScanState], list[str]]:
    """Scan session events and collect per-child state and completion sequence.

    Forward pass through events for this invocation. For each direct
    child, tracks the latest run_id and accumulates output,
    interrupt IDs, and resolved interrupt IDs.

    Returns:
      Tuple of:
        - dict of child_name → _ChildScanState.
        - list of child_name@run_id in chronological order of completion.
    """
    ic = ctx._invocation_context
    raw_results = _reconstruct_node_states(
        events=ic.session.events,
        base_path=ctx.node_path,
        group_by_direct_child=True,
        invocation_id=ic.invocation_id,
    )

    from ..events._node_path_builder import _NodePathBuilder

    # Build chronological sequence of completions
    sequence: list[str] = []
    base_path_builder = _NodePathBuilder.from_string(ctx.node_path)

    for event in ic.session.events:
      if event.invocation_id != ic.invocation_id:
        continue

      event_node_path = event.node_info.path or ''
      event_path_builder = _NodePathBuilder.from_string(event_node_path)

      if not event_path_builder.is_descendant_of(base_path_builder):
        continue

      child_path = base_path_builder.get_direct_child(event_path_builder)
      segment: str = child_path.leaf_segment

      if is_terminal_event(event):
        # Maintain unique segments ordered by their LAST terminal event.
        # If a node interrupts in turn 1 and completes in turn 2, moving it
        # to the end ensures we record the order of its final completion.
        if segment in sequence:
          sequence.remove(segment)
        sequence.append(segment)

    return raw_results, sequence

  # --- FINALIZE ---

  def _finalize(self, loop_state: _LoopState, ctx: Context) -> None:
    """Set interrupt_ids or terminal output on ctx.

    If any child interrupted, propagate their interrupt IDs to ctx
    so the parent orchestrator sees them. Otherwise, set the terminal
    node's output on ctx so the parent can read it.
    """
    if loop_state.interrupt_ids:
      ctx._interrupt_ids = set(loop_state.interrupt_ids)
      return

    # Set terminal output on ctx so parent reads ctx.output.
    # Terminal nodes = no outgoing edges.
    assert self.graph is not None
    terminal_outputs = [
        loop_state.node_outputs[name]
        for name in self.graph._terminal_node_names
        if name in loop_state.node_outputs
    ]
    if len(terminal_outputs) == 1:
      ctx.output = self._validate_output_data(terminal_outputs[0])
    elif terminal_outputs:
      raise ValueError(
          f'Workflow {self.name}: multiple terminal nodes produced'
          f' output ({len(terminal_outputs)}). A workflow must have'
          ' at most one terminal output.'
      )

  # --- Utilities ---

  def _has_terminal_output(self, loop_state: _LoopState) -> bool:
    """Check if any terminal node produced output."""
    assert self.graph is not None
    return any(
        name in loop_state.node_outputs
        for name in self.graph._terminal_node_names
    )

  def _get_static_node_by_name(self, name: str) -> BaseNode:
    """Find a node in the graph by name."""
    assert self.graph is not None
    for node in self.graph.nodes:
      if node.name == name:
        return node
    raise ValueError(f'Node {name} not found in graph.')

  def _pop_completed_task(
      self, loop_state: _LoopState, task: asyncio.Task[Context]
  ) -> str:
    """Remove a completed task and return its node name."""
    for name, t in loop_state.pending_tasks.items():
      if t is task:
        del loop_state.pending_tasks[name]
        return name
    raise ValueError('Task not found in pending_tasks.')

  async def _cleanup_all_tasks(self, loop_state: _LoopState) -> None:
    """Cancel remaining tasks to prevent leaks."""
    dynamic_tasks = loop_state.get_dynamic_tasks()

    all_tasks = list(loop_state.pending_tasks.values()) + dynamic_tasks
    if all_tasks:
      logger.warning(
          'Workflow %s: cancelling %d leftover tasks.',
          self.name,
          len(all_tasks),
      )
    for task in all_tasks:
      if not task.done():
        task.cancel()
    if all_tasks:
      await asyncio.gather(*all_tasks, return_exceptions=True)
      for task in all_tasks:
        if task.cancelled():
          # Mark static nodes as CANCELLED
          for name, t in loop_state.pending_tasks.items():
            if t is task:
              loop_state.nodes[name].status = NodeStatus.CANCELLED
              break
          # Mark dynamic nodes as CANCELLED
          for _, run in loop_state.runs.items():
            if run.task is task:
              run.state.status = NodeStatus.CANCELLED
              break
