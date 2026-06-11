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

"""NodeRunner — per-node executor class.

Converts BaseNode.run() (async generator) into an awaitable that returns
the child Context with output, route, and interrupt_ids set. Used
internally by orchestrators (Workflow, SingleLlmAgentReactNode, etc.).

User-facing ctx.run_node() wraps this and returns just ctx.output.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from typing import TYPE_CHECKING

from ..events._node_path_builder import _NodePathBuilder
from ..telemetry import node_tracing

if TYPE_CHECKING:
  from ..agents.context import Context
  from ..events.event import Event
  from ._base_node import BaseNode


logger = logging.getLogger("google_adk." + __name__)


class NodeRunner:
  """Per-node executor. Drives BaseNode.run(), enriches events.

  Creates child Context, iterates node.run(), enqueues events to
  ic.event_queue, writes output/route/interrupt_ids to ctx, and
  returns the child Context.
  """

  def __init__(
      self,
      *,
      node: BaseNode,
      parent_ctx: Context,
      run_id: str | None = None,
      # Output delegation (use_as_output)
      use_as_output: bool = False,
      # Resume state from a previous run
      prior_output: Any = None,
      prior_interrupt_ids: set[str] | None = None,
      use_sub_branch: bool = False,
      override_branch: str | None = None,
      override_isolation_scope: str | None = None,
  ) -> None:
    """Initialize a NodeRunner.

    Args:
      node: The BaseNode to execute.
      parent_ctx: The parent node's Context.
      run_id: Unique ID for this run. Should be a sequential
        counter string ("1", "2", …) unique per node path.
        Falls back to "1" if not provided.

      use_as_output: If True, this node's output also represents the parent
        node's output.
      prior_output: Output from a previous run, carried
        forward on resume when the node had both output and
        interrupts.
      prior_interrupt_ids: Unresolved interrupt IDs (set) from a
        previous run, carried forward on resume.
      use_sub_branch: Whether the node should use a sub-branch.
      override_branch: Optional branch to use instead of parent's branch.
    """
    # Core
    self._node = node
    self._parent_ctx = parent_ctx

    self._run_id = str(run_id) if run_id else "1"
    self._use_sub_branch = use_sub_branch
    self._override_branch = override_branch
    self._override_isolation_scope = override_isolation_scope

    # Output delegation
    self._use_as_output = use_as_output

    # Resume state
    self._prior_output = prior_output
    self._prior_interrupt_ids = prior_interrupt_ids

  @property
  def run_id(self) -> str:
    """The run ID assigned to this node run."""
    return self._run_id

  async def run(
      self,
      node_input: Any = None,
      *,
      resume_inputs: dict[str, Any] | None = None,
  ) -> Context:
    """Drive node.run(), enqueue events, return child Context.

    The caller reads ctx.output, ctx.route, and ctx.interrupt_ids
    for the node's results.
    """
    attempt_count = 1
    while True:
      ctx = self._create_child_context(
          resume_inputs, attempt_count=attempt_count
      )
      logger.debug("node %s started.", ctx.node_path)
      try:
        # Start the span within try-except block to record exceptions on the span
        async with node_tracing.start_as_current_node_span(
            self._parent_ctx, self._node
        ) as telemetry_context:
          ctx._telemetry_context = telemetry_context
          await self._execute_node(ctx, node_input)
          await self._flush_output_and_deltas(ctx)
          logger.debug("node %s end.", ctx.node_path)
          return ctx
      except Exception as e:
        from ._errors import DynamicNodeFailError

        if isinstance(e, DynamicNodeFailError):
          # TODO: consider to retry upon dynamic node failures later. This may
          # require thorough design to consider a workflow dynamic node and a
          # normal node.
          ctx._error = e.error
          ctx._error_node_path = e.error_node_path
          logger.debug("node %s end.", ctx.node_path)
          return ctx

        from ..events.event import Event

        logger.exception("Node execution failed with exception")
        error_event = Event(
            error_code=type(e).__name__,
            error_message=str(e),
        )
        await self._enqueue_event(error_event, ctx)

        if not await self._attempt_retry(e, ctx, attempt_count):
          ctx._error = e
          ctx._error_node_path = ctx.node_path
          logger.debug("node %s end.", ctx.node_path)
          return ctx
        logger.warning(
            "Node %s failed and is being retried locally. Note: retry count is"
            " not persisted across resuming.",
            self._node.name,
        )
        attempt_count += 1

  async def _attempt_retry(
      self, e: Exception, ctx: Context, attempt_count: int
  ) -> bool:
    """Checks if node should retry and sleeps if so."""
    from ._node_state import NodeState
    from .utils._retry_utils import _get_retry_delay
    from .utils._retry_utils import _should_retry_node

    node_state = NodeState(attempt_count=attempt_count)

    if not _should_retry_node(e, self._node.retry_config, node_state):
      return False

    delay = _get_retry_delay(self._node.retry_config, node_state)

    await asyncio.sleep(delay)
    return True

  def _create_child_context(
      self,
      resume_inputs: dict[str, Any] | None,
      attempt_count: int = 1,
  ) -> Context:
    """Create a child Context for the node, inheriting from parent.

    If prior_output or prior_interrupt_ids were provided at
    construction (resume scenario), pre-populates ctx with state
    from the previous run.
    """
    from ..agents.context import Context

    ic = self._parent_ctx._invocation_context
    base_branch = (
        self._override_branch
        if self._override_branch is not None
        else ic.branch
    )

    if self._use_sub_branch:
      segment = f"{self._node.name}@{self._run_id}"
      branch = f"{base_branch}.{segment}" if base_branch else segment
      ic = ic.model_copy(update={"branch": branch})
    elif self._override_branch is not None:
      ic = ic.model_copy(update={"branch": self._override_branch})

    ctx = Context(
        ic,
        parent_ctx=self._parent_ctx,
        node=self._node,
        run_id=self._run_id,
        resume_inputs=resume_inputs,
        use_as_output=self._use_as_output,
        attempt_count=attempt_count,
    )

    # override the inherited isolation_scope when explicitly set.
    if self._override_isolation_scope is not None:
      ctx.isolation_scope = self._override_isolation_scope

    # Carry forward state from a previous run (resume scenario).
    if self._prior_output is not None:
      ctx._output_value = self._prior_output
      ctx._output_emitted = True
    if self._prior_interrupt_ids:
      ctx._interrupt_ids.update(self._prior_interrupt_ids)

    return ctx

  async def _execute_node(
      self,
      ctx: Context,
      node_input: Any,
  ) -> None:
    """Iterate node.run(), enqueue events, write results to ctx."""
    from ._errors import NodeInterruptedError
    from ._errors import NodeTimeoutError

    try:
      timeout = self._node.timeout
      if timeout is not None:
        await self._run_node_loop_with_timeout(ctx, node_input, timeout)
      else:
        await self._run_node_loop(ctx, node_input)
    except NodeInterruptedError:
      # A dynamic child interrupted via ctx.run_node().
      # The child's interrupt_ids are already on ctx
      # (set by the schedule callback). Nothing more to do —
      # the caller reads ctx.interrupt_ids.
      pass

  async def _run_node_loop(self, ctx: Context, node_input: Any) -> None:
    """Iterate node.run(), track events in context, and enqueue them."""
    from ..utils.context_utils import Aclosing

    logger.debug("node %s execute loop start.", ctx.node_path)
    async with Aclosing(self._node.run(ctx=ctx, node_input=node_input)) as agen:
      async for event in agen:
        self._track_event_in_context(event, ctx)
        await self._enqueue_event(event, ctx)

    logger.debug("node %s execute loop end.", ctx.node_path)

  async def _run_node_loop_with_timeout(
      self, ctx: Context, node_input: Any, timeout: float
  ) -> None:
    try:
      await asyncio.wait_for(
          self._run_node_loop(ctx, node_input), timeout=timeout
      )
    except asyncio.TimeoutError as e:
      from ._errors import NodeTimeoutError

      raise NodeTimeoutError(node_name=self._node.name, timeout=timeout) from e

  def _track_event_in_context(self, event: Event, ctx: Context) -> None:
    """Write yielded event results to ctx (source of truth)."""
    if event.output is not None:
      ctx.output = event.output
    elif event.node_info and event.node_info.message_as_output:
      ctx.output = event.content
    if event.long_running_tool_ids is not None:
      ctx._interrupt_ids.update(event.long_running_tool_ids)
    # Only propagate decisions from native events (authored by this node or unspecified).
    # This prevents structured parent nodes (e.g. SequentialAgent) from intercepting
    # and bubbling up actions already handled internally by their nested sub-agents.
    is_native_node_event = not event.author or event.author == self._node.name
    if event.actions and is_native_node_event:
      if event.actions.route is not None:
        ctx.route = event.actions.route
        ctx._route_emitted = True
      if event.actions.transfer_to_agent is not None:
        ctx.actions.transfer_to_agent = event.actions.transfer_to_agent

    ctx.telemetry_context.add_event(event)

    # Validate state_delta if schema is present
    if (
        event.actions
        and event.actions.state_delta
        and ctx.state._schema is not None
    ):
      from ..sessions.state import _validate_state_entry

      for key, value in event.actions.state_delta.items():
        _validate_state_entry(ctx.state._schema, key, value)

  async def _enqueue_event(self, event: Event, ctx: Context) -> None:
    """Enrich and enqueue event to the session.

    Skips enqueueing if output is delegated via use_as_output —
    the child already emitted it. Pending deltas stay in ctx for
    _flush_output_and_deltas.
    """
    if event.output is not None and ctx._output_delegated:
      return

    self._enrich_event(event, ctx)
    if not event.partial:
      self._flush_deltas(event, ctx)
    await ctx._invocation_context._enqueue_event(event)

    if event.output is not None:
      ctx._output_emitted = True
    if event.node_info.message_as_output:
      ctx._output_delegated = True

  async def _flush_output_and_deltas(self, ctx: Context) -> None:
    """Emit deferred output and/or unflushed state/artifact deltas."""
    from ..events.event import Event
    from ..events.event_actions import EventActions

    state_delta = ctx.actions.state_delta
    artifact_delta = ctx.actions.artifact_delta
    has_deferred_output = (
        ctx._output_value is not None
        and not ctx._output_emitted
        and not ctx._output_delegated
    )
    has_unflushed_route = (
        ctx._route_value is not None and not ctx._route_emitted
    )
    has_deltas = bool(state_delta or artifact_delta)

    if not has_deferred_output and not has_deltas and not has_unflushed_route:
      return

    # Build the event — output + route + deltas, or a subset.
    event = Event(
        output=ctx._output_value if has_deferred_output else None,
        route=ctx._route_value if has_unflushed_route else None,
    )
    if has_deltas:
      event.actions = EventActions(
          state_delta=dict(state_delta),
          artifact_delta=dict(artifact_delta),
      )
      state_delta.clear()
      artifact_delta.clear()

    self._enrich_event(event, ctx)
    await ctx._invocation_context._enqueue_event(event)
    if has_deferred_output:
      ctx._output_emitted = True
    if has_unflushed_route:
      ctx._route_emitted = True

  def _flush_deltas(self, event: Event, ctx: Context) -> None:
    """Move pending state/artifact deltas from ctx onto the event.

    TODO: Handle non-persisted states (e.g. `temp:` prefixed keys)
    that should flow through ctx but not be written to session events.
    """
    from ..events.event_actions import EventActions

    state_delta = ctx.actions.state_delta
    artifact_delta = ctx.actions.artifact_delta
    if not state_delta and not artifact_delta:
      return

    if not event.actions:
      event.actions = EventActions()
    if state_delta:
      event.actions.state_delta.update(state_delta)
      state_delta.clear()
    if artifact_delta:
      event.actions.artifact_delta.update(artifact_delta)
      artifact_delta.clear()

  def _enrich_event(self, event: Event, ctx: Context) -> None:
    """Set author, node_info, invocation_id on the event."""
    # TODO: revisit after we settle Event.author logic for content/message.
    event.author = ctx.event_author or self._node.name
    event.invocation_id = ctx._invocation_context.invocation_id
    event.node_info.path = ctx.node_path
    if event.branch is None:
      event.branch = ctx._invocation_context.branch
    elif event.branch == "":
      event.branch = None
      ctx._invocation_context.branch = None
    else:
      ctx._invocation_context.branch = event.branch
    if event.output is not None:
      event.node_info.output_for = [ctx.node_path] + ctx._output_for_ancestors
    # stamp the scope tag.
    if event.isolation_scope is None and ctx.isolation_scope is not None:
      event.isolation_scope = ctx.isolation_scope
