# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay interceptor for workflow rehydration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

from ...agents.context import Context
from .._base_node import BaseNode
from .._node_state import NodeState
from .._node_status import NodeStatus
from ._rehydration_utils import _ChildScanState
from ._rehydration_utils import _process_rehydrated_output

if TYPE_CHECKING:
  from .._dynamic_node_scheduler import DynamicNodeRun


@dataclass(kw_only=True)
class InterceptionResult:
  """Result of replay interception check."""

  should_run: bool
  """Whether the node should be executed natively."""

  output: Any = None
  """The cached output to fast-forward or auto-complete with."""

  route: Any = None
  """The cached route to fast-forward with."""

  interrupts: set[str] = field(default_factory=set)
  """Unresolved interrupts if the node should stay WAITING."""

  resume_inputs: dict[str, Any] | None = None
  """Resolved responses to feed into the node if it is rerun."""

  transfer_to_agent: str | None = None
  """Target agent name if fast-forwarding same-turn transfer."""


def check_interception(
    *,
    node_path: str,
    node: BaseNode,
    recovered: _ChildScanState | None = None,
    current_run: DynamicNodeRun | None = None,
    curr_parent_ctx: Context,
) -> InterceptionResult:
  """Determine if a node execution should be intercepted based on history."""
  from .._workflow import Workflow  # pylint: disable=g-import-not-at-top

  # Case 1: Same-turn completed or waiting interception (dynamic nodes only).
  # If a node already successfully executed or is currently blocked in the
  # current turn, bypass execution and return its current turn results.
  if current_run:
    if current_run.state.status == NodeStatus.COMPLETED:
      return InterceptionResult(
          should_run=False,
          output=current_run.output,
          transfer_to_agent=current_run.transfer_to_agent,
      )
    if current_run.state.status == NodeStatus.WAITING:
      if current_run.state.interrupts:
        return InterceptionResult(
            should_run=False,
            interrupts=set(current_run.state.interrupts),
        )

  # Intercept executions based on historical session events (cross-turn replay).
  if not recovered:
    return InterceptionResult(should_run=True)

  unresolved = recovered.interrupt_ids - recovered.resolved_ids

  should_run = False
  output = None
  route = None
  interrupts = set()
  resume_inputs = None

  if unresolved:
    # Case 2: Cross-turn unresolved interrupts remain.
    # Rerun natively with resolved inputs if the node supports rerun and some
    # progress was made; otherwise remain waiting and bubble unresolved interrupts.
    if node.rerun_on_resume and recovered.resolved_ids:
      should_run = True
      resume_inputs = recovered.resolved_responses
    else:
      interrupts = unresolved

  elif (
      recovered.route is not None
      or recovered.output is not None
      or recovered.transfer_to_agent is not None
  ):
    # Case 3: Cross-turn successfully completed in a prior turn (fast-forward).
    # Bypass execution completely and return the cached output and route.
    output = _process_rehydrated_output(node, recovered.output)
    route = recovered.route

  elif recovered.interrupt_ids:
    # Case 4: Cross-turn all prior interrupts are resolved, but no output yet.
    # Extract responses directly if the node does not support rerun; otherwise
    # rerun natively with resolved responses to produce output.
    if not node.rerun_on_resume:
      child_resume_inputs = recovered.resolved_responses
      if len(child_resume_inputs) == 1:
        output = list(child_resume_inputs.values())[0]
      else:
        output = dict(child_resume_inputs)
    else:
      should_run = True
      resume_inputs = recovered.resolved_responses

  else:
    # Case 5: Cross-turn no events, or events contain no output, route, or interrupts.
    # Rerun Workflow nodes to guide nested children; otherwise fall through.
    if (
        isinstance(node, Workflow)
        and node.wait_for_output
        and recovered.output is None
    ):
      should_run = True
      resume_inputs = recovered.resolved_responses
    else:
      # Allow fresh execution for crashed/timeout dynamic nodes;
      # static nodes with no outcome (e.g. return None) should be fast-forwarded.
      if current_run is not None:
        should_run = True
      else:
        should_run = False

  return InterceptionResult(
      should_run=should_run,
      output=output,
      route=route,
      interrupts=interrupts,
      resume_inputs=resume_inputs,
      transfer_to_agent=recovered.transfer_to_agent if recovered else None,
  )


def create_mock_context(
    *,
    parent_ctx: Context,
    node: BaseNode,
    run_id: str,
    result: InterceptionResult,
    ancestors: list[str],
    node_path: str | None = None,
    branch: str | None = None,
) -> Context:
  """Build a Context with cached results (no execution)."""
  ic = parent_ctx._invocation_context  # pylint: disable=protected-access
  if branch:
    ic = ic.model_copy(update={"branch": branch})

  mock_ctx = Context(
      ic,
      parent_ctx=parent_ctx,
      node=node,
      run_id=run_id,
      node_path=node_path,
  )
  mock_ctx._output_for_ancestors = ancestors  # pylint: disable=protected-access

  if result.output is not None:
    mock_ctx._output_value = result.output  # pylint: disable=protected-access
    mock_ctx._output_emitted = True  # pylint: disable=protected-access

  if result.transfer_to_agent is not None:
    mock_ctx.actions.transfer_to_agent = result.transfer_to_agent

  mock_ctx.route = result.route
  mock_ctx._interrupt_ids = result.interrupts  # pylint: disable=protected-access

  return mock_ctx
