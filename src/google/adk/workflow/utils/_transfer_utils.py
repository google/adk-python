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

"""Utilities for agent transfer in ADK workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._errors import WorkflowDataError

if TYPE_CHECKING:
  from ...agents.base_agent import BaseAgent
  from ...agents.context import Context


def _find_direct_sub_agent(
    agent: BaseAgent | None, name: str
) -> BaseAgent | None:
  """Returns the direct sub-agent of the given agent with the given name."""
  if agent is None:
    return None
  for sub_agent in agent.sub_agents:
    if sub_agent.name == name:
      return sub_agent
  return None


def resolve_and_derive_transfer_context(
    target_name: str,
    current_agent: BaseAgent,
    root_agent: BaseAgent,
    curr_ctx: Context,
    curr_parent_ctx: Context | None,
) -> tuple[BaseAgent, Context | None] | tuple[None, None]:
  """Resolves the target agent and derives its parent context in a single pass.

  Args:
    target_name: The name of the target agent to transfer to.
    current_agent: The agent initiating the transfer.
    root_agent: The root agent of the application.
    curr_ctx: The current execution context of the current_agent.
    curr_parent_ctx: The parent context of the current_agent.

  Returns:
    A tuple of (target_agent, next_parent_context) or (None, None) if target not
    found. If target is found but cannot be logically routed (unrelated
    transfer),
    returns (target_agent, None).

  Raises:
    WorkflowDataError: If target_agent is the same as current_agent, or if
      current_agent forbids transferring to the target.
  """
  # Each routing case below picks the target out of the current agent's own
  # children, parent or siblings. Searching the whole tree first and then
  # comparing names would let an agent in an unrelated branch that happens to
  # share the name be routed to instead.

  # Case 1: SELF (invalid transfer target)
  if target_name == current_agent.name:
    raise WorkflowDataError(f"Agent '{target_name}' cannot transfer to itself.")

  # Case 2: Direct CHILD (nests deeper under the current context)
  if child_agent := _find_direct_sub_agent(current_agent, target_name):
    return child_agent, curr_ctx

  parent_agent = current_agent.parent_agent

  # Case 3: Direct PARENT (climbs up the context chain to find the parent's parent)
  # Checked before the sibling case so that a parent holding a sub-agent named
  # after itself still resolves as a parent transfer.
  if parent_agent and parent_agent.name == target_name:
    if getattr(current_agent, "disallow_transfer_to_parent", False):
      raise WorkflowDataError(
          f"Cannot transfer from '{current_agent.name}' to parent agent"
          f" '{target_name}': disallow_transfer_to_parent is set."
      )
    # Walk up the context chain to find the target parent agent's context
    curr: Context | None = curr_ctx
    while curr is not None and curr.node is not None:
      if curr.node.name == target_name:
        return parent_agent, curr.parent_ctx
      curr = curr.parent_ctx

    # Root Coordinator / Bypassed parent fallback: returns the outermost root context of this turn
    root_ctx = curr_ctx
    while root_ctx.parent_ctx is not None and root_ctx.node is not None:
      root_ctx = root_ctx.parent_ctx
    return parent_agent, root_ctx

  # Case 4: SIBLING (runs under the same parent context)
  if sibling_agent := _find_direct_sub_agent(parent_agent, target_name):
    if getattr(current_agent, "disallow_transfer_to_peers", False):
      raise WorkflowDataError(
          f"Cannot transfer from '{current_agent.name}' to peer agent"
          f" '{target_name}': disallow_transfer_to_peers is set."
      )
    return sibling_agent, curr_parent_ctx

  # Fallback: the target has no routing relationship with the current agent.
  target_agent = root_agent.find_agent(target_name)
  if not target_agent:
    return None, None
  return target_agent, None
