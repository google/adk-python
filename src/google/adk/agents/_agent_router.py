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

"""Private helper module for multi-agent routing and branch path recovery."""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from ..events._branch_path import _BranchPath
from ..events._node_path_builder import _NodePathBuilder
from ..events.event import Event
from ..flows.llm_flows.agent_transfer import _get_transfer_targets
from ..flows.llm_flows.functions import _collect_function_call_ids
from ..flows.llm_flows.functions import find_matching_function_call

if TYPE_CHECKING:
  from ..agents.base_agent import BaseAgent
  from ..agents.invocation_context import InvocationContext
  from ..apps.app import ResumabilityConfig
  from ..sessions.session import Session
  from ..workflow._base_node import BaseNode

logger = logging.getLogger("google_adk." + __name__)


def can_transfer_between_agents(root: Any) -> bool:
  """Reports whether any agent in the tree can transfer to another agent."""
  pending = [root]
  while pending:
    agent = pending.pop()
    sub_agents = getattr(agent, "sub_agents", None)
    if not isinstance(sub_agents, list):
      continue
    if hasattr(agent, "disallow_transfer_to_parent") and _get_transfer_targets(
        agent
    ):
      return True
    pending.extend(sub_agents)
  return False


def is_transferable_across_agent_tree(agent_to_run: BaseAgent) -> bool:
  """Whether the agent to run can transfer to any other agent in the agent tree.

  This typically means all agent_to_run's ancestor can transfer to their
  parent_agent all the way to the root_agent.

  Args:
      agent_to_run: The agent to check for transferability.

  Returns:
      True if the agent can transfer, False otherwise.
  """
  agent: BaseAgent | None = agent_to_run
  while agent:
    if not hasattr(agent, "disallow_transfer_to_parent"):
      # Only agents with transfer capability can transfer.
      return False
    if agent.disallow_transfer_to_parent:
      return False
    agent = agent.parent_agent
  return True


def find_agent_to_run(
    session: Session,
    root_agent: BaseAgent,
    resumability_config: Optional[ResumabilityConfig] = None,
) -> BaseAgent:
  """Finds the agent to run to continue the session.

  A qualified agent must be either of:

  - The agent that returned a function call and the last user message is a
    function response to this function call.
  - The root agent.
  - An LlmAgent who replied last and is capable to transfer to any other agent
    in the agent hierarchy.

  TODO: use wait_for_output to decide the agent to run

  Args:
      session: The session to find the agent for.
      root_agent: The root agent of the runner.
      resumability_config: Optional resumability configuration.

  Returns:
    The agent to run. (the active agent that should reply to the latest user
    message)
  """
  # Mesh and Workflow Agents handle their own internal routing.
  # Workflow will figure which node is interrupted and should be resumed.
  from ..workflow._workflow import Workflow

  if isinstance(root_agent, Workflow):
    return root_agent

  # If the last event is a function response, should send this response to
  # the agent that returned the corresponding function call regardless the
  # type of the agent. e.g. a remote a2a agent may surface a credential
  # request as a special long-running function tool call.
  event = find_matching_function_call(session.events)
  is_resumable = resumability_config and resumability_config.is_resumable
  # Only route based on a past function response if resumability is enabled.
  # In non-resumable scenarios, a turn ending with function call response
  # shouldn't trap the next turn on that same agent if it's not transferable.
  # Falling through allows it to return to root.
  if event and event.author and is_resumable:
    # `find_agent` returns None when the author does not correspond to any
    # agent in the current hierarchy (e.g. the author is "user" or a stale or
    # foreign agent name carried over from a previous turn/session). Returning
    # None here would propagate to `build_node`, raising a confusing
    # "Invalid node type: <class 'NoneType'>" error. Fall through to the
    # event-scan logic below (which ultimately falls back to the root agent)
    # whenever the author cannot be resolved.
    if (resumed_agent := root_agent.find_agent(event.author)) is not None:
      return resumed_agent

  def _event_filter(event: Event) -> bool:
    """Filters out user-authored events and agent state change events."""
    if event.author == "user":
      return False
    if event.actions.agent_state is not None or event.actions.end_of_agent:
      return False
    return True

  for event in filter(_event_filter, reversed(session.events)):
    if event.author == root_agent.name:
      # Found root agent.
      return root_agent
    if not (agent := root_agent.find_sub_agent(event.author)):
      # Agent not found, continue looking.
      logger.warning(
          "Event from an unknown agent: %s, event id: %s",
          event.author,
          event.id,
      )
      continue
    transferable = is_transferable_across_agent_tree(agent)
    if transferable:
      return agent
  # Falls back to root agent if no suitable agents are found in the session.
  return root_agent


def restore_branch_from_history(
    invocation_context: InvocationContext,
    node: BaseNode,
    *,
    root: BaseNode,
    invocation_id: Optional[str] = None,
) -> None:
  """Restores a non-root node's branch from its latest matching event.

  A freshly created ``InvocationContext`` has no branch, so a node that
  previously ran on a sub-branch (e.g. a resumed sub-agent, or an agent
  resolved by ``_find_agent_to_run``) would otherwise continue on the root
  branch.

  Tool branches are skipped. A tool's user-facing message is authored under
  the agent's name (``functions.py``) and stamped with the agent's node path
  (``base_agent.py``), so it is indistinguishable from the agent's own turns
  by author or path; what gives it away is its branch, which the same code
  builds as ``<tool>@<function_call_id>`` from a call in this session. Such a
  branch is therefore recognised and skipped, while every other event the node
  authored -- including a plain text turn, which is all a non-resumable
  sub-agent may leave behind -- still carries ``ctx.branch`` and is eligible.
  A branch whose leaf names the node itself is kept even when its id is a
  function call id, since that is how ``AgentTool`` scopes a real sub-agent.

  Nodes are matched by their static path (run ids stripped) so that two nodes
  sharing a name (e.g. the same sub-agent mounted under two parents) are
  disambiguated; events that predate node paths fall back to author/name
  matching. When ``invocation_id`` is provided (resuming a known invocation),
  only that invocation's events are considered, so a resumed node cannot
  inherit a stale branch authored in an earlier invocation. When it is None
  (a fresh direct-node turn, or a new invocation continuing a sub-agent), the
  most recent matching event across the session is used.
  """
  from ..workflow._base_node import find_static_node_path

  expected_static_path = find_static_node_path(root, node)
  tool_call_ids = _collect_function_call_ids(invocation_context.session.events)
  for event in reversed(invocation_context.session.events):
    if invocation_id is not None and event.invocation_id != invocation_id:
      continue
    if not event.branch:
      continue
    if _BranchPath.is_tool_branch(event.branch, node.name, tool_call_ids):
      continue
    matched = False
    if expected_static_path and event.node_info.path:
      event_static_path = _NodePathBuilder.from_string(
          event.node_info.path
      ).static_path
      if event_static_path == expected_static_path:
        matched = True
    elif event.author == node.name or event.node_info.name == node.name:
      matched = True
    if matched:
      invocation_context.branch = event.branch
      break
