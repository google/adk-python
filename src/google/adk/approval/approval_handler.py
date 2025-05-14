# Copyright 2025 Google LLC
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

"""Approval handler for managing tool approvals."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Union

from typing import Literal, Optional

from google.genai.types import FunctionCall
from pydantic import BaseModel

from .approval_grant import ApprovalAction, ApprovalActor, ApprovalEffect, ApprovalGrant
from .approval_policy import ApprovalPolicyRegistry, TOOL_NAMESPACE
from .approval_request import ApprovalChallenge, ApprovalDenied, ApprovalRequest, ApprovalResponse
from ..tools import ToolContext

if TYPE_CHECKING:
  from ..sessions.state import State


class ApprovalHandler(object):
  """Handles approval requests and responses."""

  @classmethod
  def parse_and_store_approval_responses(
      cls, initial_grants: list[ApprovalGrant], approval_responses: list[ApprovalResponse]
  ) -> list[ApprovalGrant]:
    extra_grants = []
    for approval_response in approval_responses:
      added_grants = [
        grant for grant in approval_response.grants
        if grant not in initial_grants and grant not in extra_grants
      ]
      extra_grants = extra_grants + added_grants

    return extra_grants

  @classmethod
  def get_approval_request(
      cls,
      function_call: FunctionCall,
      state: Dict[str, Any],
      tool_context: ToolContext,
      user_id: str,
      session_id: str,
  ) -> Optional[Dict[str, Any]]:
    try:
      if approval_request := cls._get_pending_challenges(
          state=state, tool_call=function_call, user_id=user_id, session_id=session_id,
      ):
        # Create state delta to approve function calls - the actions will be merged
        if "approvals" not in tool_context.state:
          tool_context.state['approvals'] = {}
        if "suspended_function_calls" not in tool_context.state["approvals"]:
          tool_context.state["approvals"]["suspended_function_calls"] = []

        tool_context.actions.state_delta["approvals"] = {
          "suspended_function_calls": [
            *tool_context.state["approvals"]["suspended_function_calls"],
            function_call.model_dump(mode="json")
          ]
        }
        tool_context.state["approvals"]["suspended_function_calls"] = tool_context.actions.state_delta["approvals"]["suspended_function_calls"]

        tool_context.request_approval(approval_request)
        return {
            "status": "approval_requested",
            "pending_challenges": approval_request.challenges,
        }
      else:
        return None
    except ApprovalDenied as e:
      return {
          "status": "denied",
          "denied_challenges": e.denied_challenges,
      }

  @classmethod
  def _get_pending_challenges(
      cls,
      state: Union[State, Dict[str, Any]],
      tool_call: FunctionCall,
      user_id: str,
      session_id: str,
  ) -> Optional[ApprovalRequest]:
    """Check if an approval exists for the given tool call.

    Args:
        state: The session state.
        tool_name: The name of the tool.
        tool_args: The arguments being passed to the tool.
        function_call_id: The ID of the function call.

    Raises:
      ApprovalDenied: if the an approval with effect Deny matches.
    """

    policies = ApprovalPolicyRegistry.get_tool_policies(tool_call.name)
    grants = cls._get_existing_grants(state)

    user_actor = ApprovalActor(id=user_id, type="user")
    agent_actor = ApprovalActor(
        id=session_id, type="agent", on_behalf_of=user_actor
    )
    function_call_actor = ApprovalActor(
        id=f"{TOOL_NAMESPACE}:{tool_call.name}:{tool_call.id}", type="tool", on_behalf_of=agent_actor
    )

    approved_policy_pairs = []
    denied_policy_pairs = []
    challenges = []

    for policy in policies:
      unmet_policy_pairs = []
      for policy_action, policy_resource in policy.get_action_resources(
          tool_call.args
      ):
        # Check against function call grantee, then check against the agent and whether delegation is possible
        if effect := cls._check_action_on_resource_against_grants(
            policy_action, policy_resource, grants, function_call_actor
        ):
          if effect == ApprovalEffect.deny:
            denied_policy_pairs.append((policy_action, policy_resource))
          elif effect == ApprovalEffect.allow:
            approved_policy_pairs.append((policy_action, policy_resource))
          continue
        unmet_policy_pairs.append((policy_action, policy_resource))

      grouped_challenges = {}

      for action, resource in unmet_policy_pairs:
        if resource not in grouped_challenges:
          grouped_challenges[resource] = [action]
        else:
          grouped_challenges[resource].append(action)
      for resource, actions in grouped_challenges.items():
        challenges.append(
            ApprovalChallenge(
                actions=actions,
                resources=[resource],
            )
        )

    if denied_policy_pairs:
      raise ApprovalDenied(
          denied_challenges=[
              ApprovalChallenge(
                  grantee=function_call_actor,
                  actions=[action],
                  resources=[resource],
              )
              for action, resource in denied_policy_pairs
          ],
      )
    if challenges:
      return ApprovalRequest(
        function_call=tool_call,
        challenges=challenges,
        grantee=function_call_actor,
      )

  @staticmethod
  def _get_approvals_state(state: State) -> dict[str, Any]:
    if "approvals" not in state:
      state["approvals"] = {
          "grants": [],
          "suspended_function_calls": [],
      }
    if "grants" not in state["approvals"]:
      state["approvals"]["grants"] = []
    if "suspended_function_calls" not in state["approvals"]:
      state["approvals"]["suspended_function_calls"] = []
    return state["approvals"]

  @classmethod
  def _get_existing_grants(cls, state: State) -> list[ApprovalGrant]:
    return [ApprovalGrant.model_validate(grant) for grant in cls._get_approvals_state(state)["grants"]]

  @staticmethod
  def _resource_met(policy_resource, grant_resource):
    """Check if a policy resource matches a grant resource.

    Grant resources can contain wildcards (*) which match any value
    in that position.

    Args:
        policy_resource: The resource being accessed
        grant_resource: The resource in the grant, may contain wildcards

    Returns:
        bool: True if the policy resource matches the grant resource pattern
    """
    # Full wildcard matches anything
    if grant_resource == "*":
      return True

    # Split resources into segments
    policy_segments = policy_resource.split(":")
    grant_segments = grant_resource.split(":")

    # If we have different segment counts and not covered by the special case above
    if len(policy_segments) != len(grant_segments):
      return False

    # Compare each segment
    for policy_segment, grant_segment in zip(policy_segments, grant_segments):
      if grant_segment == "*":
        continue

      if "*" in grant_segment:
        prefix, suffix = grant_segment.split("*", maxsplit=1)
        if not policy_segment.startswith(prefix):
          return False
        if not policy_segment.endswith(suffix):
          return False
        continue

      if policy_segment != grant_segment:
        return False

    return True

  @classmethod
  def _check_actor(cls, actor, grantee) -> bool:
    # Check if the grantee IDs match
    if not cls._check_actor_id(actor_id=actor.id, grantee_id=grantee.id):
      return False

    # Check if the grantee types match
    if actor.type != grantee.type:
      return False

    # If on_behalf_of is specified in the grant, check that too
    if grantee.on_behalf_of is not None:
      if actor.on_behalf_of is None:
        return False

      return cls._check_actor(actor.on_behalf_of, grantee.on_behalf_of)

    return True

  @classmethod
  def _check_actor_id(cls, actor_id, grantee_id) -> bool:
    if actor_id == grantee_id:
      return True

    if grantee_id == "*":
      return True

    actor_segments = actor_id.split(":")
    grantee_segments = grantee_id.split(":")

    if len(actor_segments) != len(grantee_segments):
      return False

    # Compare each segment
    for actor_segment, grantee_segment in zip(actor_segments, grantee_segments):
      if grantee_segment == "*":
        continue

      if actor_segment == grantee_segment:
        continue

      if "*" in grantee_segment:
        prefix, suffix = grantee_segment.split("*", maxsplit=1)
        if not actor_segment.startswith(prefix):
          return False
        if not actor_segment.endswith(suffix):
          return False
        continue

      if actor_segment != grantee_segment:
        return False

    return True

  @classmethod
  def _action_granted_for_resource(
      cls,
      grant: ApprovalGrant,
      policy_action: ApprovalAction,
      policy_resource: str,
      actor: ApprovalActor,
  ) -> Optional[Literal[ApprovalEffect.allow, ApprovalEffect.deny]]:
    """Check if the given action on the resource is granted by this grant.

    Args:
        grant: The approval grant to check against
        policy_action: The action being requested
        policy_resource: The resource being accessed
        actor: The actor requesting the action

    Returns:
        The effect (allow/deny) if granted, None if not applicable
    """
    # Check if the action is in the grant's actions
    if policy_action not in grant.actions:
      return None

    # Check if the grantee matches
    if not cls._check_actor(actor, grant.grantee):
      return None

    # Check if any of the resources match
    for grant_resource in grant.resources:
      if cls._resource_met(policy_resource, grant_resource):
        return grant.effect

    return None

  @classmethod
  def _check_action_on_resource_against_grants(
      cls,
      action: ApprovalAction,
      resource: str,
      grants: list[ApprovalGrant],
      actor: ApprovalActor,
  ) -> Optional[Literal[ApprovalEffect.allow, ApprovalEffect.deny]]:
    """Check if the given action on the resource is granted by any of the grants.

    Args:
        action: The action being requested
        resource: The resource being accessed
        grants: The list of grants to check against
        actor: The actor requesting the action

    Returns:
        The effect (allow/deny) if granted, None if not applicable
    """
    # Prioritize deny grants, then check allow grants
    allow_grants = [
        grant for grant in grants if grant.effect == ApprovalEffect.allow
    ]
    deny_grants = [
        grant for grant in grants if grant.effect == ApprovalEffect.deny
    ]

    # Check deny grants first (they take precedence)
    for grant in deny_grants:
      if effect := cls._action_granted_for_resource(
          grant, action, resource, actor
      ):
        return effect

    # Then check allow grants
    for grant in allow_grants:
      if effect := cls._action_granted_for_resource(
          grant, action, resource, actor
      ):
        return effect

    return None
