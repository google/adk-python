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

"""Handles the logic for managing and evaluating approval requests against policies and grants.

This module provides the `ApprovalHandler` class, which is responsible for:
- Parsing and storing approval responses.
- Determining if a given function call requires approval based on registered policies and existing grants.
- Generating approval requests (challenges) when necessary.
- Checking if actions on resources are permitted by existing grants.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING, Union

from typing import Literal, Optional

from google.genai.types import FunctionCall
from pydantic import BaseModel

from .approval_grant import ApprovalAction, ApprovalActor, ApprovalEffect, ApprovalGrant
from .approval_policy import ApprovalPolicyRegistry, TOOL_NAMESPACE
from .approval_request import ApprovalChallenge, ApprovalDenied, ApprovalRequest, ApprovalResponse, FunctionCallStatus
from ..tools import ToolContext

if TYPE_CHECKING:
  from ..sessions.state import State


class ApprovalHandler(object):
  """Manages the lifecycle of approval requests, from checking policies to storing grants.

  This class provides static methods to interact with the approval system. It does not
  maintain its own state but operates on the state provided to its methods (typically
  session state).
  """

  @classmethod
  def parse_and_store_approval_responses(
      cls,
      initial_grants: list[ApprovalGrant],
      approval_responses: list[ApprovalResponse],
  ) -> list[ApprovalGrant]:
    """Parses approval responses and extracts new grants.

    Compares grants from approval responses with existing initial grants to identify
    and return only the newly added grants.

    Args:
        initial_grants: A list of already existing approval grants.
        approval_responses: A list of approval responses, each potentially containing grants.

    Returns:
        A list of new `ApprovalGrant` objects that were not present in `initial_grants`.
    """
    extra_grants = []
    for approval_response in approval_responses:
      added_grants = [
          grant
          for grant in approval_response.grants
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
    """Determines if a function call requires approval and generates a request if so.

    This method checks the given `function_call` against registered approval policies
    and existing grants in the `state`. If approval is required (i.e., there are
    pending challenges), it updates the `tool_context` to suspend the function call
    and request approval.

    If the function call is already approved or doesn't require approval, it updates
    the `tool_context` to mark the function call as resumed.

    If a deny grant explicitly forbids the call, an `ApprovalDenied` exception is caught,
    and the function call is marked as cancelled.

    Args:
        function_call: The `FunctionCall` object to be checked.
        state: The current session state, containing existing grants and suspended calls.
        tool_context: The context for the current tool call, used to update its status
                      and request approval.
        user_id: The ID of the user initiating the call.
        session_id: The ID of the current session.

    Returns:
        A dictionary with a "status" key if approval is requested or denied:
        - {"status": "approval_requested"} if challenges are pending.
        - {"status": "denied", "denied_challenges": ...} if the call is denied.
        Returns `None` if the function call can proceed without further approval.
    """
    try:
      if approval_request := cls._get_pending_challenges(
          state=state,
          tool_call=function_call,
          user_id=user_id,
          session_id=session_id,
      ):
        tool_context.state["approvals__suspended_function_calls"] = (
            FunctionCallStatus.update_status(
                tool_context.state.get(
                    "approvals__suspended_function_calls", []
                ),
                function_call,
                "suspended",
            )
        )

        tool_context.request_approval(approval_request)
        return {
            "status": "approval_requested",
        }
      else:
        tool_context.state["approvals__suspended_function_calls"] = (
            FunctionCallStatus.update_status(
                tool_context.state.get(
                    "approvals__suspended_function_calls", []
                ),
                function_call,
                "resumed",
            )
        )
        return None
    except ApprovalDenied as e:
      tool_context.state["approvals__suspended_function_calls"] = (
          FunctionCallStatus.update_status(
              tool_context.state.get("approvals__suspended_function_calls", []),
              function_call,
              "cancelled",
          )
      )

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
    """Checks a tool call against policies and grants to identify pending challenges.

    This method evaluates the `tool_call` against all registered `ApprovalPolicy`
    objects relevant to the tool. It then checks existing `ApprovalGrant` objects
    in the `state` to see if the required actions on resources are permitted.

    If any required action/resource pair is explicitly denied by a grant, this method
    raises an `ApprovalDenied` exception.

    If there are action/resource pairs required by policies that are not covered by
    any existing allow grants, these are collected into an `ApprovalRequest`.

    Args:
        state: The session state or a dictionary containing at least `approvals__grants`.
        tool_call: The `FunctionCall` to be evaluated.
        user_id: The ID of the user initiating the call.
        session_id: The ID of the current session.

    Returns:
        An `ApprovalRequest` object if there are unmet challenges, otherwise `None`.

    Raises:
        ApprovalDenied: If an existing grant explicitly denies one of the required
                        action/resource pairs.
    """

    policies = ApprovalPolicyRegistry.get_tool_policies(tool_call.name)
    grants = cls._get_existing_grants(state)

    user_actor = ApprovalActor(id=user_id, type="user")
    agent_actor = ApprovalActor(
        id=session_id, type="agent", on_behalf_of=user_actor
    )
    function_call_actor = ApprovalActor(
        id=f"{TOOL_NAMESPACE}:{tool_call.name}:{tool_call.id}",
        type="tool",
        on_behalf_of=agent_actor,
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

  @classmethod
  def _get_existing_grants(cls, state: State) -> list[ApprovalGrant]:
    """Retrieves and validates existing approval grants from the state.

    Args:
        state: The session state, expected to contain an `approvals__grants` key
               with a list of grant dictionaries.

    Returns:
        A list of `ApprovalGrant` objects.
    """
    return [
        ApprovalGrant.model_validate(grant)
        for grant in state.get("approvals__grants", [])
    ]

  @staticmethod
  def _resource_met(policy_resource: str, grant_resource: str) -> bool:
    """Checks if a policy resource string matches a grant resource string, supporting wildcards.

    The `grant_resource` can contain wildcards (`*`) which match any sequence of characters
    within a segment, or an entire segment if the segment itself is `*`.
    Resource strings are colon-separated (e.g., "namespace:type:identifier").

    Examples:
      - `_resource_met("tool:files:read", "tool:files:*")` -> `True`
      - `_resource_met("tool:files:read", "tool:*:read")` -> `True`
      - `_resource_met("tool:files:read", "*")` -> `True`
      - `_resource_met("foo:bar", "foo:baz")` -> `False`

    Args:
        policy_resource: The specific resource being accessed (e.g., "tool:files:/data/my_file.txt").
        grant_resource: The resource pattern from a grant (e.g., "tool:files:*", "*").

    Returns:
        `True` if the `policy_resource` matches the `grant_resource` pattern, `False` otherwise.
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
  def _check_actor(cls, actor: ApprovalActor, grantee: ApprovalActor) -> bool:
    """Recursively checks if an `actor` matches a `grantee` definition, including `on_behalf_of`.

    This method verifies that the `id` and `type` of the `actor` match the `grantee`.
    If `grantee.on_behalf_of` is set, it recursively checks that `actor.on_behalf_of`
    also matches.

    Args:
        actor: The `ApprovalActor` requesting access.
        grantee: The `ApprovalActor` specified in a grant.

    Returns:
        `True` if the actor matches the grantee definition, `False` otherwise.
    """
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
  def _check_actor_id(cls, actor_id: str, grantee_id: str) -> bool:
    """Checks if an actor ID matches a grantee ID, supporting wildcards.

    Similar to `_resource_met`, but for actor IDs. The `grantee_id` can contain
    wildcards (`*`) to match parts of or the entire `actor_id`.

    Args:
        actor_id: The ID of the actor requesting access.
        grantee_id: The ID pattern from a grant.

    Returns:
        `True` if the `actor_id` matches the `grantee_id` pattern, `False` otherwise.
    """
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
    """Checks if a specific action on a resource by an actor is permitted or denied by a single grant.

    This method verifies:
    1. The `policy_action` is listed in `grant.actions`.
    2. The `actor` matches `grant.grantee` (using `_check_actor`).
    3. The `policy_resource` matches one of the `grant.resources` (using `_resource_met`).

    If all conditions are met, it returns the `grant.effect` (allow or deny).

    Args:
        grant: The `ApprovalGrant` to check against.
        policy_action: The action being attempted (e.g., "tool:files:read").
        policy_resource: The resource being accessed (e.g., "tool:files:/data/my_doc.txt").
        actor: The `ApprovalActor` attempting the action.

    Returns:
        `ApprovalEffect.allow` or `ApprovalEffect.deny` if the grant applies and matches,
        otherwise `None`.
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
    """Evaluates an action on a resource against a list of grants to determine its effective status.

    Deny grants are prioritized. If any deny grant matches the action, resource, and actor,
    `ApprovalEffect.deny` is returned immediately.
    If no deny grants match, allow grants are checked. If an allow grant matches,
    `ApprovalEffect.allow` is returned.
    If no grants match, `None` is returned, indicating the action is not explicitly
    allowed or denied by the provided grants.

    Args:
        action: The `ApprovalAction` being attempted.
        resource: The resource string being accessed.
        grants: A list of `ApprovalGrant` objects to check against.
        actor: The `ApprovalActor` attempting the action.

    Returns:
        `ApprovalEffect.allow` if an allow grant matches and no deny grant matches.
        `ApprovalEffect.deny` if a deny grant matches.
        `None` if no grants explicitly cover the action/resource/actor combination.
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
