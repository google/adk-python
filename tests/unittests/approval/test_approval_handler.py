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
import pytest
from google.genai.types import FunctionCall

from google.adk.approval.approval_grant import ApprovalAction, ApprovalActor, ApprovalEffect, ApprovalGrant
from google.adk.approval.approval_handler import ApprovalHandler
from google.adk.approval.approval_policy import FunctionToolPolicy, register_tool_policy
from google.adk.approval.approval_request import ApprovalDenied, ApprovalResponse
from google.adk.sessions import State


class TestResourceMet:
  """Tests for the _resource_met method."""

  def test_exact_match(self):
    """Test when resources match exactly."""
    assert (
        ApprovalHandler._resource_met("test:resource", "test:resource") is True
    )

  def test_no_match(self):
    """Test when resources don't match at all."""
    assert (
        ApprovalHandler._resource_met("test:resource1", "test:resource2")
        is False
    )

  def test_wildcard_prefix(self):
    """Test wildcard at the beginning of grant resource."""
    assert ApprovalHandler._resource_met("test:resource", "*:resource") is True
    assert ApprovalHandler._resource_met("other:resource", "*:resource") is True
    assert ApprovalHandler._resource_met("test:other", "*:resource") is False

  def test_wildcard_suffix(self):
    """Test wildcard at the end of grant resource."""
    assert ApprovalHandler._resource_met("test:resource", "test:*") is True
    assert ApprovalHandler._resource_met("test:other", "test:*") is True
    assert ApprovalHandler._resource_met("other:resource", "test:*") is False

  def test_wildcard_middle(self):
    """Test wildcard in the middle of grant resource."""
    assert (
        ApprovalHandler._resource_met("test:abc:resource", "test:*:resource")
        is True
    )
    assert (
        ApprovalHandler._resource_met("test:xyz:resource", "test:*:resource")
        is True
    )
    assert (
        ApprovalHandler._resource_met("test:abc:other", "test:*:resource")
        is False
    )

  def test_full_wildcard(self):
    """Test full wildcard grant resource."""
    assert ApprovalHandler._resource_met("test:resource", "*") is True
    assert ApprovalHandler._resource_met("any:thing:at:all", "*") is True

  def test_multiple_wildcards(self):
    """Test grant resource with multiple wildcards."""
    assert (
        ApprovalHandler._resource_met("test:abc:resource", "test:*:*") is True
    )
    assert ApprovalHandler._resource_met("test:abc:xyz", "test:*:*") is True
    assert (
        ApprovalHandler._resource_met("other:abc:resource", "test:*:*") is False
    )

  def test_wildcard_segments(self):
    """Test wildcard matching entire segments."""
    # Wildcard matching a single segment
    assert (
        ApprovalHandler._resource_met("test:abc:resource", "test:*:resource")
        is True
    )

    # Wildcard matching multiple segments
    assert (
        ApprovalHandler._resource_met(
            "test:abc:def:resource", "test:*:resource"
        )
        is False
    )
    assert (
        ApprovalHandler._resource_met("test:resource", "test:*:resource")
        is False
    )

  def test_complex_patterns(self):
    """Test more complex wildcard patterns."""
    assert (
        ApprovalHandler._resource_met(
            "tool_call:read_file:/path/to/file", "tool_call:read_file:*"
        )
        is True
    )
    assert (
        ApprovalHandler._resource_met(
            "tool_call:read_file:/path/to/file", "tool_call:*:/path/to/*"
        )
        is True
    )
    assert (
        ApprovalHandler._resource_met(
            "tool_call:write_file:/path/to/file", "tool_call:read_*:*"
        )
        is False
    )


@pytest.fixture
def sample_action():
  """Create a sample approval action for testing."""
  return ApprovalAction("test:read")


@pytest.fixture
def different_action():
  """Create a different approval action for testing."""
  return ApprovalAction("test:write")


@pytest.fixture
def sample_actor():
  """Create a sample approval actor for testing."""
  return ApprovalActor(id="test_user", type="user")


@pytest.fixture
def allow_grant(sample_action, sample_actor):
  """Create an allow grant for testing."""
  return ApprovalGrant(
      effect=ApprovalEffect.allow,
      actions=[sample_action],
      resources=["test:resource:123", "test:resource:*"],
      grantee=sample_actor,
      grantor=sample_actor,
  )


@pytest.fixture
def deny_grant(sample_action, sample_actor):
  """Create a deny grant for testing."""
  return ApprovalGrant(
      effect=ApprovalEffect.deny,
      actions=[sample_action],
      resources=["test:resource:456", "test:special:*"],
      grantee=sample_actor,
      grantor=sample_actor,
  )


class TestActionGrantedForResource:
  """Tests for the _action_granted_for_resource method."""

  def test_action_not_in_grant(
      self, allow_grant, different_action, sample_actor
  ):
    """Test when the requested action is not in the grant's actions."""
    result = ApprovalHandler._action_granted_for_resource(
        allow_grant,
        different_action,
        "test:resource:123",
        sample_actor,
    )
    assert result is None

  def test_resource_not_in_grant(
      self, allow_grant, sample_action, sample_actor
  ):
    """Test when the requested resource is not in the grant's resources."""
    result = ApprovalHandler._action_granted_for_resource(
        allow_grant,
        sample_action,
        "test:different:789",
        sample_actor,
    )
    assert result is None

  def test_exact_resource_match_allow(
      self, allow_grant, sample_action, sample_actor
  ):
    """Test when there's an exact match with an allow grant."""
    result = ApprovalHandler._action_granted_for_resource(
        allow_grant, sample_action, "test:resource:123", sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_exact_resource_match_deny(
      self, deny_grant, sample_action, sample_actor
  ):
    """Test when there's an exact match with a deny grant."""
    result = ApprovalHandler._action_granted_for_resource(
        deny_grant, sample_action, "test:resource:456", sample_actor
    )
    assert result == ApprovalEffect.deny

  def test_wildcard_resource_match_allow(
      self, allow_grant, sample_action, sample_actor
  ):
    """Test when there's a wildcard match with an allow grant."""
    result = ApprovalHandler._action_granted_for_resource(
        allow_grant, sample_action, "test:resource:anything", sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_wildcard_resource_match_deny(
      self, deny_grant, sample_action, sample_actor
  ):
    """Test when there's a wildcard match with a deny grant."""
    result = ApprovalHandler._action_granted_for_resource(
        deny_grant, sample_action, "test:special:anything", sample_actor
    )
    assert result == ApprovalEffect.deny

  def test_first_resource_match_is_returned(self, sample_action, sample_actor):
    """Test that the first matching resource's effect is returned."""
    # Create a grant with multiple matching resources
    grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:first", "test:resource:second"],
        grantee=sample_actor,
        grantor=sample_actor,
    )

    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:first", sample_actor
    )
    assert result == ApprovalEffect.allow

    # The second resource should also match if checked first
    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:second", sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_grantee_match(self, sample_action, sample_actor):
    """Test when the grantee matches the grant's grantee."""
    # Create a grant for a specific grantee
    grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )

    # Use the same grantee in the check
    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:123", actor=sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_grantee_mismatch(self, sample_action, sample_actor):
    """Test when the grantee doesn't match the grant's grantee."""
    # Create a grant for a specific grantee
    grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )

    # Use a different grantee in the check
    different_actor = ApprovalActor(id="different_user", type="user")
    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:123", actor=different_actor
    )
    assert result is None

  def test_grantee_with_on_behalf_of(self, sample_action, sample_actor):
    """Test when the grant's grantee has an on_behalf_of field."""
    # Create an actor acting on behalf of another actor
    original_actor = ApprovalActor(id="original_user", type="user")
    delegated_actor = ApprovalActor(
        id="delegate_user", type="agent", on_behalf_of=original_actor
    )

    # Create a grant for a delegated actor
    grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=delegated_actor,
        grantor=original_actor,
    )

    # Use the same delegated actor in the check
    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:123", actor=delegated_actor
    )
    assert result == ApprovalEffect.allow

    # Use a different delegated actor in the check
    different_delegated = ApprovalActor(
        id="delegate_user",
        type="agent",
        on_behalf_of=ApprovalActor(id="different_user", type="user"),
    )
    result = ApprovalHandler._action_granted_for_resource(
        grant, sample_action, "test:resource:123", actor=different_delegated
    )
    assert result is None


class TestCheckActionOnResourceAgainstGrants:
  # Tests for _check_action_on_resource_against_grants
  def test_no_grants(self, sample_action, sample_actor):
    """Test when there are no grants available."""
    grants = []
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action,
        "test:resource:123",
        grants,
        sample_actor,
    )
    assert result is None

  def test_allow_grant_match(self, sample_action, sample_actor):
    """Test when an allow grant matches the action and resource."""
    allow_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [allow_grant]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action,
        "test:resource:123",
        grants,
        sample_actor,
    )
    assert result == ApprovalEffect.allow

  def test_deny_grant_match(self, sample_action, sample_actor):
    """Test when a deny grant matches the action and resource."""
    deny_grant = ApprovalGrant(
        effect=ApprovalEffect.deny,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [deny_grant]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, sample_actor
    )
    assert result == ApprovalEffect.deny

  def test_deny_takes_precedence(self, sample_action, sample_actor):
    """Test that deny grants take precedence over allow grants."""
    allow_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    deny_grant = ApprovalGrant(
        effect=ApprovalEffect.deny,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    # Note that deny_grant is after allow_grant in the list
    grants = [allow_grant, deny_grant]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, sample_actor
    )
    assert result == ApprovalEffect.deny

  def test_first_match_returned(
      self, sample_action, different_action, sample_actor
  ):
    """Test that the first matching grant's effect is returned."""
    allow_grant1 = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    allow_grant2 = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:456"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [allow_grant1, allow_grant2]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_no_match(self, sample_action, different_action, sample_actor):
    """Test when no grants match the action and resource."""
    allow_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[different_action],  # Different action
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [allow_grant]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, sample_actor
    )
    assert result is None

  def test_wildcard_resource(self, sample_action, sample_actor):
    """Test with a wildcard resource in a grant."""
    allow_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:*"],  # Wildcard resource
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [allow_grant]
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, sample_actor
    )
    assert result == ApprovalEffect.allow

  def test_with_grantee(self, sample_action, sample_actor):
    """Test that the grantee is correctly passed to _action_granted_for_resource."""
    # Create a grant for a specific grantee
    grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=sample_actor,
        grantor=sample_actor,
    )
    grants = [grant]

    # Test with the matching grantee
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, actor=sample_actor
    )
    assert result == ApprovalEffect.allow

    # Test with a different grantee
    different_actor = ApprovalActor(id="different_user", type="user")
    result = ApprovalHandler._check_action_on_resource_against_grants(
        sample_action, "test:resource:123", grants, actor=different_actor
    )
    assert result is None


def create_tool_policy(tool_name, actions, static_resources):
  return FunctionToolPolicy(
      policy_name=FunctionToolPolicy.format_name(tool_name=tool_name),
      actions=actions,
      resource_mappers=lambda args: static_resources,
  )


class TestGetPendingChallenges:
  """Tests for the get_pending_challenges method."""

  @pytest.fixture
  def mock_state(self):
    """Create a mock state for testing."""
    return State(value={}, delta={})

  @pytest.fixture
  def sample_action(self):
    """Create a sample approval action for testing."""
    return ApprovalAction("test:read")

  @pytest.fixture
  def sample_actor(self):
    """Create a sample approval actor for testing."""
    return ApprovalActor(id="test_user", type="user")

  @pytest.fixture
  def function_call_id(self):
    """Create a function call ID for testing."""
    return "test_function_id"

  @pytest.fixture
  def allow_grant(self, sample_action, sample_actor):
    """Create an allow grant for testing."""
    function_call_actor = ApprovalActor(
        id="tool:test_tool:test_function_id",
        type="tool",
        on_behalf_of=ApprovalActor(
            id="test_session", type="agent", on_behalf_of=sample_actor
        ),
    )

    return ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:123"],
        grantee=function_call_actor,
        grantor=sample_actor,
    )

  @pytest.fixture
  def deny_grant(self, sample_action, sample_actor):
    """Create a deny grant for testing."""
    function_call_actor = ApprovalActor(
        id="tool:test_tool:test_function_id",
        type="tool",
        on_behalf_of=ApprovalActor(
            id="test_session", type="agent", on_behalf_of=sample_actor
        ),
    )

    return ApprovalGrant(
        effect=ApprovalEffect.deny,
        actions=[sample_action],
        resources=["test:resource:456"],
        grantee=function_call_actor,
        grantor=sample_actor,
    )

  def test_allow_when_no_policies(self, mock_state, function_call_id):
    """Test that approval is granted when there are no policies."""
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert not approval_request

  def test_challenge_when_no_grants(
      self, mock_state, sample_action, function_call_id
  ):
    """Test that a challenge is created when there are no grants."""
    # Create a mock policy with one action/resource pair
    policy = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:123"]
    )
    register_tool_policy(policy)

    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert len(approval_request.challenges) == 1
    assert sample_action in approval_request.challenges[0].actions
    assert "test:resource:123" in approval_request.challenges[0].resources

  def test_allow_when_all_actions_granted(
      self, sample_action, allow_grant, function_call_id, mock_state
  ):
    """Test that approval is granted when all actions are allowed."""
    # Add grant to state
    mock_state["approvals__grants"] = [allow_grant]

    # Create a mock policy with one action/resource pair that matches the grant
    policy = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:123"]
    )
    register_tool_policy(policy)
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert not approval_request

  def test_deny_when_any_action_denied(
      self,
      sample_action,
      mock_state,
      allow_grant,
      deny_grant,
      function_call_id,
  ):
    """Test that approval is denied when any action is denied."""
    # Add grants to state
    mock_state["approvals__grants"] = [allow_grant, deny_grant]

    # Create a mock policy with two action/resource pairs, one allowed and one denied
    policy = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:123", "test:resource:456"]
    )
    register_tool_policy(policy)
    with pytest.raises(ApprovalDenied) as exc_info:
      ApprovalHandler._get_pending_challenges(
          mock_state,
          FunctionCall(
              name="test_tool",
              args={"param": "value"},
              id=function_call_id,
          ),
          user_id="test_user",
          session_id="test_session",
      )
    assert exc_info.value.denied_challenges

  def test_challenge_when_some_actions_not_granted(
      self,
      sample_action,
      mock_state,
      allow_grant,
      function_call_id,
  ):
    """Test that a challenge is created when some actions are not granted."""
    # Add grant to state
    mock_state["approvals__grants"] = [allow_grant]

    # Create a different action that is not granted
    different_action = ApprovalAction("test:write")

    # Create a mock policy with two action/resource pairs, one allowed and one not
    policy = create_tool_policy(
        "test_tool", [sample_action, different_action], ["test:resource:123"]
    )
    register_tool_policy(policy)
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert len(approval_request.challenges) == 1
    assert different_action in approval_request.challenges[0].actions

  def test_multiple_policies(
      self,
      mock_state,
      sample_action,
      allow_grant,
      function_call_id,
  ):
    """Test handling multiple policies."""
    # Add grant to state
    mock_state["approvals__grants"] = [allow_grant]

    # Create a different action that is not granted
    different_action = ApprovalAction("test:write")

    # Create two mock policies with different action/resource pairs
    policy1 = create_tool_policy(
        "test_tool", [sample_action, different_action], ["test:resource:123"]
    )
    policy2 = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:789"]
    )
    register_tool_policy(policy1)
    register_tool_policy(policy2)
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert len(approval_request.challenges) == 2
    assert approval_request.challenges[0].actions == [different_action]
    assert approval_request.challenges[0].resources == ["test:resource:123"]
    assert approval_request.challenges[1].actions == [sample_action]
    assert approval_request.challenges[1].resources == ["test:resource:789"]

  def test_with_wildcard_resources(
      self, mock_state, sample_action, function_call_id
  ):
    """Test with grants that use wildcard resources."""
    # Create a wildcard grant
    function_call_actor = ApprovalActor(
        id="tool:test_tool:test_function_id",
        type="tool",
        on_behalf_of=ApprovalActor(
            id="test_session",
            type="agent",
            on_behalf_of=ApprovalActor(id="test_user", type="user"),
        ),
    )

    wildcard_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:*"],  # Wildcard resource
        grantee=function_call_actor,
        grantor=ApprovalActor(id="test_user", type="user"),
    )

    # Add grant to state
    mock_state["approvals__grants"] = [wildcard_grant]

    # Create a mock policy with a specific resource that should match the wildcard
    policy = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:specific"]
    )
    register_tool_policy(policy)
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id=function_call_id,
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert not approval_request

  @pytest.mark.parametrize(
      "function_call_actor_id",
      [
          "tool:test_tool:test_function_id",
          "tool:test_tool:*",
          "tool:*:*",
          "*:*:*",
          "*",
      ],
  )
  def test_with_wildcard_grantee(
      self, mock_state, sample_action, function_call_actor_id
  ):
    """Test with grants that use wildcard resources."""
    # Create a wildcard grant
    function_call_actor = ApprovalActor(
        id=function_call_actor_id,
        type="tool",
        on_behalf_of=ApprovalActor(
            id="test_session",
            type="agent",
            on_behalf_of=ApprovalActor(id="test_user", type="user"),
        ),
    )

    wildcard_grant = ApprovalGrant(
        effect=ApprovalEffect.allow,
        actions=[sample_action],
        resources=["test:resource:*"],  # Wildcard resource
        grantee=function_call_actor,
        grantor=ApprovalActor(id="test_user", type="user"),
    )

    # Add grant to state
    mock_state["approvals__grants"] = [wildcard_grant]

    # Create a mock policy with a specific resource that should match the wildcard
    policy = create_tool_policy(
        "test_tool", [sample_action], ["test:resource:specific"]
    )
    register_tool_policy(policy)
    approval_request = ApprovalHandler._get_pending_challenges(
        mock_state,
        FunctionCall(
            name="test_tool",
            args={"param": "value"},
            id="test_function_id",
        ),
        user_id="test_user",
        session_id="test_session",
    )

    assert not approval_request
