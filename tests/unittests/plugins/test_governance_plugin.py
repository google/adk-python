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

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import PropertyMock

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.governance_plugin import _hash_dict
from google.adk.plugins.governance_plugin import AuditAction
from google.adk.plugins.governance_plugin import AuditEvent
from google.adk.plugins.governance_plugin import AuditHandler
from google.adk.plugins.governance_plugin import Decision
from google.adk.plugins.governance_plugin import DelegationScope
from google.adk.plugins.governance_plugin import GovernancePlugin
from google.adk.plugins.governance_plugin import LoggingAuditHandler
from google.adk.plugins.governance_plugin import PolicyDecision
from google.adk.plugins.governance_plugin import PolicyEvaluator
from google.adk.plugins.governance_plugin import ToolPolicy
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
import pytest

# -- Helpers ---------------------------------------------------------------


class AllowAllEvaluator:
  """Policy evaluator that allows everything."""

  async def evaluate_tool_call(self, **kwargs) -> PolicyDecision:
    return PolicyDecision.allow(evaluator="allow_all")

  async def evaluate_agent_delegation(self, **kwargs) -> PolicyDecision:
    return PolicyDecision.allow(evaluator="allow_all")


class DenyAllEvaluator:
  """Policy evaluator that denies everything."""

  async def evaluate_tool_call(self, **kwargs) -> PolicyDecision:
    return PolicyDecision.deny(reason="denied by policy", evaluator="deny_all")

  async def evaluate_agent_delegation(self, **kwargs) -> PolicyDecision:
    return PolicyDecision.deny(reason="delegation denied", evaluator="deny_all")


class CollectingAuditHandler:
  """Audit handler that collects events for assertion."""

  def __init__(self):
    self.events: list[AuditEvent] = []

  async def handle(self, event: AuditEvent) -> None:
    self.events.append(event)


def _make_tool_context(
    agent_name="test_agent", invocation_id="inv-1", session_id="sess-1"
):
  """Create a mock ToolContext."""
  mock_session = Mock()
  mock_session.id = session_id

  mock_inv_ctx = Mock(spec=InvocationContext)
  mock_inv_ctx.invocation_id = invocation_id
  mock_inv_ctx.session = mock_session

  ctx = Mock(spec=ToolContext)
  ctx.agent_name = agent_name
  ctx.invocation_id = invocation_id
  ctx._invocation_context = mock_inv_ctx
  ctx.function_call_id = "fc-1"
  return ctx


def _make_invocation_context(
    agent_name="root_agent", invocation_id="inv-1", session_id="sess-1"
):
  """Create a mock InvocationContext."""
  mock_session = Mock()
  mock_session.id = session_id

  mock_agent = Mock(spec=BaseAgent)
  mock_agent.name = agent_name

  ctx = Mock(spec=InvocationContext)
  ctx.invocation_id = invocation_id
  ctx.session = mock_session
  ctx.agent = mock_agent
  return ctx


def _make_callback_context(
    agent_name="test_agent", invocation_id="inv-1", session_id="sess-1"
):
  """Create a mock CallbackContext."""
  mock_session = Mock()
  mock_session.id = session_id

  mock_inv_ctx = Mock(spec=InvocationContext)
  mock_inv_ctx.invocation_id = invocation_id
  mock_inv_ctx.session = mock_session

  ctx = Mock(spec=CallbackContext)
  ctx.agent_name = agent_name
  ctx.invocation_id = invocation_id
  ctx._invocation_context = mock_inv_ctx
  return ctx


# -- PolicyDecision tests -------------------------------------------------


class TestPolicyDecision:

  def test_allow_factory(self):
    d = PolicyDecision.allow(reason="ok")
    assert d.decision == Decision.ALLOW
    assert d.reason == "ok"

  def test_deny_factory(self):
    d = PolicyDecision.deny(reason="blocked", evaluator="test")
    assert d.decision == Decision.DENY
    assert d.reason == "blocked"
    assert d.evaluator == "test"

  def test_timestamp_auto_set(self):
    d = PolicyDecision.allow()
    assert d.timestamp > 0


# -- GovernancePlugin initialization tests ---------------------------------


class TestGovernancePluginInit:

  def test_default_init(self):
    plugin = GovernancePlugin()
    assert plugin.name == "governance_plugin"
    assert plugin._policy_evaluator is None
    assert plugin._tool_policies == {}
    assert plugin._delegation_scopes == {}
    assert plugin._blocked_tools == set()
    assert isinstance(plugin._audit_handler, LoggingAuditHandler)

  def test_custom_init(self):
    evaluator = AllowAllEvaluator()
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        name="custom",
        policy_evaluator=evaluator,
        tool_policies={"t": ToolPolicy(allowed=False)},
        delegation_scopes={"a": DelegationScope()},
        audit_handler=handler,
        blocked_tools={"bad_tool"},
    )
    assert plugin.name == "custom"
    assert plugin._policy_evaluator is evaluator
    assert "t" in plugin._tool_policies
    assert "a" in plugin._delegation_scopes
    assert "bad_tool" in plugin._blocked_tools


# -- Tool governance tests -------------------------------------------------


class TestToolGovernance:

  @pytest.mark.asyncio
  async def test_blocked_tool_denied(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        blocked_tools={"dangerous"}, audit_handler=handler
    )
    tool = Mock(spec=BaseTool)
    tool.name = "dangerous"
    ctx = _make_tool_context()

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={"a": 1}, tool_context=ctx
    )

    assert result is not None
    assert "error" in result
    assert any(e.action == AuditAction.TOOL_CALL_DENIED for e in handler.events)

  @pytest.mark.asyncio
  async def test_allowed_tool_passes(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(audit_handler=handler)

    # Initialize invocation counters
    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "safe_tool"
    ctx = _make_tool_context()

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )

    assert result is None
    assert any(
        e.action == AuditAction.TOOL_CALL_REQUESTED for e in handler.events
    )

  @pytest.mark.asyncio
  async def test_tool_policy_disabled(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        tool_policies={"sql": ToolPolicy(allowed=False)},
        audit_handler=handler,
    )
    tool = Mock(spec=BaseTool)
    tool.name = "sql"
    ctx = _make_tool_context()

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )

    assert result is not None
    assert "error" in result

  @pytest.mark.asyncio
  async def test_call_count_limit_enforced(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        tool_policies={"api": ToolPolicy(max_calls_per_invocation=2)},
        audit_handler=handler,
    )

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "api"
    ctx = _make_tool_context()

    # First two calls should pass
    assert (
        await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )
        is None
    )
    assert (
        await plugin.before_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx
        )
        is None
    )

    # Third call should be denied
    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )
    assert result is not None
    assert "error" in result

  @pytest.mark.asyncio
  async def test_custom_evaluator_deny(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        policy_evaluator=DenyAllEvaluator(), audit_handler=handler
    )

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "any_tool"
    ctx = _make_tool_context()

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={"q": "test"}, tool_context=ctx
    )

    assert result is not None
    assert "denied by policy" in result["reason"]

  @pytest.mark.asyncio
  async def test_custom_evaluator_allow(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        policy_evaluator=AllowAllEvaluator(), audit_handler=handler
    )

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "any_tool"
    ctx = _make_tool_context()

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_delegation_scope_restricts_tools(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        delegation_scopes={
            "restricted_agent": DelegationScope(allowed_tools={"read_only"})
        },
        audit_handler=handler,
    )

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "write_tool"
    ctx = _make_tool_context(agent_name="restricted_agent")

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )

    assert result is not None
    assert "not in delegation scope" in result["reason"]

  @pytest.mark.asyncio
  async def test_delegation_scope_allows_listed_tool(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        delegation_scopes={
            "restricted_agent": DelegationScope(allowed_tools={"read_only"})
        },
        audit_handler=handler,
    )

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)

    tool = Mock(spec=BaseTool)
    tool.name = "read_only"
    ctx = _make_tool_context(agent_name="restricted_agent")

    result = await plugin.before_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx
    )

    assert result is None


# -- After tool callback tests --------------------------------------------


class TestAfterToolCallback:

  @pytest.mark.asyncio
  async def test_audit_receipt_emitted(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(audit_handler=handler)

    tool = Mock(spec=BaseTool)
    tool.name = "my_tool"
    ctx = _make_tool_context()

    result = await plugin.after_tool_callback(
        tool=tool,
        tool_args={"a": 1},
        tool_context=ctx,
        result={"output": "done"},
    )

    assert result is None
    assert any(
        e.action == AuditAction.TOOL_CALL_COMPLETED for e in handler.events
    )
    completed = [
        e for e in handler.events if e.action == AuditAction.TOOL_CALL_COMPLETED
    ][0]
    assert completed.args_hash != ""
    assert completed.result_hash != ""

  @pytest.mark.asyncio
  async def test_audit_skipped_when_not_required(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        tool_policies={"quiet": ToolPolicy(requires_audit=False)},
        audit_handler=handler,
    )

    tool = Mock(spec=BaseTool)
    tool.name = "quiet"
    ctx = _make_tool_context()

    await plugin.after_tool_callback(
        tool=tool, tool_args={}, tool_context=ctx, result={}
    )

    assert not any(
        e.action == AuditAction.TOOL_CALL_COMPLETED for e in handler.events
    )


# -- Tool error callback tests --------------------------------------------


class TestToolErrorCallback:

  @pytest.mark.asyncio
  async def test_error_audited(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(audit_handler=handler)

    tool = Mock(spec=BaseTool)
    tool.name = "failing_tool"
    ctx = _make_tool_context()

    result = await plugin.on_tool_error_callback(
        tool=tool,
        tool_args={"x": 1},
        tool_context=ctx,
        error=ValueError("boom"),
    )

    assert result is None  # Does not block error propagation
    assert any(e.action == AuditAction.TOOL_CALL_ERROR for e in handler.events)
    error_event = [
        e for e in handler.events if e.action == AuditAction.TOOL_CALL_ERROR
    ][0]
    assert "boom" in error_event.metadata["error"]


# -- Agent delegation tests ------------------------------------------------


class TestAgentDelegation:

  @pytest.mark.asyncio
  async def test_delegation_denied_by_evaluator(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        policy_evaluator=DenyAllEvaluator(),
        delegation_scopes={"sub_agent": DelegationScope()},
        audit_handler=handler,
    )

    agent = Mock(spec=BaseAgent)
    agent.name = "sub_agent"
    ctx = _make_callback_context()

    result = await plugin.before_agent_callback(
        agent=agent, callback_context=ctx
    )

    assert result is not None
    assert any(
        e.action == AuditAction.AGENT_DELEGATION_DENIED for e in handler.events
    )

  @pytest.mark.asyncio
  async def test_delegation_allowed(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(
        policy_evaluator=AllowAllEvaluator(),
        delegation_scopes={"sub_agent": DelegationScope()},
        audit_handler=handler,
    )

    agent = Mock(spec=BaseAgent)
    agent.name = "sub_agent"
    ctx = _make_callback_context()

    result = await plugin.before_agent_callback(
        agent=agent, callback_context=ctx
    )

    assert result is None
    assert any(e.action == AuditAction.AGENT_DELEGATION for e in handler.events)

  @pytest.mark.asyncio
  async def test_no_scope_means_no_check(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(audit_handler=handler)

    agent = Mock(spec=BaseAgent)
    agent.name = "unconstrained_agent"
    ctx = _make_callback_context()

    result = await plugin.before_agent_callback(
        agent=agent, callback_context=ctx
    )

    assert result is None
    assert len(handler.events) == 0  # No audit for unconstrained agents


# -- Lifecycle tests -------------------------------------------------------


class TestLifecycle:

  @pytest.mark.asyncio
  async def test_invocation_start_and_end(self):
    handler = CollectingAuditHandler()
    plugin = GovernancePlugin(audit_handler=handler)

    inv_ctx = _make_invocation_context()

    await plugin.before_run_callback(invocation_context=inv_ctx)
    await plugin.after_run_callback(invocation_context=inv_ctx)

    actions = [e.action for e in handler.events]
    assert AuditAction.INVOCATION_START in actions
    assert AuditAction.INVOCATION_END in actions

  @pytest.mark.asyncio
  async def test_counters_cleaned_up_after_run(self):
    plugin = GovernancePlugin()

    inv_ctx = _make_invocation_context()
    await plugin.before_run_callback(invocation_context=inv_ctx)
    assert "inv-1" in plugin._tool_call_counts

    await plugin.after_run_callback(invocation_context=inv_ctx)
    assert "inv-1" not in plugin._tool_call_counts

  @pytest.mark.asyncio
  async def test_audit_log_accumulates(self):
    plugin = GovernancePlugin()
    inv_ctx = _make_invocation_context()

    await plugin.before_run_callback(invocation_context=inv_ctx)
    await plugin.after_run_callback(invocation_context=inv_ctx)

    assert len(plugin.audit_log) == 2


# -- Utility tests ---------------------------------------------------------


class TestUtilities:

  def test_hash_dict_deterministic(self):
    d = {"b": 2, "a": 1}
    h1 = _hash_dict(d)
    h2 = _hash_dict(d)
    assert h1 == h2
    assert len(h1) == 16

  def test_hash_dict_different_for_different_input(self):
    assert _hash_dict({"a": 1}) != _hash_dict({"a": 2})

  def test_hash_dict_handles_non_serializable(self):
    # default=str makes most objects serializable; verify it still returns
    # a hash rather than crashing.
    result = _hash_dict(object())
    assert isinstance(result, str)
    assert len(result) == 16
