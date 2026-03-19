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

"""Governance plugin for ADK agent lifecycle.

Provides runtime governance for ADK agents — policy-based tool filtering,
delegation scope enforcement, and structured audit trails — without modifying
agent logic.

Usage:
    >>> from google.adk.plugins.governance_plugin import (
    ...     GovernancePlugin, PolicyEvaluator, PolicyDecision, ToolPolicy,
    ... )
    >>>
    >>> class MyPolicyEvaluator(PolicyEvaluator):
    ...     async def evaluate_tool_call(self, *, tool_name, tool_args,
    ...                                  agent_name, context):
    ...         if tool_name == "dangerous_tool":
    ...             return PolicyDecision.deny("Tool not allowed by policy")
    ...         return PolicyDecision.allow()
    ...
    >>> plugin = GovernancePlugin(
    ...     policy_evaluator=MyPolicyEvaluator(),
    ...     tool_policies={"sql_tool": ToolPolicy(allowed_arg_patterns={"query": r"^SELECT"})},
    ... )
    >>> # runner = Runner(..., plugins=[plugin])
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import hashlib
import json
import logging
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import runtime_checkable
from typing import Set
from typing import TYPE_CHECKING

from google.genai import types
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.callback_context import CallbackContext
from ..models.llm_request import LlmRequest
from ..models.llm_response import LlmResponse
from ..tools.base_tool import BaseTool
from .base_plugin import BasePlugin

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..tools.tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Decision(str, Enum):
  """The outcome of a policy evaluation."""

  ALLOW = "ALLOW"
  DENY = "DENY"


@dataclass(frozen=True)
class PolicyDecision:
  """Structured result of a policy evaluation.

  Attributes:
    decision: Whether the action is allowed or denied.
    reason: Human-readable explanation for the decision.
    evaluator: Name of the evaluator that produced the decision.
    timestamp: Unix timestamp when the decision was made.
  """

  decision: Decision
  reason: str = ""
  evaluator: str = ""
  timestamp: float = field(default_factory=time.time)

  @staticmethod
  def allow(reason: str = "", evaluator: str = "") -> PolicyDecision:
    return PolicyDecision(
        decision=Decision.ALLOW, reason=reason, evaluator=evaluator
    )

  @staticmethod
  def deny(reason: str = "", evaluator: str = "") -> PolicyDecision:
    return PolicyDecision(
        decision=Decision.DENY, reason=reason, evaluator=evaluator
    )


class AuditAction(str, Enum):
  """Types of governance audit events."""

  TOOL_CALL_REQUESTED = "tool_call_requested"
  TOOL_CALL_COMPLETED = "tool_call_completed"
  TOOL_CALL_DENIED = "tool_call_denied"
  TOOL_CALL_ERROR = "tool_call_error"
  AGENT_DELEGATION = "agent_delegation"
  AGENT_DELEGATION_DENIED = "agent_delegation_denied"
  MODEL_REQUEST = "model_request"
  INVOCATION_START = "invocation_start"
  INVOCATION_END = "invocation_end"


@dataclass
class AuditEvent:
  """Structured audit event for governance compliance.

  Attributes:
    action: The type of action being audited.
    agent_name: The agent that triggered the action.
    tool_name: The tool involved, if applicable.
    policy_decision: The governance decision made.
    invocation_id: The ADK invocation ID.
    session_id: The session ID.
    timestamp: Unix timestamp of the event.
    args_hash: SHA-256 hash of tool arguments (not the args themselves).
    result_hash: SHA-256 hash of tool result (not the result itself).
    metadata: Additional context for the audit event.
  """

  action: AuditAction
  agent_name: str = ""
  tool_name: str = ""
  policy_decision: Optional[PolicyDecision] = None
  invocation_id: str = ""
  session_id: str = ""
  timestamp: float = field(default_factory=time.time)
  args_hash: str = ""
  result_hash: str = ""
  metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolPolicy:
  """Per-tool governance policy.

  Attributes:
    allowed: Whether the tool is allowed at all.
    allowed_arg_patterns: Regex patterns that args must match (key -> pattern).
    max_calls_per_invocation: Max times this tool can be called per invocation.
    requires_audit: Whether calls to this tool must be audited (default True).
  """

  allowed: bool = True
  allowed_arg_patterns: Optional[Dict[str, str]] = None
  max_calls_per_invocation: int = 0  # 0 = unlimited
  requires_audit: bool = True


@dataclass
class DelegationScope:
  """Defines the authority scope for sub-agent delegation.

  Implements monotonic narrowing — sub-agents can only operate within
  the authority explicitly granted by the parent.

  Attributes:
    allowed_tools: Set of tool names the sub-agent may use. Empty = all.
    allowed_sub_agents: Set of sub-agent names allowed. Empty = all.
    max_delegation_depth: Maximum depth of delegation chain. 0 = unlimited.
    custom_constraints: Additional key-value constraints for policy evaluation.
  """

  allowed_tools: Set[str] = field(default_factory=set)
  allowed_sub_agents: Set[str] = field(default_factory=set)
  max_delegation_depth: int = 0
  custom_constraints: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy evaluator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PolicyEvaluator(Protocol):
  """Protocol for policy engines to implement.

  Provides the standard integration point for governance policy evaluation.
  Implementations can be as simple as a static allowlist or as complex as
  an external policy service.
  """

  async def evaluate_tool_call(
      self,
      *,
      tool_name: str,
      tool_args: Dict[str, Any],
      agent_name: str,
      context: Optional[CallbackContext] = None,
  ) -> PolicyDecision:
    """Evaluate whether a tool call should be allowed.

    Args:
      tool_name: Name of the tool being called.
      tool_args: Arguments to the tool.
      agent_name: Name of the agent making the call.
      context: The callback context, if available.

    Returns:
      A PolicyDecision indicating ALLOW or DENY.
    """
    ...

  async def evaluate_agent_delegation(
      self,
      *,
      parent_agent_name: str,
      child_agent_name: str,
      delegation_scope: Optional[DelegationScope] = None,
      context: Optional[CallbackContext] = None,
  ) -> PolicyDecision:
    """Evaluate whether an agent delegation should be allowed.

    Args:
      parent_agent_name: Name of the delegating agent.
      child_agent_name: Name of the sub-agent being delegated to.
      delegation_scope: The scope constraints for the delegation.
      context: The callback context, if available.

    Returns:
      A PolicyDecision indicating ALLOW or DENY.
    """
    ...


# ---------------------------------------------------------------------------
# Audit handler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AuditHandler(Protocol):
  """Protocol for audit event handlers.

  Implementations can write to files, send to external systems, or
  attach to OTel spans.
  """

  async def handle(self, event: AuditEvent) -> None:
    """Handle an audit event.

    Args:
      event: The audit event to process.
    """
    ...


class LoggingAuditHandler:
  """Default audit handler that logs events via Python logging."""

  async def handle(self, event: AuditEvent) -> None:
    decision_str = ""
    if event.policy_decision:
      decision_str = (
          f" decision={event.policy_decision.decision.value}"
          f" reason='{event.policy_decision.reason}'"
      )
    logger.info(
        "[AUDIT] action=%s agent=%s tool=%s invocation=%s%s",
        event.action.value,
        event.agent_name,
        event.tool_name,
        event.invocation_id,
        decision_str,
    )


# ---------------------------------------------------------------------------
# GovernancePlugin
# ---------------------------------------------------------------------------


class GovernancePlugin(BasePlugin):
  """Plugin that enforces runtime governance on ADK agent execution.

  Provides three core governance capabilities:

  1. **Policy-based tool filtering**: Evaluates tool calls against a policy
     engine before execution. Denied calls are blocked and audited.

  2. **Delegation scope enforcement**: Validates sub-agent delegations
     against defined authority scopes, enforcing monotonic narrowing.

  3. **Structured audit trail**: Emits structured audit events at every
     governance decision point, queryable via standard logging or
     custom audit handlers.

  Example:
      >>> plugin = GovernancePlugin(
      ...     policy_evaluator=my_evaluator,
      ...     tool_policies={"sql": ToolPolicy(max_calls_per_invocation=5)},
      ...     delegation_scopes={"sub_agent": DelegationScope(
      ...         allowed_tools={"safe_tool"},
      ...     )},
      ... )
      >>> runner = Runner(
      ...     agent=root_agent,
      ...     plugins=[plugin],
      ... )
  """

  def __init__(
      self,
      *,
      name: str = "governance_plugin",
      policy_evaluator: Optional[PolicyEvaluator] = None,
      tool_policies: Optional[Dict[str, ToolPolicy]] = None,
      delegation_scopes: Optional[Dict[str, DelegationScope]] = None,
      audit_handler: Optional[AuditHandler] = None,
      denied_tool_message: str = "Tool call denied by governance policy.",
      blocked_tools: Optional[Set[str]] = None,
  ):
    """Initialize the governance plugin.

    Args:
      name: Unique name for this plugin instance.
      policy_evaluator: Custom policy engine implementing PolicyEvaluator.
      tool_policies: Per-tool policy configurations.
      delegation_scopes: Per-agent delegation scope constraints.
      audit_handler: Custom handler for audit events. Defaults to logging.
      denied_tool_message: Message returned when a tool call is denied.
      blocked_tools: Set of tool names that are always blocked.
    """
    super().__init__(name)
    self._policy_evaluator = policy_evaluator
    self._tool_policies = tool_policies or {}
    self._delegation_scopes = delegation_scopes or {}
    self._audit_handler = audit_handler or LoggingAuditHandler()
    self._denied_tool_message = denied_tool_message
    self._blocked_tools = blocked_tools or set()

    # Per-invocation tool call counters: {invocation_id: {tool_name: count}}
    self._tool_call_counts: Dict[str, Dict[str, int]] = {}
    # Audit log for the current session
    self._audit_log: List[AuditEvent] = []

  @property
  def audit_log(self) -> List[AuditEvent]:
    """Access the accumulated audit events."""
    return list(self._audit_log)

  # -- Lifecycle callbacks -------------------------------------------------

  @override
  async def before_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> Optional[types.Content]:
    """Record invocation start and initialize counters."""
    inv_id = invocation_context.invocation_id
    self._tool_call_counts[inv_id] = {}

    await self._emit_audit(
        AuditEvent(
            action=AuditAction.INVOCATION_START,
            agent_name=getattr(invocation_context.agent, "name", ""),
            invocation_id=inv_id,
            session_id=invocation_context.session.id,
        )
    )
    return None

  @override
  async def after_run_callback(
      self, *, invocation_context: InvocationContext
  ) -> None:
    """Record invocation end and clean up counters."""
    inv_id = invocation_context.invocation_id

    await self._emit_audit(
        AuditEvent(
            action=AuditAction.INVOCATION_END,
            agent_name=getattr(invocation_context.agent, "name", ""),
            invocation_id=inv_id,
            session_id=invocation_context.session.id,
            metadata={
                "tool_call_counts": dict(
                    self._tool_call_counts.get(inv_id, {})
                ),
            },
        )
    )

    # Clean up invocation state
    self._tool_call_counts.pop(inv_id, None)

  # -- Agent delegation governance -----------------------------------------

  @override
  async def before_agent_callback(
      self, *, agent: BaseAgent, callback_context: CallbackContext
  ) -> Optional[types.Content]:
    """Enforce delegation scope when sub-agents are invoked."""
    agent_name = getattr(agent, "name", "")
    inv_id = callback_context.invocation_id

    # Check if this agent has a delegation scope constraint
    if agent_name not in self._delegation_scopes:
      return None

    scope = self._delegation_scopes[agent_name]
    decision = PolicyDecision.allow()

    # Check via policy evaluator if available
    if self._policy_evaluator:
      parent_name = callback_context.agent_name
      decision = await self._policy_evaluator.evaluate_agent_delegation(
          parent_agent_name=parent_name,
          child_agent_name=agent_name,
          delegation_scope=scope,
          context=callback_context,
      )

    if decision.decision == Decision.DENY:
      await self._emit_audit(
          AuditEvent(
              action=AuditAction.AGENT_DELEGATION_DENIED,
              agent_name=agent_name,
              policy_decision=decision,
              invocation_id=inv_id,
              session_id=callback_context._invocation_context.session.id,
          )
      )
      logger.warning(
          "Delegation to agent '%s' denied: %s",
          agent_name,
          decision.reason,
      )
      return types.Content(
          role="model",
          parts=[
              types.Part(text=f"Agent delegation denied: {decision.reason}")
          ],
      )

    await self._emit_audit(
        AuditEvent(
            action=AuditAction.AGENT_DELEGATION,
            agent_name=agent_name,
            policy_decision=decision,
            invocation_id=inv_id,
            session_id=callback_context._invocation_context.session.id,
        )
    )
    return None

  # -- Tool governance -----------------------------------------------------

  @override
  async def before_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
  ) -> Optional[dict]:
    """Evaluate tool call against governance policy before execution."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    inv_id = tool_context.invocation_id

    # 1. Check blocked tools list
    if tool_name in self._blocked_tools:
      decision = PolicyDecision.deny(
          reason=f"Tool '{tool_name}' is in the blocked tools list.",
          evaluator="blocked_tools",
      )
      return await self._deny_tool(
          tool_name, agent_name, inv_id, tool_context, tool_args, decision
      )

    # 2. Check per-tool policy
    tool_policy = self._tool_policies.get(tool_name)
    if tool_policy and not tool_policy.allowed:
      decision = PolicyDecision.deny(
          reason=f"Tool '{tool_name}' is disabled by policy.",
          evaluator="tool_policy",
      )
      return await self._deny_tool(
          tool_name, agent_name, inv_id, tool_context, tool_args, decision
      )

    # 3. Check call count limits
    if tool_policy and tool_policy.max_calls_per_invocation > 0:
      counts = self._tool_call_counts.get(inv_id, {})
      current = counts.get(tool_name, 0)
      if current >= tool_policy.max_calls_per_invocation:
        decision = PolicyDecision.deny(
            reason=(
                f"Tool '{tool_name}' exceeded max calls"
                f" ({tool_policy.max_calls_per_invocation}) for this"
                " invocation."
            ),
            evaluator="call_limit",
        )
        return await self._deny_tool(
            tool_name, agent_name, inv_id, tool_context, tool_args, decision
        )

    # 4. Check delegation scope constraints
    if agent_name in self._delegation_scopes:
      scope = self._delegation_scopes[agent_name]
      if scope.allowed_tools and tool_name not in scope.allowed_tools:
        decision = PolicyDecision.deny(
            reason=(
                f"Tool '{tool_name}' not in delegation scope for agent"
                f" '{agent_name}'."
            ),
            evaluator="delegation_scope",
        )
        return await self._deny_tool(
            tool_name, agent_name, inv_id, tool_context, tool_args, decision
        )

    # 5. Run custom policy evaluator
    if self._policy_evaluator:
      decision = await self._policy_evaluator.evaluate_tool_call(
          tool_name=tool_name,
          tool_args=tool_args,
          agent_name=agent_name,
          context=tool_context,
      )
      if decision.decision == Decision.DENY:
        return await self._deny_tool(
            tool_name, agent_name, inv_id, tool_context, tool_args, decision
        )

    # All checks passed — record the request and increment counter
    self._tool_call_counts.setdefault(inv_id, {})
    self._tool_call_counts[inv_id][tool_name] = (
        self._tool_call_counts[inv_id].get(tool_name, 0) + 1
    )

    await self._emit_audit(
        AuditEvent(
            action=AuditAction.TOOL_CALL_REQUESTED,
            agent_name=agent_name,
            tool_name=tool_name,
            policy_decision=PolicyDecision.allow(),
            invocation_id=inv_id,
            session_id=tool_context._invocation_context.session.id,
            args_hash=_hash_dict(tool_args),
        )
    )
    return None

  @override
  async def after_tool_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      result: dict,
  ) -> Optional[dict]:
    """Record audit receipt after tool execution."""
    tool_name = tool.name
    tool_policy = self._tool_policies.get(tool_name)

    # Only emit if auditing is required (default True)
    if tool_policy and not tool_policy.requires_audit:
      return None

    await self._emit_audit(
        AuditEvent(
            action=AuditAction.TOOL_CALL_COMPLETED,
            agent_name=tool_context.agent_name,
            tool_name=tool_name,
            invocation_id=tool_context.invocation_id,
            session_id=tool_context._invocation_context.session.id,
            args_hash=_hash_dict(tool_args),
            result_hash=_hash_dict(result),
        )
    )
    return None

  @override
  async def on_tool_error_callback(
      self,
      *,
      tool: BaseTool,
      tool_args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> Optional[dict]:
    """Record audit event for tool errors."""
    await self._emit_audit(
        AuditEvent(
            action=AuditAction.TOOL_CALL_ERROR,
            agent_name=tool_context.agent_name,
            tool_name=tool.name,
            invocation_id=tool_context.invocation_id,
            session_id=tool_context._invocation_context.session.id,
            args_hash=_hash_dict(tool_args),
            metadata={"error": str(error)},
        )
    )
    return None

  # -- Internal helpers ----------------------------------------------------

  async def _deny_tool(
      self,
      tool_name: str,
      agent_name: str,
      invocation_id: str,
      tool_context: ToolContext,
      tool_args: dict[str, Any],
      decision: PolicyDecision,
  ) -> dict:
    """Handle a denied tool call — emit audit and return denial response."""
    await self._emit_audit(
        AuditEvent(
            action=AuditAction.TOOL_CALL_DENIED,
            agent_name=agent_name,
            tool_name=tool_name,
            policy_decision=decision,
            invocation_id=invocation_id,
            session_id=tool_context._invocation_context.session.id,
            args_hash=_hash_dict(tool_args),
        )
    )
    logger.warning(
        "Tool '%s' denied for agent '%s': %s",
        tool_name,
        agent_name,
        decision.reason,
    )
    return {"error": self._denied_tool_message, "reason": decision.reason}

  async def _emit_audit(self, event: AuditEvent) -> None:
    """Emit an audit event to the handler and the internal log."""
    self._audit_log.append(event)
    try:
      await self._audit_handler.handle(event)
    except Exception as e:
      logger.error("Audit handler failed: %s", e)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _hash_dict(d: Any) -> str:
  """Produce a stable SHA-256 hash of a dictionary for audit purposes."""
  try:
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]
  except (TypeError, ValueError):
    return ""
