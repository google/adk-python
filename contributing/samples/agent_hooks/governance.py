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

"""An example agent-hooks interceptor: a small tool-governance policy.

The interceptor implements the async
``intercept(AgentContext) -> Verdict`` contract.
It demonstrates the two enforcement primitives that matter most for tool use:

- ``deny``: a destructive tool (``delete_account``) is blocked before it runs.
- ``transform``: sensitive fields returned by a tool are redacted before the
  model (and the transcript) ever see them.

An interceptor is framework-neutral: this same class works against any
agent-hooks host (ADK, crewAI, ...), not just ADK.
"""

from __future__ import annotations

import re
from typing import Any

from agent_hooks import AgentContext
from agent_hooks import Decision
from agent_hooks import Transform
from agent_hooks import Verdict

#: Tools that must never execute under this policy.
_DENIED_TOOLS = frozenset({"delete_account"})

#: Result fields whose values are masked before the model sees them.
_SENSITIVE_KEYS = frozenset(
    {"email", "api_key", "password", "secret", "token", "ssn"}
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _redact(value: Any) -> tuple[Any, bool]:
  """Return ``(redacted_value, changed)`` for a tool result.

  Masks the values of sensitive keys and any email address found in a string.
  """
  changed = False

  def walk(node: Any) -> Any:
    nonlocal changed
    if isinstance(node, dict):
      out: dict[str, Any] = {}
      for key, item in node.items():
        if key in _SENSITIVE_KEYS and isinstance(item, str):
          out[key] = "[REDACTED]"
          changed = True
        else:
          out[key] = walk(item)
      return out
    if isinstance(node, list):
      return [walk(item) for item in node]
    if isinstance(node, str):
      masked = _EMAIL_RE.sub("[REDACTED_EMAIL]", node)
      if masked != node:
        changed = True
      return masked
    return node

  return walk(value), changed


class ToolGovernanceInterceptor:
  """Deny destructive tools and redact sensitive tool results."""

  name = "tool_governance"

  async def intercept(self, ctx: AgentContext) -> Verdict:
    point = ctx["interception_point"]

    if point == "pre_tool_call":
      tool_name = ctx["tool_call"]["name"]
      if tool_name in _DENIED_TOOLS:
        return Verdict.deny(
            reason="tool_denied",
            message=f"Tool '{tool_name}' is disabled by policy.",
        )
      return Verdict(decision=Decision.ALLOW)

    if point == "post_tool_call":
      # ``ctx["target"]`` is the tool result value at post_tool_call.
      redacted, changed = _redact(ctx["target"])
      if changed:
        return Verdict(
            decision=Decision.TRANSFORM,
            transform=Transform(path="$target", value=redacted),
        )
      return Verdict(decision=Decision.ALLOW)

    return Verdict(decision=Decision.ALLOW)
