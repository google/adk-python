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

"""Reference recipe: Deterministic pre-execution tool validation via before_tool_callback.

Demonstrates an operator-defined, spec-bound policy check executed ahead of any
tool body execution. Covers action allowlists, argument bounds, canonical URI
normalization, refusal event generation, and decision-ledger recording in
session state (Issue #6099).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Optional
import urllib.parse

from google.adk.agents.llm_agent import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


def normalize_resource_uri(uri: str) -> str:
    """Canonicalizes resource URIs for consistent policy matching."""
    parsed = urllib.parse.urlparse(uri)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]
    path = urllib.parse.urljoin("/", parsed.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))


@dataclasses.dataclass(frozen=True)
class NumericBound:
    """Spec boundary constraint for numeric arguments."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def is_valid(self, value: float | int) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclasses.dataclass
class PolicySpec:
    """Declared operator policy defining allowed operations, resources, and bounds."""

    allowed_operations: set[str] = dataclasses.field(default_factory=set)
    allowed_resources: set[str] = dataclasses.field(default_factory=set)
    numeric_bounds: dict[str, NumericBound] = dataclasses.field(default_factory=dict)
    allowed_values: dict[str, set[Any]] = dataclasses.field(default_factory=dict)
    resource_normalizer: Callable[[str], str] = normalize_resource_uri


class DeterministicToolValidator:
    """Deterministic tool validator callable suitable for before_tool_callback.

    Short-circuits execution before the tool body runs when an action or parameter
    violates declared constraints. Also maintains an audit ledger in session state.
    """

    def __init__(
        self,
        policy: PolicySpec,
        authority: str = "operator_policy",
        resource_arg_name: str = "resource_uri",
    ):
        self.policy = policy
        self.authority = authority
        self.resource_arg_name = resource_arg_name

    def _record_ledger(
        self,
        tool_context: Optional[ToolContext],
        tool_name: str,
        args: dict[str, Any],
        verdict: str,
        rule: str,
        reason: Optional[str] = None,
    ) -> None:
        if tool_context is None or not hasattr(tool_context, "state"):
            return
        ledger = tool_context.state.setdefault("decision_ledger", [])
        ledger.append(
            {
                "timestamp": time.time(),
                "authority": self.authority,
                "tool": tool_name,
                "args": dict(args),
                "verdict": verdict,
                "rule": rule,
                "reason": reason,
            }
        )

    def __call__(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        tool_name = tool.name

        # 1. Operation allowlist
        if (
            self.policy.allowed_operations
            and tool_name not in self.policy.allowed_operations
        ):
            reason = f"out-of-plan: tool '{tool_name}' not permitted"
            self._record_ledger(
                tool_context,
                tool_name,
                args,
                verdict="reject",
                rule="operation_allowlist",
                reason=reason,
            )
            return {
                "status": "rejected",
                "code": "OUT_OF_PLAN",
                "reason": reason,
            }

        # 2. Resource target allowlist with normalization
        if self.resource_arg_name in args and self.policy.allowed_resources:
            raw_resource = str(args[self.resource_arg_name])
            canonical_resource = self.policy.resource_normalizer(raw_resource)
            if canonical_resource not in self.policy.allowed_resources:
                reason = f"out-of-plan: resource '{canonical_resource}' not permitted"
                self._record_ledger(
                    tool_context,
                    tool_name,
                    args,
                    verdict="reject",
                    rule="resource_allowlist",
                    reason=reason,
                )
                return {
                    "status": "rejected",
                    "code": "OUT_OF_PLAN_RESOURCE",
                    "reason": reason,
                    "resource": canonical_resource,
                }

        # 3. Numeric bounds
        for param_name, bound in self.policy.numeric_bounds.items():
            if param_name in args:
                val = args[param_name]
                if isinstance(val, (int, float)) and not bound.is_valid(val):
                    reason = (
                        f"spec-bound breach: argument '{param_name}'={val} violates"
                        f" [{bound.min_value}, {bound.max_value}]"
                    )
                    self._record_ledger(
                        tool_context,
                        tool_name,
                        args,
                        verdict="reject",
                        rule=f"numeric_bound:{param_name}",
                        reason=reason,
                    )
                    return {
                        "status": "rejected",
                        "code": "SPEC_BOUND_BREACH",
                        "reason": reason,
                        "parameter": param_name,
                        "value": val,
                    }

        # 4. Discrete values
        for param_name, allowed_set in self.policy.allowed_values.items():
            if param_name in args:
                val = args[param_name]
                if val not in allowed_set:
                    reason = (
                        f"forbidden-value: argument '{param_name}'='{val}' not in allowed"
                        " set"
                    )
                    self._record_ledger(
                        tool_context,
                        tool_name,
                        args,
                        verdict="reject",
                        rule=f"allowed_values:{param_name}",
                        reason=reason,
                    )
                    return {
                        "status": "rejected",
                        "code": "FORBIDDEN_VALUE",
                        "reason": reason,
                        "parameter": param_name,
                        "value": val,
                    }

        # Allowed: record verdict and return None to execute tool
        self._record_ledger(
            tool_context,
            tool_name,
            args,
            verdict="allow",
            rule="policy_passed",
        )
        return None


# Define sample tools
def adjust_thermostat(temperature: float, mode: str = "cool") -> str:
    """Adjusts the environment thermostat temperature in Fahrenheit."""
    return f"Thermostat set to {temperature}°F in {mode} mode."


def fetch_sensor_data(resource_uri: str) -> dict[str, Any]:
    """Fetches sensor telemetry from an authorized endpoint."""
    return {
        "endpoint": resource_uri,
        "status": "healthy",
        "reading": 72.4,
    }


# Operator policy configuration
hvac_policy = PolicySpec(
    allowed_operations={"adjust_thermostat", "fetch_sensor_data"},
    allowed_resources={
        "https://sensors.facility.internal/v1/temperature",
        "https://sensors.facility.internal/v1/humidity",
    },
    numeric_bounds={
        "temperature": NumericBound(min_value=60.0, max_value=85.0),
    },
    allowed_values={
        "mode": {"cool", "heat", "eco"},
    },
)

policy_validator = DeterministicToolValidator(policy=hvac_policy)

root_agent = Agent(
    name="hvac_facility_agent",
    model="gemini-2.5-flash",
    description=(
        "Facilities management agent with deterministic pre-execution tool validation."
    ),
    instruction=(
        "You are an HVAC facility assistant. Help operators monitor sensors"
        " and adjust temperature safely within operating envelopes."
    ),
    before_tool_callback=policy_validator,
    tools=[adjust_thermostat, fetch_sensor_data],
)
