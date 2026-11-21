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

"""Tests and reference recipe for deterministic pre-execution tool validation.

Addresses Issue #7520: Demonstrates a deterministic, spec-bound policy check
inside before_tool_callback and BasePlugin without an LLM in the loop.
Covers allowed calls, out-of-plan calls, argument-bound violations,
resource normalization, tool-body short-circuiting, refusal recording,
decision-ledger auditing (Issue #6099), and composition with
ToolCallIntegrityPlugin.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Optional
import urllib.parse

from google.adk.agents.llm_agent import Agent
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins._tool_call_integrity_plugin import ToolCallIntegrityPlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import Part
import pytest

from ... import testing_utils


# ---------------------------------------------------------------------------
# Reference Recipe: Deterministic Policy Validator
# ---------------------------------------------------------------------------


def normalize_resource_uri(uri: str) -> str:
    """Normalizes a resource URI into a canonical representation."""
    parsed = urllib.parse.urlparse(uri)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Remove default ports if present
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]
    # Normalize path: collapse slashes, strip trailing slash unless root
    path = urllib.parse.urljoin("/", parsed.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urllib.parse.urlunparse((scheme, netloc, path, "", "", ""))


@dataclasses.dataclass(frozen=True)
class NumericBound:
    """Fixed numeric bounds for argument validation."""

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
    """Declared operator policy specifying allowed actions and constraints."""

    allowed_operations: set[str] = dataclasses.field(default_factory=set)
    allowed_resources: set[str] = dataclasses.field(default_factory=set)
    numeric_bounds: dict[str, NumericBound] = dataclasses.field(default_factory=dict)
    allowed_values: dict[str, set[Any]] = dataclasses.field(default_factory=dict)
    resource_normalizer: Callable[[str], str] = normalize_resource_uri


class DeterministicToolValidator:
    """A deterministic pre-execution tool validator for before_tool_callback.

    Enforces operator-declared plan allowlists and fixed argument bounds without
    relying on an LLM. Emits structured refusals that short-circuit tool execution
    and records decision-ledger audit entries in tool_context.state.
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

    def validate(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: Optional[ToolContext] = None,
    ) -> Optional[dict[str, Any]]:
        tool_name = tool.name

        # 1. Check operation allowlist
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

        # 2. Check resource key if present
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

        # 3. Check fixed numeric bounds
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

        # 4. Check allowed discrete values
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

        # Allowed: record ledger entry and return None to proceed with execution
        self._record_ledger(
            tool_context,
            tool_name,
            args,
            verdict="allow",
            rule="policy_passed",
        )
        return None

    def __call__(
        self,
        tool: BaseTool,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        return self.validate(tool, args, tool_context)


class DeterministicPolicyPlugin(BasePlugin):
    """Plugin packaging for the deterministic policy validator."""

    def __init__(
        self,
        policy: PolicySpec,
        name: str = "deterministic_policy_plugin",
        authority: str = "operator_policy",
        resource_arg_name: str = "resource_uri",
    ):
        super().__init__(name=name)
        self.validator = DeterministicToolValidator(
            policy=policy,
            authority=authority,
            resource_arg_name=resource_arg_name,
        )

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Optional[dict[str, Any]]:
        return self.validator.validate(tool, tool_args, tool_context)


# ---------------------------------------------------------------------------
# Test Suite: 16 Focused Verification Tests
# ---------------------------------------------------------------------------


class ToolCallSpy:
    """Spy to verify whether a tool body was executed."""

    def __init__(self, return_value: Any = "executed"):
        self.call_count = 0
        self.last_args: dict[str, Any] = {}
        self.return_value = return_value

    def __call__(self, **kwargs) -> dict[str, Any]:
        self.call_count += 1
        self.last_args = kwargs
        return {"result": self.return_value, "received": kwargs}


def make_spy_tool(
    name: str, return_value: Any = "executed"
) -> tuple[FunctionTool, ToolCallSpy]:
    spy = ToolCallSpy(return_value=return_value)

    def tool_fn(
        temperature: float = 70.0,
        mode: str = "cool",
        resource_uri: str = "",
        target: str = "",
        db: str = "",
    ):
        kwargs = {}
        kwargs["temperature"] = temperature
        if mode:
            kwargs["mode"] = mode
        if resource_uri:
            kwargs["resource_uri"] = resource_uri
        if target:
            kwargs["target"] = target
        if db:
            kwargs["db"] = db
        return spy(**kwargs)

    tool_fn.__name__ = name
    tool_fn.__doc__ = f"Mock tool for {name}"
    tool = FunctionTool(tool_fn)
    return tool, spy


@pytest.fixture
def default_policy() -> PolicySpec:
    return PolicySpec(
        allowed_operations={"adjust_thermostat", "fetch_data"},
        allowed_resources={"https://api.example.com/v1/sensors"},
        numeric_bounds={
            "temperature": NumericBound(min_value=60.0, max_value=85.0),
            "limit": NumericBound(min_value=1, max_value=100),
        },
        allowed_values={
            "mode": {"cool", "heat", "eco"},
        },
    )


def test_1_allowed_call_executes_tool(default_policy):
    """1. In-plan action within spec bounds executes tool body."""
    tool, spy = make_spy_tool("adjust_thermostat", return_value="temp_adjusted")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 72.0, "mode": "cool"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("Set temp to 72")

    assert spy.call_count == 1
    assert spy.last_args == {"temperature": 72.0, "mode": "cool"}

    # Confirm genuine tool response event
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    assert len(responses) == 1
    assert responses[0].content.parts[0].function_response.response == {
        "result": "temp_adjusted",
        "received": {"temperature": 72.0, "mode": "cool"},
    }


def test_2_out_of_plan_tool_rejected(default_policy):
    """2. Unapproved tool name is rejected before execution."""
    tool, spy = make_spy_tool("delete_database")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="delete_database",
                args={"target": "prod_db"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="admin_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("delete db")

    # Tool body must not be executed
    assert spy.call_count == 0

    # Refusal response emitted in events
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    assert len(responses) == 1
    resp_dict = responses[0].content.parts[0].function_response.response
    assert resp_dict["status"] == "rejected"
    assert resp_dict["code"] == "OUT_OF_PLAN"


def test_3_out_of_plan_resource_rejected(default_policy):
    """3. Approved tool targeting unauthorized resource is rejected."""
    tool, spy = make_spy_tool("fetch_data")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="fetch_data",
                args={"resource_uri": "https://internal.vault.corp/keys"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="data_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("fetch secret keys")

    assert spy.call_count == 0


def test_4_argument_upper_bound_violation_rejected(default_policy):
    """4. Argument exceeding upper bound is rejected."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 99.0, "mode": "heat"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("set temperature to 99")

    assert spy.call_count == 0
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    resp_dict = responses[0].content.parts[0].function_response.response
    assert resp_dict["code"] == "SPEC_BOUND_BREACH"
    assert resp_dict["parameter"] == "temperature"


def test_5_argument_lower_bound_violation_rejected(default_policy):
    """5. Argument below lower bound is rejected."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 45.0, "mode": "cool"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("set temperature to 45")

    assert spy.call_count == 0
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    resp_dict = responses[0].content.parts[0].function_response.response
    assert resp_dict["code"] == "SPEC_BOUND_BREACH"


def test_6_forbidden_parameter_value_rejected(default_policy):
    """6. Denied or unapproved categorical parameter is rejected."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 70.0, "mode": "extreme_blast"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("blast mode")

    assert spy.call_count == 0
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    resp_dict = responses[0].content.parts[0].function_response.response
    assert resp_dict["code"] == "FORBIDDEN_VALUE"
    assert resp_dict["value"] == "extreme_blast"


def test_7_resource_normalization_matching(default_policy):
    """7. Un-normalized URI matching canonical allowed resource succeeds."""
    tool, spy = make_spy_tool("fetch_data", return_value="sensor_ok")
    validator = DeterministicToolValidator(default_policy)

    # Raw URI has uppercase scheme/host and trailing slash, but canonically matches
    raw_uri = "HTTPS://API.EXAMPLE.COM:443/v1/sensors/"
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="fetch_data",
                args={"resource_uri": raw_uri},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="data_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("fetch sensor data")

    assert spy.call_count == 1


def test_8_resource_normalization_traversal_rejected(default_policy):
    """8. Path traversal attempt normalizing outside envelope is rejected."""
    tool, spy = make_spy_tool("fetch_data")
    validator = DeterministicToolValidator(default_policy)

    traversal_uri = "https://api.example.com/v1/sensors/../../v2/admin"
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="fetch_data",
                args={"resource_uri": traversal_uri},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="data_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("fetch admin")

    assert spy.call_count == 0


def test_9_tool_body_short_circuit_spy(default_policy):
    """9. Verifies tool body is never executed across multiple rejections."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    for bad_temp in [10.0, 50.0, 95.0, 1000.0]:
        mock_model = testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name="adjust_thermostat",
                    args={"temperature": bad_temp, "mode": "cool"},
                ),
                "done",
            ]
        )
        agent = Agent(
            name="hvac_agent",
            model=mock_model,
            before_tool_callback=validator,
            tools=[tool],
        )
        runner = testing_utils.InMemoryRunner(agent)
        runner.run(f"set temp to {bad_temp}")

    assert spy.call_count == 0


def test_10_event_stream_records_rejection(default_policy):
    """10. Session event stream records structured rejection for auditing."""
    tool, spy = make_spy_tool("delete_database")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="delete_database",
                args={"db": "production"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="root_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    events = runner.run("delete")

    # Verify FunctionCall followed by FunctionResponse with refusal
    simplified = testing_utils.simplify_events(events)
    assert simplified[0] == (
        "root_agent",
        Part.from_function_call(name="delete_database", args={"db": "production"}),
    )
    resp_part = simplified[1][1]
    assert resp_part.function_response.response["status"] == "rejected"


def test_11_decision_ledger_recorded_in_state(default_policy):
    """11. Decision ledger records audit trace in tool_context.state (#6099)."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 70.0, "mode": "cool"},
            ),
            "done1",
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 150.0, "mode": "heat"},
            ),
            "done2",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("turn 1")
    runner.run("turn 2")

    ledger = runner.session.state.get("decision_ledger", [])
    assert len(ledger) >= 2

    # First call allowed
    first = ledger[0]
    assert first["tool"] == "adjust_thermostat"
    assert first["verdict"] == "allow"
    assert first["rule"] == "policy_passed"

    # Second call rejected
    second = ledger[1]
    assert second["tool"] == "adjust_thermostat"
    assert second["verdict"] == "reject"
    assert "numeric_bound" in second["rule"]


@pytest.mark.asyncio
async def test_12_async_callback_support(default_policy):
    """12. Supports async coroutine before_tool_callback."""
    tool, spy = make_spy_tool("adjust_thermostat")
    sync_validator = DeterministicToolValidator(default_policy)

    async def async_validator(tool, args, tool_context):
        return sync_validator(tool, args, tool_context)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 95.0, "mode": "cool"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=async_validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("test")

    assert spy.call_count == 0


def test_13_multi_callback_chain_short_circuit(default_policy):
    """13. Multi-callback pipeline: validator rejection stops chain."""
    tool, spy = make_spy_tool("adjust_thermostat")
    validator = DeterministicToolValidator(default_policy)

    secondary_spy_called = False

    def secondary_callback(tool, args, tool_context):
        nonlocal secondary_spy_called
        secondary_spy_called = True
        return None

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 110.0, "mode": "heat"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        before_tool_callback=[validator, secondary_callback],
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent)
    runner.run("overheat")

    assert spy.call_count == 0
    # Secondary callback should never be reached because first rejected
    assert not secondary_spy_called


def test_14_plugin_before_tool_callback(default_policy):
    """14. Encapsulation as BasePlugin enforces policy at plugin layer."""
    tool, spy = make_spy_tool("adjust_thermostat")
    plugin = DeterministicPolicyPlugin(default_policy)

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 120.0, "mode": "heat"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="hvac_agent",
        model=mock_model,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent, plugins=[plugin])
    runner.run("too hot")

    assert spy.call_count == 0


def test_15_integrity_plugin_composition_honest_call(default_policy):
    """15. ToolCallIntegrityPlugin + Deterministic Policy: honest call succeeds."""
    key = b"test_hmac_secret_key"
    integrity_plugin = ToolCallIntegrityPlugin(secret_key=key)
    policy_validator = DeterministicToolValidator(default_policy)

    tool, spy = make_spy_tool(
        "adjust_thermostat", return_value="executed_with_integrity"
    )

    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 74.0, "mode": "cool"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="secure_hvac_agent",
        model=mock_model,
        before_tool_callback=policy_validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent, plugins=[integrity_plugin])
    events = runner.run("set 74")

    # Call is stamped by integrity plugin and passed by policy validator
    assert spy.call_count == 1
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    assert (
        responses[0].content.parts[0].function_response.response["result"]
        == "executed_with_integrity"
    )


def test_16_integrity_plugin_composition_valid_stamp_out_of_bounds(
    default_policy,
):
    """16. ToolCallIntegrityPlugin + Policy: valid stamp on bad args caught by policy."""
    key = b"test_hmac_secret_key"
    integrity_plugin = ToolCallIntegrityPlugin(secret_key=key)
    policy_validator = DeterministicToolValidator(default_policy)

    tool, spy = make_spy_tool("adjust_thermostat")

    # Model emits call with physically invalid temperature (e.g. 99999)
    # HMAC will be stamped honestly by the framework on model emission,
    # but the deterministic policy gate must catch the out-of-bounds argument.
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(
                name="adjust_thermostat",
                args={"temperature": 99999.0, "mode": "cool"},
            ),
            "done",
        ]
    )
    agent = Agent(
        name="secure_hvac_agent",
        model=mock_model,
        before_tool_callback=policy_validator,
        tools=[tool],
    )
    runner = testing_utils.InMemoryRunner(agent, plugins=[integrity_plugin])
    events = runner.run("set 99999")

    # Integrity passed (valid HMAC), but policy vetoed: tool body did not run!
    assert spy.call_count == 0
    responses = [
        e
        for e in events
        if e.content and any(p.function_response for p in e.content.parts)
    ]
    assert len(responses) == 1
    resp_dict = responses[0].content.parts[0].function_response.response
    assert resp_dict["status"] == "rejected"
    assert resp_dict["code"] == "SPEC_BOUND_BREACH"
