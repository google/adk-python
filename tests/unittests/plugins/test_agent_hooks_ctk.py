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

"""Agent Hooks conformance vectors driven through ADK's production runner."""

from __future__ import annotations

import copy
import inspect
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import AgentHooksPlugin
from google.adk.tools.function_tool import FunctionTool
from google.genai import types
import pytest

agent_hooks = pytest.importorskip("agent_hooks")

from agent_hooks import AgentContext  # noqa: E402
from agent_hooks import EnforcementMode  # noqa: E402
from agent_hooks.composition import CompositionConfig  # noqa: E402
from agent_hooks.ctk import Capability  # noqa: E402
from agent_hooks.ctk import load_vectors  # noqa: E402
from agent_hooks.ctk import run_vector  # noqa: E402
from agent_hooks.ctk import RunOutcome  # noqa: E402
from agent_hooks.ctk import RunRecord  # noqa: E402
from agent_hooks.ctk import Scenario  # noqa: E402
from agent_hooks.ctk.harness import ToolSpec  # noqa: E402
from agent_hooks.emitter import IdentityProvider  # noqa: E402
from agent_hooks.interceptor import Interceptor  # noqa: E402

from tests.unittests import testing_utils  # noqa: E402

_SKIPPED_VECTOR_CATEGORIES = {
    "approval resolver unsupported": frozenset({
        "AH-CTK-030",
        "AH-CTK-031",
        "AH-CTK-072",
        "AH-CTK-073",
        "AH-CTK-080",
        "AH-CTK-082",
        "AH-CTK-083",
        "AH-CTK-086",
        "AH-CTK-088",
        "AH-CTK-089",
        "AH-CTK-098",
        "AH-CTK-099",
        "AH-CTK-103",
        "AH-CTK-105",
    }),
    "model-facing reasons are redacted to policy_denied": frozenset(
        {"AH-CTK-100"}
    ),
}
_SKIPPED_VECTOR_IDS = frozenset().union(*_SKIPPED_VECTOR_CATEGORIES.values())

_VECTORS = load_vectors()
_EXECUTED_VECTOR_IDS = tuple(
    vector["id"]
    for vector in _VECTORS
    if vector["id"] not in _SKIPPED_VECTOR_IDS
)


class _AsyncInterceptor:
  """Adapt the CTK's synchronous scripted interceptor to the timed async API."""

  def __init__(self, interceptor: Interceptor) -> None:
    self._interceptor = interceptor

  async def intercept(self, context: AgentContext) -> Any:
    result = self._interceptor.intercept(context)
    if inspect.isawaitable(result):
      return await result
    return result


class _AdkAgentHooksHarness:
  """Run CTK scenarios through ADK with no model or tool I/O."""

  name = "google-adk-agent-hooks"
  capabilities = {
      Capability.MODEL_CALLS,
      Capability.TOOL_CALLS,
      Capability.INT64_JSON,
      Capability.BIGINT_JSON,
  }

  def __init__(self) -> None:
    self._scenario: Scenario | None = None
    self._interceptors: list[_AsyncInterceptor] = []
    self._mode = EnforcementMode.ENFORCE
    self._composition = CompositionConfig.default()
    self._identity_provider: str | IdentityProvider | None = "jcs-sha256"
    self._records: list[Any] = []
    self._tool_invocations: list[dict[str, Any]] = []

  def setup(
      self,
      scenario: Scenario,
      interceptors: list[Interceptor],
      resolver: Any,
      mode: EnforcementMode,
      composition: CompositionConfig,
      identity_provider: str | None,
      redact_for_approval: list[str] | None = None,
  ) -> None:
    if resolver is not None or redact_for_approval:
      raise ValueError("AgentHooksPlugin does not expose an approval resolver")
    self._scenario = scenario
    self._interceptors = [_AsyncInterceptor(item) for item in interceptors]
    self._mode = mode
    self._composition = composition
    self._identity_provider = self._provider(identity_provider)
    self._records = []
    self._tool_invocations = []

  async def run(self) -> RunRecord:
    scenario = self._require_scenario()
    input_content = scenario.input.get("content")
    if scenario.input.get("role") != "user" or not isinstance(
        input_content, str
    ):
      raise ValueError("ADK CTK scenarios require string user input")

    model = testing_utils.MockModel.create(
        responses=[self._model_response(item) for item in scenario.model_script]
    )
    tools = [self._tool(tool) for tool in scenario.tools.values()]
    plugin = AgentHooksPlugin(
        interceptors=self._interceptors,
        mode=self._mode.value,
        composition=self._composition,
        identity_provider=self._identity_provider,
        record_sink=self._records.append,
    )
    agent = Agent(name="ctk_agent", model=model, tools=tools)
    runner = testing_utils.InMemoryRunner(
        app=App(name="ctk_app", root_agent=agent, plugins=[plugin])
    )

    events = await runner.run_async(input_content)
    terminal_block = any(
        not record.proceeds
        and record.interception_point.value
        in {
            "agent_startup",
            "input",
            "pre_model_call",
            "post_model_call",
            "output",
        }
        for record in self._records
    )
    final_output = None if terminal_block else self._final_output(events)
    return RunRecord(
        outcome=(
            RunOutcome.BLOCKED if terminal_block else RunOutcome.COMPLETED
        ),
        final_output=final_output,
        tool_invocations=copy.deepcopy(self._tool_invocations),
        identities=[
            (record.input_identity, record.enforced_identity)
            for record in self._records
        ],
        records=[record.to_wire() for record in self._records],
    )

  def teardown(self) -> None:
    self._scenario = None
    self._interceptors = []
    self._records = []
    self._tool_invocations = []

  def _require_scenario(self) -> Scenario:
    if self._scenario is None:
      raise RuntimeError("CTK harness setup was not called")
    return self._scenario

  def _tool(self, tool: ToolSpec) -> FunctionTool:
    async def invoke(**tool_args: Any) -> Any:
      self._tool_invocations.append({
          "name": tool.name,
          "args": copy.deepcopy(tool_args),
      })
      result, is_error = tool.invoke(tool_args)
      if is_error:
        raise RuntimeError(str(result))
      return result

    invoke.__name__ = tool.name
    invoke.__doc__ = f"CTK scripted tool {tool.name}."
    argument_names = set(tool.schema)
    for behavior in tool.behavior:
      if behavior.when_args is not None:
        argument_names.update(behavior.when_args)
    for response in self._require_scenario().model_script:
      for tool_call in response.tool_calls:
        if tool_call["name"] == tool.name:
          argument_names.update(tool_call["args"])
    invoke.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Any,
            )
            for name in sorted(argument_names)
        ],
        return_annotation=Any,
    )
    return FunctionTool(invoke)

  @staticmethod
  def _model_response(response: Any) -> LlmResponse:
    parts: list[types.Part] = []
    if response.content is not None:
      if not isinstance(response.content, str):
        raise ValueError("ADK CTK model content must be text")
      parts.append(types.Part.from_text(text=response.content))
    for tool_call in response.tool_calls:
      parts.append(
          types.Part(
              function_call=types.FunctionCall(
                  id=tool_call["id"],
                  name=tool_call["name"],
                  args=copy.deepcopy(tool_call["args"]),
              )
          )
      )
    finish_reason = (
        None
        if response.finish_reason == "tool_calls"
        else types.FinishReason.STOP
    )
    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        finish_reason=finish_reason,
        model_version="ctk-model",
    )

  @staticmethod
  def _final_output(events: list[Any]) -> Any:
    for event in reversed(events):
      if event.is_final_response() and event.content is not None:
        text = "".join(
            part.text or "" for part in event.content.parts or [] if part.text
        )
        return text or None
    return None

  @staticmethod
  def _provider(
      identity_provider: str | None,
  ) -> str | IdentityProvider | None:
    if identity_provider != "ctk-fault":
      return identity_provider

    def fail_identity(_context: AgentContext) -> str:
      raise RuntimeError("ctk identity provider fault")

    return IdentityProvider("ctk-fault", fail_identity)


def test_ctk_inventory_is_explicit_and_exhaustive() -> None:
  official_ids = {vector["id"] for vector in _VECTORS}

  assert set(_EXECUTED_VECTOR_IDS).isdisjoint(_SKIPPED_VECTOR_IDS)
  assert set(_EXECUTED_VECTOR_IDS) | set(_SKIPPED_VECTOR_IDS) == official_ids
  assert all(
      vector.get("approval_script")
      for vector in _VECTORS
      if vector["id"]
      in _SKIPPED_VECTOR_CATEGORIES["approval resolver unsupported"]
  )


@pytest.mark.parametrize("vector_id", _EXECUTED_VECTOR_IDS)
async def test_applicable_ctk_vector_uses_adk_runner(vector_id: str) -> None:
  vector = next(item for item in _VECTORS if item["id"] == vector_id)

  result = await run_vector(_AdkAgentHooksHarness(), vector)

  assert result.status == "pass", {
      "detail": result.detail,
      "failures": result.failures,
  }
