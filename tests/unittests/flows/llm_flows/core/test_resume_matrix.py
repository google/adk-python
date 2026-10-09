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

"""18-cell pause/resume scenario matrix and event-stream invariant tests.

Covers `{lro, confirmation, request_input} x {root, sub_agent, node_tool} x
{resumable_true, resumable_false}` end to end through `InMemoryRunner`.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import copy
from typing import Any

from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.flows.llm_flows.tools._functions import REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
from google.adk.tools._node_tool import NodeTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.adk.workflow.utils._workflow_hitl_utils import create_request_input_response
from google.adk.workflow.utils._workflow_hitl_utils import get_request_input_interrupt_ids
from google.adk.workflow.utils._workflow_hitl_utils import REQUEST_INPUT_FUNCTION_CALL_NAME
from google.genai import types
from pydantic import BaseModel
import pytest

from .... import testing_utils


class _EmptyInput(BaseModel):
  pass


PAUSE_KINDS = ("lro", "confirmation", "request_input")
POSITIONS = ("root", "sub_agent", "node_tool")
RESUMABLE_VALUES = (False, True)
_BASELINE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _normalize_events(
    events: list[Event], id_map: dict[str, str] | None = None
) -> list[dict[str, Any]]:
  """Normalizes an event stream into deterministic JSON-serializable records."""
  if id_map is None:
    id_map = {}

  def _norm_id(raw_id: str | None) -> str | None:
    if raw_id is None:
      return None
    if raw_id not in id_map:
      id_map[raw_id] = f"fc-{len(id_map) + 1}"
    return id_map[raw_id]

  def _norm_branch(branch: str | None) -> str | None:
    if not branch:
      return None
    segments: list[str] = []
    for seg in branch.split("."):
      if "@" in seg:
        name, run_id = seg.split("@", 1)
        segments.append(f"{name}@{_norm_id(run_id)}")
      else:
        segments.append(seg)
    return ".".join(segments)

  records: list[dict[str, Any]] = []
  for ev in events:
    calls: list[dict[str, Any]] = []
    for fc in ev.get_function_calls():
      norm_fc_id = _norm_id(fc.id)
      args = copy.deepcopy(fc.args or {})
      if (
          fc.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
          and isinstance(args.get("originalFunctionCall"), dict)
          and args["originalFunctionCall"].get("id")
      ):
        args["originalFunctionCall"]["id"] = _norm_id(
            args["originalFunctionCall"]["id"]
        )
      calls.append({
          "id": norm_fc_id,
          "name": fc.name,
          "args": args,
      })

    responses: list[dict[str, Any]] = []
    for fr in ev.get_function_responses():
      responses.append({
          "id": _norm_id(fr.id),
          "name": fr.name,
          "response": copy.deepcopy(fr.response or {}),
      })

    texts: list[str] = []
    if ev.content and ev.content.parts:
      for part in ev.content.parts:
        if part.text:
          texts.append(part.text)

    lro_ids = sorted(
        _norm_id(lro_id) or "" for lro_id in ev.long_running_tool_ids or []
    )

    records.append({
        "author": ev.author,
        "branch": _norm_branch(ev.branch),
        "calls": calls,
        "responses": responses,
        "texts": texts,
        "long_running_tool_ids": lro_ids,
        "state_delta": dict(ev.actions.state_delta),
        "end_of_agent": bool(ev.actions.end_of_agent),
    })
  return records


def _build_leaf_tool_and_responses(
    pause_kind: str,
) -> tuple[Any, list[Any], str]:
  """Builds the leaf tool, mocked LLM responses, and expected final text."""
  if pause_kind == "lro":

    def _lro_action(task: str) -> None:
      """Starts a long-running action."""
      del task
      return None

    tool = LongRunningFunctionTool(func=_lro_action)
    responses = [
        types.Part.from_function_call(
            name="_lro_action", args={"task": "deploy"}
        ),
        "Leaf completed lro",
    ]
    return tool, responses, "Leaf completed lro"

  if pause_kind == "confirmation":

    def _confirmed_action(target: str) -> dict[str, str]:
      """Runs a sensitive action after confirmation."""
      return {"deleted": target}

    tool = FunctionTool(func=_confirmed_action, require_confirmation=True)
    responses = [
        types.Part.from_function_call(
            name="_confirmed_action", args={"target": "db"}
        ),
        "Leaf completed confirmation",
    ]
    return tool, responses, "Leaf completed confirmation"

  if pause_kind == "request_input":

    async def _ask_input_tool(
        topic: str, tool_context: ToolContext
    ) -> AsyncGenerator[RequestInput | dict[str, Any], None]:
      """Asks the user for additional input via RequestInput."""
      answer = tool_context.resume_inputs.get("clarify_topic")
      if answer is None:
        yield RequestInput(
            interrupt_id="clarify_topic",
            message=f"Clarify {topic}:",
        )
        return
      val = answer.get("value") if isinstance(answer, dict) else answer
      yield {"topic": topic, "clarified": val}

    tool = FunctionTool(func=_ask_input_tool)
    responses = [
        types.Part.from_function_call(
            name="_ask_input_tool", args={"topic": "budget"}
        ),
        "Leaf completed request_input",
    ]
    return tool, responses, "Leaf completed request_input"

  raise ValueError(f"Unknown pause_kind: {pause_kind}")


def _build_scenario_app(
    pause_kind: str,
    position: str,
    resumable: bool,
) -> tuple[App, str]:
  """Constructs the App and expected final text for a matrix cell."""
  leaf_tool, leaf_responses, leaf_final_text = _build_leaf_tool_and_responses(
      pause_kind
  )

  if position == "root":
    root_agent = LlmAgent(
        name="root_agent",
        model=testing_utils.MockModel.create(responses=leaf_responses),
        tools=[leaf_tool],
    )
    expected_final_text = leaf_final_text
  elif position == "sub_agent":
    sub_agent = LlmAgent(
        name="sub_agent",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        model=testing_utils.MockModel.create(responses=leaf_responses),
        tools=[leaf_tool],
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name="transfer_to_agent", args={"agent_name": "sub_agent"}
                ),
            ]
        ),
        sub_agents=[sub_agent],
    )
    expected_final_text = leaf_final_text
  elif position == "node_tool":
    child_agent = LlmAgent(
        name="child_agent",
        model=testing_utils.MockModel.create(responses=leaf_responses),
        tools=[leaf_tool],
    )
    sub_workflow = Workflow(
        name="sub_workflow",
        edges=[(START, child_agent)],
    )
    sub_workflow.input_schema = _EmptyInput
    wf_tool = NodeTool(
        node=sub_workflow,
        name="sub_workflow_tool",
        description="Runs sub_workflow.",
    )
    root_agent = LlmAgent(
        name="root_agent",
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name="sub_workflow_tool", args={}
                ),
                f"Root finished via node_tool ({pause_kind})",
            ]
        ),
        tools=[wf_tool],
    )
    expected_final_text = f"Root finished via node_tool ({pause_kind})"
  else:
    raise ValueError(f"Unknown position: {position}")

  app = App(
      name=f"matrix_{pause_kind}_{position}_{resumable}",
      root_agent=root_agent,
      resumability_config=ResumabilityConfig(is_resumable=resumable),
  )
  return app, expected_final_text


def _build_turn2_reply(
    pause_kind: str, turn1_events: list[Event]
) -> types.Content:
  """Constructs the scripted user FunctionResponse message for Turn 2."""
  if pause_kind == "lro":
    calls = [
        fc
        for ev in turn1_events
        for fc in ev.get_function_calls()
        if fc.name == "_lro_action"
    ]
    assert len(calls) == 1
    part = types.Part.from_function_response(
        name="_lro_action",
        response={"status": "done"},
    )
    part.function_response.id = calls[0].id
    return testing_utils.UserContent(part)

  if pause_kind == "confirmation":
    calls = [
        fc
        for ev in turn1_events
        for fc in ev.get_function_calls()
        if fc.name == REQUEST_CONFIRMATION_FUNCTION_CALL_NAME
    ]
    assert len(calls) == 1
    part = types.Part.from_function_response(
        name=REQUEST_CONFIRMATION_FUNCTION_CALL_NAME,
        response={"confirmed": True},
    )
    part.function_response.id = calls[0].id
    return testing_utils.UserContent(part)

  if pause_kind == "request_input":
    req_events = [
        ev
        for ev in turn1_events
        if any(
            fc.name == REQUEST_INPUT_FUNCTION_CALL_NAME
            for fc in ev.get_function_calls()
        )
    ]
    assert len(req_events) == 1
    interrupt_id = get_request_input_interrupt_ids(req_events[0])[0]
    part = create_request_input_response(interrupt_id, {"value": "approved"})
    return testing_utils.UserContent(part)

  raise ValueError(f"Unknown pause_kind: {pause_kind}")


async def _run_scenario(
    pause_kind: str,
    position: str,
    resumable: bool,
) -> dict[str, Any]:
  """Runs Turn 1 (pause) and Turn 2 (resume) and returns normalized records."""
  app, expected_final_text = _build_scenario_app(
      pause_kind, position, resumable
  )
  runner = testing_utils.InMemoryRunner(app=app)

  # Turn 1: Trigger the pause.
  turn1_events = await runner.run_async("Start task")
  pause_events = [ev for ev in turn1_events if ev.long_running_tool_ids]
  assert len(pause_events) == 1

  expected_pause_author = (
      "_ask_input_tool"
      if pause_kind == "request_input"
      else {
          "root": "root_agent",
          "sub_agent": "sub_agent",
          "node_tool": "child_agent",
      }[position]
  )
  assert pause_events[0].author == expected_pause_author
  if position == "node_tool":
    assert pause_events[0].branch is not None
    assert pause_events[0].branch.startswith("sub_workflow_tool@")
  elif pause_kind == "request_input":
    assert pause_events[0].branch is not None
    assert pause_events[0].branch.startswith("_ask_input_tool@")
  else:
    assert pause_events[0].branch is None

  # Turn 2: Supply the user FunctionResponse and complete the flow.
  reply = _build_turn2_reply(pause_kind, turn1_events)
  turn2_events = await runner.run_async(reply)

  turn2_texts = [
      part.text
      for ev in turn2_events
      if ev.content and ev.content.parts
      for part in ev.content.parts
      if part.text
  ]
  assert expected_final_text in turn2_texts

  session = runner.session
  assert session is not None
  id_map: dict[str, str] = {}
  return {
      "turn1_events": _normalize_events(turn1_events, id_map),
      "turn2_events": _normalize_events(turn2_events, id_map),
      "session_events": _normalize_events(session.events, id_map),
  }


def _strip_lifecycle_events(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
  return [
      rec
      for rec in records
      if not rec["end_of_agent"]
      and (
          rec["calls"]
          or rec["responses"]
          or rec["texts"]
          or rec["state_delta"]
          or rec["long_running_tool_ids"]
      )
  ]


@pytest.mark.asyncio
@pytest.mark.parametrize("resumable", RESUMABLE_VALUES)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("pause_kind", PAUSE_KINDS)
async def test_resume_matrix_and_event_stream_invariants(
    pause_kind: str,
    position: str,
    resumable: bool,
) -> None:
  """Pausing and resuming preserves event-stream parity across topologies.

  Setup: App configured with `pause_kind` at `position` and `is_resumable`.
  Act:
    - Turn 1: Trigger tool pause.
    - Turn 2: Supply matching user `FunctionResponse`.
  Assert:
    - Pause event carries expected `author`, `branch`, and `long_running_tool_ids`.
    - Resumable runs emit `end_of_agent` checkpoints and match non-resumable
      substantive events once lifecycle markers are stripped.
  """
  result = await _run_scenario(pause_kind, position, resumable)
  session_events = result["session_events"]

  has_end_of_agent = any(rec["end_of_agent"] for rec in session_events)
  assert has_end_of_agent is resumable

  cache_key = (pause_kind, position)
  if not resumable:
    _BASELINE_CACHE[cache_key] = result
  else:
    baseline = _BASELINE_CACHE.get(cache_key)
    if baseline is None:
      baseline = await _run_scenario(pause_kind, position, resumable=False)
    for key in ("turn1_events", "turn2_events", "session_events"):
      assert _strip_lifecycle_events(result[key]) == _strip_lifecycle_events(
          baseline[key]
      )
