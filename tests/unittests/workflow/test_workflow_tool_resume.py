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

"""Tests that a resumed Workflow node receives its own ordinary tool result."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
import uuid

from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.apps import ResumabilityConfig
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools import FunctionTool
from google.adk.workflow import BaseNode
from google.adk.workflow import FunctionNode
from google.adk.workflow import JoinNode
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._node_status import NodeStatus
from google.genai import types
from pydantic import Field
import pytest

SESSION = dict(app_name="boundary_test", user_id="tester", session_id="session")
STOPPED_IN_TOOL = 73
ACTIVE = "active-invocation"


class ScriptedModel(BaseLlm):
  """Record the request and return the marker carried by the tool call."""

  model: str = "offline"
  resuming: bool = False
  seen: list = Field(default_factory=list)

  async def generate_content_async(self, llm_request, stream=False):
    parts = [p for content in llm_request.contents for p in content.parts or []]
    texts = [p.text for p in parts if p.text]
    calls = [
        p.function_call.model_dump(exclude_none=True)
        for p in parts
        if p.function_call
    ]
    replies = [
        p.function_response.model_dump(exclude_none=True)
        for p in parts
        if p.function_response
    ]
    self.seen.append(dict(texts=texts, calls=calls, replies=replies))
    if self.resuming:
      # Always finish so an error in one sibling cannot suppress the other's observation.
      text = "missing tool result"
      if calls and replies and calls[-1]["id"] == replies[-1]["id"]:
        text = "resumed:" + calls[-1]["args"]["token"]
      part = types.Part(text=text)
    elif replies:
      part = types.Part(text="finished:" + replies[-1]["response"]["echo"])
    else:
      token = next(t for t in texts if t.startswith("marker/"))
      part = types.Part(
          function_call=types.FunctionCall(
              id="call_" + uuid.uuid4().hex, name="echo", args={"token": token}
          )
      )
    yield LlmResponse(content=types.Content(role="model", parts=[part]))


class LapRouter(BaseNode):

  async def _run_impl(self, *, ctx, node_input):
    if node_input == "finished:marker/earlier/a":
      yield Event(output="marker/active/a", route="again")
    else:
      yield Event(output=node_input, route="stop")


def finish(node_input: str) -> str:
  return node_input


class Scenario:
  """Build the same graph for the interrupted and resumed Runner processes."""

  def __init__(self, service, topology, *, resuming=False):
    self.topology = topology
    self.parallel = topology in ("parallel", "completed-sibling")
    self.labels = ("a", "b") if self.parallel else ("a",)
    self.pending_labels = (
        ("b",) if topology == "completed-sibling" else self.labels
    )
    self.entered = asyncio.Event()
    self.tool_calls = []
    self.models = {}
    self.resuming = resuming
    self.active_calls = 0
    edges = []
    join = JoinNode(name="join") if self.parallel else None
    for label in self.labels:
      model = ScriptedModel(resuming=resuming)
      self.models[label] = model
      agent = Agent(
          name="worker_" + label, model=model, tools=[FunctionTool(self.echo)]
      )
      if self.parallel:
        prepare = FunctionNode(
            name="prepare_" + label, func=self.make_input(label)
        )
        edges.append((START, prepare, agent, join))
      else:
        edges.append((START, agent))
      if topology == "loop":
        router = LapRouter(name="router")
        terminal = FunctionNode(name="finish", func=finish)
        edges.extend(
            [(agent, router), (router, {"again": agent, "stop": terminal})]
        )
    workflow = Workflow(name="root", edges=edges)
    self.runner = Runner(
        app=App(
            name=SESSION["app_name"],
            root_agent=workflow,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        session_service=service,
    )

  @staticmethod
  def make_input(label):
    def prepare(node_input):
      return "marker/active/" + label

    return prepare

  async def echo(self, token: str, tool_context) -> dict:
    """Return the caller's marker."""
    self.tool_calls.append((tool_context.function_call_id, token))
    if (
        token in {"marker/active/" + lane for lane in self.pending_labels}
        and not self.resuming
    ):
      self.active_calls += 1
      if self.active_calls == len(self.pending_labels):
        self.entered.set()
      await asyncio.Event().wait()
    return {"echo": token}

  async def invoke(self, invocation, text):
    return [
        event
        async for event in self.runner.run_async(
            user_id=SESSION["user_id"],
            session_id=SESSION["session_id"],
            invocation_id=invocation,
            new_message=types.UserContent(text),
        )
    ]


def initial_text(topology):
  if topology in ("parallel", "completed-sibling"):
    return "start parallel workers"
  return "marker/earlier/a" if topology == "loop" else "marker/active/a"


async def stop_first_process(db, topology, interruption):
  service = SqliteSessionService(db)
  await service.create_session(**SESSION)
  scenario = Scenario(service, topology)
  if topology == "reuse":
    # The very same Runner, graph and Agent will enter another invocation.
    await scenario.invoke("previous-invocation", "marker/earlier/a")
    assert len(scenario.tool_calls) == 1
  task = asyncio.create_task(scenario.invoke(ACTIVE, initial_text(topology)))
  try:
    await asyncio.wait_for(scenario.entered.wait(), 6)
    if topology == "completed-sibling":
      async with asyncio.timeout(6):
        while True:
          session = await service.get_session(**SESSION)
          states = [
              e.actions.agent_state["nodes"]
              for e in session.events
              if e.author == "root"
              and e.actions.agent_state
              and "nodes" in e.actions.agent_state
          ]
          if (
              states
              and states[-1].get("worker_a", {}).get("status")
              == NodeStatus.COMPLETED.value
          ):
            break
          await asyncio.sleep(0.01)
    if interruption == "process-loss":
      # Only this test child exits. The original tool call is already in SQLite.
      os._exit(STOPPED_IN_TOOL)
  finally:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def history(session):
  calls = []
  replies = []
  for event in session.events:
    for call in event.get_function_calls():
      calls.append((
          call.id,
          call.args["token"],
          event.node_info.path,
          event.invocation_id,
      ))
    for reply in event.get_function_responses():
      replies.append((reply.id, reply.response, event.error_code))
  return calls, replies


def terminal_outputs(events):
  values = []
  for event in events:
    info = event.node_info
    if "root@1" not in (info.output_for or []) and info.path != "root@1":
      continue
    if info.message_as_output and event.content:
      values.extend(p.text for p in event.content.parts or [] if p.text)
    elif event.output is not None:
      values.append(event.output)
  return values


@pytest.mark.parametrize(
    "topology", ["parallel", "completed-sibling", "reuse", "loop"]
)
@pytest.mark.parametrize("interruption", ["cancel", "process-loss"])
@pytest.mark.asyncio
async def test_resumed_node_receives_its_own_result(
    tmp_path, topology, interruption
):
  db = str(tmp_path / "history.sqlite3")
  child = await asyncio.create_subprocess_exec(
      sys.executable, str(Path(__file__).resolve()), db, topology, interruption
  )
  try:
    code = await asyncio.wait_for(child.wait(), 20)
  except asyncio.TimeoutError:
    child.kill()
    await child.wait()
    raise
  assert code == (STOPPED_IN_TOOL if interruption == "process-loss" else 0)
  service = SqliteSessionService(db)
  before, replies_before = history(await service.get_session(**SESSION))
  active = {
      token: (call_id, path)
      for call_id, token, path, inv in before
      if inv == ACTIVE and token.startswith("marker/active/")
  }
  scenario = Scenario(service, topology, resuming=True)
  expected_tokens = {"marker/active/" + label for label in scenario.labels}
  assert set(active) == expected_tokens and len(before) == 2
  if topology in ("reuse", "loop"):
    old = [x for x in before if x[1] == "marker/earlier/a"]
    assert len(old) == 1
    assert (old[0][2] == active["marker/active/a"][1]) == (topology == "reuse")
    assert (old[0][3] == ACTIVE) == (topology == "loop")
    assert any(
        r[0] == old[0][0] and r[1] == {"echo": old[0][1]}
        for r in replies_before
    )
  pending = {
      token: active[token]
      for token in ("marker/active/" + lane for lane in scenario.pending_labels)
  }
  if topology == "completed-sibling":
    assert (
        active["marker/active/a"][0],
        {"echo": "marker/active/a"},
        None,
    ) in replies_before
  for call_id, _ in pending.values():
    matching = [r for r in replies_before if r[0] == call_id]
    if interruption == "cancel":
      assert len(matching) == 1 and matching[0][2] == "INVOCATION_ABORTED"
    else:
      assert not matching

  events = await asyncio.wait_for(
      scenario.invoke(ACTIVE, initial_text(topology)), 15
  )
  after, replies_after = history(await service.get_session(**SESSION))
  observations = {label: model.seen for label, model in scenario.models.items()}
  assert after == before
  executions = [(call_id, token) for token, (call_id, _) in pending.items()]
  assert sorted(scenario.tool_calls) == (
      sorted(executions) if interruption == "process-loss" else []
  )
  for label, requests in observations.items():
    token = "marker/active/" + label
    if label not in scenario.pending_labels:
      assert not requests
      continue
    call_id, _ = active[token]
    assert len(requests) == 1, (label, requests)
    request = requests[0]
    assert [t for t in request["texts"] if t.startswith("marker/")] == [token]
    assert [(c["id"], c["name"], c["args"]) for c in request["calls"]] == [
        (call_id, "echo", {"token": token})
    ]
    saved = [r[1] for r in replies_after if r[0] == call_id]
    assert len(saved) == 1
    assert [
        (r["id"], r["name"], r["response"]) for r in request["replies"]
    ] == [(call_id, "echo", saved[0])]
    if interruption == "process-loss":
      assert saved[0] == {"echo": token}
    else:
      assert "aborted" in saved[0]["error"]
  if scenario.parallel:
    expected = {
        f"worker_{lane}": (
            ("resumed:" if lane in scenario.pending_labels else "finished:")
            + "marker/active/"
            + lane
        )
        for lane in scenario.labels
    }
  else:
    expected = "resumed:marker/active/a"
  assert expected in terminal_outputs(events)


if __name__ == "__main__":
  asyncio.run(stop_first_process(*sys.argv[1:]))
