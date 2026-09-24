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

"""End-to-end tests for Runner.rewind_async with Workflow and DynamicNodeScheduler."""

from __future__ import annotations

from google.adk.agents.context import Context
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow import node as workflow_node
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest


def _user_msg(text: str) -> types.Content:
  return types.Content(role="user", parts=[types.Part(text=text)])


@pytest.mark.asyncio
async def test_static_workflow_rewind_discards_rewound_node_completions():
  """Rewinding an interrupted workflow invocation drops completed upstream nodes and recovers new user message."""
  step_a_runs: list[tuple[int, str]] = []

  @workflow_node
  async def step_a(ctx: Context, node_input: types.Content):
    text = node_input.parts[0].text if node_input and node_input.parts else ""
    step_a_runs.append((len(step_a_runs) + 1, text))
    return {"a_run": step_a_runs[-1][0], "text": text}

  @workflow_node
  async def step_b(ctx: Context, node_input: dict[str, object]):
    if len(step_a_runs) == 1:
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="confirm", args={}, id="fc-pause-1"
                      )
                  )
              ]
          ),
          long_running_tool_ids={"fc-pause-1"},
      )
      return
    yield Event(
        output={"b_saw": node_input["a_run"], "text": node_input["text"]}
    )

  wf = Workflow(name="wf", edges=[(START, step_a), (step_a, step_b)])
  session_service = InMemorySessionService()
  app = App(
      name="rewind_static_app",
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)
  session = await session_service.create_session(
      app_name="rewind_static_app", user_id="u"
  )

  # Turn 1: seed an initial completed invocation so rewind has a prior anchor.
  step_a_runs.append((0, "seed"))  # placeholder so Turn 1 completes
  turn_1 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("turn1")
      )
  ]
  assert turn_1
  step_a_runs.clear()

  # Turn 2: step_a completes (a_run=1), step_b pauses on fc-pause-1.
  turn_2 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("turn2")
      )
  ]
  paused_inv_id = turn_2[0].invocation_id
  assert step_a_runs == [(1, "turn2")]

  # Rewind Turn 2 completely.
  await runner.rewind_async(
      user_id="u",
      session_id=session.id,
      rewind_before_invocation_id=paused_inv_id,
  )

  # Turn 3: re-run using the rewound invocation_id to verify ReplayManager
  # does not replay the rewound step_a output (a_run=1) from Turn 2, and
  # _find_user_message_for_invocation does not recover "turn2" as node_input.
  turn_3 = [
      e
      async for e in runner.run_async(
          user_id="u",
          session_id=session.id,
          invocation_id=paused_inv_id,
          new_message=_user_msg("turn3"),
      )
  ]
  assert step_a_runs == [(1, "turn2"), (2, "turn3")]
  outputs = [e.output for e in turn_3 if e.output is not None]
  assert {"b_saw": 2, "text": "turn3"} in outputs


@pytest.mark.asyncio
async def test_dynamic_scheduler_rewind_rebuilds_index_and_reruns_child():
  """DynamicNodeScheduler does not replay rewound child executions."""
  child_calls: list[int] = []

  @workflow_node
  async def child_step(ctx: Context):
    child_calls.append(len(child_calls) + 1)
    return {"call": child_calls[-1]}

  @workflow_node(rerun_on_resume=True)
  async def parent_driver(ctx: Context):
    res = await ctx.run_node(child_step, node_input="go")
    if res["call"] == 2:
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name="wait_tool", args={}, id="fc-dyn-1"
                      )
                  )
              ]
          ),
          long_running_tool_ids={"fc-dyn-1"},
      )
      return
    yield Event(output=res)

  wf = Workflow(name="wf", edges=[(START, parent_driver)])
  session_service = InMemorySessionService()
  app = App(
      name="rewind_dyn_app",
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)
  session = await session_service.create_session(
      app_name="rewind_dyn_app", user_id="u"
  )

  # Turn 1: completes normally (call=1).
  async for _ in runner.run_async(
      user_id="u", session_id=session.id, new_message=_user_msg("t1")
  ):
    pass
  assert child_calls == [1]

  # Turn 2: child_step completes (call=2), parent_driver pauses.
  turn_2 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("t2")
      )
  ]
  inv_2 = turn_2[0].invocation_id
  assert child_calls == [1, 2]

  # Rewind Turn 2.
  await runner.rewind_async(
      user_id="u", session_id=session.id, rewind_before_invocation_id=inv_2
  )

  # Turn 3 with same invocation_id: child_step must execute fresh (call=3)
  # rather than replaying call=2 from the rewound Turn 2 events.
  turn_3 = [
      e
      async for e in runner.run_async(
          user_id="u",
          session_id=session.id,
          invocation_id=inv_2,
          new_message=_user_msg("t3"),
      )
  ]
  assert child_calls == [1, 2, 3]
  assert {"call": 3} in [e.output for e in turn_3 if e.output is not None]


@pytest.mark.asyncio
async def test_rewound_active_task_scope_is_not_rejoined_on_next_turn():
  """Rewinding a turn with an active isolation_scope does not leak that scope into the next turn."""
  seen_turns: list[tuple[str, str | None]] = []

  @workflow_node
  async def scoped_worker(ctx: Context, node_input: types.Content):
    text = node_input.parts[0].text if node_input and node_input.parts else ""
    seen_turns.append((text, ctx.isolation_scope))
    if text == "t2":
      yield Event(
          content=types.Content(
              parts=[types.Part(text="paused in task scope")], role="model"
          ),
          isolation_scope="wf@1/task_agent@1",
      )
      return
    yield Event(output={"text": text, "scope": ctx.isolation_scope})

  wf = Workflow(name="wf", edges=[(START, scoped_worker)])
  session_service = InMemorySessionService()
  app = App(
      name="rewind_scope_app",
      root_agent=wf,
      resumability_config=ResumabilityConfig(is_resumable=True),
  )
  runner = Runner(app=app, session_service=session_service)
  session = await session_service.create_session(
      app_name="rewind_scope_app", user_id="u"
  )

  # Turn 1: anchor turn.
  turn_1 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("t1")
      )
  ]
  assert turn_1

  # Turn 2: emits an unclosed isolation_scope ("wf@1/task_agent@1").
  turn_2 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("t2")
      )
  ]
  inv_2 = turn_2[0].invocation_id

  # Rewind Turn 2.
  await runner.rewind_async(
      user_id="u", session_id=session.id, rewind_before_invocation_id=inv_2
  )

  # Turn 3 (without explicit invocation_id): _find_active_task_scope must ignore
  # the rewound scope from Turn 2, so Turn 3 starts a fresh invocation and does
  # not stamp the user event with "wf@1/task_agent@1".
  turn_3 = [
      e
      async for e in runner.run_async(
          user_id="u", session_id=session.id, new_message=_user_msg("t3")
      )
  ]
  assert turn_3[0].invocation_id != inv_2
  updated_session = await session_service.get_session(
      app_name="rewind_scope_app", user_id="u", session_id=session.id
  )
  assert updated_session is not None
  user_events_t3 = [
      e
      for e in updated_session.events
      if e.invocation_id == turn_3[0].invocation_id and e.author == "user"
  ]
  assert user_events_t3
  assert user_events_t3[0].isolation_scope is None
  assert {"text": "t3", "scope": None} in [
      e.output for e in turn_3 if e.output is not None
  ]
