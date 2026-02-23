"""Example 5: All Interrupt Actions — Concurrent Injection

Demonstrates all 8 interrupt action types via asyncio.create_task injection:

  1. continue        — interrupt logged, workflow proceeds normally
  2. rerun           — current node re-executes before continuing
  3. pause + resume  — execution blocks until resume() is called
  4. skip            — BEFORE-mode: node execution is skipped entirely
  5. go_back         — rewinds N steps in the execution path
  6. defer           — interrupt saved as a todo in session.state; continues
  7. update_state    — injects key/value pairs into GraphState.data
  8. change_condition — stores overrides in agent_state.conditions

Timing pattern (scenarios 1–3, 5):
  • SlowNode runs for 2 seconds (2 × 1s sub-steps)
  • Interrupt injected at t=0.8s via asyncio.create_task
  • AFTER-interrupt check fires when node completes at t=2s
  → The interrupt was queued mid-execution but processed at the AFTER checkpoint.

Note: Between sub-steps the GraphAgent only checks `is_active()` (cancellation),
NOT the message queue. The message queue is consumed once at the AFTER checkpoint.

Run modes:
- Default: python -m contributing.samples.graph_examples.05_interrupts_basic.agent
- LLM: python -m contributing.samples.graph_examples.05_interrupts_basic.agent --use-llm
  or: USE_LLM=1 python -m contributing.samples.graph_examples.05_interrupts_basic.agent
"""

import asyncio
import time

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.graph import GraphAgent
from google.adk.agents.graph import InterruptConfig
from google.adk.agents.graph import InterruptMode
from google.adk.agents.graph.interrupt_service import InterruptService
from google.adk.events.event import Event
from google.adk.events.event import EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from contributing.samples.graph_examples.example_utils import create_llm_agent
from contributing.samples.graph_examples.example_utils import use_llm_mode

APP_NAME = "interrupt_demo"
USER_ID = "user1"


# ---------------------------------------------------------------------------
# Deterministic Agents (BaseAgent)
# ---------------------------------------------------------------------------


class SlowNode(BaseAgent):
  """2 sub-steps × 1s each = 2s total. Shows interrupt queued mid-execution."""

  def __init__(self, name: str, label: str = "", **kwargs):
    super().__init__(name=name, **kwargs)
    self._label = label or name

  async def _run_async_impl(self, ctx):
    run_count = ctx.session.state.get(f"{self.name}_runs", 0) + 1
    for step in range(1, 3):
      await asyncio.sleep(1.0)
      yield Event(
          author=self.name,
          content=types.Content(
              parts=[
                  types.Part(
                      text=(
                          f"[{self._label}] sub-step {step}/2"
                          + (f" (run #{run_count})" if run_count > 1 else "")
                      )
                  )
              ]
          ),
          actions=EventActions(state_delta={f"{self.name}_runs": run_count}),
      )


class QuickNode(BaseAgent):
  """Instant node — used for surrounding workflow nodes."""

  def __init__(self, name: str, label: str = "", **kwargs):
    super().__init__(name=name, **kwargs)
    self._label = label or name

  async def _run_async_impl(self, ctx):
    yield Event(
        author=self.name,
        content=types.Content(parts=[types.Part(text=f"[{self._label}] done")]),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0: float = 0.0


def _ts() -> str:
  return f"t={time.time() - _T0:.1f}s"


async def _run(
    runner: Runner,
    session_id: str,
    *,
    user_id: str = USER_ID,
) -> None:
  msg = types.Content(parts=[types.Part(text="Start")])
  async for event in runner.run_async(
      user_id=user_id, session_id=session_id, new_message=msg
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if part.text and "#metadata" not in event.author:
          prefix = "  🛑" if "INTERRUPT" in part.text else "    "
          print(f"  {prefix} [{_ts()}] [{event.author}] {part.text}")


async def _make_session(
    session_service: InMemorySessionService,
    interrupt_service: InterruptService,
    session_id: str,
) -> None:
  await session_service.create_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=session_id
  )
  interrupt_service.register_session(session_id)


# ---------------------------------------------------------------------------
# Scenario 1: continue
# ---------------------------------------------------------------------------


async def scenario_continue() -> None:
  print("\n" + "-" * 55)
  print("Scenario 1: continue — interrupt logged, execution proceeds")
  print("-" * 55)
  print("  Interrupt injected at t=0.8s (node runs until t=2s)")
  print("  AFTER check fires at t=2s — message consumed, continue\n")

  sid = "s1_continue"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_continue",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=SlowNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  async def _inject():
    await asyncio.sleep(0.8)
    print(f"  >>> [{_ts()}] injecting 'continue' interrupt")
    await interrupt_service.send_message(
        sid, "Looks good — continue", action="continue"
    )

  global _T0
  _T0 = time.time()
  task = asyncio.create_task(_inject())
  await _run(runner, sid)
  await task
  print("  Result: workflow completed normally after interrupt logged\n")


# ---------------------------------------------------------------------------
# Scenario 2: rerun
# ---------------------------------------------------------------------------


async def scenario_rerun() -> None:
  print("-" * 55)
  print("Scenario 2: rerun — current node re-executes")
  print("-" * 55)
  print("  Interrupt injected at t=0.8s; AFTER check at t=2s → rerun")
  print("  Draft runs again (run #2), then finalize\n")

  sid = "s2_rerun"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_rerun",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=SlowNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  async def _inject():
    await asyncio.sleep(0.8)
    print(f"  >>> [{_ts()}] injecting 'rerun' interrupt")
    await interrupt_service.send_message(
        sid,
        "Revise — rerun",
        action="rerun",
        metadata={"guidance": "Add more detail"},
    )

  global _T0
  _T0 = time.time()
  task = asyncio.create_task(_inject())
  await _run(runner, sid)
  await task

  session = await session_service.get_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=sid
  )
  runs = session.state.get("draft_runs", 0)
  print(f"  Result: draft_runs={runs} (rerun worked)\n")


# ---------------------------------------------------------------------------
# Scenario 3: pause + resume
# ---------------------------------------------------------------------------


async def scenario_pause_resume() -> None:
  print("-" * 55)
  print("Scenario 3: pause + resume — execution blocks for human review")
  print("-" * 55)
  print("  Node completes at t=2s; 'pause' queued at t=0.8s → blocks")
  print("  resume() called at t=3.5s → continues to finalize\n")

  sid = "s3_pause"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_pause",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=SlowNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  async def _inject():
    await asyncio.sleep(0.8)
    print(
        f"  >>> [{_ts()}] injecting 'pause' interrupt (queued; processed at"
        " t=2s AFTER check)"
    )
    await interrupt_service.send_message(
        sid, "Hold — human review", action="pause"
    )
    # Simulate human review delay — resume after 1.5s of "paused" state
    await asyncio.sleep(1.5)
    print(f"  >>> [{_ts()}] calling resume() — human review complete")
    await interrupt_service.resume(sid)

  global _T0
  _T0 = time.time()
  task = asyncio.create_task(_inject())
  await _run(runner, sid)
  await task
  print("  Result: workflow paused for human review then resumed\n")


# ---------------------------------------------------------------------------
# Scenario 4: skip (BEFORE mode)
# ---------------------------------------------------------------------------


async def scenario_skip() -> None:
  print("-" * 55)
  print("Scenario 4: skip — node execution skipped entirely (BEFORE mode)")
  print("-" * 55)
  print("  'review' node skipped via BEFORE interrupt (pre-queued)")
  print("  Graph: draft -> review -> finalize (review skipped)\n")

  sid = "s4_skip"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_skip",
          interrupt_service=interrupt_service,
          # BEFORE mode — interrupt check happens BEFORE 'review' runs
          interrupt_config=InterruptConfig(
              mode=InterruptMode.BEFORE, nodes=["review"]
          ),
      )
      .add_node("draft", agent=QuickNode(name="draft", label="draft"))
      .add_node("review", agent=QuickNode(name="review", label="review"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "review")
      .add_edge("review", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  # Pre-queue the skip interrupt (BEFORE check → message must be in queue before node starts)
  await interrupt_service.send_message(
      sid, "Skip review — auto-approved", action="skip"
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  global _T0
  _T0 = time.time()
  await _run(runner, sid)
  print("  Result: 'review' node was skipped, finalize ran directly\n")


# ---------------------------------------------------------------------------
# Scenario 5: go_back
# ---------------------------------------------------------------------------


async def scenario_go_back() -> None:
  print("-" * 55)
  print("Scenario 5: go_back — rewind N steps in execution path")
  print("-" * 55)
  print("  AFTER 'review': go_back 1 step → reruns 'draft' then 'review'")
  print("  Interrupt injected at t=0.8s while review node runs (2s)\n")

  sid = "s5_goback"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_goback",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["review"]
          ),
      )
      .add_node("draft", agent=QuickNode(name="draft", label="draft"))
      .add_node("review", agent=SlowNode(name="review", label="review"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "review")
      .add_edge("review", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  _injected = False

  async def _inject():
    nonlocal _injected
    await asyncio.sleep(0.8)
    print(f"  >>> [{_ts()}] injecting 'go_back' (steps=1) while review runs")
    await interrupt_service.send_message(
        sid,
        "Needs rework — go back to draft",
        action="go_back",
        metadata={"steps": 1},
    )
    _injected = True

  global _T0
  _T0 = time.time()
  task = asyncio.create_task(_inject())
  await _run(runner, sid)
  await task
  print(
      "  Result: execution rewound to 'draft', then re-ran"
      " draft→review→finalize\n"
  )


# ---------------------------------------------------------------------------
# Scenario 6: defer
# ---------------------------------------------------------------------------


async def scenario_defer() -> None:
  print("-" * 55)
  print("Scenario 6: defer — interrupt saved as todo, execution continues")
  print("-" * 55)
  print("  'defer' adds the message to session.state['_interrupt_todos']\n")

  sid = "s6_defer"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_defer",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=QuickNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  # Pre-queue defer interrupt
  await interrupt_service.send_message(
      sid,
      "Non-urgent: add metadata later",
      action="defer",
      metadata={"message": "Add citation metadata", "priority": "low"},
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  global _T0
  _T0 = time.time()
  await _run(runner, sid)

  session = await session_service.get_session(
      app_name=APP_NAME, user_id=USER_ID, session_id=sid
  )
  todos = session.state.get("_interrupt_todos", [])
  print(f"  Result: todos in session.state = {[t['message'] for t in todos]}\n")


# ---------------------------------------------------------------------------
# Scenario 7: update_state
# ---------------------------------------------------------------------------


async def scenario_update_state() -> None:
  print("-" * 55)
  print("Scenario 7: update_state — inject key/value into GraphState.data")
  print("-" * 55)
  print("  Injects priority='high' into GraphState.data via interrupt\n")

  sid = "s7_update"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_update",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=QuickNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  # Pre-queue update_state interrupt: injects priority into GraphState.data
  await interrupt_service.send_message(
      sid,
      "Boost priority",
      action="update_state",
      metadata={"priority": "high", "escalated": True},
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  global _T0
  _T0 = time.time()
  await _run(runner, sid)
  print(
      "  Result: priority='high' and escalated=True injected into"
      " GraphState.data\n"
  )
  print(
      "  (Edge conditions on subsequent nodes can use s.data.get('priority'))\n"
  )


# ---------------------------------------------------------------------------
# Scenario 8: change_condition
# ---------------------------------------------------------------------------


async def scenario_change_condition() -> None:
  print("-" * 55)
  print("Scenario 8: change_condition — store condition overrides in metadata")
  print("-" * 55)
  print("  Stores named condition overrides in agent_state.conditions\n")

  sid = "s8_cond"
  interrupt_service = InterruptService()
  session_service = InMemorySessionService()
  await _make_session(session_service, interrupt_service, sid)

  graph = (
      GraphAgent(
          name="g_cond",
          interrupt_service=interrupt_service,
          interrupt_config=InterruptConfig(
              mode=InterruptMode.AFTER, nodes=["draft"]
          ),
      )
      .add_node("draft", agent=QuickNode(name="draft", label="draft"))
      .add_node("finalize", agent=QuickNode(name="finalize", label="finalize"))
      .add_edge("draft", "finalize")
      .set_start("draft")
      .set_end("finalize")
  )

  # Pre-queue change_condition: stores override flags in agent_state.conditions
  await interrupt_service.send_message(
      sid,
      "Override routing conditions",
      action="change_condition",
      metadata={"allow_fast_track": True, "require_approval": False},
  )

  runner = Runner(
      app_name=APP_NAME,
      agent=graph,
      session_service=session_service,
      auto_create_session=False,
  )

  global _T0
  _T0 = time.time()
  await _run(runner, sid)
  print("  Result: condition overrides stored in agent_state.conditions")
  print("  Edge conditions can read: data.get('_conditions', {})\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main():
  print("\n" + "=" * 55)
  print("Example 5: All Interrupt Actions — Concurrent Injection")
  print("=" * 55)
  print()
  print("Interrupt timing model:")
  print("  • AFTER mode: interrupt checked ONCE after all node events complete")
  print("  • BEFORE mode: interrupt checked ONCE before node starts")
  print(
      "  • Mid-execution injection: message queued during node, consumed at"
      " checkpoint"
  )
  print()

  await scenario_continue()
  await scenario_rerun()
  await scenario_pause_resume()
  await scenario_skip()
  await scenario_go_back()
  await scenario_defer()
  await scenario_update_state()
  await scenario_change_condition()

  print("=" * 55)
  print("All 8 interrupt actions demonstrated.")
  print("=" * 55 + "\n")


if __name__ == "__main__":
  asyncio.run(main())
