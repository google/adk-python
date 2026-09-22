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

"""Tests that a live agent stops its background tools when its run ends.

Live mode runs two kinds of tools as background tasks that outlive the model
turn that started them: streaming tools and non-blocking tools. Both belong to
the agent run that started them, and both stop when it ends -- including when
it ends by handing off to another agent, which is when the next agent takes
over the live request queue they write to.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from typing import Any
from typing import AsyncGenerator
from typing import Callable

from google.adk.agents.active_streaming_tool import ActiveStreamingTool
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.flows.llm_flows import base_llm_flow
from google.adk.flows.llm_flows.single_flow import SingleFlow
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .. import testing_utils

_MONITOR = 'monitor'
_MAX_EVENTS = 50
# Slow enough that a monitor left running is unmistakable, yet quick enough
# that a turn produces only a handful of ticks before it ends.
_TICK_SECONDS = 0.1


def _call(name: str) -> LlmResponse:
  """A model turn that calls ``name`` with no arguments."""
  return LlmResponse(
      content=types.Content(
          role='model',
          parts=[types.Part.from_function_call(name=name, args={})],
      ),
      turn_complete=False,
  )


async def _run_live_turn(
    tools: list[Any],
    *,
    calls: list[str],
    stop_when: Callable[[list[Event]], bool] | None = None,
    timeout: float = 5.0,
) -> tuple[list[Event], bool]:
  """Runs a live turn in which the model makes ``calls``, in order.

  Returns the events the caller saw and whether the stream ended on its own.
  It does not end on its own if ``stop_when`` asked to stop early, or if the
  turn is still producing events once ``timeout`` elapses.
  """
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create([_call(name) for name in calls]),
      tools=tools,
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(app_name='app', agent=agent, session_service=session_service)
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      types.Blob(data=b'question', mime_type='audio/pcm')
  )

  events: list[Event] = []
  ended = False

  async def _consume() -> None:
    nonlocal ended
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['TEXT']),
        )
    ) as agen:
      async for event in agen:
        events.append(event)
        if len(events) >= _MAX_EVENTS or (stop_when and stop_when(events)):
          return
      ended = True

  try:
    await asyncio.wait_for(_consume(), timeout=timeout)
  except asyncio.TimeoutError:
    pass
  # Let the teardown that the generator's closure kicked off finish.
  for _ in range(10):
    await asyncio.sleep(0)
  return events, ended


def _task_completed() -> str:
  """The signal a live agent uses to end its own turn."""
  return 'done'


@pytest.mark.asyncio
async def test_teardown_empties_both_registries(
    monkeypatch: pytest.MonkeyPatch,
):
  """Neither registry keeps a tool of a run that is over.

  Tools that stop on request retire themselves, so this uses two that refuse
  to: what is left behind is exactly what teardown has to sweep up.
  """
  monkeypatch.setattr(base_llm_flow, '_TOOL_SHUTDOWN_TIMEOUT_SECONDS', 0.05)

  def refuses_to_stop() -> Any:
    """Ignores the first cancellation; honors the second, so this test ends."""
    swallowed = False

    async def run() -> None:
      nonlocal swallowed
      while True:
        try:
          await asyncio.sleep(0.01)
        except asyncio.CancelledError:
          if swallowed:
            raise
          swallowed = True

    return run()

  streaming_task = asyncio.create_task(refuses_to_stop())
  non_blocking_task = asyncio.create_task(refuses_to_stop())
  await asyncio.sleep(0)

  invocation_context = await testing_utils.create_invocation_context(
      agent=Agent(name='agent', model=testing_utils.MockModel.create([]))
  )
  invocation_context.active_streaming_tools = {
      _MONITOR: ActiveStreamingTool(
          task=streaming_task, stream=LiveRequestQueue()
      )
  }
  invocation_context.active_non_blocking_tool_tasks = {
      'lookup_1': non_blocking_task
  }

  await SingleFlow()._stop_background_tool_tasks(invocation_context)

  assert not invocation_context.active_streaming_tools
  assert not invocation_context.active_non_blocking_tool_tasks

  # The registry no longer holds them, so this test owns their disposal: a
  # task left pending here would stall the event loop's shutdown.
  for task in (streaming_task, non_blocking_task):
    task.cancel()
  await asyncio.gather(
      streaming_task, non_blocking_task, return_exceptions=True
  )
  assert streaming_task.done() and non_blocking_task.done()


@pytest.mark.asyncio
async def test_streaming_tool_stops_when_its_agent_hands_off():
  """A handoff ends the agent's run, so its background tools end with it.

  The sub agent takes over the live request queue: a tool still running for
  the previous agent would push function responses at a model that never
  called it.
  """
  tasks: list[asyncio.Task[Any]] = []
  ticks = 0
  seen_by_sub_agent: dict[str, Any] = {}

  async def monitor() -> AsyncGenerator[Any, None]:
    nonlocal ticks
    tasks.append(asyncio.current_task())
    while True:
      ticks += 1
      yield {'tick': ticks}
      await asyncio.sleep(_TICK_SECONDS)

  def report() -> str:
    """Records, from inside the sub agent, what the handoff left running."""
    seen_by_sub_agent['monitor_stopped'] = tasks[0].done()
    seen_by_sub_agent['ticks'] = ticks
    return 'reported'

  sub_agent = Agent(
      name='sub_agent',
      model=testing_utils.MockModel.create([_call('report')]),
      tools=[report],
  )
  root_agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create([
          _call(_MONITOR),
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part.from_function_call(
                          name='transfer_to_agent',
                          args={'agent_name': 'sub_agent'},
                      )
                  ],
              ),
              turn_complete=False,
          ),
      ]),
      tools=[monitor],
      sub_agents=[sub_agent],
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(
      app_name='app', agent=root_agent, session_service=session_service
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      types.Blob(data=b'question', mime_type='audio/pcm')
  )

  async def _consume() -> None:
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['TEXT']),
        )
    ) as agen:
      seen = 0
      async for _ in agen:
        seen += 1
        # Stop once the sub agent has run, or the replaying mock loops.
        if 'monitor_stopped' in seen_by_sub_agent or seen >= _MAX_EVENTS:
          return

  try:
    await asyncio.wait_for(_consume(), timeout=10.0)
  except asyncio.TimeoutError:
    pass

  assert seen_by_sub_agent.get('monitor_stopped'), (
      'the monitor was still running while the sub agent held the live'
      ' request queue'
  )
  # It stopped at the handoff, not merely by the end of the session.
  await asyncio.sleep(_TICK_SECONDS * 3)
  assert ticks == seen_by_sub_agent['ticks']


@pytest.mark.asyncio
async def test_handoff_stops_feeding_the_stopped_tools_stream():
  """A stopped tool's stream is dropped, not left collecting live input.

  ``_send_to_model`` duplicates every live request into each registered
  stream, so an entry left behind after the tool is gone grows for the rest of
  the session -- one entry per audio chunk the user speaks.
  """
  contexts: list[InvocationContext] = []
  handed_off = asyncio.Event()

  async def monitor(
      tool_context: ToolContext, input_stream: LiveRequestQueue
  ) -> AsyncGenerator[Any, None]:
    # Declaring `input_stream` is what gets this tool a dedicated queue.
    contexts.append(tool_context._invocation_context)
    while True:
      await input_stream.get()
      yield {'saw': 'input'}

  def report() -> str:
    handed_off.set()
    return 'sub agent is live'

  sub_agent = Agent(
      name='sub_agent',
      model=testing_utils.MockModel.create([_call('report')]),
      tools=[report],
  )
  root_agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create([
          _call(_MONITOR),
          LlmResponse(
              content=types.Content(
                  role='model',
                  parts=[
                      types.Part.from_function_call(
                          name='transfer_to_agent',
                          args={'agent_name': 'sub_agent'},
                      )
                  ],
              ),
              turn_complete=False,
          ),
      ]),
      tools=[monitor],
      sub_agents=[sub_agent],
  )

  session_service = InMemorySessionService()
  session = await session_service.create_session(app_name='app', user_id='u')
  runner = Runner(
      app_name='app', agent=root_agent, session_service=session_service
  )
  live_request_queue = LiveRequestQueue()
  live_request_queue.send_realtime(
      types.Blob(data=b'question', mime_type='audio/pcm')
  )

  async def _consume() -> None:
    async with aclosing(
        runner.run_live(
            user_id='u',
            session_id=session.id,
            live_request_queue=live_request_queue,
            run_config=RunConfig(response_modalities=['TEXT']),
        )
    ) as agen:
      seen = 0
      async for _ in agen:
        seen += 1
        if handed_off.is_set():
          # The user keeps talking while the sub agent is in charge.
          for _ in range(25):
            live_request_queue.send_realtime(
                types.Blob(data=b'...', mime_type='audio/pcm')
            )
            await asyncio.sleep(0.005)
          return
        if seen >= _MAX_EVENTS:
          return

  try:
    await asyncio.wait_for(_consume(), timeout=10.0)
  except asyncio.TimeoutError:
    pass

  assert _MONITOR not in (contexts[0].active_streaming_tools or {}), (
      'the stopped tool is still registered, so every live request the user'
      ' sends for the rest of the session is copied into its stream'
  )


@pytest.mark.asyncio
async def test_streaming_tool_stops_when_the_live_turn_ends():
  """A streaming tool that never stops on its own is stopped for it."""
  tasks: list[asyncio.Task[Any]] = []
  ticks = 0
  started = asyncio.Event()

  async def monitor() -> AsyncGenerator[Any, None]:
    nonlocal ticks
    tasks.append(asyncio.current_task())
    started.set()
    while True:
      ticks += 1
      yield {'tick': ticks}
      await asyncio.sleep(_TICK_SECONDS)

  async def task_completed() -> str:
    # Ends the turn only once the monitor is up, so the turn cannot end
    # before there is anything to stop.
    await started.wait()
    return _task_completed()

  events, ended = await _run_live_turn(
      [monitor, task_completed], calls=[_MONITOR, 'task_completed']
  )

  assert ended, (
      'the live stream never ended: the streaming tool kept producing after'
      f' the agent turn was over. Saw: {len(events)} events.'
  )
  assert tasks[0].done()
  # And it really is stopped, not merely between ticks.
  ticks_at_the_end = ticks
  await asyncio.sleep(_TICK_SECONDS * 3)
  assert ticks == ticks_at_the_end


@pytest.mark.asyncio
async def test_streaming_tool_stops_when_the_caller_stops_listening():
  """Abandoning the stream stops the tool too, rather than leaking it."""
  tasks: list[asyncio.Task[Any]] = []
  ticks = 0
  started = asyncio.Event()

  async def monitor() -> AsyncGenerator[Any, None]:
    nonlocal ticks
    tasks.append(asyncio.current_task())
    started.set()
    while True:
      ticks += 1
      yield {'tick': ticks}
      await asyncio.sleep(_TICK_SECONDS)

  async def sync() -> str:
    # Answers only once the monitor is up, so the event that makes the caller
    # walk away cannot arrive before there is something to leak.
    await started.wait()
    return 'ok'

  _, ended = await _run_live_turn(
      [monitor, sync],
      calls=[_MONITOR, 'sync'],
      stop_when=lambda _: started.is_set(),
  )

  assert not ended  # The caller walked away mid-stream.
  assert tasks[0].done()
  ticks_at_the_end = ticks
  await asyncio.sleep(_TICK_SECONDS * 3)
  assert ticks == ticks_at_the_end


@pytest.mark.asyncio
async def test_non_blocking_tool_stops_when_the_live_turn_ends():
  """A non-blocking tool's task is cancelled with the invocation."""
  started = asyncio.Event()
  cancelled = asyncio.Event()

  async def slow_lookup() -> str:
    started.set()
    try:
      await asyncio.sleep(30)
    except asyncio.CancelledError:
      cancelled.set()
      raise
    return 'never'

  async def task_completed() -> str:
    await started.wait()
    return _task_completed()

  scheduled = FunctionTool(func=slow_lookup)
  scheduled.response_scheduling = types.FunctionResponseScheduling.SILENT

  _, ended = await _run_live_turn(
      [scheduled, task_completed], calls=['slow_lookup', 'task_completed']
  )

  assert ended
  assert cancelled.is_set()
