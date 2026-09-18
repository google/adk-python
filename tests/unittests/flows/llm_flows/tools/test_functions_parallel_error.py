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

import asyncio
from contextlib import aclosing
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import StreamingMode
from google.adk.flows.llm_flows import functions
from google.adk.live import LiveRequestQueue
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .... import testing_utils


def function_call(function_call_id, name, args: dict[str, Any]) -> types.Part:
  part = types.Part.from_function_call(name=name, args=args)
  part.function_call.id = function_call_id
  return part


@pytest.mark.asyncio
async def test_parallel_function_call_error_fail_fast():
  id_1 = 'id_1'
  id_2 = 'id_2'
  responses = [
      [
          function_call(id_1, 'fail_tool', {}),
          function_call(id_2, 'sleep_tool', {}),
      ],
      [
          types.Part.from_text(text='final response'),
      ],
  ]

  mock_model = testing_utils.MockModel.create(responses=responses)

  fail_called = False
  sleep_started = False
  sleep_completed = False
  sleep_cancelled = False

  async def fail_tool(tool_context: ToolContext) -> str:
    nonlocal fail_called
    fail_called = True
    raise ValueError('Tool failed intentionally')

  async def sleep_tool(tool_context: ToolContext) -> str:
    nonlocal sleep_started, sleep_completed, sleep_cancelled
    sleep_started = True
    try:
      await asyncio.sleep(10)  # Sleep long enough to be cancelled
      sleep_completed = True
      return 'Tool succeeded'
    except asyncio.CancelledError:
      sleep_cancelled = True
      raise

  agent = Agent(
      name='root_agent',
      model=mock_model,
      tools=[fail_tool, sleep_tool],
  )

  runner = testing_utils.InMemoryRunner(agent)

  with pytest.raises(ValueError, match='Tool failed intentionally'):
    await runner.run_async(
        new_message=types.Content(parts=[types.Part(text='test')]),
    )

  assert fail_called
  assert sleep_started
  assert not sleep_completed
  assert sleep_cancelled


def _parallel_tool_responses() -> list[list[types.Part]]:
  return [
      [
          function_call('id_1', 'stopper', {}),
          function_call('id_2', 'sibling', {}),
      ],
      [
          types.Part.from_text(text='final response'),
      ],
  ]


async def _run_parallel_child_cancel(
    stopper,
    sibling,
    execution: str = 'run_async',
) -> None:
  agent = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(
          responses=_parallel_tool_responses()
      ),
      tools=[stopper, sibling],
  )
  runner = testing_utils.InMemoryRunner(agent)
  session = runner.session
  user_message = types.Content(
      role='user', parts=[types.Part(text='test')]
  )
  # Runner swallows CancelledError at root-task cleanup so the caller is
  # not itself cancelled; the invariant is that sibling tools are torn down
  # before that iterator returns.
  if execution == 'live':
    live_queue = LiveRequestQueue()
    live_queue.send_content(user_message)
    live_queue.close()

    async def _consume_live() -> None:
      async with aclosing(
          runner.runner.run_live(
              user_id=session.user_id,
              session_id=session.id,
              live_request_queue=live_queue,
              run_config=RunConfig(response_modalities=['TEXT']),
          )
      ) as agen:
        async for _ in agen:
          pass

    await asyncio.wait_for(_consume_live(), timeout=10)
    return

  streaming_mode = (
      StreamingMode.SSE if execution == 'sse' else StreamingMode.NONE
  )
  async with aclosing(
      runner.runner.run_async(
          user_id=session.user_id,
          session_id=session.id,
          new_message=user_message,
          run_config=RunConfig(streaming_mode=streaming_mode),
      )
  ) as agen:
    async for _ in agen:
      pass


_EXECUTIONS = [
    pytest.param('run_async', id='run-async'),
    pytest.param('sse', id='run-async-sse'),
    pytest.param('live', id='run-live'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('execution', _EXECUTIONS)
async def test_parallel_function_call_cancels_siblings_on_cancelled_error(
    execution: str,
):
  """A tool that raises CancelledError cancels unfinished sibling tools."""
  started = asyncio.Event()
  release = asyncio.Event()
  sibling_task = None
  sibling_completed = False
  sibling_cancelled = False

  async def stopper(tool_context: ToolContext) -> str:
    await started.wait()
    raise asyncio.CancelledError()

  async def sibling(tool_context: ToolContext) -> str:
    nonlocal sibling_task, sibling_completed, sibling_cancelled
    sibling_task = asyncio.current_task()
    started.set()
    try:
      await release.wait()
      sibling_completed = True
      return 'ok'
    except asyncio.CancelledError:
      sibling_cancelled = True
      raise

  await _run_parallel_child_cancel(stopper, sibling, execution=execution)

  pending = sibling_task is not None and not sibling_task.done()
  release.set()
  if sibling_task is not None:
    await asyncio.gather(sibling_task, return_exceptions=True)

  assert sibling_task is not None
  assert sibling_cancelled
  assert not sibling_completed
  assert not pending


@pytest.mark.asyncio
@pytest.mark.parametrize('execution', _EXECUTIONS)
async def test_parallel_function_call_cancels_siblings_when_tool_cancels_itself(
    execution: str,
):
  """A tool that cancels its own task cancels unfinished sibling tools."""
  started = asyncio.Event()
  release = asyncio.Event()
  sibling_task = None
  sibling_completed = False
  sibling_cancelled = False

  async def stopper(tool_context: ToolContext) -> str:
    await started.wait()
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    await asyncio.sleep(0)
    return 'should not reach'

  async def sibling(tool_context: ToolContext) -> str:
    nonlocal sibling_task, sibling_completed, sibling_cancelled
    sibling_task = asyncio.current_task()
    started.set()
    try:
      await release.wait()
      sibling_completed = True
      return 'ok'
    except asyncio.CancelledError:
      sibling_cancelled = True
      raise

  await _run_parallel_child_cancel(stopper, sibling, execution=execution)

  pending = sibling_task is not None and not sibling_task.done()
  release.set()
  if sibling_task is not None:
    await asyncio.gather(sibling_task, return_exceptions=True)

  assert sibling_task is not None
  assert sibling_cancelled
  assert not sibling_completed
  assert not pending
