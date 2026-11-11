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

"""Tests for thread pool execution of tools in Live API mode."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import gc
import threading
import time
from typing import Any
from unittest import mock

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import ToolThreadPoolConfig
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.flows.llm_flows import functions
from google.adk.flows.llm_flows.functions import _call_tool_in_thread_pool
from google.adk.flows.llm_flows.functions import _get_tool_thread_pool
from google.adk.flows.llm_flows.functions import _is_sync_tool
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import InMemoryRunner
from google.adk.tools._node_tool import NodeTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.set_model_response_tool import SetModelResponseTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.adk.utils._sync_runner import _SYNC_CALLABLE_RUNNER
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._function_node import FunctionNode
from google.adk.workflow._workflow import Workflow
from google.genai import types
from pydantic import BaseModel
import pytest

from .... import testing_utils


@pytest.fixture(autouse=True)
def cleanup_thread_pools():
  yield
  # Shutdown all pools
  for pools in list(functions._TOOL_THREAD_POOLS.values()):
    for pool in pools.values():
      pool.shutdown(wait=False)
  functions._TOOL_THREAD_POOLS.clear()


async def _wait_until(predicate, timeout: float = 5.0) -> None:
  """Waits for a condition set by a background thread."""
  deadline = time.time() + timeout
  while not predicate():
    assert time.time() < deadline, 'timed out waiting for background threads'
    await asyncio.sleep(0.01)


class TestIsSyncTool:
  """Tests for the _is_sync_tool helper function."""

  def test_sync_function_is_sync(self):
    """Test that a synchronous function is detected as sync."""

    def sync_func(x: int) -> int:
      return x + 1

    tool = FunctionTool(sync_func)
    assert _is_sync_tool(tool) is True

  def test_async_function_is_not_sync(self):
    """Test that an async function is detected as not sync."""

    async def async_func(x: int) -> int:
      return x + 1

    tool = FunctionTool(async_func)
    assert _is_sync_tool(tool) is False

  def test_async_generator_is_not_sync(self):
    """Test that an async generator function is detected as not sync."""

    async def async_gen_func(x: int):
      yield x + 1

    tool = FunctionTool(async_gen_func)
    assert _is_sync_tool(tool) is False

  def test_tool_without_func_returns_false(self):
    """Test that a tool without func attribute returns False."""
    tool = BaseTool(name='test', description='test tool')
    assert _is_sync_tool(tool) is False

  def test_sync_function_node_tool_is_sync(self):
    """NodeTool wrapping a sync FunctionNode is detected as sync."""

    def sync_fn(x: int) -> int:
      return x + 1

    tool = NodeTool(node=FunctionNode(func=sync_fn, name='sync_fn'))
    assert _is_sync_tool(tool) is True

  def test_sync_generator_function_node_tool_is_sync(self):
    """NodeTool wrapping a sync generator FunctionNode is detected as sync."""

    def sync_gen_fn(x: int):
      yield x + 1

    tool = NodeTool(node=FunctionNode(func=sync_gen_fn, name='sync_gen_fn'))
    assert _is_sync_tool(tool) is True

  def test_async_function_node_tool_is_not_sync(self):
    """NodeTool wrapping an async FunctionNode is detected as not sync."""

    async def async_fn(x: int) -> int:
      return x + 1

    tool = NodeTool(node=FunctionNode(func=async_fn, name='async_fn'))
    assert _is_sync_tool(tool) is False

  def test_async_generator_function_node_tool_is_not_sync(self):
    """NodeTool wrapping an async generator FunctionNode is not sync."""

    async def async_gen_fn(x: int):
      yield x + 1

    tool = NodeTool(node=FunctionNode(func=async_gen_fn, name='async_gen_fn'))
    assert _is_sync_tool(tool) is False

  def test_workflow_node_tool_is_sync(self):
    """NodeTool wrapping a Workflow is detected as sync."""

    class _WfInput(BaseModel):
      query: str

    def step_fn() -> str:
      return 'done'

    wf = Workflow(
        name='wf_tool', input_schema=_WfInput, edges=[(START, step_fn)]
    )
    tool = NodeTool(node=wf)
    assert _is_sync_tool(tool) is True


class TestGetToolThreadPool:
  """Tests for the _get_tool_thread_pool function."""

  @pytest.mark.asyncio
  async def test_returns_thread_pool_executor(self):
    """Test that the function returns a ThreadPoolExecutor."""
    pool = _get_tool_thread_pool()
    assert isinstance(pool, ThreadPoolExecutor)

  @pytest.mark.asyncio
  async def test_returns_same_pool_on_multiple_calls(self):
    """Test that the same pool is returned on multiple calls (singleton)."""
    pool1 = _get_tool_thread_pool()
    pool2 = _get_tool_thread_pool()
    assert pool1 is pool2

  @pytest.mark.asyncio
  async def test_different_max_workers_creates_different_pools(self):
    """Test that different max_workers values create separate pools."""
    pool_4 = _get_tool_thread_pool(max_workers=4)
    pool_8 = _get_tool_thread_pool(max_workers=8)
    assert pool_4 is not pool_8

  @pytest.mark.asyncio
  async def test_same_max_workers_returns_same_pool(self):
    """Test that same max_workers returns the cached pool."""
    pool1 = _get_tool_thread_pool(max_workers=16)
    pool2 = _get_tool_thread_pool(max_workers=16)
    assert pool1 is pool2

  @pytest.mark.asyncio
  async def test_pool_is_isolated_from_the_loop_default_executor(self):
    """Tool work must not share the executor the loop uses for its own work."""
    loop = asyncio.get_running_loop()

    tool_thread = await loop.run_in_executor(
        _get_tool_thread_pool(), lambda: threading.current_thread().name
    )
    default_thread = await asyncio.to_thread(
        lambda: threading.current_thread().name
    )

    assert tool_thread.startswith('adk_tool_executor')
    assert not default_thread.startswith('adk_tool_executor')

  def test_separate_event_loops_get_separate_pools(self):
    """Each event loop owns its pool rather than a process-wide one."""

    async def get_pool() -> ThreadPoolExecutor:
      return _get_tool_thread_pool(max_workers=3)

    assert asyncio.run(get_pool()) is not asyncio.run(get_pool())

  def test_pool_is_shut_down_when_its_event_loop_is_gone(self):
    """Regression test: idle tool threads must not outlive their loop."""

    pre_existing = {
        thread
        for thread in threading.enumerate()
        if thread.name.startswith('adk_tool_executor')
    }

    def tool_threads() -> set:
      return {
          thread
          for thread in threading.enumerate()
          if thread.name.startswith('adk_tool_executor')
      } - pre_existing

    async def run_tool_and_return_pool() -> ThreadPoolExecutor:
      def sync_func() -> dict:
        return {'result': 'success'}

      tool = FunctionTool(sync_func)
      model = testing_utils.MockModel.create(responses=[])
      agent = Agent(name='test_agent', model=model, tools=[tool])
      invocation_context = await testing_utils.create_invocation_context(
          agent=agent, user_content=''
      )
      tool_context = ToolContext(
          invocation_context=invocation_context,
          function_call_id='test_id',
      )
      await _call_tool_in_thread_pool(tool, {}, tool_context)
      assert tool_threads()
      return _get_tool_thread_pool()

    pool = asyncio.run(run_tool_and_return_pool())
    gc.collect()

    with pytest.raises(RuntimeError):
      pool.submit(lambda: None)
    deadline = time.time() + 5
    while tool_threads() and time.time() < deadline:
      time.sleep(0.01)
    assert not tool_threads()


class TestCallToolInThreadPool:
  """Tests for the _call_tool_in_thread_pool function."""

  @pytest.mark.asyncio
  async def test_sync_tool_runs_in_thread_pool(self):
    """Test that sync tools run in a separate thread."""
    main_thread_id = threading.current_thread().ident
    tool_thread_id = None

    def sync_func() -> dict:
      nonlocal tool_thread_id
      tool_thread_id = threading.current_thread().ident
      return {'result': 'success'}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == {'result': 'success'}
    assert tool_thread_id is not None
    assert tool_thread_id != main_thread_id

  @pytest.mark.asyncio
  async def test_sync_tool_preserves_function_tool_confirmation(self):
    calls: list[str] = []

    def write(value: str) -> dict[str, str]:
      calls.append(value)
      return {'value': value}

    tool = FunctionTool(write, require_confirmation=True)
    agent = Agent(
        name='test_agent',
        model=testing_utils.MockModel.create(responses=[]),
        tools=[tool],
    )
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(
        tool, {'value': 'write'}, tool_context
    )

    assert result['error'].startswith('This tool call requires confirmation')
    assert calls == []
    assert 'test_id' in tool_context.actions.requested_tool_confirmations

  @pytest.mark.asyncio
  async def test_sync_tool_runs_in_thread_pool_once_confirmed(self):
    main_thread_id = threading.current_thread().ident
    tool_thread_id = None
    calls: list[str] = []

    def write(value: str) -> dict[str, str]:
      nonlocal tool_thread_id
      tool_thread_id = threading.current_thread().ident
      calls.append(value)
      return {'value': value}

    tool = FunctionTool(write, require_confirmation=True)
    agent = Agent(
        name='test_agent',
        model=testing_utils.MockModel.create(responses=[]),
        tools=[tool],
    )
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
        tool_confirmation=ToolConfirmation(confirmed=True),
    )

    result = await _call_tool_in_thread_pool(
        tool, {'value': 'write'}, tool_context
    )

    assert result == {'value': 'write'}
    assert calls == ['write']
    assert tool_thread_id is not None
    assert tool_thread_id != main_thread_id

  @pytest.mark.asyncio
  async def test_sync_tool_does_not_leak_runner_into_nested_function_tool(self):
    inner_tool = FunctionTool(lambda: 'nested result')
    agent = Agent(
        name='test_agent',
        model=testing_utils.MockModel.create(responses=[]),
        tools=[inner_tool],
    )
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    def outer() -> str:
      return asyncio.run(
          inner_tool.run_async(args={}, tool_context=tool_context)
      )

    result = await _call_tool_in_thread_pool(
        FunctionTool(outer), {}, tool_context
    )

    assert result == 'nested result'

  @pytest.mark.asyncio
  async def test_async_tool_runs_in_thread_pool(self):
    """Test that async tools run in a separate thread with new event loop."""
    main_thread_id = threading.current_thread().ident
    tool_thread_id = None

    async def async_func() -> dict:
      nonlocal tool_thread_id
      tool_thread_id = threading.current_thread().ident
      return {'result': 'async_success'}

    tool = FunctionTool(async_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == {'result': 'async_success'}
    assert tool_thread_id is not None
    assert tool_thread_id != main_thread_id

  @pytest.mark.asyncio
  async def test_sync_tool_with_args(self):
    """Test that sync tools receive arguments correctly."""

    def sync_func(x: int, y: str) -> dict:
      return {'sum': x, 'text': y}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(
        tool, {'x': 42, 'y': 'hello'}, tool_context
    )

    assert result == {'sum': 42, 'text': 'hello'}

  @pytest.mark.asyncio
  async def test_sync_tool_missing_mandatory_args(self):
    """Test sync tools return error dict when mandatory args are missing."""

    def sync_func(x: int, y: str) -> dict:
      return {'sum': x, 'text': y}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {'x': 42}, tool_context)

    assert 'error' in result
    assert 'mandatory input parameters are not present' in result['error']

  @pytest.mark.asyncio
  async def test_sync_tool_calling_asyncio_run(self):
    """Test that sync tools can call asyncio.run internally."""

    def sync_func_with_loop(x: int) -> dict:
      async def inner_async():
        return {'result': x * 2}

      return asyncio.run(inner_async())

    tool = FunctionTool(sync_func_with_loop)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {'x': 21}, tool_context)
    assert result == {'result': 42}

  @pytest.mark.asyncio
  async def test_async_tool_with_args(self):
    """Test that async tools receive arguments correctly."""

    async def async_func(x: int, y: str) -> dict:
      return {'sum': x, 'text': y}

    tool = FunctionTool(async_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(
        tool, {'x': 42, 'y': 'hello'}, tool_context
    )

    assert result == {'sum': 42, 'text': 'hello'}

  @pytest.mark.asyncio
  async def test_sync_tool_with_tool_context(self):
    """Test that sync tools receive tool_context when requested."""

    def sync_func_with_context(x: int, tool_context: ToolContext) -> dict:
      return {'x': x, 'has_context': tool_context is not None}

    tool = FunctionTool(sync_func_with_context)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {'x': 10}, tool_context)

    assert result == {'x': 10, 'has_context': True}

  @pytest.mark.asyncio
  async def test_blocking_io_does_not_block_event_loop(self):
    """Test that blocking I/O in thread pool doesn't block main event loop."""
    event_loop_ticks = 0

    async def ticker():
      nonlocal event_loop_ticks
      for _ in range(10):
        await asyncio.sleep(0.01)
        event_loop_ticks += 1

    def blocking_sleep() -> dict:
      time.sleep(0.15)  # Blocking sleep for 150ms
      return {'result': 'done'}

    tool = FunctionTool(blocking_sleep)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    # Run both ticker and blocking tool concurrently
    ticker_task = asyncio.create_task(ticker())
    result = await _call_tool_in_thread_pool(tool, {}, tool_context)
    await ticker_task

    assert result == {'result': 'done'}
    # Ticker should have run multiple times while tool was sleeping
    assert (
        event_loop_ticks >= 5
    ), f'Event loop should have ticked at least 5 times, got {event_loop_ticks}'

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      'return_value,use_implicit_return',
      [
          (None, True),  # implicit None (no return statement)
          (None, False),  # explicit `return None`
          (0, False),  # falsy int
          ('', False),  # falsy str
          ({}, False),  # falsy dict
          (False, False),  # falsy bool
      ],
  )
  async def test_sync_tool_falsy_return_executes_exactly_once(
      self, return_value, use_implicit_return
  ):
    """FunctionTools returning None or falsy values must execute once.

    Previously, a None return was mistaken for the internal sentinel used to
    signal 'non-FunctionTool, fall back to run_async', causing a second
    invocation. The fix uses an identity-based sentinel so that None and other
    falsy values (0, '', {}, False) are treated as valid results.
    """
    call_count = 0

    def sync_func():
      nonlocal call_count
      call_count += 1
      if not use_implicit_return:
        return return_value
      # implicit None — no return statement

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == return_value
    assert (
        call_count == 1
    ), f'Tool function executed {call_count} time(s); expected exactly 1.'

  @pytest.mark.asyncio
  async def test_sync_tool_exception_propagates(self):
    """Test that exceptions from sync tools propagate correctly."""

    def sync_func_raises() -> dict:
      raise ValueError('Test error from sync tool')

    tool = FunctionTool(sync_func_raises)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    with pytest.raises(ValueError, match='Test error from sync tool'):
      await _call_tool_in_thread_pool(tool, {}, tool_context)

  @pytest.mark.asyncio
  async def test_async_tool_exception_propagates(self):
    """Test that exceptions from async tools propagate correctly."""

    async def async_func_raises() -> dict:
      raise RuntimeError('Test error from async tool')

    tool = FunctionTool(async_func_raises)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    with pytest.raises(RuntimeError, match='Test error from async tool'):
      await _call_tool_in_thread_pool(tool, {}, tool_context)

  @pytest.mark.asyncio
  async def test_custom_max_workers_used(self):
    """Test that custom max_workers parameter is passed to thread pool."""
    tool_thread_name = None

    def sync_func() -> dict:
      nonlocal tool_thread_name
      tool_thread_name = threading.current_thread().name
      return {'result': 'success'}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    # Call with custom max_workers
    result = await _call_tool_in_thread_pool(
        tool, {}, tool_context, max_workers=12
    )

    assert result == {'result': 'success'}
    # The call ran on the dedicated pool for that worker count.
    assert tool_thread_name.startswith('adk_tool_executor')
    assert _get_tool_thread_pool(max_workers=12)._max_workers == 12

  @pytest.mark.asyncio
  async def test_max_workers_bounds_concurrent_calls(self):
    """Only max_workers background tool calls run at once."""
    lock = threading.Lock()
    release = threading.Event()
    running = 0
    peak = 0

    def sync_func() -> dict:
      nonlocal running, peak
      with lock:
        running += 1
        peak = max(peak, running)
      release.wait(timeout=5)
      with lock:
        running -= 1
      return {'result': 'success'}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    calls = [
        asyncio.create_task(
            _call_tool_in_thread_pool(tool, {}, tool_context, max_workers=2)
        )
        for _ in range(4)
    ]
    try:
      await _wait_until(lambda: running == 2)
      await asyncio.sleep(0.05)
      assert peak == 2
    finally:
      release.set()

    assert await asyncio.gather(*calls) == [{'result': 'success'}] * 4
    assert peak == 2

  @pytest.mark.asyncio
  async def test_concurrent_invocations_share_the_loop_pool(self):
    """Extra invocations must not each add their own worker threads."""
    lock = threading.Lock()
    release = threading.Event()
    thread_names = set()

    def sync_func() -> dict:
      with lock:
        thread_names.add(threading.current_thread().name)
      release.wait(timeout=5)
      return {'result': 'success'}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    tool_contexts = []
    for index in range(2):
      invocation_context = await testing_utils.create_invocation_context(
          agent=agent, user_content=''
      )
      tool_contexts.append(
          ToolContext(
              invocation_context=invocation_context,
              function_call_id=f'test_id_{index}',
          )
      )

    calls = [
        asyncio.create_task(
            _call_tool_in_thread_pool(tool, {}, tool_context, max_workers=2)
        )
        for tool_context in tool_contexts
        for _ in range(2)
    ]
    try:
      await _wait_until(lambda: len(thread_names) == 2)
      await asyncio.sleep(0.05)
      assert len(thread_names) == 2
    finally:
      release.set()

    assert await asyncio.gather(*calls) == [{'result': 'success'}] * 4
    assert len(thread_names) == 2

  @pytest.mark.asyncio
  async def test_failed_call_frees_its_worker_thread(self):
    """A raising tool must not permanently consume a worker thread."""

    def sync_func_raises() -> dict:
      raise ValueError('Test error from sync tool')

    tool = FunctionTool(sync_func_raises)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    for _ in range(3):
      with pytest.raises(ValueError, match='Test error from sync tool'):
        await asyncio.wait_for(
            _call_tool_in_thread_pool(tool, {}, tool_context, max_workers=1),
            timeout=5,
        )

  @pytest.mark.asyncio
  async def test_cancelled_call_holds_its_worker_thread_until_it_returns(self):
    """Python cannot stop a running thread, so it keeps its worker."""
    lock = threading.Lock()
    first_worker_started = threading.Event()
    finish_first_worker = threading.Event()
    second_worker_started = threading.Event()
    call_count = 0

    def sync_func() -> dict:
      nonlocal call_count
      with lock:
        call_count += 1
        is_first = call_count == 1
      if is_first:
        first_worker_started.set()
        finish_first_worker.wait(timeout=5)
      else:
        second_worker_started.set()
      return {'result': 'success'}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    first_call = asyncio.create_task(
        _call_tool_in_thread_pool(tool, {}, tool_context, max_workers=1)
    )
    try:
      await _wait_until(first_worker_started.is_set)
      first_call.cancel()
      with pytest.raises(asyncio.CancelledError):
        await first_call

      second_call = asyncio.create_task(
          _call_tool_in_thread_pool(tool, {}, tool_context, max_workers=1)
      )
      await asyncio.sleep(0.05)
      assert not second_worker_started.is_set()
    finally:
      finish_first_worker.set()

    assert await asyncio.wait_for(second_call, timeout=5) == {
        'result': 'success'
    }
    assert second_worker_started.is_set()

  @pytest.mark.asyncio
  async def test_cancelled_call_that_never_started_does_not_run(self):
    """A call still queued for a worker is dropped rather than run later."""
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    calls = []

    def blocking_func() -> dict:
      blocker_started.set()
      release_blocker.wait(timeout=5)
      return {'result': 'success'}

    def queued_func() -> dict:
      calls.append('queued')
      return {'result': 'success'}

    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    blocking_call = asyncio.create_task(
        _call_tool_in_thread_pool(
            FunctionTool(blocking_func), {}, tool_context, max_workers=1
        )
    )
    try:
      await _wait_until(blocker_started.is_set)
      queued_call = asyncio.create_task(
          _call_tool_in_thread_pool(
              FunctionTool(queued_func), {}, tool_context, max_workers=1
          )
      )
      await asyncio.sleep(0.05)
      queued_call.cancel()
      with pytest.raises(asyncio.CancelledError):
        await queued_call
    finally:
      release_blocker.set()

    await asyncio.wait_for(blocking_call, timeout=5)
    await asyncio.sleep(0.1)
    assert not calls

  @pytest.mark.asyncio
  async def test_contextvars_propagation_sync_tool(self):
    """Test that contextvars propagate to sync tools in thread pool."""
    test_var = contextvars.ContextVar('test_var', default='default')
    test_var.set('main_thread_value')

    def sync_func() -> dict[str, str]:
      return {'value': test_var.get()}

    tool = FunctionTool(sync_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == {'value': 'main_thread_value'}

  @pytest.mark.asyncio
  async def test_contextvars_propagation_async_tool(self):
    """Test that contextvars propagate to async tools in thread pool."""
    test_var = contextvars.ContextVar('test_var', default='default')
    test_var.set('main_thread_value')

    async def async_func() -> dict[str, str]:
      return {'value': test_var.get()}

    tool = FunctionTool(async_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == {'value': 'main_thread_value'}

  @pytest.mark.asyncio
  async def test_sync_tool_returning_none_runs_exactly_once(self):
    """Regression test for double invocation of a None-returning sync tool.

    A sync FunctionTool whose underlying function returns None must not
    be re-invoked through the run_async fallback path.
    """
    call_count = 0

    def side_effect_only_func() -> None:
      nonlocal call_count
      call_count += 1

    tool = FunctionTool(side_effect_only_func)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result is None
    assert call_count == 1

  @pytest.mark.asyncio
  async def test_non_function_tool_sync_falls_back_to_run_async(self):
    """Sync tools that aren't FunctionTool subclasses go through run_async.

    Covers the fall-through path used by tools like SetModelResponseTool
    that have a sync ``func`` attribute but aren't FunctionTool instances.
    """
    run_async_call_count = 0

    class _SyncNonFunctionTool(BaseTool):

      def __init__(self):
        super().__init__(name='custom_tool', description='desc')
        # Sync attribute so _is_sync_tool returns True.
        self.func = lambda: 'unused'

      async def run_async(self, *, args, tool_context):
        nonlocal run_async_call_count
        run_async_call_count += 1
        return {'via': 'run_async'}

    tool = _SyncNonFunctionTool()
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(tool, {}, tool_context)

    assert result == {'via': 'run_async'}
    assert run_async_call_count == 1

  @pytest.mark.asyncio
  async def test_set_model_response_tool_falls_back_to_run_async(self):
    """SetModelResponseTool — the real-world non-FunctionTool sync tool."""

    class _Schema(BaseModel):
      answer: str

    tool = SetModelResponseTool(output_schema=_Schema)
    # Precondition: this is the code path the bug report referenced.
    assert _is_sync_tool(tool)

    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='test_id',
    )

    result = await _call_tool_in_thread_pool(
        tool, {'answer': 'hello'}, tool_context
    )

    assert result == {'answer': 'hello'}


class TestToolThreadPoolConfig:
  """Tests for the tool_thread_pool_config in RunConfig."""

  def test_default_is_none(self):
    """Test that tool_thread_pool_config defaults to None."""
    config = RunConfig()
    assert config.tool_thread_pool_config is None

  def test_can_be_set_with_defaults(self):
    """Test that tool_thread_pool_config can be set with default values."""
    config = RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    assert config.tool_thread_pool_config is not None
    assert config.tool_thread_pool_config.max_workers == 4

  def test_can_set_custom_max_workers(self):
    """Test that max_workers can be customized."""
    config = RunConfig(
        tool_thread_pool_config=ToolThreadPoolConfig(max_workers=8)
    )
    assert config.tool_thread_pool_config.max_workers == 8

  def test_max_workers_must_be_positive(self):
    """Test that max_workers must be >= 1."""
    with pytest.raises(ValueError):
      ToolThreadPoolConfig(max_workers=0)

  def test_max_workers_rejects_negative(self):
    """Test that negative max_workers is rejected."""
    with pytest.raises(ValueError):
      ToolThreadPoolConfig(max_workers=-1)


class TestNodeAndGeneratorThreadPoolExecution:
  """Tests for NodeTool and generator tools with the thread pool."""

  @pytest.mark.asyncio
  async def test_sync_node_tool_runs_callable_on_thread_pool(self):
    """NodeTool wrapping a sync FunctionNode runs on adk_tool_executor."""
    worker_thread_name = None

    def sync_node_fn(x: int) -> dict[str, int]:
      nonlocal worker_thread_name
      worker_thread_name = threading.current_thread().name
      return {'doubled': x * 2}

    node_tool = NodeTool(node=FunctionNode(func=sync_node_fn, name='double_it'))
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[node_tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    invocation_context._enqueue_event = mock.AsyncMock()
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='call_node_1',
    )

    result = await _call_tool_in_thread_pool(node_tool, {'x': 21}, tool_context)

    assert result == {'doubled': 42}
    assert worker_thread_name is not None
    assert worker_thread_name.startswith('adk_tool_executor')

  @pytest.mark.asyncio
  async def test_sync_generator_node_tool_runs_iterations_on_thread_pool(self):
    """Sync generator FunctionNode steps run on one dedicated thread."""
    worker_threads: list[str] = []

    def sync_gen_fn(x: int):
      worker_threads.append(threading.current_thread().name)
      yield Event(message='progress 1')
      worker_threads.append(threading.current_thread().name)
      yield {'final': x + 10}

    node_tool = NodeTool(node=FunctionNode(func=sync_gen_fn, name='gen_it'))
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[node_tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    invocation_context._enqueue_event = mock.AsyncMock()
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='call_gen_1',
    )

    result = await _call_tool_in_thread_pool(node_tool, {'x': 5}, tool_context)

    assert result == {'final': 15}
    assert len(worker_threads) == 2
    assert len(set(worker_threads)) == 1
    assert worker_threads[0].startswith(
        'adk_sync_generator_for_adk_tool_executor'
    )

  @pytest.mark.asyncio
  async def test_async_node_tool_runs_on_caller_loop_without_pool(self):
    """NodeTool wrapping an async node runs on the caller loop without pool."""
    caller_loop = asyncio.get_running_loop()
    enqueued_loops: list[asyncio.AbstractEventLoop] = []
    runners_seen: list[object] = []

    async def async_node_fn(x: int):
      runners_seen.append(_SYNC_CALLABLE_RUNNER.get())
      yield Event(message='progress')
      yield {'doubled': x * 2}

    node_tool = NodeTool(
        node=FunctionNode(func=async_node_fn, name='double_async')
    )
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[node_tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )

    async def record_enqueue(event: Event) -> None:
      del event
      enqueued_loops.append(asyncio.get_running_loop())

    object.__setattr__(invocation_context, '_enqueue_event', record_enqueue)
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='call_async_node_1',
    )

    result = await _call_tool_in_thread_pool(node_tool, {'x': 4}, tool_context)

    assert result == {'doubled': 8}
    assert runners_seen == [None]
    assert enqueued_loops
    assert all(loop is caller_loop for loop in enqueued_loops)

  @pytest.mark.asyncio
  async def test_sync_node_tool_runs_on_thread_pool_via_prepared_call_async(
      self,
  ):
    """Prepared tool call runs a sync FunctionNode on adk_tool_executor."""
    worker_thread_name = None

    def sync_node_fn(x: int) -> dict[str, int]:
      nonlocal worker_thread_name
      worker_thread_name = threading.current_thread().name
      return {'doubled': x * 2}

    node_tool = NodeTool(node=FunctionNode(func=sync_node_fn, name='double_it'))
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='double_it', args={'x': 21}),
            'done',
        ]
    )
    agent = Agent(name='test_agent', model=mock_model, tools=[node_tool])
    runner = testing_utils.TestInMemoryRunner(agent)

    events = await runner.run_async_with_new_session(
        'test', RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    )

    assert worker_thread_name is not None
    assert worker_thread_name.startswith('adk_tool_executor')
    fn_responses = [
        part.function_response
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response is not None
    ]
    assert len(fn_responses) == 1
    assert fn_responses[0].response == {'doubled': 42}

  @pytest.mark.asyncio
  async def test_sync_gen_node_tool_runs_on_pool_via_prepared_call_async(self):
    """Prepared tool call runs a sync generator node on its dedicated thread."""
    worker_threads: list[str] = []

    def sync_gen_fn(x: int):
      worker_threads.append(threading.current_thread().name)
      yield Event(message='progress 1')
      worker_threads.append(threading.current_thread().name)
      yield {'final': x + 10}

    node_tool = NodeTool(node=FunctionNode(func=sync_gen_fn, name='gen_it'))
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='gen_it', args={'x': 5}),
            'done',
        ]
    )
    agent = Agent(name='test_agent', model=mock_model, tools=[node_tool])
    runner = testing_utils.TestInMemoryRunner(agent)

    events = await runner.run_async_with_new_session(
        'test', RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    )

    assert len(worker_threads) == 2
    assert len(set(worker_threads)) == 1
    assert worker_threads[0].startswith(
        'adk_sync_generator_for_adk_tool_executor'
    )
    fn_responses = [
        part.function_response
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response is not None
    ]
    assert len(fn_responses) == 1
    assert fn_responses[0].response == {'final': 15}

  @pytest.mark.asyncio
  async def test_async_node_tool_runs_on_caller_loop_via_prepared_call_async(
      self,
  ):
    """Prepared tool call keeps an async FunctionNode on the caller loop."""
    caller_loop = asyncio.get_running_loop()
    seen_loops: list[asyncio.AbstractEventLoop] = []
    runners_seen: list[object] = []

    async def async_node_fn(x: int):
      seen_loops.append(asyncio.get_running_loop())
      runners_seen.append(_SYNC_CALLABLE_RUNNER.get())
      yield Event(message='progress')
      yield {'doubled': x * 2}

    node_tool = NodeTool(
        node=FunctionNode(func=async_node_fn, name='double_async')
    )
    mock_model = testing_utils.MockModel.create(
        responses=[
            types.Part.from_function_call(name='double_async', args={'x': 4}),
            'done',
        ]
    )
    agent = Agent(name='test_agent', model=mock_model, tools=[node_tool])
    runner = testing_utils.TestInMemoryRunner(agent)

    events = await runner.run_async_with_new_session(
        'test', RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    )

    assert seen_loops == [caller_loop]
    assert runners_seen == [None]
    fn_responses = [
        part.function_response
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if part.function_response is not None
    ]
    assert len(fn_responses) == 1
    assert fn_responses[0].response == {'doubled': 8}

  @pytest.mark.asyncio
  async def test_workflow_node_tool_runs_sync_nodes_on_thread_pool(self):
    """NodeTool wrapping a Workflow runs sync nodes on adk_tool_executor."""

    class _WfInput(BaseModel):
      x: int

    worker_thread_name = None

    def sync_step(node_input: _WfInput) -> dict[str, int]:
      nonlocal worker_thread_name
      worker_thread_name = threading.current_thread().name
      return {'tripled': node_input.x * 3}

    wf = Workflow(
        name='triple_wf',
        input_schema=_WfInput,
        edges=[(START, sync_step)],
    )
    node_tool = NodeTool(node=wf)
    model = testing_utils.MockModel.create(responses=[])
    agent = Agent(name='test_agent', model=model, tools=[node_tool])
    invocation_context = await testing_utils.create_invocation_context(
        agent=agent, user_content=''
    )
    invocation_context._enqueue_event = mock.AsyncMock()
    tool_context = ToolContext(
        invocation_context=invocation_context,
        function_call_id='call_wf_1',
    )

    result = await _call_tool_in_thread_pool(node_tool, {'x': 7}, tool_context)

    assert result == {'tripled': 21}
    assert worker_thread_name is not None
    assert worker_thread_name.startswith('adk_tool_executor')


class TestWorkflowDirectThreadPoolExecution:
  """Tests for tool_thread_pool_config in direct Workflow execution."""

  @pytest.mark.asyncio
  async def test_workflow_runs_sync_function_and_tool_nodes_on_thread_pool(
      self,
  ):
    """Standalone Workflow runs sync nodes and tools on adk_tool_executor."""
    caller_thread = threading.get_ident()
    barrier = threading.Barrier(2, timeout=5)
    recorded_threads: dict[str, object] = {}
    callback_threads: list[int] = []
    async_runners_seen: list[object] = []

    class _RecordingPlugin(BasePlugin):

      def __init__(self):
        super().__init__(name='recording_plugin')

      async def before_tool_callback(self, *, tool, tool_args, tool_context):
        del tool, tool_args, tool_context
        callback_threads.append(threading.get_ident())
        return None

      async def after_tool_callback(
          self, *, tool, tool_args, tool_context, result
      ):
        del tool, tool_args, tool_context, result
        callback_threads.append(threading.get_ident())
        return None

    def parallel_sync_a() -> None:
      barrier.wait()
      recorded_threads['sync_a'] = threading.current_thread().name

    def parallel_sync_b() -> None:
      barrier.wait()
      recorded_threads['sync_b'] = threading.current_thread().name

    def sync_gen_step():
      recorded_threads['sync_gen'] = threading.current_thread().name
      yield Event(message='gen_progress')

    def sync_tool_fn() -> dict[str, str]:
      recorded_threads['sync_tool'] = threading.current_thread().name
      return {'tool': 'ok'}

    async def async_step() -> None:
      recorded_threads['async_step'] = threading.get_ident()
      async_runners_seen.append(_SYNC_CALLABLE_RUNNER.get())

    async def async_tool_fn() -> None:
      recorded_threads['async_tool'] = threading.get_ident()
      async_runners_seen.append(_SYNC_CALLABLE_RUNNER.get())

    branches = [
        parallel_sync_a,
        parallel_sync_b,
        sync_gen_step,
        FunctionTool(sync_tool_fn),
        async_step,
        FunctionTool(async_tool_fn),
    ]
    wf = Workflow(
        name='direct_pool_wf',
        edges=[(START, branches)],
    )
    runner = InMemoryRunner(
        app=App(
            name='direct_pool_app',
            root_agent=wf,
            plugins=[_RecordingPlugin()],
        )
    )

    await runner.run_debug(
        '{}',
        run_config=RunConfig(
            tool_thread_pool_config=ToolThreadPoolConfig(max_workers=2)
        ),
        quiet=True,
    )

    assert recorded_threads['sync_a'].startswith('adk_tool_executor')
    assert recorded_threads['sync_b'].startswith('adk_tool_executor')
    assert recorded_threads['sync_gen'].startswith(
        'adk_sync_generator_for_adk_tool_executor'
    )
    assert recorded_threads['sync_tool'].startswith('adk_tool_executor')
    assert recorded_threads['async_step'] == caller_thread
    assert recorded_threads['async_tool'] == caller_thread
    assert async_runners_seen == [None, None]
    assert callback_threads == [caller_thread] * 4

  @pytest.mark.asyncio
  async def test_confirmation_predicates_run_on_pool_in_workflow_and_agent(
      self,
  ):
    """Sync require_confirmation runs on adk_tool_executor in all paths."""
    predicate_threads: dict[str, str] = {}

    def node_pred() -> bool:
      predicate_threads['fn_node'] = threading.current_thread().name
      return True

    @node(require_confirmation=node_pred)
    def gated_node() -> str:
      return 'ran'

    def tool_node_pred() -> bool:
      predicate_threads['tool_node'] = threading.current_thread().name
      return True

    def agent_tool_pred() -> bool:
      predicate_threads['agent_tool'] = threading.current_thread().name
      return True

    def tool_node_op() -> dict[str, str]:
      return {'status': 'ok'}

    def guarded_op() -> dict[str, str]:
      return {'status': 'ok'}

    wf = Workflow(
        name='confirm_wf',
        edges=[(
            START,
            [
                gated_node,
                FunctionTool(tool_node_op, require_confirmation=tool_node_pred),
            ],
        )],
    )
    wf_runner = InMemoryRunner(agent=wf)
    await wf_runner.run_debug(
        '{}',
        run_config=RunConfig(tool_thread_pool_config=ToolThreadPoolConfig()),
        quiet=True,
    )

    agent = Agent(
        name='confirm_agent',
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(name='guarded_op', args={}),
                'paused',
            ]
        ),
        tools=[
            FunctionTool(
                func=guarded_op,
                require_confirmation=agent_tool_pred,
            )
        ],
    )
    agent_runner = testing_utils.TestInMemoryRunner(agent)
    await agent_runner.run_async_with_new_session(
        'check', RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    )

    assert predicate_threads['fn_node'].startswith('adk_tool_executor')
    assert predicate_threads['tool_node'].startswith('adk_tool_executor')
    assert predicate_threads['agent_tool'].startswith('adk_tool_executor')

  @pytest.mark.asyncio
  async def test_custom_run_async_override_uses_pool_in_agent_and_workflow(
      self,
  ):
    """Custom FunctionTool.run_async overrides use pool in Agent & Workflow."""
    seen_threads: dict[str, str] = {}

    def sync_helper(label: str) -> dict[str, str]:
      seen_threads[label] = threading.current_thread().name
      return {'ran': label}

    class _CustomRunAsyncTool(FunctionTool):

      def __init__(self, label: str):
        async def placeholder() -> dict[str, str]:
          return {}

        placeholder.__name__ = f'custom_tool_{label}'
        super().__init__(placeholder)
        self._label = label

      async def run_async(
          self, *, args: dict[str, Any], tool_context: ToolContext
      ) -> Any:
        del args, tool_context
        return await self._invoke_callable(sync_helper, {'label': self._label})

    agent = Agent(
        name='custom_tool_agent',
        model=testing_utils.MockModel.create(
            responses=[
                types.Part.from_function_call(
                    name='custom_tool_agent', args={}
                ),
                'done',
            ]
        ),
        tools=[_CustomRunAsyncTool('agent')],
    )
    agent_runner = testing_utils.TestInMemoryRunner(agent)
    await agent_runner.run_async_with_new_session(
        'run', RunConfig(tool_thread_pool_config=ToolThreadPoolConfig())
    )

    wf = Workflow(
        name='custom_tool_wf',
        edges=[(START, _CustomRunAsyncTool('workflow'))],
    )
    wf_runner = InMemoryRunner(agent=wf)
    await wf_runner.run_debug(
        '{}',
        run_config=RunConfig(tool_thread_pool_config=ToolThreadPoolConfig()),
        quiet=True,
    )

    assert seen_threads['agent'].startswith('adk_tool_executor')
    assert seen_threads['workflow'].startswith('adk_tool_executor')

  @pytest.mark.asyncio
  async def test_workflow_runs_sync_nodes_inline_when_pool_unconfigured(self):
    """Sync nodes and tools stay inline when tool_thread_pool_config is None."""
    caller_thread = threading.get_ident()
    seen_threads: list[int] = []

    def sync_step() -> None:
      seen_threads.append(threading.get_ident())

    def sync_tool() -> dict[str, str]:
      seen_threads.append(threading.get_ident())
      return {'ok': '1'}

    wf = Workflow(
        name='inline_wf',
        edges=[(START, sync_step, FunctionTool(sync_tool))],
    )
    runner = InMemoryRunner(agent=wf)
    await runner.run_debug('{}', run_config=RunConfig(), quiet=True)

    assert seen_threads == [caller_thread, caller_thread]
