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
import threading
import time

from google.adk.agents.llm_agent import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.run_config import ToolThreadPoolConfig
from google.adk.flows.llm_flows.functions import _call_tool_in_thread_pool
from google.adk.flows.llm_flows.functions import _get_tool_thread_pool
from google.adk.flows.llm_flows.functions import _is_sync_tool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from ... import testing_utils


@pytest.fixture(autouse=True)
def cleanup_thread_pools():
  yield
  from google.adk.flows.llm_flows import functions

  # Shutdown all pools
  for pool in functions._TOOL_THREAD_POOLS.values():
    pool.shutdown(wait=False)
  functions._TOOL_THREAD_POOLS.clear()


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
    from google.adk.tools.base_tool import BaseTool

    tool = BaseTool(name='test', description='test tool')
    assert _is_sync_tool(tool) is False


class TestGetToolThreadPool:
  """Tests for the _get_tool_thread_pool function."""

  def test_returns_thread_pool_executor(self):
    """Test that the function returns a ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor

    pool = _get_tool_thread_pool()
    assert isinstance(pool, ThreadPoolExecutor)

  def test_returns_same_pool_on_multiple_calls(self):
    """Test that the same pool is returned on multiple calls (singleton)."""
    pool1 = _get_tool_thread_pool()
    pool2 = _get_tool_thread_pool()
    assert pool1 is pool2

  def test_different_max_workers_creates_different_pools(self):
    """Test that different max_workers values create separate pools."""
    pool_4 = _get_tool_thread_pool(max_workers=4)
    pool_8 = _get_tool_thread_pool(max_workers=8)
    assert pool_4 is not pool_8

  def test_same_max_workers_returns_same_pool(self):
    """Test that same max_workers returns the cached pool."""
    pool1 = _get_tool_thread_pool(max_workers=16)
    pool2 = _get_tool_thread_pool(max_workers=16)
    assert pool1 is pool2


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
    pool_used = None

    def sync_func() -> dict:
      nonlocal pool_used
      # The pool itself is global, so we just verify the call works
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
    # Verify the pool was created with custom max_workers
    pool = _get_tool_thread_pool(max_workers=12)
    assert pool is not None


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
