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

"""Unit tests for _tool_caller."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
import concurrent.futures
import contextvars
from typing import Any
from typing import Callable
from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event_actions import EventActions
from google.adk.flows.llm_flows import functions
from google.adk.flows.llm_flows.tools import _caller as _tool_caller
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.adk.tools.tool_context import ToolContext
from google.genai import types
import pytest

from .... import testing_utils


def test_normalize_tool_result() -> None:
  assert _tool_caller._normalize_tool_result({'foo': 'bar'}) == {'foo': 'bar'}
  assert _tool_caller._normalize_tool_result('hello') == {'result': 'hello'}
  assert _tool_caller._normalize_tool_result(123) == {'result': 123}
  assert _tool_caller._normalize_tool_result([1, 2]) == {'result': [1, 2]}


def test_as_callback_result() -> None:
  assert _tool_caller._as_callback_result({'a': 1}) == {'a': 1}


def test_build_function_response_content() -> None:
  tool = BaseTool(name='my_tool', description='desc')
  content = _tool_caller._build_function_response_content(
      tool=tool,
      function_result={'status': 'ok'},
      function_call_id='call-123',
  )
  assert content.role == 'user'
  assert content.parts is not None
  assert len(content.parts) == 1
  fr = content.parts[0].function_response
  assert fr is not None
  assert fr.name == 'my_tool'
  assert fr.id == 'call-123'
  assert fr.response == {'status': 'ok'}


@pytest.mark.asyncio
async def test_execute_single_prepared_call_runs_tool_runner() -> None:
  tool = BaseTool(name='echo_tool', description='echo')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-1'

  fc = types.FunctionCall(name='echo_tool', id='call-1', args={'val': 42})
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={'val': 42},
      contextvars_snapshot=contextvars.copy_context(),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.run_config = None
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_after_tool_callback.return_value = None

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = []
  agent.canonical_after_tool_callbacks = []

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {'val': 84}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  assert runner_called
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response == {'val': 84}


@pytest.mark.asyncio
async def test_execute_single_prepared_call_lookup_failure() -> None:
  tool = BaseTool(name='missing_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-missing'

  fc = types.FunctionCall(name='missing_tool', id='call-missing')
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
      tools_dict={},
      tool_lookup_error=ValueError('Tool missing_tool not found'),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_on_tool_error_callback.return_value = (
      None
  )

  after_tool_calls: list[str] = []

  def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    after_tool_calls.append(tool.name)

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = []
  agent.canonical_on_tool_error_callbacks = []
  agent.canonical_after_tool_callbacks = [after_tool]

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  # Tool runner must NOT be called on lookup failure
  assert not runner_called
  # Nor the after-tool callbacks: they describe a run that never happened.
  assert not after_tool_calls
  invocation_context.plugin_manager.run_after_tool_callback.assert_not_awaited()
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response is not None
  assert 'missing_tool' in fr.response['error']


@pytest.mark.asyncio
async def test_lookup_failure_answerable_by_before_callback() -> None:
  tool = BaseTool(name='missing_tool', description='desc')
  tool_context = mock.create_autospec(ToolContext, instance=True)
  tool_context.actions = EventActions()
  tool_context.function_call_id = 'call-missing'

  fc = types.FunctionCall(name='missing_tool', id='call-missing')
  prepared = _tool_caller._PreparedFunctionCall(
      function_call=fc,
      tool=tool,
      tool_context=tool_context,
      function_args={},
      contextvars_snapshot=contextvars.copy_context(),
      tools_dict={},
      tool_lookup_error=ValueError('Tool missing_tool not found'),
  )

  invocation_context = mock.create_autospec(InvocationContext, instance=True)
  invocation_context.invocation_id = 'inv-1'
  invocation_context.branch = 'main'
  invocation_context.agent = mock.Mock()
  invocation_context.agent.name = 'test_agent'
  invocation_context.plugin_manager = mock.AsyncMock()
  invocation_context.plugin_manager.run_before_tool_callback.return_value = None
  invocation_context.plugin_manager.run_after_tool_callback.return_value = None

  def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> dict[str, Any]:
    return {'answered': True}

  agent = mock.create_autospec(LlmAgent, instance=True)
  agent.name = 'test_agent'
  agent.canonical_before_tool_callbacks = [before_tool]
  agent.canonical_after_tool_callbacks = []

  runner_called = False

  async def mock_runner() -> dict[str, Any]:
    nonlocal runner_called
    runner_called = True
    return {}

  event = await _tool_caller._execute_single_prepared_call(
      invocation_context,
      prepared,
      agent,
      tool_runner=mock_runner,
  )

  assert not runner_called
  assert event is not None
  assert event.content is not None
  assert event.content.parts is not None
  fr = event.content.parts[0].function_response
  assert fr is not None
  assert fr.response == {'answered': True}


@pytest.mark.asyncio
async def test_tool_callbacks_pair_up_when_nothing_in_the_call_awaits() -> None:
  order: list[str] = []
  bookkeeping: dict[str, Any] = {}
  pairings: list[bool] = []

  def record(value: int) -> dict[str, int]:
    return {'value': value}

  def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    bookkeeping['value'] = args['value']
    order.append(f'before:{args["value"]}')

  def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    pairings.append(bookkeeping['value'] == args['value'])
    order.append(f'after:{args["value"]}')

  agent = LlmAgent(
      name='test_agent',
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  invocation_context = await testing_utils.create_invocation_context(agent)

  await functions.handle_function_call_list_async(
      invocation_context,
      [
          types.FunctionCall(name='record', id='call-1', args={'value': 1}),
          types.FunctionCall(name='record', id='call-2', args={'value': 2}),
      ],
      {'record': FunctionTool(record)},
  )

  assert order == ['before:1', 'after:1', 'before:2', 'after:2']
  assert pairings == [True, True]


@pytest.mark.asyncio
async def test_awaiting_tool_callbacks_keep_their_state_per_call() -> None:
  order: list[str] = []
  pairings: list[bool] = []

  async def record(value: int) -> dict[str, int]:
    order.append(f'tool-start:{value}')
    await asyncio.sleep(0)
    order.append(f'tool-end:{value}')
    return {'value': value}

  async def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    await asyncio.sleep(0)
    tool_context.state['seen'] = args['value']
    order.append(f'before:{args["value"]}')

  async def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    await asyncio.sleep(0)
    pairings.append(tool_context.state['seen'] == args['value'])
    order.append(f'after:{args["value"]}')

  agent = LlmAgent(
      name='test_agent',
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  invocation_context = await testing_utils.create_invocation_context(agent)

  await functions.handle_function_call_list_async(
      invocation_context,
      [
          types.FunctionCall(name='record', id='call-1', args={'value': 1}),
          types.FunctionCall(name='record', id='call-2', args={'value': 2}),
      ],
      {'record': FunctionTool(record)},
  )

  assert pairings == [True, True]
  for value in (1, 2):
    assert (
        order.index(f'before:{value}')
        < order.index(f'tool-start:{value}')
        < order.index(f'after:{value}')
    )
  # The tools still overlap; awaiting callbacks must not serialize the batch.
  assert order.index('tool-start:2') < order.index('tool-end:1')


def _run_with_own_loop(
    coro_fn: Callable[[], Awaitable[None]], *, raise_after: bool
) -> dict[str, Any]:
  """Runs coro_fn on a fresh loop; returns that loop and its tool pool.

  Mirrors a server that calls asyncio.run per request. The returned dict keeps
  the loop referenced after asyncio.run has closed it, standing in for whatever
  holds it in production -- a traceback on a log record, most often.

  Args:
    coro_fn: Awaited on the fresh loop, after its tool pool is acquired.
    raise_after: Whether the coroutine should raise once coro_fn returns, so
      that the run ends the way a failed request does.

  Returns:
    A dict with the run's event loop under 'loop' and the tool pool it
    acquired under 'pool'.
  """
  captured: dict[str, Any] = {}

  async def main() -> None:
    captured['loop'] = asyncio.get_running_loop()
    captured['pool'] = _tool_caller._get_tool_thread_pool()
    await coro_fn()
    if raise_after:
      raise RuntimeError('request failed')

  try:
    asyncio.run(main())
  except RuntimeError:
    pass
  return captured


async def _noop() -> None:
  await asyncio.sleep(0)


def _is_shut_down(pool: concurrent.futures.ThreadPoolExecutor) -> bool:
  """Whether the pool refuses new work, via public API rather than _shutdown."""
  try:
    pool.submit(bool).cancel()
  except RuntimeError:
    return True
  return False


def test_tool_thread_pool_is_released_when_its_loop_closes() -> None:
  """A closed-but-uncollected loop must not keep its tool threads alive."""
  # Hold the result -- and so the loop -- for the whole test, the way a
  # traceback held by a log record would.
  failed = _run_with_own_loop(_noop, raise_after=True)
  stranded_loop = failed['loop']
  stranded_pool = failed['pool']

  assert stranded_loop.is_closed()
  # The weakref finalizer cannot have fired: the loop is still referenced.
  assert stranded_loop in _tool_caller._TOOL_THREAD_POOLS
  assert not _is_shut_down(stranded_pool)

  # A later acquisition sweeps it.
  _run_with_own_loop(_noop, raise_after=False)

  assert stranded_loop not in _tool_caller._TOOL_THREAD_POOLS
  assert _is_shut_down(stranded_pool)


def test_tool_thread_pool_is_reused_within_one_loop() -> None:
  """Sweeping must not disturb the pool of the loop that is still running."""

  async def main() -> None:
    first = _tool_caller._get_tool_thread_pool()
    second = _tool_caller._get_tool_thread_pool()
    assert first is second
    assert not _is_shut_down(first)
    # A different max_workers is a different pool on the same loop.
    assert _tool_caller._get_tool_thread_pool(max_workers=2) is not first
    # Still live after the acquisition that created the second pool swept.
    assert not _is_shut_down(first)

  asyncio.run(main())


def _record(x: int) -> dict[str, int]:
  """A plain tool for exercising the dedupe key and predicate helpers."""
  return {'x': x}


def _prepared_call(
    invocation_context: InvocationContext,
    tool: BaseTool,
    *,
    args: dict[str, Any] | None = None,
    name: str | None = None,
    tool_lookup_error: Exception | None = None,
    tool_confirmation: ToolConfirmation | None = None,
) -> _tool_caller._PreparedFunctionCall:
  """A prepared call of `tool` with real contexts, as the prepare phase builds."""
  function_call = types.FunctionCall(
      name=name or tool.name, id='call-1', args=args or {}
  )
  return _tool_caller._PreparedFunctionCall(
      function_call=function_call,
      tool=tool,
      tool_context=_tool_caller._create_tool_context(
          invocation_context, function_call, tool_confirmation
      ),
      function_args=dict(args or {}),
      contextvars_snapshot=contextvars.copy_context(),
      tool_lookup_error=tool_lookup_error,
  )


async def _dedupe_context() -> InvocationContext:
  return await testing_utils.create_invocation_context(
      LlmAgent(name='test_agent'),
      run_config=RunConfig(dedupe_tool_calls=True),
  )


async def test_dedupe_is_off_unless_the_run_opts_in() -> None:
  """A regular tool is deduped only when RunConfig.dedupe_tool_calls is set."""
  plain = await testing_utils.create_invocation_context(
      LlmAgent(name='test_agent')
  )
  opted_in = await _dedupe_context()
  tool = FunctionTool(_record)

  assert not _tool_caller._should_dedupe_tool_call(
      plain, _prepared_call(plain, tool, args={'x': 1})
  )
  assert _tool_caller._should_dedupe_tool_call(
      opted_in, _prepared_call(opted_in, tool, args={'x': 1})
  )


async def test_long_running_function_tool_is_a_candidate_by_default() -> None:
  """A LongRunningFunctionTool is deduped even when the run did not opt in."""
  plain = await testing_utils.create_invocation_context(
      LlmAgent(name='test_agent')
  )
  tool = LongRunningFunctionTool(func=_record)

  assert _tool_caller._should_dedupe_tool_call(
      plain, _prepared_call(plain, tool, args={'x': 1})
  )


async def test_other_long_running_tools_follow_the_run_setting() -> None:
  """A tool merely flagged long-running, like a wrapped workflow node, needs the opt-in."""
  plain = await testing_utils.create_invocation_context(
      LlmAgent(name='test_agent')
  )
  opted_in = await _dedupe_context()
  tool = BaseTool(
      name='run_node', description='Runs a workflow node.', is_long_running=True
  )

  assert not _tool_caller._should_dedupe_tool_call(
      plain, _prepared_call(plain, tool)
  )
  assert _tool_caller._should_dedupe_tool_call(
      opted_in, _prepared_call(opted_in, tool)
  )


async def test_tool_that_defers_its_response_is_never_deduped() -> None:
  """A tool whose response another orchestrator synthesizes is never deduped."""
  invocation_context = await _dedupe_context()

  class _DeferringTool(BaseTool):

    def __init__(self) -> None:
      super().__init__(name='delegate', description='Runs a sub-agent.')
      self._defers_response = True

  assert not _tool_caller._should_dedupe_tool_call(
      invocation_context, _prepared_call(invocation_context, _DeferringTool())
  )


async def test_call_carrying_a_confirmation_answer_is_never_deduped() -> None:
  """A call re-run with the user's confirmation answer is never deduped."""
  invocation_context = await _dedupe_context()
  tool = FunctionTool(_record, require_confirmation=True)

  assert not _tool_caller._should_dedupe_tool_call(
      invocation_context,
      _prepared_call(
          invocation_context,
          tool,
          args={'x': 1},
          tool_confirmation=ToolConfirmation(confirmed=True),
      ),
  )


async def test_stop_streaming_call_is_never_deduped() -> None:
  """The stop_streaming live control operation is never deduped."""
  invocation_context = await _dedupe_context()

  def stop_streaming(function_name: str) -> None:
    del function_name

  assert not _tool_caller._should_dedupe_tool_call(
      invocation_context,
      _prepared_call(
          invocation_context,
          FunctionTool(stop_streaming),
          args={'function_name': 'monitor'},
      ),
  )


async def test_streaming_tool_is_never_deduped() -> None:
  """A live streaming tool, whose results arrive on the live queue, is never deduped."""
  invocation_context = await _dedupe_context()

  async def monitor(x: int) -> AsyncGenerator[dict[str, int], None]:
    yield {'x': x}

  assert not _tool_caller._should_dedupe_tool_call(
      invocation_context,
      _prepared_call(invocation_context, FunctionTool(monitor), args={'x': 1}),
  )


async def test_unresolved_tool_is_never_deduped() -> None:
  """A call whose tool name resolved to nothing is never deduped."""
  invocation_context = await _dedupe_context()
  tool = BaseTool(name='missing_tool', description='Tool not found')

  assert not _tool_caller._should_dedupe_tool_call(
      invocation_context,
      _prepared_call(
          invocation_context,
          tool,
          tool_lookup_error=ValueError('Tool missing_tool not found'),
      ),
  )


async def test_cache_key_ignores_argument_order() -> None:
  """Calls whose arguments differ only in key order share a cache key."""
  invocation_context = await _dedupe_context()
  tool = FunctionTool(_record)
  first = _prepared_call(
      invocation_context, tool, args={'a': 1, 'b': [1, {'c': 2, 'd': 3}]}
  )
  second = _prepared_call(
      invocation_context, tool, args={'b': [1, {'d': 3, 'c': 2}], 'a': 1}
  )

  assert _tool_caller._tool_call_cache_key(
      invocation_context, first
  ) == _tool_caller._tool_call_cache_key(invocation_context, second)


@pytest.mark.parametrize('other_x', [True, 1.0, '1'])
async def test_cache_key_tells_equal_values_of_different_types_apart(
    other_x: Any,
) -> None:
  """``1`` and a value that compares equal to it are different arguments."""
  invocation_context = await _dedupe_context()
  tool = FunctionTool(_record)
  first = _prepared_call(invocation_context, tool, args={'x': 1})
  second = _prepared_call(invocation_context, tool, args={'x': other_x})

  assert _tool_caller._tool_call_cache_key(
      invocation_context, first
  ) != _tool_caller._tool_call_cache_key(invocation_context, second)


async def test_cache_key_tells_an_empty_dict_from_an_empty_list() -> None:
  """``{}`` and ``[]`` are different arguments although both are empty."""
  invocation_context = await _dedupe_context()
  tool = FunctionTool(_record)
  first = _prepared_call(invocation_context, tool, args={'x': {}})
  second = _prepared_call(invocation_context, tool, args={'x': []})

  assert _tool_caller._tool_call_cache_key(
      invocation_context, first
  ) != _tool_caller._tool_call_cache_key(invocation_context, second)


async def test_cache_key_differs_between_branches() -> None:
  """The same call on two agent branches has two cache keys."""
  invocation_context = await _dedupe_context()
  left = invocation_context.model_copy(update={'branch': 'root.left'})
  right = invocation_context.model_copy(update={'branch': 'root.right'})
  tool = FunctionTool(_record)

  assert _tool_caller._tool_call_cache_key(
      left, _prepared_call(left, tool, args={'x': 1})
  ) != _tool_caller._tool_call_cache_key(
      right, _prepared_call(right, tool, args={'x': 1})
  )


async def test_cache_key_differs_between_agents() -> None:
  """The same call made by two agents on one branch has two cache keys."""
  invocation_context = await _dedupe_context()
  other = invocation_context.model_copy(
      update={'agent': LlmAgent(name='other_agent')}
  )
  tool = FunctionTool(_record)

  assert _tool_caller._tool_call_cache_key(
      invocation_context,
      _prepared_call(invocation_context, tool, args={'x': 1}),
  ) != _tool_caller._tool_call_cache_key(
      other, _prepared_call(other, tool, args={'x': 1})
  )


@pytest.mark.parametrize(
    ('actions', 'shareable'),
    [
        pytest.param(EventActions(), True, id='nothing'),
        pytest.param(
            EventActions(state_delta={'runs': 1}, artifact_delta={'report': 1}),
            True,
            id='deltas',
        ),
        pytest.param(
            EventActions(transfer_to_agent='child'), False, id='transfer'
        ),
        pytest.param(
            EventActions(escalate=True, skip_summarization=True),
            False,
            id='exit_loop',
        ),
        pytest.param(
            EventActions(skip_summarization=True),
            False,
            id='skip_summarization',
        ),
        pytest.param(
            EventActions(
                requested_tool_confirmations={
                    'call-1': ToolConfirmation(hint='Approve?')
                }
            ),
            False,
            id='confirmation',
        ),
    ],
)
def test_result_is_shareable_unless_its_run_acted_beyond_deltas(
    actions: EventActions, shareable: bool
) -> None:
  """A result is shared unless its run recorded an action beyond state and artifact deltas."""
  assert _tool_caller._tool_result_is_shareable(actions) is shareable
