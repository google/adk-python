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

"""Tests for deduping identical tool calls within one invocation.

Verifies that, once a run opts in with ``RunConfig.dedupe_tool_calls`` (or the
tool is a long-running function tool), identical tool calls share a single
tool execution while every call keeps its own callbacks, response event and
function call id.
"""

import asyncio
from typing import Any

from google.adk.agents.llm_agent import Agent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.run_config import RunConfig
from google.adk.events.event import Event
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .... import testing_utils

_CACHE_HIT_KEY = 'adk_tool_call_cache_hit'
_CONFIRMATION_CALL = 'adk_request_confirmation'


def _call(args: dict[str, Any], name: str = 'slow_tool') -> types.Part:
  """A fresh function call part; each call needs its own id, so never reuse one."""
  return types.Part.from_function_call(name=name, args=args)


async def _run(
    runner: testing_utils.InMemoryRunner,
    *,
    dedupe: bool,
    new_message: types.Content | None = None,
) -> list[Event]:
  """Runs one invocation of the runner's agent, opting into deduping or not."""
  events = []
  async for event in runner.runner.run_async(
      user_id=runner.session.user_id,
      session_id=runner.session.id,
      new_message=new_message or testing_utils.get_user_content('run'),
      run_config=RunConfig(dedupe_tool_calls=dedupe),
  ):
    events.append(event)
  return events


def _response_events(events: list[Event]) -> list[Event]:
  """The events that carry at least one function response."""
  return [event for event in events if event.get_function_responses()]


def _responses(events: list[Event]) -> list[types.FunctionResponse]:
  """The function responses of the events, in the order they were emitted."""
  return [
      response
      for event in events
      for response in event.get_function_responses()
  ]


def _call_ids(events: list[Event], name: str) -> list[str]:
  """The ids of the `name` function calls, in emission order."""
  return [
      call.id
      for event in events
      for call in event.get_function_calls()
      if call.name == name and call.id is not None
  ]


def _is_cache_hit(event: Event) -> bool:
  """Whether the event is marked as reusing the result of an earlier call."""
  return bool((event.custom_metadata or {}).get(_CACHE_HIT_KEY))


async def test_identical_call_in_a_later_step_reuses_the_first_result():
  """A call repeated in the next step reuses the result instead of running again."""
  runs = 0

  def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 1
  first, second = _response_events(events)
  assert first.get_function_responses()[0].response == {'result': 1}
  assert second.get_function_responses()[0].response == {'result': 1}
  assert not _is_cache_hit(first)
  assert second.custom_metadata is not None
  assert second.custom_metadata[_CACHE_HIT_KEY] is True


async def test_identical_calls_in_one_step_share_one_execution():
  """Two identical calls in one step run the tool once and each answer their own id."""
  runs = 0

  async def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    # Yields so that the second call finds the first one still running.
    await asyncio.sleep(0.01)
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[[_call({'x': 1}), _call({'x': 1})], 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 1
  call_ids = _call_ids(events, 'slow_tool')
  assert len(set(call_ids)) == 2
  (merged,) = _response_events(events)
  responses = merged.get_function_responses()
  assert [response.response for response in responses] == [
      {'result': 1},
      {'result': 1},
  ]
  assert [response.id for response in responses] == call_ids
  assert merged.custom_metadata is not None
  assert merged.custom_metadata[_CACHE_HIT_KEY] is True


async def test_identical_calls_run_separately_by_default():
  """Without opting in, a repeated call runs the tool again and is not marked."""
  runs = 0

  def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=False)

  assert runs == 2
  assert [response.response for response in _responses(events)] == [
      {'result': 1},
      {'result': 2},
  ]
  assert not any(_is_cache_hit(event) for event in events)


async def test_long_running_tool_is_deduped_without_opting_in():
  """Identical long-running calls in one step run once; both ids stay long-running."""
  runs = 0

  def start_job(x: int) -> dict[str, str]:
    nonlocal runs
    runs += 1
    return {'status': 'pending'}

  model = testing_utils.MockModel.create(
      responses=[
          [_call({'x': 1}, 'start_job'), _call({'x': 1}, 'start_job')],
          'done',
      ]
  )
  agent = Agent(
      name='root_agent',
      model=model,
      tools=[LongRunningFunctionTool(func=start_job)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=False)

  assert runs == 1
  call_ids = _call_ids(events, 'start_job')
  assert len(set(call_ids)) == 2
  call_event = next(event for event in events if event.get_function_calls())
  assert call_event.long_running_tool_ids == set(call_ids)
  assert [response.response for response in _responses(events)] == [
      {'status': 'pending'},
      {'status': 'pending'},
  ]


async def test_calls_with_different_arguments_run_separately():
  """Only calls whose arguments match share an execution."""
  runs = 0

  def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 2}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  assert not any(_is_cache_hit(event) for event in events)


async def test_argument_type_is_part_of_a_call_identity():
  """``{'x': 1}`` and ``{'x': True}`` are different calls although ``1 == True``."""
  runs = 0

  def slow_tool(x: Any) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': True}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  assert not any(_is_cache_hit(event) for event in events)


async def test_argument_order_is_not_part_of_a_call_identity():
  """Calls whose arguments differ only in key order share one execution."""
  runs = 0

  def slow_tool(a: int, b: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  model = testing_utils.MockModel.create(
      responses=[_call({'a': 1, 'b': 2}), _call({'b': 2, 'a': 1}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 1
  assert [_is_cache_hit(event) for event in _response_events(events)] == [
      False,
      True,
  ]


async def test_side_effects_of_a_reused_result_apply_once():
  """The state the tool writes is applied by the first call only."""

  def slow_tool(x: int, tool_context: ToolContext) -> dict[str, int]:
    tool_context.state['runs'] = tool_context.state.get('runs', 0) + 1
    return {'result': tool_context.state['runs']}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runner.session.state['runs'] == 1
  first, second = _response_events(events)
  assert first.actions.state_delta == {'runs': 1}
  assert second.actions.state_delta == {}
  assert second.get_function_responses()[0].response == {'result': 1}


async def test_failed_execution_is_not_reused():
  """A call identical to one whose tool raised runs the tool again."""
  runs = 0

  def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    if runs == 1:
      raise RuntimeError('transient failure')
    return {'result': runs}

  def on_tool_error(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, str]:
    return {'error': str(error)}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(
      name='root_agent',
      model=model,
      tools=[slow_tool],
      on_tool_error_callback=on_tool_error,
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  assert [response.response for response in _responses(events)] == [
      {'error': 'transient failure'},
      {'result': 2},
  ]
  assert not any(_is_cache_hit(event) for event in events)


async def test_failure_while_identical_calls_wait_reaches_each_call_then_retries():
  """A failure is reported to every call waiting on it; the next identical call retries."""
  runs = 0
  failed_call_ids: list[str] = []

  async def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    # Yields so that the second call finds the first one still running.
    await asyncio.sleep(0.01)
    if runs == 1:
      raise RuntimeError('transient failure')
    return {'result': runs}

  def on_tool_error(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      error: Exception,
  ) -> dict[str, str]:
    failed_call_ids.append(tool_context.function_call_id)
    return {'error': str(error)}

  model = testing_utils.MockModel.create(
      responses=[[_call({'x': 1}), _call({'x': 1})], _call({'x': 1}), 'done']
  )
  agent = Agent(
      name='root_agent',
      model=model,
      tools=[slow_tool],
      on_tool_error_callback=on_tool_error,
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  assert [response.response for response in _responses(events)] == [
      {'error': 'transient failure'},
      {'error': 'transient failure'},
      {'result': 2},
  ]
  assert sorted(failed_call_ids) == sorted(_call_ids(events, 'slow_tool')[:2])
  assert not any(_is_cache_hit(event) for event in events)


async def test_callbacks_run_for_every_call_while_the_tool_runs_once():
  """Before- and after-tool callbacks run per call; only the tool run is shared."""
  runs = 0
  before_calls = 0
  after_calls = 0

  def slow_tool(x: int) -> dict[str, Any]:
    nonlocal runs
    runs += 1
    return {'result': runs, 'seen_by': []}

  def before_tool(
      tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
  ) -> None:
    nonlocal before_calls
    before_calls += 1

  def after_tool(
      tool: BaseTool,
      args: dict[str, Any],
      tool_context: ToolContext,
      tool_response: dict[str, Any],
  ) -> None:
    nonlocal after_calls
    after_calls += 1
    # Alters the response in place rather than returning a new one.
    tool_response['seen_by'].append(after_calls)

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(
      name='root_agent',
      model=model,
      tools=[slow_tool],
      before_tool_callback=before_tool,
      after_tool_callback=after_tool,
  )
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 1
  assert before_calls == 2
  assert after_calls == 2
  # Each call's callback sees the result as the tool returned it: neither
  # call's in-place edit reaches the other call's response.
  first, second = _responses(events)
  assert first.response == {'result': 1, 'seen_by': [1]}
  assert second.response == {'result': 1, 'seen_by': [2]}


async def test_result_that_cannot_be_copied_is_shared_as_is():
  """A reused result that cannot be deep-copied is shared rather than failing the call."""
  runs = 0

  class _Handle(dict):
    """A value, such as a live client handle, that refuses to be copied."""

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
      raise TypeError('cannot copy a live handle')

  def slow_tool(x: int) -> dict[str, Any]:
    nonlocal runs
    runs += 1
    return {'result': runs, 'handle': _Handle(connection=runs)}

  model = testing_utils.MockModel.create(
      responses=[_call({'x': 1}), _call({'x': 1}), 'done']
  )
  agent = Agent(name='root_agent', model=model, tools=[slow_tool])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 1
  first, second = _responses(events)
  assert second.response['result'] == 1
  assert second.response['handle'] is first.response['handle']
  assert [_is_cache_hit(event) for event in _response_events(events)] == [
      False,
      True,
  ]


async def test_calls_that_need_confirmation_each_request_their_own():
  """Identical calls held for confirmation each request it, deduped or not.

  The flow answers a call whose tool requires confirmation before the tool
  would run, so deduping never sees such a call; the events must come out the
  same with and without deduping.
  """
  runs = 0

  def guarded(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': x}

  def make_runner() -> testing_utils.InMemoryRunner:
    model = testing_utils.MockModel.create(
        responses=[
            [_call({'x': 1}, 'guarded'), _call({'x': 1}, 'guarded')],
            'done',
        ]
    )
    agent = Agent(
        name='root_agent',
        model=model,
        tools=[FunctionTool(guarded, require_confirmation=True)],
    )
    return testing_utils.InMemoryRunner(agent)

  def shape(events: list[Event]) -> list[tuple[str, list[str]]]:
    """The events as (author, part kinds), ignoring ids and payloads."""
    return [
        (
            event.author,
            [
                f'call:{part.function_call.name}'
                if part.function_call
                else f'response:{part.function_response.name}'
                if part.function_response
                else 'text'
                for part in event.content.parts
            ],
        )
        for event in events
        if event.content and event.content.parts
    ]

  def confirmation_targets(events: list[Event]) -> set[str]:
    return {
        call.args['originalFunctionCall']['id']
        for event in events
        for call in event.get_function_calls()
        if call.name == _CONFIRMATION_CALL and call.args
    }

  deduped = await _run(make_runner(), dedupe=True)
  plain = await _run(make_runner(), dedupe=False)

  assert runs == 0
  call_ids = set(_call_ids(deduped, 'guarded'))
  assert len(call_ids) == 2
  assert confirmation_targets(deduped) == call_ids
  (response_event,) = _response_events(deduped)
  assert set(response_event.actions.requested_tool_confirmations) == call_ids
  assert shape(deduped) == shape(plain)
  assert not any(_is_cache_hit(event) for event in deduped)


async def test_calls_whose_tool_asks_for_confirmation_itself_each_run():
  """A tool that requests confirmation in its own body runs for every identical call.

  Setup: an async tool that yields, then records a confirmation request and
    returns a pending answer, the way a tool that gates itself does; the model
    calls it twice with the same arguments in one step, deduping on.
  Assert: the tool runs twice, since the first run's confirmation request is
    the effect of that call alone; each call id has its own request; nothing
    is marked as reused.
  """
  runs = 0

  async def guarded(x: int, tool_context: ToolContext) -> dict[str, Any]:
    nonlocal runs
    runs += 1
    # Yields so that the second call finds the first one still running.
    await asyncio.sleep(0.01)
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(hint='Approve?')
      tool_context.actions.skip_summarization = True
      return {'error': 'This tool call requires confirmation.'}
    return {'result': x}

  model = testing_utils.MockModel.create(
      responses=[
          [_call({'x': 1}, 'guarded'), _call({'x': 1}, 'guarded')],
          'done',
      ]
  )
  agent = Agent(name='root_agent', model=model, tools=[guarded])
  runner = testing_utils.InMemoryRunner(agent)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  call_ids = set(_call_ids(events, 'guarded'))
  assert len(call_ids) == 2
  (response_event,) = _response_events(events)
  assert set(response_event.actions.requested_tool_confirmations) == call_ids
  assert not any(_is_cache_hit(event) for event in events)


async def test_confirmed_and_rejected_identical_calls_are_answered_apart():
  """Of two identical calls, the confirmed one runs and the rejected one is refused.

  Setup: a confirmation-gated tool called twice with the same arguments.
  Act:
    - Turn 1: both calls request confirmation.
    - Turn 2: the user confirms the first request and rejects the second.
  Assert: the tool runs once, for the confirmed call id; the rejected call id
    is answered with the rejection error.
  """
  runs = 0

  def guarded(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': x}

  model = testing_utils.MockModel.create(
      responses=[
          [_call({'x': 1}, 'guarded'), _call({'x': 1}, 'guarded')],
          'done',
      ]
  )
  agent = Agent(
      name='root_agent',
      model=model,
      tools=[FunctionTool(guarded, require_confirmation=True)],
  )
  runner = testing_utils.InMemoryRunner(agent)

  first_turn = await _run(runner, dedupe=True)
  requests = [
      call
      for event in first_turn
      for call in event.get_function_calls()
      if call.name == _CONFIRMATION_CALL
  ]
  confirmed_id, rejected_id = [
      call.args['originalFunctionCall']['id'] for call in requests
  ]
  answers = types.Content(
      role='user',
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name=_CONFIRMATION_CALL,
                  id=request.id,
                  response={'confirmed': confirmed},
              )
          )
          for request, confirmed in zip(requests, [True, False])
      ],
  )
  second_turn = await _run(runner, dedupe=True, new_message=answers)

  assert runs == 1
  assert {
      response.id: response.response
      for response in _responses(second_turn)
      if response.name == 'guarded'
  } == {
      confirmed_id: {'result': 1},
      rejected_id: {'error': 'This tool call is rejected.'},
  }


async def test_repeating_a_transfer_hands_off_again():
  """A transfer identical to an earlier one still hands control to its target."""

  def transfer(agent_name: str) -> types.Part:
    return _call({'agent_name': agent_name}, 'transfer_to_agent')

  child = Agent(
      name='child',
      model=testing_utils.MockModel.create(
          responses=[transfer('root_agent'), 'child done']
      ),
  )
  root = Agent(
      name='root_agent',
      model=testing_utils.MockModel.create(
          responses=[transfer('child'), transfer('child'), 'root done']
      ),
      sub_agents=[child],
  )
  runner = testing_utils.InMemoryRunner(root)

  events = await _run(runner, dedupe=True)

  assert [
      event.actions.transfer_to_agent for event in _response_events(events)
  ] == ['child', 'root_agent', 'child']
  assert testing_utils.simplify_events(events)[-1] == ('child', 'child done')
  assert not any(_is_cache_hit(event) for event in events)


async def test_same_named_tools_of_two_agents_run_separately():
  """An agent's tool is not answered from another agent's tool of the same name."""

  def agent_with_lookup(
      name: str, responses: list[Any], sub_agents: list[Agent] | None = None
  ) -> Agent:
    def lookup(x: int) -> dict[str, str]:
      return {'source': name}

    return Agent(
        name=name,
        model=testing_utils.MockModel.create(responses=responses),
        tools=[lookup],
        sub_agents=sub_agents or [],
    )

  child = agent_with_lookup('child', [_call({'x': 1}, 'lookup'), 'child done'])
  root = agent_with_lookup(
      'root_agent',
      [
          _call({'x': 1}, 'lookup'),
          _call({'agent_name': 'child'}, 'transfer_to_agent'),
      ],
      sub_agents=[child],
  )
  runner = testing_utils.InMemoryRunner(root)

  events = await _run(runner, dedupe=True)

  assert [
      (event.author, response.response, _is_cache_hit(event))
      for event in _response_events(events)
      for response in event.get_function_responses()
      if response.name == 'lookup'
  ] == [
      ('root_agent', {'source': 'root_agent'}, False),
      ('child', {'source': 'child'}, False),
  ]


async def test_parallel_sub_agents_each_run_the_call_once():
  """Deduping is scoped to an agent branch; sibling branches run the tool themselves."""
  runs = 0

  def slow_tool(x: int) -> dict[str, int]:
    nonlocal runs
    runs += 1
    return {'result': runs}

  def sub_agent(name: str) -> Agent:
    model = testing_utils.MockModel.create(
        responses=[_call({'x': 1}), f'done by {name}']
    )
    return Agent(name=name, model=model, tools=[slow_tool])

  root = ParallelAgent(
      name='root_agent', sub_agents=[sub_agent('left'), sub_agent('right')]
  )
  runner = testing_utils.InMemoryRunner(root)

  events = await _run(runner, dedupe=True)

  assert runs == 2
  assert sorted(
      response.response['result'] for response in _responses(events)
  ) == [1, 2]
  assert not any(_is_cache_hit(event) for event in events)
