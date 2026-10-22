# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Delivery and native event correlation for parallel task messaging."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events._internal_metadata import public_event
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow._parallel_task_context import _bind_parallel_task
from google.adk.workflow._parallel_task_context import _deliver_parallel_task_messages
from google.adk.workflow._parallel_task_context import _parallel_task_cancellation_requested
from google.adk.workflow._parallel_task_context import _ParallelTaskControl
from google.adk.workflow._parallel_task_context import _persist_parallel_task_cancelled_tools
from google.adk.workflow._parallel_task_context import _tag_parallel_task_event
from google.adk.workflow._workflow import Workflow
from google.genai import types
import pytest

from . import testing_utils


def _request_text(request: LlmRequest) -> list[str]:
  return [
      part.text
      for content in request.contents
      for part in content.parts or []
      if part.text
  ]


def test_receipts_are_detached_and_pending_messages_finish_undelivered():
  """Finishing preserves accepted receipts without promising delivery."""
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  receipt = control.send_message('First instruction')
  receipt['status'] = 'caller edit'

  control.finish('completed')
  snapshot = control.receipts
  snapshot[0]['status'] = 'another caller edit'

  assert control.message_receipt(receipt['id'])['status'] == 'undelivered'
  with pytest.raises(ValueError, match='no longer accepting'):
    control.send_message('Too late')
  with pytest.raises(ValueError, match='Unknown message receipt'):
    control.message_receipt('another task receipt')


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
@pytest.mark.parametrize('limit', ['max_messages', 'max_message_bytes'])
def test_mailbox_limits_must_be_positive_integers(limit, value):
  """Invalid mailbox limits are rejected before task execution."""
  with pytest.raises(ValueError, match='positive integer'):
    _ParallelTaskControl(
        task_id='task', root_node_path='root/worker', **{limit: value}
    )


def test_mailbox_count_and_utf8_bytes_are_bounded_before_acceptance():
  """Admission counts UTF-8 bytes and retains only accepted receipts."""
  control = _ParallelTaskControl(
      task_id='task',
      root_node_path='root/worker',
      max_messages=1,
      max_message_bytes=4,
  )
  with pytest.raises(ValueError, match='byte limit'):
    control.send_message('ééé')
  receipt = control.send_message('éé')
  with pytest.raises(ValueError, match='message limit'):
    control.send_message('next')

  assert control.receipts == [receipt]


def test_progress_tags_copy_events_and_binding_restores_after_nested_scope():
  """Progress metadata never mutates original or previously tagged events."""
  event = Event(author='worker', custom_metadata={'application': 'kept'})
  outer = _ParallelTaskControl(task_id='outer', root_node_path='root/outer')
  inner = _ParallelTaskControl(
      task_id='inner', root_node_path='root/outer/inner'
  )
  outer.status = 'running'

  with _bind_parallel_task(outer):
    first = _tag_parallel_task_event(event)
    with _bind_parallel_task(inner):
      nested = _tag_parallel_task_event(event)
      assert not _parallel_task_cancellation_requested()
      outer.cancellation_requested = True
      assert _parallel_task_cancellation_requested()
    outer.finish('completed')
    last = _tag_parallel_task_event(event)

  assert event.custom_metadata == {'application': 'kept'}
  assert first is not event
  assert first.custom_metadata['parallel_task'] == {
      'task_id': 'outer',
      'status': 'running',
  }
  assert nested.custom_metadata['parallel_task']['task_id'] == 'inner'
  assert first.custom_metadata['__adk_internal_parallel_task_owners'] == [
      'outer'
  ]
  assert nested.custom_metadata['__adk_internal_parallel_task_owners'] == [
      'outer',
      'inner',
  ]
  public = public_event(nested)
  assert '__adk_internal_parallel_task_owners' not in public.custom_metadata
  assert public.custom_metadata['parallel_task']['task_id'] == 'inner'
  assert last.custom_metadata['__adk_internal_parallel_task_owners'] == [
      'outer'
  ]
  assert last.custom_metadata['parallel_task']['status'] == 'completed'
  assert _tag_parallel_task_event(event) is event
  assert not _parallel_task_cancellation_requested()


async def test_delivery_waits_for_native_persistence_and_keeps_later_messages():
  """Only the accepted batch is acknowledged after native event persistence.

  Setup: A worker pauses while its message event awaits Runner persistence.
  Act: Queue a second message, persist the first event, and finish the worker.
  Assert: The first receipt is delivered; the later receipt is undelivered.
  """
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  first = control.send_message('First instruction')
  request = LlmRequest()

  with _bind_parallel_task(control):
    delivery = asyncio.create_task(
        _deliver_parallel_task_messages(invocation, request)
    )
  event, persisted = await asyncio.wait_for(invocation._event_queue.get(), 2)
  assert request.contents == []
  assert control.message_receipt(first['id'])['status'] == 'queued'
  later = control.send_message('Arrived during persistence')
  await invocation.session_service.append_event(invocation.session, event)
  persisted.set()
  await asyncio.wait_for(delivery, 2)
  control.finish('completed')

  assert _request_text(request) == [
      'Message from the coordinating task:\nFirst instruction'
  ]
  assert control.message_receipt(first['id'])['status'] == 'delivered'
  assert control.message_receipt(later['id'])['status'] == 'undelivered'
  assert event.node_info.path == 'root/worker'
  assert event.custom_metadata['parallel_task_message']['message_ids'] == [
      first['id']
  ]
  assert event in invocation.session.events


async def test_cancel_during_persistence_does_not_claim_delivery():
  """A cancelled persistence wait leaves its message explicitly undelivered."""
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  receipt = control.send_message('Instruction')
  request = LlmRequest()

  with _bind_parallel_task(control):
    delivery = asyncio.create_task(
        _deliver_parallel_task_messages(invocation, request)
    )
  await asyncio.wait_for(invocation._event_queue.get(), 2)
  delivery.cancel()
  with pytest.raises(asyncio.CancelledError):
    await delivery
  control.finish('cancelled')

  assert request.contents == []
  assert control.message_receipt(receipt['id'])['status'] == 'undelivered'


async def test_finish_during_persistence_does_not_resume_delivery():
  """Persisted mail is not falsely acknowledged after its worker has closed."""
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  receipt = control.send_message('Instruction')
  request = LlmRequest()

  with _bind_parallel_task(control):
    delivery = asyncio.create_task(
        _deliver_parallel_task_messages(invocation, request)
    )
  event, persisted = await asyncio.wait_for(invocation._event_queue.get(), 2)
  await invocation.session_service.append_event(invocation.session, event)
  control.finish('cancelled')
  persisted.set()
  await asyncio.wait_for(delivery, 2)

  assert event in invocation.session.events
  assert request.contents == []
  assert control.message_receipt(receipt['id'])['status'] == 'undelivered'


async def test_delivered_receipts_still_count_toward_lifetime_admission_limit():
  """A caller cannot grow retained receipts by draining and refilling inboxes."""
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  control = _ParallelTaskControl(
      task_id='task', root_node_path='root/worker', max_messages=1
  )
  receipt = control.send_message('Instruction')

  with _bind_parallel_task(control):
    delivery = asyncio.create_task(
        _deliver_parallel_task_messages(invocation, LlmRequest())
    )
  _, persisted = await asyncio.wait_for(invocation._event_queue.get(), 2)
  persisted.set()
  await asyncio.wait_for(delivery, 2)

  assert control.message_receipt(receipt['id'])['status'] == 'delivered'
  with pytest.raises(ValueError, match='message limit'):
    control.send_message('Another instruction')


async def test_native_runner_delivers_fifo_only_to_root_and_retains_history():
  """Nested agents cannot consume messages addressed to their task root.

  Setup: The task root pauses in a tool before invoking a nested agent.
  Act: Its coordinator sends two messages, then releases the nested agent.
  Assert: Only later root requests contain both messages in FIFO order. Their
    persisted native events retain task identity, and root progress is live.
  """
  before_nested = asyncio.Event()
  release_nested = asyncio.Event()
  observations: dict[str, Any] = {}
  progress: list[Event] = []
  nested_model = testing_utils.MockModel.create(['nested complete'])
  nested = LlmAgent(name='nested', model=nested_model)

  class Observer(BasePlugin):

    async def on_event_callback(self, *, invocation_context, event):
      progress.append(event.model_copy(deep=True))
      return None

  async def consult(ctx: Context) -> str:
    before_nested.set()
    await release_nested.wait()
    return await ctx.run_node(nested, 'Nested task', use_sub_branch=True)

  def another_step() -> str:
    return 'One more model boundary'

  root_model = testing_utils.MockModel.create([
      types.Part.from_function_call(name='consult', args={}),
      types.Part.from_function_call(name='another_step', args={}),
      'root complete',
  ])
  worker = LlmAgent(
      name='worker', model=root_model, tools=[consult, another_step]
  )

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as group:
      task = group.spawn(worker, 'Original task')
      await asyncio.wait_for(before_nested.wait(), 2)
      observations['live_progress'] = any(
          (event.custom_metadata or {}).get('parallel_task', {}).get('task_id')
          == task.id
          for event in progress
      )
      first = task.send_message('Root-only first instruction')
      second = task.send_message('Root-only second instruction')
      release_nested.set()
      output = await task.result()
      observations['task_id'] = task.id
      observations['receipts'] = [
          task.message_receipt(first['id']),
          task.message_receipt(second['id']),
      ]
      return output

  workflow = Workflow(name='team', edges=[(START, coordinate)])
  runner = testing_utils.InMemoryRunner(
      node=workflow, plugins=[Observer(name='parallel_observer')]
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 5)
  finally:
    await runner.runner.close()

  assert observations['live_progress']
  assert len(nested_model.requests) == 1
  assert not any(
      'Root-only' in text for text in _request_text(nested_model.requests[0])
  )
  assert len(root_model.requests) == 3
  expected = [
      'Message from the coordinating task:\nRoot-only first instruction',
      'Message from the coordinating task:\nRoot-only second instruction',
  ]
  for request in root_model.requests[1:]:
    assert [
        text for text in _request_text(request) if 'Root-only' in text
    ] == expected
  assert [receipt['status'] for receipt in observations['receipts']] == [
      'delivered',
      'delivered',
  ]
  message_events = [
      event
      for event in events
      if 'parallel_task_message' in (event.custom_metadata or {})
  ]
  assert len(message_events) == 1
  assert message_events[0].node_info.path == observations['task_id']


async def test_native_runner_final_model_race_reports_undelivered_message():
  """Messages accepted after the final boundary never silently disappear."""
  entered = asyncio.Event()
  release = asyncio.Event()
  observations: dict[str, Any] = {}

  class GatedModel(testing_utils.MockModel):

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      self.requests.append(llm_request.model_copy(deep=True))
      entered.set()
      await release.wait()
      yield LlmResponse(
          content=types.Content(
              role='model', parts=[types.Part.from_text(text='complete')]
          )
      )

  model = GatedModel(responses=[])
  worker = LlmAgent(name='worker', model=model)

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as group:
      task = group.spawn(worker, 'Task')
      await asyncio.wait_for(entered.wait(), 2)
      receipt = task.send_message('After the final model boundary')
      release.set()
      result = await task.result()
      observations['receipt'] = task.message_receipt(receipt['id'])
      return result

  workflow = Workflow(name='team', edges=[(START, coordinate)])
  runner = testing_utils.InMemoryRunner(node=workflow)
  try:
    await asyncio.wait_for(runner.run_async('start'), 5)
  finally:
    await runner.runner.close()

  assert observations['receipt']['status'] == 'undelivered'
  assert len(model.requests) == 1
  assert not any(
      'After the final' in text for text in _request_text(model.requests[0])
  )


async def test_native_runner_rebuilds_delivered_message_history_after_pause():
  """A resumed native task reads persisted mail without retaining its mailbox.

  Setup: A task receives mail before pausing on a long-running tool.
  Act: Resume the workflow with the tool result, recreating the task scope.
  Assert: The resumed model request contains exactly one copy of the message.
  """
  coordinator_runs = 0
  handles = []

  def await_approval() -> None:
    return None

  model = testing_utils.MockModel.create([
      types.Part.from_function_call(name='await_approval', args={}),
      'Approved output',
  ])
  worker = LlmAgent(
      name='worker',
      model=model,
      tools=[LongRunningFunctionTool(func=await_approval)],
  )

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    nonlocal coordinator_runs
    coordinator_runs += 1
    async with ctx.parallel_tasks() as group:
      task = group.spawn(worker, 'Task', run_id='owned')
      handles.append(task)
      if coordinator_runs == 1:
        task.send_message('Remember this after approval')
      return await task.result()

  workflow = Workflow(name='team', edges=[(START, coordinate)])
  runner = testing_utils.InMemoryRunner(node=workflow)
  try:
    first_events = await asyncio.wait_for(runner.run_async('start'), 5)
    function_calls = [
        part.function_call
        for event in first_events
        if event.content
        for part in event.content.parts or []
        if part.function_call and part.function_call.name == 'await_approval'
    ]
    assert len(function_calls) == 1
    assert handles[0].status == 'waiting'
    assert handles[0].message_receipts[0]['status'] == 'delivered'
    await asyncio.wait_for(
        runner.run_async(
            types.Content(
                role='user',
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name='await_approval',
                            id=function_calls[0].id,
                            response={'approved': True},
                        )
                    )
                ],
            )
        ),
        5,
    )
  finally:
    await runner.runner.close()

  assert coordinator_runs == 2
  assert handles[0] is not handles[1]
  assert handles[1].message_receipts == []
  assert handles[1].status == 'completed'
  assert len(model.requests) == 2
  assert [
      text
      for text in _request_text(model.requests[1])
      if 'Remember this after approval' in text
  ] == ['Message from the coordinating task:\nRemember this after approval']


@pytest.mark.parametrize('boundary', ['model', 'tool'])
@pytest.mark.parametrize('nested', [False, True])
async def test_native_cancellation_closes_generators_while_parent_can_work(
    boundary,
    nested,
):
  """An active agent does not block root work or outlive deliberate cancel."""
  entered = asyncio.Event()
  released = asyncio.Event()
  closed = asyncio.Event()
  observations = {}

  class GatedModel(testing_utils.MockModel):

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
      try:
        entered.set()
        await released.wait()
        yield LlmResponse(
            content=types.Content(
                role='model', parts=[types.Part.from_text(text='complete')]
            )
        )
      finally:
        closed.set()

  async def gated_tool() -> AsyncGenerator[str, None]:
    try:
      entered.set()
      await released.wait()
      yield 'complete'
    finally:
      closed.set()

  worker = LlmAgent(
      name='worker',
      model=GatedModel(responses=[])
      if boundary == 'model'
      else testing_utils.MockModel.create([
          types.Part.from_function_call(name='gated_tool', args={}),
          'complete',
      ]),
      tools=[] if boundary == 'model' else [gated_tool],
  )

  async def parent_work() -> str:
    return 'parent made progress'

  @node(rerun_on_resume=True)
  async def nested_team(ctx: Context) -> str:
    async with ctx.parallel_tasks() as group:
      return await group.spawn(worker, 'Nested task').result()

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as group:
      task = group.spawn(nested_team if nested else worker, 'Task')
      await asyncio.wait_for(entered.wait(), 2)
      observations['parent_output'] = await ctx.run_node(parent_work)
      observations['worker_status_before_cancel'] = task.status
      observations['closed_before_cancel'] = closed.is_set()
      receipt = task.send_message('Queued before cancellation')
      await group.cancel(task)
      observations['closed_after_cancel'] = closed.is_set()
      observations['worker_status_after_cancel'] = task.status
      observations['receipt'] = task.message_receipt(receipt['id'])
      with pytest.raises(asyncio.CancelledError):
        await task.result()
      return 'parent complete'

  runner = testing_utils.InMemoryRunner(
      node=Workflow(name='team', edges=[(START, coordinate)])
  )
  try:
    await asyncio.wait_for(runner.run_async('start'), 5)
    history = runner.session.events
  finally:
    released.set()
    await runner.runner.close()

  assert observations['parent_output'] == 'parent made progress'
  assert observations['worker_status_before_cancel'] == 'running'
  assert not observations['closed_before_cancel']
  assert observations['closed_after_cancel']
  assert observations['worker_status_after_cancel'] == 'cancelled'
  assert observations['receipt']['status'] == 'undelivered'
  if boundary == 'tool':
    responses = [event for event in history if event.get_function_responses()]
    assert len(responses) == 1
    metadata = responses[0].custom_metadata
    owners = metadata['__adk_internal_parallel_task_owners']
    assert len(owners) == (2 if nested else 1)
    assert owners[-1] == metadata['parallel_task']['task_id']


async def test_first_pauses_queued_admission_for_already_completed_candidate():
  """Evaluating an existing result must not admit another queued worker."""
  predicate_entered = asyncio.Event()
  release_predicate = asyncio.Event()
  second_entered = asyncio.Event()
  observations = {}

  async def first_worker() -> str:
    return 'accepted'

  async def second_worker() -> str:
    second_entered.set()
    return 'unnecessary'

  async def parent_work() -> str:
    return 'parent progress while evaluating'

  async def accept(value: str) -> bool:
    predicate_entered.set()
    await release_predicate.wait()
    return value == 'accepted'

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks(max_parallel_workers=1) as group:
      first = group.spawn(first_worker)
      await first.result()
      second = group.spawn(second_worker)

      async def observe_pending_decision() -> None:
        try:
          await asyncio.wait_for(predicate_entered.wait(), 2)
          await ctx.run_node(parent_work)
          observations['second_started'] = second_entered.is_set()
          observations['second_status'] = second.status
        finally:
          release_predicate.set()

      observer = asyncio.create_task(observe_pending_decision())
      winner = await group.first(accept)
      await observer
      return winner.output

  runner = testing_utils.InMemoryRunner(
      node=Workflow(name='team', edges=[(START, coordinate)])
  )
  try:
    await asyncio.wait_for(runner.run_async('start'), 5)
  finally:
    release_predicate.set()
    await runner.runner.close()

  assert not observations['second_started']
  assert observations['second_status'] == 'queued'


async def test_cancelled_tool_batch_preserves_completed_sibling_and_lineage():
  """Cancellation retains completed work and pairs the unfinished sibling."""
  completed = asyncio.Event()
  entered = asyncio.Event()
  release = asyncio.Event()
  closed = asyncio.Event()

  async def completed_tool(ctx: Context) -> dict[str, int]:
    ctx.state['completed_tool_effect'] = 7
    completed.set()
    return {'value': 7}

  async def blocked_tool() -> AsyncGenerator[str, None]:
    try:
      entered.set()
      await release.wait()
      yield 'complete'
    finally:
      closed.set()

  async def parent_work() -> str:
    return 'continue coordinating'

  model = testing_utils.MockModel.create([
      [
          types.Part.from_function_call(name='completed_tool', args={}),
          types.Part.from_function_call(name='blocked_tool', args={}),
      ],
      'Should not call the model again',
  ])
  worker = LlmAgent(
      name='worker', model=model, tools=[completed_tool, blocked_tool]
  )

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as group:
      task = group.spawn(worker, 'Task')
      await asyncio.wait_for(
          asyncio.gather(completed.wait(), entered.wait()), 2
      )
      await ctx.run_node(parent_work)
      await group.cancel(task)
      return 'complete'

  runner = testing_utils.InMemoryRunner(
      node=Workflow(name='team', edges=[(START, coordinate)])
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 5)
  finally:
    release.set()
    await runner.runner.close()

  assert closed.is_set()
  assert len(model.requests) == 1
  calls = {
      call.id: (event, call)
      for event in events
      for call in event.get_function_calls()
  }
  responses = [
      (event, response)
      for event in events
      for response in event.get_function_responses()
  ]
  assert len(calls) == len(responses) == 2
  assert {response.id for _, response in responses} == calls.keys()
  for response_event, response in responses:
    call_event, call = calls[response.id]
    assert response.name == call.name
    assert response_event.author == call_event.author
    assert response_event.branch == call_event.branch
    assert response_event.node_info.path == call_event.node_info.path
    assert response_event.invocation_id == call_event.invocation_id
  values = {response.name: response.response for _, response in responses}
  assert values['completed_tool'] == {'value': 7}
  assert values['blocked_tool']['status'] == 'cancelled'
  assert 'External side effects may already have occurred' in (
      values['blocked_tool']['error']
  )
  assert any(
      event.actions.state_delta.get('completed_tool_effect') == 7
      for event, _ in responses
  )


@pytest.mark.parametrize('response_path', ['root/worker', 'root/other'])
async def test_cancellation_response_deduplicates_only_its_own_node(
    response_path,
):
  """Another node reusing a provider call ID cannot suppress this response."""
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  call_event = Event(
      author='worker',
      invocation_id=invocation.invocation_id,
      node_info=NodeInfo(path=invocation.node_path),
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='finished', name='completed_tool', args={}
                  )
              ),
              types.Part(
                  function_call=types.FunctionCall(
                      id='unfinished', name='blocked_tool', args={}
                  )
              ),
          ],
      ),
  )
  kept_event = Event(
      author='worker',
      invocation_id=invocation.invocation_id,
      node_info=NodeInfo(path=response_path),
      content=types.Content(
          role='user',
          parts=[
              types.Part(
                  function_response=types.FunctionResponse(
                      id='finished',
                      name='completed_tool',
                      response={'value': 7},
                  )
              ),
          ],
      ),
  )
  await invocation.session_service.append_event(invocation.session, kept_event)
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  with _bind_parallel_task(control):
    await _persist_parallel_task_cancelled_tools(
        invocation, call_event, kept_event
    )
    assert invocation._event_queue.empty()
    control.cancellation_requested = True
    cleanup = asyncio.create_task(
        _persist_parallel_task_cancelled_tools(
            invocation, call_event, kept_event
        )
    )
  event, persisted = await asyncio.wait_for(invocation._event_queue.get(), 2)
  persisted.set()
  await asyncio.wait_for(cleanup, 2)

  expected_ids = (
      ['unfinished']
      if response_path == 'root/worker'
      else ['finished', 'unfinished']
  )
  assert [
      response.id for response in event.get_function_responses()
  ] == expected_ids


@pytest.mark.parametrize('close_ancestor', [False, True])
async def test_late_cancellation_cleanup_stops_when_scope_delivery_closes(
    close_ancestor,
):
  """Cleanup cannot wait forever after a timed-out owner's consumer exits."""
  agent = LlmAgent(name='worker', model=testing_utils.MockModel.create(['ok']))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.node_path = 'root/worker'
  invocation._event_queue = asyncio.Queue()
  call_event = Event(
      author='worker',
      invocation_id=invocation.invocation_id,
      content=types.Content(
          role='model',
          parts=[
              types.Part(
                  function_call=types.FunctionCall(
                      id='unfinished', name='blocked_tool', args={}
                  )
              ),
          ],
      ),
  )
  control = _ParallelTaskControl(task_id='task', root_node_path='root/worker')
  inner = _ParallelTaskControl(
      task_id='inner', root_node_path='root/worker/nested'
  )
  control.cancellation_requested = True
  with _bind_parallel_task(control), _bind_parallel_task(inner):
    cleanup = asyncio.create_task(
        _persist_parallel_task_cancelled_tools(invocation, call_event, None)
    )
    await asyncio.wait_for(invocation._event_queue.get(), 2)
    (control if close_ancestor else inner).close_event_delivery()
    await asyncio.wait_for(cleanup, 2)
    await _persist_parallel_task_cancelled_tools(invocation, call_event, None)

  assert invocation._event_queue.empty()


async def test_timeout_seals_mailbox_and_allows_late_tool_cleanup_to_finish():
  """A timed-out scope rejects mail without pretending its worker has stopped."""
  entered = asyncio.Event()
  ignored_cancellation = asyncio.Event()
  release = asyncio.Event()
  closed = asyncio.Event()
  observations = {}

  async def resistant_tool() -> str:
    try:
      entered.set()
      try:
        await release.wait()
      except asyncio.CancelledError:
        ignored_cancellation.set()
        await release.wait()
      return 'Finished external work after cancellation'
    finally:
      closed.set()

  worker = LlmAgent(
      name='worker',
      model=testing_utils.MockModel.create([
          types.Part.from_function_call(name='resistant_tool', args={}),
          'Should not request another model response',
      ]),
      tools=[resistant_tool],
  )

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> None:
    async with ctx.parallel_tasks(cancellation_timeout=0.01) as group:
      task = group.spawn(worker, 'Task')
      observations['task'] = task
      await asyncio.wait_for(entered.wait(), 2)
      observations['receipt'] = task.send_message('Too late to deliver')
      await group.cancel(task)

  # A plain root node isolates the group's bounded ownership contract from a
  # Workflow's independent cleanup of every dynamic task in that workflow.
  runner = testing_utils.InMemoryRunner(node=coordinate)
  try:
    with pytest.raises(TimeoutError, match='did not stop'):
      await asyncio.wait_for(runner.run_async('start'), 2)
    task = observations['task']
    assert ignored_cancellation.is_set()
    assert not closed.is_set()
    assert task.status == 'running'
    assert task.message_receipt(observations['receipt']['id'])['status'] == (
        'undelivered'
    )
    with pytest.raises(ValueError, match='no longer accepting'):
      task.send_message('The owning scope has already closed')
    release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task.result(), 2)
    assert closed.is_set()
    assert task.status == 'cancelled'
  finally:
    release.set()
    await runner.runner.close()
