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

"""Tests for _event_queue and _enqueue_event on InvocationContext."""

from __future__ import annotations

import asyncio

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
import pytest


async def _create_ic_with_queue() -> InvocationContext:
  """Create a minimal InvocationContext with _event_queue set."""
  agent = LlmAgent(
      name='test_agent',
      model='gemini-2.5-flash',
      instruction='test',
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  ic = InvocationContext(
      invocation_id='test_invocation',
      agent=agent,
      session=session,
      session_service=session_service,
  )
  ic._event_queue = asyncio.Queue()
  return ic


async def _create_ic_without_queue() -> InvocationContext:
  """Create a minimal InvocationContext without _event_queue."""
  agent = LlmAgent(
      name='test_agent',
      model='gemini-2.5-flash',
      instruction='test',
  )
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  return InvocationContext(
      invocation_id='test_invocation',
      agent=agent,
      session=session,
      session_service=session_service,
  )


@pytest.mark.asyncio
async def test_non_partial_event_blocks_until_processed() -> None:
  """A non-partial event should block _enqueue_event until the consumer
  signals processed."""
  ic: InvocationContext = await _create_ic_with_queue()
  event: Event = Event(id=Event.new_id(), author='test')

  completed: bool = False

  async def consumer() -> None:
    nonlocal completed
    ev, processed = await ic._event_queue.get()
    assert ev is event
    assert processed is not None
    processed.set()
    completed = True

  consumer_task: asyncio.Task = asyncio.create_task(consumer())
  await ic._enqueue_event(event)
  await consumer_task

  assert completed, 'Consumer should have processed the event.'


@pytest.mark.asyncio
async def test_partial_event_does_not_block() -> None:
  """A partial event should not block — it returns immediately
  without waiting for a processed signal."""
  ic: InvocationContext = await _create_ic_with_queue()
  event: Event = Event(id=Event.new_id(), author='test', partial=True)

  await ic._enqueue_event(event)

  assert not ic._event_queue.empty()
  ev, processed = await ic._event_queue.get()
  assert ev is event
  assert processed is None, 'Partial events should have no processed signal.'


@pytest.mark.asyncio
async def test_events_arrive_in_order() -> None:
  """Multiple partial events should arrive on the queue in order."""
  ic: InvocationContext = await _create_ic_with_queue()
  events: list[Event] = [
      Event(id=Event.new_id(), author=f'test_{i}', partial=True)
      for i in range(5)
  ]

  for event in events:
    await ic._enqueue_event(event)

  for i in range(5):
    ev, _ = await ic._event_queue.get()
    assert ev.author == f'test_{i}'


@pytest.mark.asyncio
async def test_enqueue_event_raises_when_queue_not_set() -> None:
  """_enqueue_event should raise RuntimeError if _event_queue is None."""
  ic: InvocationContext = await _create_ic_without_queue()
  event: Event = Event(id=Event.new_id(), author='test')

  with pytest.raises(RuntimeError, match='_event_queue is not set'):
    await ic._enqueue_event(event)


@pytest.mark.asyncio
async def test_non_partial_event_waits_for_signal() -> None:
  """Verify that _enqueue_event for a non-partial event actually waits —
  it should not complete before the consumer signals."""
  ic: InvocationContext = await _create_ic_with_queue()
  event: Event = Event(id=Event.new_id(), author='test')

  emit_done: bool = False

  async def emitter() -> None:
    nonlocal emit_done
    await ic._enqueue_event(event)
    emit_done = True

  emit_task: asyncio.Task = asyncio.create_task(emitter())

  # Give the emitter a chance to run.
  await asyncio.sleep(0.01)
  assert not emit_done, '_enqueue_event should still be waiting.'

  # Now consume and signal.
  _, processed = await ic._event_queue.get()
  processed.set()

  await emit_task
  assert emit_done, '_enqueue_event should complete after signal.'
