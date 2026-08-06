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
from __future__ import annotations

from typing import Optional

from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import Event as A2AEvent
from a2a.server.events.event_queue import EventQueue
from a2a.types import Message
from a2a.types import TaskStatusUpdateEvent

from .. import _compat
from ...events.event import Event
from ..converters.utils import _get_adk_metadata_key as _get_adk_metadata_key
from .config import ExecuteInterceptor
from .executor_context import ExecutorContext


async def _enqueue_canceled_task_event(
    context: RequestContext,
    event_queue: EventQueue,
) -> None:
  """Publishes the terminal event required by the A2A cancellation contract."""
  if not context.task_id:
    raise ValueError('A2A cancellation must have a task ID')

  # ``final`` is load-bearing on a2a-sdk 0.3.x, which keeps consuming the queue
  # until it sees a status update with it set; 1.x has no such field.
  await event_queue.enqueue_event(
      _compat.make_task_status_update_event(
          task_id=context.task_id,
          context_id=context.context_id,
          status=_compat.make_task_status(_compat.TS_CANCELED),
          final=True,
      )
  )


def _require_request_context(
    context: RequestContext,
) -> tuple[Message, str, str]:
  """Return the values guaranteed for an executor-ready A2A request."""
  message = context.message
  if message is None:
    raise ValueError('A2A request must have a message')
  if not context.task_id:
    raise ValueError('A2A request must have a task ID')
  if not context.context_id:
    raise ValueError('A2A request must have a context ID')
  return message, context.task_id, context.context_id


async def execute_before_agent_interceptors(
    context: RequestContext,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> RequestContext:
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.before_agent:
        context = await interceptor.before_agent(context)
  return context


async def execute_after_event_interceptors(
    a2a_event: A2AEvent,
    executor_context: ExecutorContext,
    adk_event: Event,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> list[A2AEvent]:
  events = [a2a_event]
  if execute_interceptors:
    for interceptor in execute_interceptors:
      if interceptor.after_event:
        next_events: list[A2AEvent] = []
        for e in events:
          res = await interceptor.after_event(executor_context, e, adk_event)
          if res is None:
            continue
          if isinstance(res, list):
            next_events.extend(res)
          else:
            next_events.append(res)
        events = next_events
        if not events:
          return []
  return events


async def execute_after_agent_interceptors(
    executor_context: ExecutorContext,
    final_event: TaskStatusUpdateEvent,
    execute_interceptors: Optional[list[ExecuteInterceptor]],
) -> TaskStatusUpdateEvent:
  if execute_interceptors:
    for interceptor in reversed(execute_interceptors):
      if interceptor.after_agent:
        final_event = await interceptor.after_agent(
            executor_context, final_event
        )
  return final_event
