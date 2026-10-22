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

"""Execution-local messaging and event correlation for parallel task groups.

The owning group controls task lifetimes. This module does not schedule workers.
A ContextVar carries the ownership lineage into native child execution and is
reset when that execution leaves.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from google.genai import types

from ..events._internal_metadata import INTERNAL_METADATA_PREFIX
from ..platform import uuid as platform_uuid

if TYPE_CHECKING:
  from ..agents.invocation_context import InvocationContext
  from ..events.event import Event
  from ..models.llm_request import LlmRequest


class _ParallelTaskControl:
  """One logical task's bounded mailbox and delivery receipts.

  Queued messages and receipts live only as long as their owning task group.
  Delivered message content is persisted through the native event stream and
  remains part of the addressed node's conversation history. A receipt means
  delivery to the model request, not provider acknowledgement or execution of
  an instruction.
  """

  def __init__(
      self,
      *,
      task_id: str,
      root_node_path: str,
      max_messages: int = 128,
      max_message_bytes: int = 65_536,
  ) -> None:
    for name, value in (
        ('max_messages', max_messages),
        ('max_message_bytes', max_message_bytes),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    if not task_id or not root_node_path:
      raise ValueError('Task ID and root node path must be nonempty')
    self.task_id = task_id
    self.root_node_path = root_node_path
    self.status = 'queued'
    self.cancellation_requested = False
    self._event_delivery_closed = asyncio.Event()
    self._max_messages = max_messages
    self._max_message_bytes = max_message_bytes
    self._messages: deque[tuple[str, str]] = deque()
    self._receipts: dict[str, dict[str, str]] = {}
    self._finished = False

  def send_message(self, content: str) -> dict[str, str]:
    """Queue content for this task's next top-level model boundary.

    Receipt retention counts toward the limit even after delivery. This bounds
    the control's lifetime memory rather than only its current queue length.
    """
    if (
        self._finished
        or not self.event_delivery_open
        or self.status not in ('queued', 'running')
    ):
      raise ValueError('Task is no longer accepting messages')
    if not isinstance(content, str) or not content.strip():
      raise ValueError('Message content must be a nonempty string')
    if len(content.encode('utf-8')) > self._max_message_bytes:
      raise ValueError('Message content exceeds the task message byte limit')
    if len(self._receipts) >= self._max_messages:
      raise ValueError('Task message limit reached')
    message_id = platform_uuid.new_uuid()
    receipt = {
        'id': message_id,
        'task_id': self.task_id,
        'status': 'queued',
    }
    self._messages.append((message_id, content))
    self._receipts[message_id] = receipt
    return dict(receipt)

  def message_receipt(self, message_id: str) -> dict[str, str]:
    """Return a detached delivery receipt for an accepted message."""
    try:
      return dict(self._receipts[message_id])
    except KeyError as error:
      raise ValueError('Unknown message receipt for this task') from error

  @property
  def receipts(self) -> list[dict[str, str]]:
    """Return detached receipts in acceptance order."""
    return [dict(receipt) for receipt in self._receipts.values()]

  def finish(self, status: str) -> None:
    """Close admission and mark messages without another boundary undelivered."""
    if self._finished:
      return
    self._finished = True
    self.status = status
    while self._messages:
      message_id, _ = self._messages.popleft()
      self._receipts[message_id]['status'] = 'undelivered'

  @property
  def event_delivery_open(self) -> bool:
    """Whether the owning scope can still await native event persistence."""
    return not self._event_delivery_closed.is_set()

  def close_event_delivery(self) -> None:
    """Seal the mailbox and stop awaiting events after the owning scope exits.

    Closing delivery does not claim that a cancellation-resistant worker has
    stopped. Its execution status remains unchanged until it actually finishes.
    """
    self._event_delivery_closed.set()
    while self._messages:
      message_id, _ = self._messages.popleft()
      self._receipts[message_id]['status'] = 'undelivered'


_current_parallel_task: ContextVar[_ParallelTaskControl | None] = ContextVar(
    'adk_parallel_task_control', default=None
)
_parallel_task_controls: ContextVar[tuple[_ParallelTaskControl, ...]] = (
    ContextVar('adk_parallel_task_controls', default=())
)


@contextmanager
def _bind_parallel_task(control: _ParallelTaskControl) -> Iterator[None]:
  """Bind an owned control while the existing scheduler runs its child."""
  token = _current_parallel_task.set(control)
  lineage_token = _parallel_task_controls.set(
      (*_parallel_task_controls.get(), control)
  )
  try:
    yield
  finally:
    _parallel_task_controls.reset(lineage_token)
    _current_parallel_task.reset(token)


def _tag_parallel_task_event(event: Event) -> Event:
  """Correlate native progress without mutating a node's cached event."""
  control = _current_parallel_task.get()
  if control is None:
    return event
  return event.model_copy(
      update={
          'custom_metadata': {
              **(event.custom_metadata or {}),
              'parallel_task': {
                  'task_id': control.task_id,
                  'status': control.status,
              },
              INTERNAL_METADATA_PREFIX
              + 'parallel_task_owners': [
                  owner.task_id for owner in _parallel_task_controls.get()
              ],
          }
      }
  )


def _parallel_task_cancellation_requested() -> bool:
  """Whether this task or an ancestor deliberately requested cancellation."""
  return any(
      control.cancellation_requested
      for control in _parallel_task_controls.get()
  )


async def _persist_parallel_task_cancelled_tools(
    invocation_context: InvocationContext,
    function_call_event: Event,
    kept_event: Event | None,
) -> None:
  """Pair outstanding calls when deliberate task cancellation stops a batch.

  Completed siblings keep their real responses and state deltas. A cancellation
  response describes local cancellation, not whether external side effects
  occurred. No synthetic response is persisted for a call already answered.
  """
  if not _parallel_task_cancellation_requested():
    return
  controls = _parallel_task_controls.get()
  if not controls or any(
      not control.event_delivery_open for control in controls
  ):
    return

  from ..events._rewind_events import _apply_rewinds
  from ..events.event import Event
  from ..events.event import NodeInfo
  from ..events.event_actions import EventActions

  call_path = function_call_event.node_info.path or invocation_context.node_path
  answered = {
      response.id
      for event in _apply_rewinds(invocation_context.session.events)
      if event.invocation_id == invocation_context.invocation_id
      and event.node_info.path == call_path
      and event.branch == function_call_event.branch
      and event.author == function_call_event.author
      and event.isolation_scope == function_call_event.isolation_scope
      for response in event.get_function_responses()
  }
  kept = (
      {
          part.function_response.id: part
          for part in kept_event.content.parts or []
          if part.function_response
      }
      if kept_event is not None and kept_event.content is not None
      else {}
  )
  parts = []
  included_kept = False
  for call in function_call_event.get_function_calls():
    if call.id in answered:
      continue
    if call.id in kept:
      parts.append(kept[call.id].model_copy(deep=True))
      included_kept = True
    else:
      parts.append(
          types.Part(
              function_response=types.FunctionResponse(
                  id=call.id,
                  name=call.name,
                  response={
                      'status': 'cancelled',
                      'error': (
                          'The owning parallel task cancelled this local call. '
                          'External side effects may already have occurred.'
                      ),
                  },
              )
          )
      )
  if not parts:
    return
  event = Event(
      author=function_call_event.author,
      invocation_id=invocation_context.invocation_id,
      branch=function_call_event.branch,
      isolation_scope=function_call_event.isolation_scope,
      node_info=NodeInfo(path=call_path),
      content=types.Content(role='user', parts=parts),
      actions=kept_event.actions.model_copy(deep=True)
      if kept_event is not None and included_kept
      else EventActions(),
  )
  delivery = asyncio.create_task(invocation_context._enqueue_event(event))
  closures = [
      asyncio.create_task(control._event_delivery_closed.wait())
      for control in controls
  ]
  try:
    done, _ = await asyncio.wait(
        [delivery, *closures], return_when=asyncio.FIRST_COMPLETED
    )
    if delivery in done:
      await delivery
  finally:
    for waiter in (delivery, *closures):
      if not waiter.done():
        waiter.cancel()
    await asyncio.gather(delivery, *closures, return_exceptions=True)


async def _deliver_parallel_task_messages(
    invocation_context: InvocationContext,
    llm_request: LlmRequest,
) -> None:
  """Persist queued content and add it to the addressed model request.

  Only the task's root node may consume its inbox. Nested agents inherit the
  execution binding for progress correlation, but cannot steal messages meant
  for the root. Queued messages remain queued if persistence is interrupted.
  """
  control = _current_parallel_task.get()
  if (
      control is None
      or control._finished
      or invocation_context.node_path != control.root_node_path
      or not control._messages
  ):
    return

  from ..events.event import Event
  from ..events.event import NodeInfo

  messages = list(control._messages)
  content = types.Content(
      role='user',
      parts=[
          types.Part.from_text(
              text=f'Message from the coordinating task:\n{message}'
          )
          for _, message in messages
      ],
  )
  event = Event(
      author='user',
      invocation_id=invocation_context.invocation_id,
      branch=invocation_context.branch,
      isolation_scope=invocation_context.isolation_scope,
      node_info=NodeInfo(path=control.root_node_path),
      content=content.model_copy(deep=True),
      custom_metadata={
          'parallel_task_message': {
              'task_id': control.task_id,
              'message_ids': [message_id for message_id, _ in messages],
          }
      },
  )
  # This is the worker producer, not an on_event callback. The native queue
  # waits until the Runner has persisted the event before the worker continues.
  await invocation_context._enqueue_event(event)
  if control._finished:
    return
  llm_request.contents.append(content)
  for message_id, _ in messages:
    control._messages.popleft()
    control._receipts[message_id]['status'] = 'delivered'
