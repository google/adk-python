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

from a2a.server.events import Event
from a2a.types import Message
from a2a.types import TaskState
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart

from ..experimental import a2a_experimental


@a2a_experimental
class TaskResultAggregator:
  """Aggregates the task status updates and provides the final task state."""

  def __init__(self):
    self._task_state = TaskState.working
    self._task_status_message = None

  def process_event(self, event: Event):
    """Process an event from the agent run and detect signals about the task status.
    Priority of task state:
    - failed
    - auth_required
    - input_required
    - working
    """
    if isinstance(event, TaskStatusUpdateEvent):
      if event.status.state == TaskState.failed:
        self._task_state = TaskState.failed
        self._task_status_message = event.status.message
      elif (
          event.status.state == TaskState.auth_required
          and self._task_state != TaskState.failed
      ):
        self._task_state = TaskState.auth_required
        self._task_status_message = event.status.message
      elif (
          event.status.state == TaskState.input_required
          and self._task_state
          not in (TaskState.failed, TaskState.auth_required)
      ):
        self._task_state = TaskState.input_required
        self._task_status_message = event.status.message
      # final state is already recorded and make sure the intermediate state is
      # always working because other state may terminate the event aggregation
      # in a2a request handler
      elif self._task_state == TaskState.working:
        self._accumulate_message(event.status.message)
      event.status.state = TaskState.working

  def _accumulate_message(self, new_message: Message | None):
    """Accumulate content from a new message into the running result.

    For delta-style streaming, successive TextPart texts are concatenated
    rather than replaced.  Metadata dicts are merged (later values win).
    """
    if new_message is None:
      return

    if self._task_status_message is None:
      self._task_status_message = new_message
      return

    # Accumulate parts
    if new_message.parts:
      if not self._task_status_message.parts:
        self._task_status_message.parts = list(new_message.parts)
      else:
        for new_part in new_message.parts:
          new_root = getattr(new_part, 'root', new_part)
          if isinstance(new_root, TextPart):
            # Concatenate into the last existing TextPart if one exists
            appended = False
            for existing_part in reversed(self._task_status_message.parts):
              existing_root = getattr(existing_part, 'root', existing_part)
              if isinstance(existing_root, TextPart):
                existing_root.text += new_root.text
                appended = True
                break
            if not appended:
              self._task_status_message.parts.append(new_part)
          else:
            self._task_status_message.parts.append(new_part)

    # Merge metadata
    if new_message.metadata:
      if self._task_status_message.metadata is None:
        self._task_status_message.metadata = dict(new_message.metadata)
      else:
        self._task_status_message.metadata.update(new_message.metadata)

  @property
  def task_state(self) -> TaskState:
    return self._task_state

  @property
  def task_status_message(self) -> Message | None:
    return self._task_status_message
