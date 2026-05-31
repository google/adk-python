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

from ..experimental import a2a_experimental


@a2a_experimental
class TaskResultAggregator:
  """Aggregates the task status updates and provides the final task state."""

  def __init__(self):
    self._task_state = TaskState.TASK_STATE_WORKING
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
      if event.status.state == TaskState.TASK_STATE_FAILED:
        self._task_state = TaskState.TASK_STATE_FAILED
        self._task_status_message = (
            event.status.message
            if event.status.HasField('message')
            else None
        )
      elif (
          event.status.state == TaskState.TASK_STATE_AUTH_REQUIRED
          and self._task_state != TaskState.TASK_STATE_FAILED
      ):
        self._task_state = TaskState.TASK_STATE_AUTH_REQUIRED
        self._task_status_message = (
            event.status.message
            if event.status.HasField('message')
            else None
        )
      elif (
          event.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
          and self._task_state
          not in (TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_AUTH_REQUIRED)
      ):
        self._task_state = TaskState.TASK_STATE_INPUT_REQUIRED
        self._task_status_message = (
            event.status.message
            if event.status.HasField('message')
            else None
        )
      # final state is already recorded and make sure the intermediate state is
      # always working because other state may terminate the event aggregation
      # in a2a request handler
      elif self._task_state == TaskState.TASK_STATE_WORKING:
        self._task_status_message = (
            event.status.message
            if event.status.HasField('message')
            else None
        )
      event.status.state = TaskState.TASK_STATE_WORKING

  @property
  def task_state(self) -> TaskState:
    return self._task_state

  @property
  def task_status_message(self) -> Message | None:
    return self._task_status_message
